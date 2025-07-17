"""
Data Format Bridge for converting between retrain and Slime data formats.

This module handles the conversion between:
- retrain's RawRolloutData ↔ Slime's Sample objects
- retrain's conversation format ↔ Slime's message format
- retrain's action/observation cycle ↔ Slime's prompt/response format
"""

from typing import List, Dict, Any, Optional, Union
import torch
from dataclasses import dataclass, field
from loguru import logger

# retrain imports
from retrain.reward.types import RawRolloutData, ProcessedTrajectory, SingleTurnDataForTrainer
from retrain.environment.types import LLMAction, EnvironmentObservation

# Define types locally to avoid import issues
from enum import Enum

class SampleStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed" 
    TRUNCATED = "truncated"
    ABORTED = "aborted"

@dataclass
class Sample:
    index: Optional[int] = None
    prompt: Union[str, List[Dict[str, str]]] = ""
    tokens: List[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    label: Optional[str] = None
    reward: Optional[Union[float, Dict[str, float]]] = None
    loss_mask: Optional[List[int]] = None
    status: SampleStatus = SampleStatus.PENDING
    metadata: Dict = field(default_factory=dict)

# Try to import actual Slime types if available
try:
    from slime.utils.types import Sample as SlimeSample  # type: ignore
    # Use Slime's version if available, but keep our local one as fallback
    Sample = SlimeSample
except ImportError:
    pass  # Use our local definition


class DataFormatBridge:
    """
    Handles conversion between retrain and Slime data formats.
    
    This bridge is essential for making retrain's Environment and RewardCalculator
    work with Slime's distributed training system.
    """
    
    def __init__(self, tokenizer: Optional[Any] = None):
        """
        Initialize the data format bridge.
        
        Args:
            tokenizer: Optional tokenizer for token conversion operations
        """
        self.tokenizer = tokenizer
    
    def raw_rollout_to_samples(
        self, 
        raw_rollout: RawRolloutData,
        episode_index: int = 0
    ) -> List[Sample]:
        """
        Convert retrain's RawRolloutData to Slime Sample objects.
        
        Args:
            raw_rollout: retrain's RawRolloutData from Environment.rollout()
            episode_index: Index for this episode/rollout
            
        Returns:
            List of Slime Sample objects, one per LLM action
        """
        try:
            # Extract data from RawRolloutData
            conversation_history = raw_rollout["full_conversation_history"]
            llm_actions = raw_rollout["executed_llm_actions"] 
            intrinsic_rewards = raw_rollout["intrinsic_rewards_per_turn"]
            final_observation = raw_rollout["final_environment_observation"]
            step_infos = raw_rollout["step_info_dicts_per_turn"]
            
            samples = []
            
            # Convert each LLM action to a Sample
            for i, llm_action in enumerate(llm_actions):
                try:
                    # Build prompt from conversation history up to this point
                    prompt_messages = self._build_prompt_for_turn(conversation_history, i)
                    
                    # Extract response from LLM action
                    response = self._extract_response_from_action(llm_action)
                    
                    # Get reward (intrinsic for now, will be augmented later)
                    reward = intrinsic_rewards[i] if i < len(intrinsic_rewards) else 0.0
                    
                    # Get metadata from step info
                    metadata = step_infos[i] if i < len(step_infos) else {}
                    metadata["episode_index"] = episode_index
                    metadata["turn_index"] = i
                    metadata["original_llm_action"] = llm_action
                    
                    # Create Sample object
                    sample = Sample(
                        index=i,  # Turn index within this episode
                        prompt=prompt_messages,  # Use conversation format
                        tokens=[],  # Will be filled by tokenizer later
                        response=response,
                        response_length=0,  # Will be calculated later
                        label=metadata.get("expected_answer"),  # If available
                        reward=reward,
                        loss_mask=None,  # Will be calculated later
                        status=self._determine_sample_status(llm_action, final_observation if i == len(llm_actions) - 1 else None),
                        metadata=metadata
                    )
                    
                    samples.append(sample)
                    
                except Exception as e:
                    logger.error(f"[DataBridge] Error converting turn {i} to Sample: {e}", exc_info=True)
                    continue
            
            logger.debug(f"[DataBridge] Converted RawRolloutData to {len(samples)} Sample objects")
            return samples
            
        except Exception as e:
            logger.error(f"[DataBridge] Error converting RawRolloutData to Samples: {e}", exc_info=True)
            return []
    
    def samples_to_processed_trajectory(
        self, 
        samples: List[Sample]
    ) -> ProcessedTrajectory:
        """
        Convert Slime Sample objects to retrain's ProcessedTrajectory.
        
        Args:
            samples: List of Slime Sample objects
            
        Returns:
            retrain's ProcessedTrajectory for training
        """
        try:
            trajectory = []
            
            for sample in samples:
                try:
                    # Convert prompt (which might be messages) to string
                    prompt_text = self._messages_to_prompt_text(sample.prompt)
                    
                    # Extract completion text
                    completion_text = sample.response or ""
                    
                    # Use reward from sample, ensure it's a float
                    if isinstance(sample.reward, dict):
                        final_reward = sum(sample.reward.values()) if sample.reward else 0.0
                    else:
                        final_reward = float(sample.reward) if sample.reward is not None else 0.0
                    
                    # Build auxiliary data
                    aux_data = dict(sample.metadata) if sample.metadata else {}
                    aux_data["slime_sample_index"] = sample.index
                    aux_data["slime_sample_status"] = sample.status
                    
                    # Create SingleTurnDataForTrainer
                    turn_data = SingleTurnDataForTrainer(
                        prompt_text=prompt_text,
                        completion_text=completion_text,
                        final_combined_reward=final_reward,
                        auxiliary_data=aux_data,
                        old_per_token_logps=None  # May be added later
                    )
                    
                    trajectory.append(turn_data)
                    
                except Exception as e:
                    logger.error(f"[DataBridge] Error converting Sample {sample.index} to turn data: {e}", exc_info=True)
                    continue
            
            logger.debug(f"[DataBridge] Converted {len(samples)} Samples to ProcessedTrajectory with {len(trajectory)} turns")
            return trajectory
            
        except Exception as e:
            logger.error(f"[DataBridge] Error converting Samples to ProcessedTrajectory: {e}", exc_info=True)
            return []
    
    def _build_prompt_for_turn(
        self, 
        conversation_history: List[Dict[str, str]], 
        turn_index: int
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Build the prompt for a specific turn from conversation history.
        
        This recreates the context that was available to the LLM at that turn.
        """
        try:
            # Find the conversation state at this turn
            # We need to reconstruct what the LLM saw when it made this action
            
            # Strategy: Look for the turn where the LLM would have acted
            # This is typically after a "user" or "environment" message
            messages_up_to_turn = []
            
            for i, message in enumerate(conversation_history):
                # Include messages up to the point where this LLM action occurred
                if i <= turn_index * 2:  # Rough heuristic: each turn might involve 2 messages
                    messages_up_to_turn.append(message)
                else:
                    break
            
            # Return as message format (Slime can handle both)
            return messages_up_to_turn
            
        except Exception as e:
            logger.error(f"[DataBridge] Error building prompt for turn {turn_index}: {e}")
            return ""
    
    def _extract_response_from_action(self, llm_action: LLMAction) -> str:
        """Extract the response text from an LLMAction."""
        try:
            if isinstance(llm_action, dict):
                # Standard LLMAction format
                if "raw_llm_output" in llm_action:
                    return llm_action["raw_llm_output"]
                elif "tool_calls" in llm_action:
                    # Handle tool call actions
                    return str(llm_action["tool_calls"])
                else:
                    return str(llm_action)
            else:
                return str(llm_action)
        except Exception as e:
            logger.error(f"[DataBridge] Error extracting response from action: {e}")
            return ""
    
    def _determine_sample_status(
        self, 
        llm_action: LLMAction, 
        observation: Optional[EnvironmentObservation]
    ) -> SampleStatus:
        """Determine the status of a sample based on action and observation."""
        try:
            # Check if the action was truncated
            if observation and observation.get("truncated"):
                return SampleStatus.TRUNCATED
            
            # Check if the episode terminated normally
            if observation and observation.get("terminated"):
                return SampleStatus.COMPLETED
            
            # Default to completed if we have a response
            if llm_action and self._extract_response_from_action(llm_action):
                return SampleStatus.COMPLETED
            
            return SampleStatus.PENDING
            
        except Exception as e:
            logger.error(f"[DataBridge] Error determining sample status: {e}")
            return SampleStatus.COMPLETED  # Safe default
    
    def _messages_to_prompt_text(
        self, 
        prompt: Union[str, List[Dict[str, str]]]
    ) -> str:
        """Convert message format prompt to plain text."""
        try:
            if isinstance(prompt, str):
                return prompt
            
            if isinstance(prompt, list):
                # Convert message list to text
                if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
                    try:
                        return self.tokenizer.apply_chat_template(
                            prompt, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    except Exception as e:
                        logger.debug(f"[DataBridge] Chat template failed, using fallback: {e}")
                
                # Fallback: simple join
                return "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in prompt])
            
            return str(prompt)
            
        except Exception as e:
            logger.error(f"[DataBridge] Error converting messages to prompt text: {e}")
            return str(prompt)
    
    def apply_tokenization(self, samples: List[Sample]) -> List[Sample]:
        """Apply tokenization to samples if tokenizer is available."""
        if not self.tokenizer:
            logger.warning("[DataBridge] No tokenizer available for tokenization")
            return samples
        
        try:
            for sample in samples:
                # Convert prompt to text for tokenization
                prompt_text = self._messages_to_prompt_text(sample.prompt)
                
                # Tokenize prompt + response
                full_text = prompt_text + sample.response
                tokens = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
                
                # Calculate response length
                prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                response_length = len(tokens) - len(prompt_tokens)
                
                # Update sample
                sample.tokens = tokens
                sample.response_length = max(0, response_length)
                
            logger.debug(f"[DataBridge] Applied tokenization to {len(samples)} samples")
            
        except Exception as e:
            logger.error(f"[DataBridge] Error applying tokenization: {e}", exc_info=True)
        
        return samples 