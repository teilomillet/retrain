"""
Environment Bridge for connecting retrain's Environment to Slime's rollout system.

This module creates a bridge that allows retrain's Environment objects to work
with Slime's distributed rollout generation system, handling the conversion between
retrain's step-by-step interaction model and Slime's batch rollout generation.
"""

from typing import List, Dict, Any, Optional, Callable
from loguru import logger

# retrain imports
from retrain.environment.environment import Environment
from retrain.environment.types import LLMAction, EnvironmentObservation
from retrain.reward.types import RawRolloutData

# Local bridge imports
from .data_bridge import DataFormatBridge, Sample


class EnvironmentBridge:
    """
    Bridge between retrain's Environment and Slime's rollout generation.
    
    This bridge allows retrain's Environment objects to be used as data sources
    for Slime's distributed training system, handling the interface conversion
    and data format adaptation.
    """
    
    def __init__(
        self, 
        environment: Environment,
        data_bridge: DataFormatBridge,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize the environment bridge.
        
        Args:
            environment: retrain Environment instance
            data_bridge: DataFormatBridge for format conversion
            tokenizer: Optional tokenizer for text processing
        """
        self.environment = environment
        self.data_bridge = data_bridge
        self.tokenizer = tokenizer
        
        # State tracking
        self.episode_count = 0
        
        logger.debug("[EnvironmentBridge] Initialized with retrain Environment")
    
    async def generate_rollout_batch(
        self, 
        batch_size: int,
        samples_per_prompt: int = 1,
        initial_prompts: Optional[List[str]] = None
    ) -> List[List[Sample]]:
        """
        Generate a batch of rollouts using retrain's Environment.
        
        This is the main interface that Slime's rollout system will call.
        
        Args:
            batch_size: Number of rollouts to generate
            samples_per_prompt: Number of samples per prompt (for multiple attempts)
            initial_prompts: Optional list of initial prompts to use
            
        Returns:
            List of lists of Sample objects (batch_size x samples_per_prompt)
        """
        logger.debug(f"[EnvironmentBridge] Generating {batch_size} rollouts with {samples_per_prompt} samples each")
        
        rollout_groups = []
        
        for rollout_idx in range(batch_size):
            try:
                # Generate samples for this rollout
                rollout_samples = []
                
                for sample_idx in range(samples_per_prompt):
                    # Generate a single rollout episode
                    raw_rollout = await self._generate_single_rollout(
                        episode_index=self.episode_count,
                        initial_prompt=initial_prompts[rollout_idx] if initial_prompts else None
                    )
                    
                    # Convert to Slime Sample objects
                    samples = self.data_bridge.raw_rollout_to_samples(
                        raw_rollout, 
                        episode_index=self.episode_count
                    )
                    
                    # Apply tokenization if available
                    if self.tokenizer:
                        samples = self.data_bridge.apply_tokenization(samples)
                    
                    # For multi-turn environments, we typically want the final turn
                    # For single-turn, we take the only sample
                    if samples:
                        # Use the last sample as the primary one for this attempt
                        primary_sample = samples[-1]
                        primary_sample.index = self.episode_count  # Use episode index as primary index
                        primary_sample.metadata["attempt_index"] = sample_idx
                        primary_sample.metadata["total_turns"] = len(samples)
                        rollout_samples.append(primary_sample)
                    
                    self.episode_count += 1
                
                if rollout_samples:
                    rollout_groups.append(rollout_samples)
                    
            except Exception as e:
                logger.error(f"[EnvironmentBridge] Error generating rollout {rollout_idx}: {e}", exc_info=True)
                continue
        
        logger.debug(f"[EnvironmentBridge] Generated {len(rollout_groups)} rollout groups")
        return rollout_groups
    
    async def _generate_single_rollout(
        self, 
        episode_index: int,
        initial_prompt: Optional[str] = None
    ) -> RawRolloutData:
        """
        Generate a single rollout episode using retrain's Environment.
        
        Args:
            episode_index: Index of this episode
            initial_prompt: Optional initial prompt to use
            
        Returns:
            RawRolloutData from the rollout
        """
        try:
            # Reset environment for new episode
            reset_options = {}
            if initial_prompt:
                reset_options["initial_prompt"] = initial_prompt
            
            observation, info = await self.environment.reset(options=reset_options)
            
            # Initialize rollout data tracking
            conversation_history = []
            executed_actions = []
            intrinsic_rewards = []
            observations = [observation]
            step_infos = [info]
            
            # Add initial state to conversation history
            if observation.get("current_conversation"):
                conversation_history.extend(observation["current_conversation"])
            
            terminated = False
            truncated = False
            step_count = 0
            max_steps = getattr(self.environment, 'max_steps', 20)  # Default limit
            
            # Run the episode
            while not terminated and not truncated and step_count < max_steps:
                try:
                    # Generate LLM action (simulated for now)
                    # In a real implementation, this would call an LLM
                    llm_action = await self._generate_llm_action(observation, step_count)
                    
                    # Execute action in environment
                    observation, reward, terminated, truncated, info = await self.environment.step(llm_action)
                    
                    # Track data
                    executed_actions.append(llm_action)
                    intrinsic_rewards.append(reward)
                    observations.append(observation)
                    step_infos.append(info)
                    
                    # Update conversation history
                    if observation.get("current_conversation"):
                        conversation_history = observation["current_conversation"]
                    
                    step_count += 1
                    
                except Exception as e:
                    logger.error(f"[EnvironmentBridge] Error in step {step_count}: {e}", exc_info=True)
                    break
            
            # Create RawRolloutData using constructor syntax (pattern from env_fastmcp.py and env_smolagent.py)
            # This ensures proper TypedDict compliance for the type checker
            raw_rollout = RawRolloutData(
                full_conversation_history=conversation_history,
                executed_llm_actions=executed_actions,
                intrinsic_rewards_per_turn=intrinsic_rewards,
                final_environment_observation=observations[-1] if observations else EnvironmentObservation(
                    observation_type="environment_state",
                    content="No observations recorded",
                    tool_result=None,
                    current_conversation=[],
                    available_tools=[],
                    requires_llm_action=False
                ),
                step_info_dicts_per_turn=step_infos,
                rollout_metadata={
                    "episode_index": episode_index,
                    "step_count": step_count,
                    "terminated": terminated,
                    "truncated": truncated,
                    "all_observations": observations  # Store all observations in metadata
                }
            )
            
            logger.debug(f"[EnvironmentBridge] Completed rollout {episode_index} with {len(executed_actions)} actions")
            return raw_rollout
            
        except Exception as e:
            logger.error(f"[EnvironmentBridge] Error generating rollout {episode_index}: {e}", exc_info=True)
            # Return empty rollout on error using constructor syntax (matches working environments)
            # Create proper EnvironmentObservation for error state
            error_observation = EnvironmentObservation(
                observation_type="environment_state",
                content=f"Error during rollout: {str(e)}",
                tool_result=None,
                current_conversation=[],
                available_tools=[],
                requires_llm_action=False
            )
            return RawRolloutData(
                full_conversation_history=[],
                executed_llm_actions=[],
                intrinsic_rewards_per_turn=[],
                final_environment_observation=error_observation,
                step_info_dicts_per_turn=[],
                rollout_metadata={"error": str(e)}
            )
    
    async def _generate_llm_action(
        self, 
        observation: EnvironmentObservation, 
        step_count: int
    ) -> LLMAction:
        """
        Generate an LLM action based on the current observation.
        
        NOTE: This is a placeholder implementation. In a real system, this would
        call the actual LLM (via Slime's SGLang server) to generate the response.
        
        Args:
            observation: Current environment observation
            step_count: Current step number
            
        Returns:
            LLMAction representing the LLM's decision
        """
        try:
            # Extract available tools from observation
            available_tools = observation.get("available_tools", [])
            
            # For now, generate a simple placeholder response
            # In real implementation, this would be replaced by Slime's LLM generation
            
            if available_tools and step_count % 3 == 0:  # Occasionally use tools
                # Simulate tool usage using constructor syntax (pattern from env_fastmcp.py and env_smolagent.py)
                tool_name = available_tools[0].get("name", "unknown_tool")
                action = LLMAction(
                    action_type="tool_call",
                    text=None,
                    tool_call={
                        "tool_name": tool_name,
                        "tool_input": {"query": "example query"}
                    },
                    raw_llm_output=f"I'll use the {tool_name} tool to help with this task.",
                    reasoning=None,
                    old_per_token_logps=None
                )
            else:
                # Simulate text response using constructor syntax
                action = LLMAction(
                    action_type="text_response",
                    text=f"This is a simulated response for step {step_count}.",
                    tool_call=None,
                    raw_llm_output=f"This is a simulated response for step {step_count}.",
                    reasoning=None,
                    old_per_token_logps=None
                )
            
            return action
            
        except Exception as e:
            logger.error(f"[EnvironmentBridge] Error generating LLM action: {e}")
            # Return minimal action on error using constructor syntax  
            return LLMAction(
                action_type="text_response",
                text="Error occurred during generation.",
                tool_call=None,
                raw_llm_output="Error occurred during generation.",
                reasoning=None,
                old_per_token_logps=None
            )
    
    def set_llm_generator(self, llm_generator: Callable):
        """
        Set a custom LLM generator function to replace the placeholder.
        
        This allows the bridge to use Slime's actual LLM generation system.
        
        Args:
            llm_generator: Callable that takes (observation, step_count) and returns LLMAction
        """
        self._generate_llm_action = llm_generator
        logger.debug("[EnvironmentBridge] Set custom LLM generator")
    
    async def evaluate_rollout(
        self,
        initial_prompt: str,
        max_steps: Optional[int] = None
    ) -> RawRolloutData:
        """
        Generate a single evaluation rollout.
        
        Args:
            initial_prompt: Prompt to start the evaluation
            max_steps: Optional maximum steps override
            
        Returns:
            RawRolloutData from evaluation rollout
        """
        if max_steps:
            original_max_steps = getattr(self.environment, 'max_steps', 20)
            setattr(self.environment, 'max_steps', max_steps)  # Safe attribute setting
            
        try:
            raw_rollout = await self._generate_single_rollout(
                episode_index=-1,  # Special index for evaluation
                initial_prompt=initial_prompt
            )
            return raw_rollout
            
        finally:
            if max_steps:
                setattr(self.environment, 'max_steps', original_max_steps)  # Restore original value
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the underlying environment."""
        return {
            "environment_type": type(self.environment).__name__,
            "max_steps": getattr(self.environment, 'max_steps', 'unknown'),
            "episode_count": self.episode_count,
            "has_tools": hasattr(self.environment, 'active_tools'),
        } 