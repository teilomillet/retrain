"""
Reward Bridge for connecting retrain's RewardCalculator to Slime's reward system.

This module creates a bridge that allows retrain's sophisticated reward calculation
system to work with Slime's distributed training, handling the conversion between
retrain's step/rollout reward model and Slime's sample-level reward system.
"""

from typing import List, Dict, Any, Optional, Union, Callable
from loguru import logger

# retrain imports
from retrain.reward.calculator import RewardCalculator
from retrain.reward.types import RawRolloutData, ProcessedTrajectory
from retrain.environment.types import LLMAction, EnvironmentObservation

# Local bridge imports
from .data_bridge import DataFormatBridge, Sample


class RewardBridge:
    """
    Bridge between retrain's RewardCalculator and Slime's reward system.
    
    This bridge allows retrain's sophisticated multi-level reward calculation
    to be used within Slime's distributed training framework, converting between
    the different reward paradigms.
    """
    
    def __init__(
        self,
        reward_calculator: RewardCalculator,
        data_bridge: DataFormatBridge
    ):
        """
        Initialize the reward bridge.
        
        Args:
            reward_calculator: retrain's RewardCalculator instance
            data_bridge: DataFormatBridge for format conversion
        """
        self.reward_calculator = reward_calculator
        self.data_bridge = data_bridge
        
        logger.debug("[RewardBridge] Initialized with retrain RewardCalculator")
    
    async def calculate_sample_rewards(
        self, 
        samples: List[Sample],
        raw_rollout_data: Optional[RawRolloutData] = None
    ) -> List[Sample]:
        """
        Calculate rewards for a list of Slime Sample objects using retrain's system.
        
        This is the main interface that converts Slime's sample-based rewards
        to use retrain's sophisticated reward calculation.
        
        Args:
            samples: List of Slime Sample objects to calculate rewards for
            raw_rollout_data: Optional raw rollout data for context
            
        Returns:
            List of Sample objects with updated reward values
        """
        try:
            logger.debug(f"[RewardBridge] Calculating rewards for {len(samples)} samples")
            
            if not samples:
                return samples
            
            # Convert samples back to retrain format if needed
            if raw_rollout_data is None:
                raw_rollout_data = await self._samples_to_raw_rollout(samples)
            
            # Use retrain's RewardCalculator to process the rollout
            processed_trajectories = await self.reward_calculator.process_rollouts([raw_rollout_data])
            
            if not processed_trajectories or not processed_trajectories[0]:
                logger.warning("[RewardBridge] No processed trajectories returned from RewardCalculator")
                return samples
            
            processed_trajectory = processed_trajectories[0]
            
            # Update sample rewards with calculated values
            updated_samples = await self._apply_processed_rewards_to_samples(
                samples, 
                processed_trajectory
            )
            
            logger.debug(f"[RewardBridge] Updated {len(updated_samples)} samples with calculated rewards")
            return updated_samples
            
        except Exception as e:
            logger.error(f"[RewardBridge] Error calculating sample rewards: {e}", exc_info=True)
            # Return original samples on error
            return samples
    
    async def calculate_batch_rewards(
        self,
        sample_groups: List[List[Sample]]
    ) -> List[List[Sample]]:
        """
        Calculate rewards for a batch of sample groups.
        
        Args:
            sample_groups: List of sample groups (batch_size x samples_per_prompt)
            
        Returns:
            List of sample groups with updated rewards
        """
        try:
            logger.debug(f"[RewardBridge] Calculating batch rewards for {len(sample_groups)} groups")
            
            updated_groups = []
            
            for group_idx, sample_group in enumerate(sample_groups):
                try:
                    # Calculate rewards for this group
                    updated_group = await self.calculate_sample_rewards(sample_group)
                    updated_groups.append(updated_group)
                    
                except Exception as e:
                    logger.error(f"[RewardBridge] Error processing group {group_idx}: {e}", exc_info=True)
                    # Keep original group on error
                    updated_groups.append(sample_group)
            
            logger.debug(f"[RewardBridge] Completed batch reward calculation for {len(updated_groups)} groups")
            return updated_groups
            
        except Exception as e:
            logger.error(f"[RewardBridge] Error in batch reward calculation: {e}", exc_info=True)
            return sample_groups
    
    async def _samples_to_raw_rollout(self, samples: List[Sample]) -> RawRolloutData:
        """
        Convert Slime Sample objects back to RawRolloutData for reward calculation.
        
        This reconstructs the rollout data needed by retrain's RewardCalculator.
        """
        try:
            # Reconstruct conversation history and actions from samples
            conversation_history = []
            executed_actions = []
            intrinsic_rewards = []
            step_infos = []
            
            for sample in samples:
                try:
                    # Extract conversation from sample prompt
                    if isinstance(sample.prompt, list):
                        conversation_history.extend(sample.prompt)
                    else:
                        # Convert string prompt to message format
                        conversation_history.append({
                            "role": "user", 
                            "content": str(sample.prompt)
                        })
                    
                    # Add assistant response to conversation
                    if sample.response:
                        conversation_history.append({
                            "role": "assistant",
                            "content": sample.response
                        })
                    
                    # Reconstruct LLM action using constructor syntax for proper TypedDict compliance
                    # This matches the pattern used throughout retrain codebase (env_smolagent.py, env_fastmcp.py, environment_bridge.py)
                    llm_action = LLMAction(
                        action_type="text_response",
                        text=sample.response,
                        tool_call=None,
                        raw_llm_output=sample.response or "",
                        reasoning=None,
                        old_per_token_logps=None
                    )
                    executed_actions.append(llm_action)
                    
                    # Extract intrinsic reward (use 0.0 if reward is complex)
                    if isinstance(sample.reward, (int, float)):
                        intrinsic_rewards.append(float(sample.reward))
                    else:
                        intrinsic_rewards.append(0.0)  # Will be recalculated anyway
                    
                    # Extract step info from metadata
                    step_info = dict(sample.metadata) if sample.metadata else {}
                    step_info["sample_index"] = sample.index
                    step_info["sample_status"] = str(sample.status)
                    step_infos.append(step_info)
                    
                except Exception as e:
                    logger.error(f"[RewardBridge] Error processing sample {sample.index}: {e}")
                    continue
            

            
            # Create RawRolloutData using constructor syntax (matches environment_bridge.py pattern)
            # This ensures proper TypedDict construction for type checker compatibility
            final_obs = EnvironmentObservation(
                observation_type="final_answer_feedback",
                content="Episode completed",
                tool_result=None,
                current_conversation=conversation_history,
                available_tools=[],
                requires_llm_action=False
            )
            # Use constructor syntax pattern from environment_bridge.py for TypedDict compatibility
            raw_rollout_data = RawRolloutData(
                full_conversation_history=conversation_history,
                executed_llm_actions=executed_actions,
                intrinsic_rewards_per_turn=intrinsic_rewards,
                final_environment_observation=final_obs,
                step_info_dicts_per_turn=step_infos,
                rollout_metadata={
                    "source": "slime_samples_conversion",
                    "sample_count": len(samples)
                }
            )
            
            logger.debug(f"[RewardBridge] Converted {len(samples)} samples to RawRolloutData")
            return raw_rollout_data

        except Exception as e:
            logger.error(f"[RewardBridge] Error converting samples to RawRolloutData: {e}", exc_info=True)
            # Use constructor syntax for error case (matches environment_bridge.py pattern)
            error_obs = EnvironmentObservation(
                observation_type="final_answer_feedback",
                content="Error during conversion",
                tool_result=None,
                current_conversation=[],
                available_tools=[],
                requires_llm_action=False
            )
            # Constructor syntax ensures proper TypedDict compliance
            error_rollout_data = RawRolloutData(
                full_conversation_history=[],
                executed_llm_actions=[],
                intrinsic_rewards_per_turn=[],
                final_environment_observation=error_obs,
                step_info_dicts_per_turn=[],
                rollout_metadata={"error": str(e)}
            )
            return error_rollout_data
    
    async def _apply_processed_rewards_to_samples(
        self,
        samples: List[Sample],
        processed_trajectory: ProcessedTrajectory
    ) -> List[Sample]:
        """
        Apply processed reward values back to Slime Sample objects.
        
        This maps the rewards calculated by retrain's system back to the samples.
        """
        try:
            # Match processed trajectory turns to samples
            # Typically there's a 1:1 mapping, but handle mismatches gracefully
            
            for i, sample in enumerate(samples):
                try:
                    if i < len(processed_trajectory):
                        turn_data = processed_trajectory[i]
                        
                        # Use the final combined reward from retrain's calculation
                        sample.reward = turn_data["final_combined_reward"]
                        
                        # Add additional reward information to metadata
                        if sample.metadata is None:
                            sample.metadata = {}
                        
                        sample.metadata.update({
                            "retrain_reward_breakdown": turn_data.get("auxiliary_data", {}),
                            "prompt_text_used": turn_data["prompt_text"],
                            "completion_text_used": turn_data["completion_text"]
                        })
                        
                        # Update logprobs if available
                        if turn_data.get("old_per_token_logps") is not None:
                            sample.metadata["old_per_token_logps"] = turn_data["old_per_token_logps"]
                    
                    else:
                        # More samples than trajectory turns - use last reward or default
                        if processed_trajectory:
                            sample.reward = processed_trajectory[-1]["final_combined_reward"]
                        logger.debug(f"[RewardBridge] Sample {i} beyond trajectory length, using fallback reward")
                
                except Exception as e:
                    logger.error(f"[RewardBridge] Error applying reward to sample {i}: {e}")
                    # Keep original reward on error
                    continue
            
            return samples
            
        except Exception as e:
            logger.error(f"[RewardBridge] Error applying processed rewards: {e}", exc_info=True)
            return samples
    
    def create_slime_reward_function(self) -> Callable:
        """
        Create a Slime-compatible reward function that uses retrain's RewardCalculator.
        
        This can be used as a custom reward function in Slime's training setup.
        
        Returns:
            Async function compatible with Slime's reward system
        """
        async def retrain_reward_function(args, sample: Sample, **kwargs) -> Union[float, Dict[str, float]]:
            """
            Slime-compatible reward function using retrain's RewardCalculator.
            
            Args:
                args: Slime's training arguments
                sample: Slime Sample object
                **kwargs: Additional arguments
                
            Returns:
                Calculated reward value
            """
            try:
                # Convert single sample to list for batch processing
                samples_with_rewards = await self.calculate_sample_rewards([sample])
                
                if samples_with_rewards and samples_with_rewards[0].reward is not None:
                    return samples_with_rewards[0].reward
                else:
                    logger.warning("[RewardBridge] No reward calculated, returning 0.0")
                    return 0.0
                    
            except Exception as e:
                logger.error(f"[RewardBridge] Error in Slime reward function: {e}", exc_info=True)
                return 0.0
        
        return retrain_reward_function
    
    def get_reward_info(self) -> Dict[str, Any]:
        """Get information about the reward system configuration."""
        return {
            "step_reward_processors": len(self.reward_calculator.step_reward_processors),
            "rollout_reward_functions": len(self.reward_calculator.rollout_reward_functions),
            "rollout_reward_distribution": self.reward_calculator.rollout_reward_distribution,
            "has_tokenizer": self.reward_calculator.tokenizer is not None,
        } 