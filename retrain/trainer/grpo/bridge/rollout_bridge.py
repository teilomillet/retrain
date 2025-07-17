"""
Rollout Bridge for coordinating retrain components with Slime's rollout system.

This module provides the top-level coordination between all bridge components,
implementing Slime's custom rollout generation interface using retrain's
Environment and RewardCalculator systems.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

# retrain imports  
from retrain.environment.environment import Environment
from retrain.reward.calculator import RewardCalculator

# Local bridge imports
from .data_bridge import DataFormatBridge, Sample
from .environment_bridge import EnvironmentBridge
from .reward_bridge import RewardBridge


class RetrainSlimeRolloutBridge:
    """
    Main coordinator for retrain-Slime integration.
    
    This class provides the custom rollout generation function that Slime expects,
    coordinating all the bridge components to use retrain's Environment and
    RewardCalculator within Slime's distributed training framework.
    """
    
    def __init__(
        self,
        environment: Environment,
        reward_calculator: RewardCalculator,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize the complete bridge system.
        
        Args:
            environment: retrain Environment instance
            reward_calculator: retrain RewardCalculator instance  
            tokenizer: Optional tokenizer for text processing
        """
        # Initialize bridge components
        self.data_bridge = DataFormatBridge(tokenizer=tokenizer)
        self.environment_bridge = EnvironmentBridge(
            environment=environment,
            data_bridge=self.data_bridge,
            tokenizer=tokenizer
        )
        self.reward_bridge = RewardBridge(
            reward_calculator=reward_calculator,
            data_bridge=self.data_bridge
        )
        
        # Store components for direct access
        self.environment = environment
        self.reward_calculator = reward_calculator
        self.tokenizer = tokenizer
        
        logger.info("[RolloutBridge] Initialized complete retrain-Slime bridge system")
    
    async def generate_rollout(
        self,
        args,
        rollout_id: int,
        data_buffer,
        evaluation: bool = False
    ) -> List[List[Sample]]:
        """
        Custom rollout generation function for Slime using retrain components.
        
        This is the main function that Slime will call for data generation,
        implementing the interface expected by Slime's training system.
        
        Args:
            args: Slime's training arguments
            rollout_id: Unique identifier for this rollout
            data_buffer: Slime's data buffer for managing samples
            evaluation: Whether this is evaluation mode
            
        Returns:
            List of lists of Sample objects (batch_size x samples_per_prompt)
        """
        try:
            logger.info(f"[RolloutBridge] Starting rollout {rollout_id} (evaluation={evaluation})")
            
            # Extract parameters from Slime args
            batch_size = getattr(args, 'rollout_batch_size', 4)
            samples_per_prompt = getattr(args, 'n_samples_per_prompt', 1)
            
            logger.debug(f"[RolloutBridge] Generating {batch_size} rollouts with {samples_per_prompt} samples each")
            
            if evaluation:
                # Handle evaluation mode
                rollout_groups = await self._generate_evaluation_rollouts(
                    args, rollout_id, batch_size, samples_per_prompt
                )
            else:
                # Handle training mode
                rollout_groups = await self._generate_training_rollouts(
                    args, rollout_id, batch_size, samples_per_prompt, data_buffer
                )
            
            # Apply reward calculation to all generated samples
            logger.debug("[RolloutBridge] Calculating rewards for generated rollouts")
            rollout_groups_with_rewards = await self.reward_bridge.calculate_batch_rewards(rollout_groups)
            
            # Store results in data buffer if provided
            if data_buffer:
                await self._update_data_buffer(data_buffer, rollout_groups_with_rewards)
            
            logger.info(f"[RolloutBridge] Completed rollout {rollout_id} with {len(rollout_groups_with_rewards)} groups")
            return rollout_groups_with_rewards
            
        except Exception as e:
            logger.error(f"[RolloutBridge] Error in rollout {rollout_id}: {e}", exc_info=True)
            # Return empty rollouts on error to prevent training crash
            return [[] for _ in range(getattr(args, 'rollout_batch_size', 4))]
    
    async def _generate_training_rollouts(
        self,
        args,
        rollout_id: int,
        batch_size: int,
        samples_per_prompt: int,
        data_buffer
    ) -> List[List[Sample]]:
        """Generate rollouts for training mode."""
        try:
            # Get initial prompts from data buffer or generate them
            initial_prompts = await self._get_initial_prompts(
                args, data_buffer, batch_size
            )
            
            # Generate rollouts using environment bridge
            rollout_groups = await self.environment_bridge.generate_rollout_batch(
                batch_size=batch_size,
                samples_per_prompt=samples_per_prompt,
                initial_prompts=initial_prompts
            )
            
            return rollout_groups
            
        except Exception as e:
            logger.error(f"[RolloutBridge] Error generating training rollouts: {e}", exc_info=True)
            return []
    
    async def _generate_evaluation_rollouts(
        self,
        args,
        rollout_id: int,
        batch_size: int,
        samples_per_prompt: int
    ) -> List[List[Sample]]:
        """Generate rollouts for evaluation mode."""
        try:
            # For evaluation, use fixed prompts or default prompts
            eval_prompts = getattr(args, 'eval_prompts', None)
            if not eval_prompts:
                eval_prompts = [
                    "Solve this problem step by step:",
                    "What is the answer to this question?",
                    "Help me with this task:",
                    "Explain how to approach this:"
                ]
            
            # Ensure we have enough prompts
            while len(eval_prompts) < batch_size:
                eval_prompts.extend(eval_prompts)
            eval_prompts = eval_prompts[:batch_size]
            
            # Generate evaluation rollouts
            rollout_groups = await self.environment_bridge.generate_rollout_batch(
                batch_size=batch_size,
                samples_per_prompt=samples_per_prompt,
                initial_prompts=eval_prompts
            )
            
            return rollout_groups
            
        except Exception as e:
            logger.error(f"[RolloutBridge] Error generating evaluation rollouts: {e}", exc_info=True)
            return []
    
    async def _get_initial_prompts(
        self,
        args,
        data_buffer,
        batch_size: int
    ) -> List[str]:
        """Get initial prompts for rollout generation."""
        try:
            # Strategy 1: Get prompts from Slime's data buffer
            if data_buffer and hasattr(data_buffer, 'get_samples'):
                try:
                    buffer_samples = data_buffer.get_samples(batch_size)
                    if buffer_samples:
                        prompts = []
                        for sample_group in buffer_samples:
                            # Extract prompt from sample group (usually first sample)
                            if sample_group and len(sample_group) > 0:
                                sample = sample_group[0]
                                prompt_text = self.data_bridge._messages_to_prompt_text(sample.prompt)
                                prompts.append(prompt_text)
                        
                        if len(prompts) == batch_size:
                            logger.debug(f"[RolloutBridge] Got {len(prompts)} prompts from data buffer")
                            return prompts
                except Exception as e:
                    logger.debug(f"[RolloutBridge] Could not get prompts from data buffer: {e}")
            
            # Strategy 2: Use prompts from args
            if hasattr(args, 'prompt_data') and args.prompt_data:
                try:
                    # This would need to be implemented based on how Slime loads prompt data
                    logger.debug("[RolloutBridge] Would load prompts from args.prompt_data")
                except Exception as e:
                    logger.debug(f"[RolloutBridge] Could not load prompts from args: {e}")
            
            # Strategy 3: Use default prompts
            default_prompts = [
                "Help me solve this problem.",
                "What should I do next?", 
                "Please assist with this task.",
                "Can you help me understand this?",
                "What is the solution to this?",
                "How would you approach this?",
                "Please provide guidance on this.",
                "What's the best way to handle this?"
            ]
            
            # Cycle through default prompts to get batch_size prompts
            prompts = []
            for i in range(batch_size):
                prompts.append(default_prompts[i % len(default_prompts)])
            
            logger.debug(f"[RolloutBridge] Using {len(prompts)} default prompts")
            return prompts
            
        except Exception as e:
            logger.error(f"[RolloutBridge] Error getting initial prompts: {e}", exc_info=True)
            # Return minimal prompts on error
            return ["Help me with this task."] * batch_size
    
    async def _update_data_buffer(
        self,
        data_buffer,
        rollout_groups: List[List[Sample]]
    ) -> None:
        """Update Slime's data buffer with generated samples."""
        try:
            if not data_buffer or not hasattr(data_buffer, 'add_samples'):
                return
            
            # Flatten rollout groups to individual samples
            all_samples = []
            for group in rollout_groups:
                all_samples.extend(group)
            
            if all_samples:
                data_buffer.add_samples(all_samples)
                logger.debug(f"[RolloutBridge] Added {len(all_samples)} samples to data buffer")
            
        except Exception as e:
            logger.error(f"[RolloutBridge] Error updating data buffer: {e}", exc_info=True)
    
    def create_slime_rollout_function(self):
        """
        Create a function compatible with Slime's custom rollout interface.
        
        Returns:
            Function that can be used as --custom-generate-function-path in Slime
        """
        async def retrain_rollout_function(args, rollout_id, data_buffer, evaluation=False):
            """
            Slime-compatible rollout function using retrain components.
            
            This function signature matches what Slime expects for custom rollout generation.
            """
            return await self.generate_rollout(args, rollout_id, data_buffer, evaluation)
        
        return retrain_rollout_function
    
    def create_slime_reward_function(self):
        """Create a Slime-compatible reward function."""
        return self.reward_bridge.create_slime_reward_function()
    
    def set_llm_generator(self, llm_generator):
        """Set custom LLM generator for the environment bridge."""
        self.environment_bridge.set_llm_generator(llm_generator)
    
    def get_bridge_info(self) -> Dict[str, Any]:
        """Get information about all bridge components."""
        return {
            "environment_info": self.environment_bridge.get_environment_info(),
            "reward_info": self.reward_bridge.get_reward_info(),
            "has_tokenizer": self.tokenizer is not None,
            "bridge_components": [
                "DataFormatBridge",
                "EnvironmentBridge", 
                "RewardBridge"
            ]
        }


# Convenience function for creating the complete bridge system
def create_retrain_slime_bridge(
    environment: Environment,
    reward_calculator: RewardCalculator,
    tokenizer: Optional[Any] = None
) -> RetrainSlimeRolloutBridge:
    """
    Create a complete retrain-Slime bridge system.
    
    Args:
        environment: retrain Environment instance
        reward_calculator: retrain RewardCalculator instance
        tokenizer: Optional tokenizer
        
    Returns:
        Complete bridge system ready for use with Slime
    """
    bridge = RetrainSlimeRolloutBridge(
        environment=environment,
        reward_calculator=reward_calculator,
        tokenizer=tokenizer
    )
    
    logger.info("[Bridge] Created complete retrain-Slime bridge system")
    return bridge 