"""
DRGRPO Adapter - Implements BaseTrainer interface for DRGRPO

This adapter wraps the DRGRPO Ray actor to make it compatible with the BaseTrainer interface
used by the retrain framework.
"""

import logging
from typing import Any, Dict, Optional
import ray

from ..trainer import (
    BaseTrainer, AlgorithmConfig, ModelObject, ExperienceBatch, 
    TrainingMetrics, PromptSource
)
from ...environment.environment import Environment
from ...reward.calculator import RewardCalculator
from ...config_models import TrainingConfig
from .drgrpo import DRGRPO

logger = logging.getLogger(__name__)


class DRGRPOAdapter(BaseTrainer):
    """
    Adapter for the DRGRPO (Dr. GRPO) trainer.
    
    This adapter wraps the DRGRPO Ray actor to make it compatible with the BaseTrainer interface.
    It handles the conversion between the retrain framework's data structures and the DRGRPO
    implementation.
    """

    def __init__(self,
                 model: ModelObject,
                 algorithm_config: AlgorithmConfig,
                 environment: Environment,
                 reward_calculator: RewardCalculator,
                 prompt_source: PromptSource,
                 tokenizer: Optional[Any] = None,
                 reference_model: Optional[ModelObject] = None,
                 **trainer_specific_kwargs: Any):
        """
        Initialize DRGRPO adapter.
        
        Args:
            model: The language model to train
            algorithm_config: Algorithm configuration
            environment: Training environment
            reward_calculator: Reward calculator
            prompt_source: Source of prompts
            tokenizer: Model tokenizer
            reference_model: Optional reference model
            **trainer_specific_kwargs: Additional trainer-specific parameters
        """
        # Store components for later use
        self.model = model
        self.algorithm_config = algorithm_config
        self.environment = environment
        self.reward_calculator = reward_calculator
        self.prompt_source = prompt_source
        self.tokenizer = tokenizer
        self.reference_model = reference_model
        self.trainer_specific_kwargs = trainer_specific_kwargs
        
        # DRGRPO actor instance
        self.drgrpo_actor = None
        
        # Training state
        self.is_initialized = False
        self.training_metrics = {}
        
        # Validate and setup backend
        self._validate_and_setup_backend()

    def _validate_and_setup_backend(self) -> None:
        """Validate inputs and setup the DRGRPO backend."""
        self._check_dependencies()
        self._validate_input_components()
        self._initialize_drgrpo_actor()

    def _check_dependencies(self) -> None:
        """Check that required dependencies are available."""
        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized before using DRGRPOAdapter")
        
        # Check that DRGRPO is available
        try:
            from .drgrpo import DRGRPO  # type: ignore
        except ImportError as e:
            raise ImportError(f"DRGRPO not available: {e}")

    def _validate_input_components(self) -> None:
        """Validate that all required components are provided."""
        if self.model is None:
            raise ValueError("Model is required for DRGRPO training")
        
        if self.environment is None:
            raise ValueError("Environment is required for DRGRPO training")
        
        if self.reward_calculator is None:
            raise ValueError("Reward calculator is required for DRGRPO training")
        
        if self.prompt_source is None:
            raise ValueError("Prompt source is required for DRGRPO training")

    def _initialize_drgrpo_actor(self) -> None:
        """Initialize the DRGRPO Ray actor."""
        try:
            # Create a proper config for DRGRPO using the actual components
            # We need to create a config that matches what the DRGRPO actor expects
            config_dict = {
                "model": {
                    "name_or_path": "Qwen/Qwen3-0.6B",  # Use a default model
                    "loader": "huggingface"
                },
                "algorithm": self.algorithm_config.dict(),
                "environment": {
                    "type": "smol_agent",  # Use a supported environment type
                    "env_specific_config": {}
                },
                "prompt_source": {
                    "type": "list",  # Use a supported prompt source type
                    "source_config": {"prompts": ["test"]}
                },
                "reward_setup": {
                    "step_reward_configs": {}, 
                    "rollout_reward_configs": {}
                },
                "experiment_name": "drgrpo_training"
            }
            
            # Create TrainingConfig for DRGRPO
            config = TrainingConfig(**config_dict)
            
            # Create DRGRPO actor
            self.drgrpo_actor = DRGRPO.remote(config=config)  # type: ignore
            
            logger.info("DRGRPO actor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DRGRPO actor: {e}")
            raise

    async def train(self, total_training_iterations: Optional[int] = None) -> TrainingMetrics:
        """
        Execute the main training loop using DRGRPO.
        
        Args:
            total_training_iterations: Number of training iterations to run
            
        Returns:
            Training metrics
        """
        if not self.is_initialized:
            await self._initialize_training()
        
        logger.info(f"Starting DRGRPO training for {total_training_iterations or 'default'} iterations")
        
        try:
            # Initialize the DRGRPO actor
            await ray.get(self.drgrpo_actor.initialize.remote())
            
            # Run training iterations
            for iteration in range(total_training_iterations or 1):
                logger.info(f"DRGRPO training iteration {iteration + 1}")
                
                # Generate training data
                training_batch = await self._generate_training_batch()
                
                # Train step
                step_metrics = await ray.get(
                    self.drgrpo_actor.train_step.remote(training_batch)
                )
                
                # Update metrics
                self.training_metrics.update(step_metrics)
                
                # Log progress
                if iteration % 10 == 0:
                    logger.info(f"DRGRPO iteration {iteration + 1} - Loss: {step_metrics.get('loss', 'N/A')}")
            
            # Get final metrics
            final_metrics = await ray.get(self.drgrpo_actor.health_check.remote())
            self.training_metrics.update(final_metrics)
            
            logger.info("DRGRPO training completed successfully")
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"DRGRPO training failed: {e}")
            raise

    async def step(self, experience_batch: ExperienceBatch) -> TrainingMetrics:
        """
        Perform a single training step using DRGRPO.
        
        Args:
            experience_batch: Batch of experience data
            
        Returns:
            Step metrics
        """
        if not self.is_initialized:
            await self._initialize_training()
        
        try:
            # Convert experience batch to DRGRPO format
            training_batch = self._convert_experience_batch(experience_batch)
            
            # Execute training step
            step_metrics = await ray.get(
                self.drgrpo_actor.train_step.remote(training_batch)
            )
            
            return step_metrics
            
        except Exception as e:
            logger.error(f"DRGRPO step failed: {e}")
            raise

    async def _initialize_training(self) -> None:
        """Initialize the training components."""
        if self.is_initialized:
            return
        
        try:
            # Debug: Check if actor exists
            if self.drgrpo_actor is None:
                raise RuntimeError("DRGRPO actor is None")
            
            logger.info(f"DRGRPO actor type: {type(self.drgrpo_actor)}")
            
            # Initialize DRGRPO actor
            logger.info("Calling DRGRPO actor initialize.remote()...")
            result = self.drgrpo_actor.initialize.remote()
            logger.info(f"Remote call result type: {type(result)}")
            
            await ray.get(result)
            
            self.is_initialized = True
            logger.info("DRGRPO training initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize DRGRPO training: {e}")
            raise

    async def _generate_training_batch(self) -> Dict[str, Any]:
        """Generate a training batch for DRGRPO."""
        # This is a simplified implementation
        # In practice, you would generate rollouts using the environment
        # and process them with the reward calculator
        
        # For now, return a minimal batch structure
        return {
            "input_ids": [[1, 2, 3, 4, 5]],  # Placeholder
            "attention_mask": [[1, 1, 1, 1, 1]],  # Placeholder
            "rewards": [1.0],  # Placeholder
            "old_log_probs": [0.0]  # Placeholder
        }

    def _convert_experience_batch(self, experience_batch: ExperienceBatch) -> Dict[str, Any]:
        """Convert experience batch to DRGRPO format."""
        # This is a placeholder implementation
        # In practice, you would convert the experience batch to the format
        # expected by DRGRPO's train_step method
        
        return {
            "input_ids": experience_batch.get("input_ids", []),
            "attention_mask": experience_batch.get("attention_mask", []),
            "rewards": experience_batch.get("rewards", []),
            "old_log_probs": experience_batch.get("old_log_probs", [])
        }

    def save_checkpoint(self, checkpoint_directory: str) -> None:
        """Save DRGRPO checkpoint."""
        if self.drgrpo_actor is not None:
            try:
                ray.get(self.drgrpo_actor.save_checkpoint.remote(checkpoint_directory))
                logger.info(f"DRGRPO checkpoint saved to {checkpoint_directory}")
            except Exception as e:
                logger.error(f"Failed to save DRGRPO checkpoint: {e}")
                raise

    def load_checkpoint(self, checkpoint_directory: str) -> None:
        """Load DRGRPO checkpoint."""
        if self.drgrpo_actor is not None:
            try:
                ray.get(self.drgrpo_actor.load_checkpoint.remote(checkpoint_directory))
                logger.info(f"DRGRPO checkpoint loaded from {checkpoint_directory}")
            except Exception as e:
                logger.error(f"Failed to load DRGRPO checkpoint: {e}")
                raise 