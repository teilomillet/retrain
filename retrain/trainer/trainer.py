import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
import ray
import torch

from ..config_models import TrainingConfig

logger = logging.getLogger(__name__)

# Type aliases for clarity
AlgorithmConfig = Any
ModelObject = Any
ExperienceBatch = Any
TrainingMetrics = Dict[str, Any]
PromptSource = Any
RewardFunction = Callable[[List[str], List[str], Dict[str, Any]], List[float]]


class BaseTrainer(ABC):
    """
    Abstract Base for all training backends.

    This class defines a standard way for the main system (Orchestrator)
    to interact with different Reinforcement Learning algorithms. It aims to
    hide the specific details of each underlying library (like TRL).
    """

    def __init__(self,
                 model: ModelObject,
                 algorithm_config: AlgorithmConfig,
                 reward_functions: List[RewardFunction],
                 prompt_source: Optional[PromptSource] = None,
                 tokenizer: Optional[Any] = None,
                 reference_model: Optional[ModelObject] = None,
                 **trainer_specific_kwargs: Any):
        """
        Initializes the training backend.

        This is where you provide all the static components and configurations
        needed for the trainer to be set up.

        Args:
            model: The core AI model (e.g., a language model) to be trained.
            algorithm_config: Specific settings for the learning algorithm itself
                              (e.g., learning rate, batch size for the algorithm).
            reward_functions: A list of functions that will score the model's
                              outputs. This is crucial for RL.
            prompt_source: Optional. Where the trainer should get its initial
                           prompts or questions if it manages its own data generation
                           (e.g., a Hugging Face Dataset, a list of strings).
            tokenizer: Optional. The tokenizer associated with the model, often needed
                       by backend libraries.
            reference_model: Optional. A baseline version of the model, used by some
                             algorithms (like PPO or GRPO) for stabilization.
            **trainer_specific_kwargs: For any other settings that are highly specific
                                       to one particular type of trainer backend and
                                       don't fit the common parameters.
        """
        self.model = model
        self.algorithm_config = algorithm_config
        self.reward_functions = reward_functions
        self.prompt_source = prompt_source
        self.tokenizer = tokenizer
        self.reference_model = reference_model
        self.trainer_specific_kwargs = trainer_specific_kwargs

        # This internal method is called to validate inputs and
        # perform the actual setup of the underlying trainer library.
        self._validate_and_setup_backend()

    @abstractmethod
    def _validate_and_setup_backend(self) -> None:
        """
        Internal method: Validates all provided components and configurations.
        It then performs the actual setup of the specific underlying training
        engine (e.g., initializes TRL's GRPOTrainer).

        This should raise appropriate errors (e.g., TypeError, ValueError)
        if the configuration or components are unsuitable for this backend.
        This method is called automatically at the end of `__init__`.
        """
        pass

    @abstractmethod
    async def train(self, total_training_iterations: Optional[int] = None) -> TrainingMetrics:
        """
        Starts and executes the main training process asynchronously.

        The trainer might run its own internal loop (generating data, calculating
        rewards, and updating the model) or it might expect data to be fed
        if designed differently (though `step` is more for that).

        Args:
            total_training_iterations: Optional. An indication of how long or how many
                                      cycles the training should run. The specific
                                      meaning (e.g., epochs, steps, episodes)
                                      can be interpreted by the backend.

        Returns:
            A dictionary containing a summary of training results or final metrics.
        """
        pass

    @abstractmethod
    async def step(self, experience_batch: ExperienceBatch) -> TrainingMetrics:
        """
        Performs a single learning update using a provided batch of experience asynchronously.

        This method is primarily for scenarios where the main system (Orchestrator)
        is controlling the experience generation loop and feeds data to the trainer
        incrementally.

        Note: Not all backend libraries or algorithms are easily adapted to an
              external step-by-step control if they are designed for a monolithic
              `train()` call. In such cases, this method might raise a
              `NotImplementedError` or have limited functionality.

        Args:
            experience_batch: A package of data from recent model interactions,
                              typically including prompts, the model's responses,
                              and the scores (rewards) it received.

        Returns:
            A dictionary containing metrics specific to this single learning step.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint_directory: str) -> None:
        """Saves the current state of the trainer and model."""
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_directory: str) -> None:
        """Loads the trainer and model state from a checkpoint."""
        pass


@ray.remote
class ReTrainer:
    """
    Distributed Training Actor for Retrain.
    
    ReTrainer is a Ray actor that coordinates algorithm-specific training.
    It spawns algorithm-specific actors (GRPO, PPO, etc.) and manages:
    1. Training step coordination through DataBuffer
    2. Model weight management and updates
    3. Checkpointing and recovery
    4. Health monitoring and metrics collection
    
    This follows the Ray-first architecture where the trainer group
    delegates to specialized algorithm actors based on configuration.
    """
    
    def __init__(self, config: TrainingConfig, databuffer: ray.ObjectRef):
        """
        Initialize ReTrainer with configuration and databuffer reference.
        
        Args:
            config: Complete training configuration
            databuffer: Reference to the ReDataBuffer actor for data coordination
        """
        self.config = config
        self.databuffer = databuffer
        self.algorithm_actor = None
        self.current_weights = None
        
        # Training state
        self.training_step_count = 0
        self.training_metrics = {}
        self.is_initialized = False
        
        # Performance tracking
        self.step_times = []
        self.memory_usage = []
        
        logger.info(f"ReTrainer initialized for algorithm: {config.algorithm.name}")
        
    async def initialize(self) -> None:
        """
        Initialize the trainer and spawn algorithm-specific actors.
        """
        logger.info("Initializing ReTrainer...")
        
        # Create algorithm-specific actor based on configuration
        algorithm_name = self.config.algorithm.name.lower()
        
        if algorithm_name == "grpo":
            # Import and create hardware-appropriate GRPO actor
            from .grpo.grpo import create_grpo_actor
            self.algorithm_actor = create_grpo_actor(self.config, self.databuffer)
            
        elif algorithm_name == "rloo":
            # Import and create RLOO actor
            from .rloo.rloo import RLOOActor  # type: ignore
            self.algorithm_actor = RLOOActor.options(
                num_gpus=1,
                num_cpus=2
            ).remote(
                self.config,
                self.databuffer
            )
            
        elif algorithm_name == "ppo":
            # Import and create PPO actor (future implementation)
            from .ppo.ppo import PPOActor  # type: ignore
            self.algorithm_actor = PPOActor.options(
                num_gpus=1,
                num_cpus=2
            ).remote(
                self.config,
                self.databuffer
            )
            
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")
        
        # Initialize the algorithm actor
        await self.algorithm_actor.initialize.remote()  # type: ignore
        
        # Get initial model weights
        self.current_weights = await self.algorithm_actor.get_model_weights.remote()  # type: ignore
        
        self.is_initialized = True
        logger.info(f"ReTrainer initialization complete for {algorithm_name}")
        
    async def train_step(self, training_batch: Dict[str, Any], episode_id: int) -> TrainingMetrics:
        """
        Execute a single training step using the algorithm-specific actor.
        
        Args:
            training_batch: Processed training data from DataBuffer
            episode_id: Current episode identifier
            
        Returns:
            Training metrics from this step
        """
        if not self.is_initialized:
            raise RuntimeError("ReTrainer not initialized. Call initialize() first.")
            
        step_start_time = time.time()
        
        logger.info(f"Executing training step for episode {episode_id}")
        
        # Execute training step through algorithm actor
        step_metrics = await self.algorithm_actor.train_step.remote(  # type: ignore
            training_batch=training_batch,
            episode_id=episode_id
        )
        
        # Update training state
        self.training_step_count += 1
        step_time = time.time() - step_start_time
        self.step_times.append(step_time)
        
        # Get updated model weights
        self.current_weights = await self.algorithm_actor.get_model_weights.remote()  # type: ignore
        
        # Enhance metrics with trainer-level information
        enhanced_metrics = {
            **step_metrics,
            'trainer_step_count': self.training_step_count,
            'step_time': step_time,
            'episode_id': episode_id,
            'algorithm': self.config.algorithm.name,
            'timestamp': time.time()
        }
        
        # Store metrics for health monitoring
        self.training_metrics[f'step_{self.training_step_count}'] = enhanced_metrics
        
        # Keep only recent metrics to prevent memory growth
        if len(self.training_metrics) > 100:
            oldest_key = min(self.training_metrics.keys())
            del self.training_metrics[oldest_key]
            
        logger.info(f"Training step {self.training_step_count} completed in {step_time:.2f}s")
        return enhanced_metrics
        
    async def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get current model weights for inference actors.
        
        Returns:
            Current model state dict
        """
        if not self.is_initialized:
            raise RuntimeError("ReTrainer not initialized")
            
        if self.current_weights is None:
            self.current_weights = await self.algorithm_actor.get_model_weights.remote()  # type: ignore
            
        return self.current_weights
        
    async def update_inference_weights(self, inference_actor: ray.ObjectRef) -> None:
        """
        Update weights in an inference actor.
        
        Args:
            inference_actor: Reference to inference actor to update
        """
        if not self.is_initialized:
            raise RuntimeError("ReTrainer not initialized")
            
        current_weights = await self.get_model_weights()
        await inference_actor.update_model_weights.remote(current_weights)  # type: ignore
        
        logger.info("Model weights updated in inference actor")
        
    async def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save trainer and algorithm state to checkpoint.
        
        Args:
            checkpoint_path: Directory path for checkpoint
        """
        if not self.is_initialized:
            raise RuntimeError("ReTrainer not initialized")
            
        logger.info(f"Saving ReTrainer checkpoint to {checkpoint_path}")
        
        # Save algorithm-specific state
        await self.algorithm_actor.save_checkpoint.remote(checkpoint_path)  # type: ignore
        
        # Save trainer state
        trainer_state = {
            'training_step_count': self.training_step_count,
            'training_metrics': self.training_metrics,
            'algorithm_name': self.config.algorithm.name,
            'step_times': self.step_times[-50:],  # Keep recent step times
            'config': self.config.dict()
        }
        
        import pickle
        trainer_checkpoint = f"{checkpoint_path}/retrainer_state.pkl"
        with open(trainer_checkpoint, 'wb') as f:
            pickle.dump(trainer_state, f)
            
        logger.info(f"ReTrainer checkpoint saved to {trainer_checkpoint}")
        
    async def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load trainer and algorithm state from checkpoint.
        
        Args:
            checkpoint_path: Directory path for checkpoint
        """
        logger.info(f"Loading ReTrainer checkpoint from {checkpoint_path}")
        
        # Load algorithm-specific state
        if self.algorithm_actor:
            await self.algorithm_actor.load_checkpoint.remote(checkpoint_path)  # type: ignore
            
        # Load trainer state
        import pickle
        trainer_checkpoint = f"{checkpoint_path}/retrainer_state.pkl"
        
        try:
            with open(trainer_checkpoint, 'rb') as f:
                trainer_state = pickle.load(f)
                
            self.training_step_count = trainer_state.get('training_step_count', 0)
            self.training_metrics = trainer_state.get('training_metrics', {})
            self.step_times = trainer_state.get('step_times', [])
            
            # Update current weights after loading
            if self.algorithm_actor:
                self.current_weights = await self.algorithm_actor.get_model_weights.remote()  # type: ignore
                
            logger.info("ReTrainer checkpoint loaded successfully")
            
        except FileNotFoundError:
            logger.warning(f"Trainer checkpoint not found: {trainer_checkpoint}")
            
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.
        
        Returns:
            Health status and performance metrics
        """
        health_status = {
            'status': 'healthy',
            'initialized': self.is_initialized,
            'training_step_count': self.training_step_count,
            'algorithm': self.config.algorithm.name if self.config else 'unknown',
            'timestamp': time.time()
        }
        
        # Add performance metrics
        if self.step_times:
            recent_times = self.step_times[-10:]  # Last 10 steps
            health_status.update({
                'avg_step_time': sum(recent_times) / len(recent_times),
                'min_step_time': min(recent_times),
                'max_step_time': max(recent_times),
                'total_steps': len(self.step_times)
            })
            
        # Check algorithm actor health
        if self.algorithm_actor and self.is_initialized:
            try:
                algorithm_health = await self.algorithm_actor.health_check.remote()  # type: ignore
                health_status['algorithm_health'] = algorithm_health
            except Exception as e:
                health_status['status'] = 'warning'
                health_status['algorithm_error'] = str(e)
                
        # Performance warnings
        if self.step_times:
            avg_time = sum(self.step_times[-10:]) / min(10, len(self.step_times))
            if avg_time > 30.0:  # Step taking longer than 30 seconds
                health_status['status'] = 'warning'
                health_status['warning'] = f'Slow training steps detected: {avg_time:.2f}s average'
                
        return health_status
        
    async def get_training_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive training metrics.
        
        Returns:
            Dictionary with training performance metrics
        """
        metrics = {
            'total_steps': self.training_step_count,
            'recent_metrics': list(self.training_metrics.values())[-10:],
            'algorithm': self.config.algorithm.name,
        }
        
        if self.step_times:
            metrics.update({
                'performance': {
                    'avg_step_time': sum(self.step_times) / len(self.step_times),
                    'recent_avg_time': sum(self.step_times[-10:]) / min(10, len(self.step_times)),
                    'total_training_time': sum(self.step_times),
                    'steps_completed': len(self.step_times)
                }
            })
            
        return metrics
        
    async def shutdown(self) -> None:
        """
        Shutdown the trainer and algorithm actors.
        """
        logger.info("Shutting down ReTrainer...")
        
        if self.algorithm_actor and self.is_initialized:
            try:
                await self.algorithm_actor.shutdown.remote()  # type: ignore
                logger.info("Algorithm actor shutdown successfully")
            except Exception as e:
                logger.warning(f"Error shutting down algorithm actor: {e}")
                
        # Clear state
        self.training_metrics.clear()
        self.step_times.clear()
        self.current_weights = None
        self.is_initialized = False
        
        logger.info("ReTrainer shutdown complete")

    # Methods for connecting with other actor groups (called by manager)
    async def connect_inference(self, inference_actor: ray.ObjectRef) -> None:
        """Connect with inference actors for weight updates."""
        logger.info("Connecting ReTrainer with inference actors")
        # Store reference for future weight updates
        self.connected_inference_actors = getattr(self, 'connected_inference_actors', [])
        self.connected_inference_actors.append(inference_actor)
        
    async def sync_weights_to_inference(self) -> None:
        """Sync current model weights to all connected inference actors."""
        if hasattr(self, 'connected_inference_actors'):
            for inference_actor in self.connected_inference_actors:
                await self.update_inference_weights(inference_actor)
