from typing import Any, List, Optional, Dict, TYPE_CHECKING
from loguru import logger

# Import retrain's base trainer interface
from ..trainer import (
    BaseTrainer,
    AlgorithmConfig, 
    ModelObject,
    ExperienceBatch,
    TrainingMetrics,
    PromptSource,
    RewardFunction
)

# Handle Ray imports with proper type checking and runtime availability checks
# Initialize availability flag at module level
_slime_available = False

if TYPE_CHECKING:
    # Type aliases for optional Slime components - avoid actual imports to prevent linter errors
    RayType = Any  # ray module when available
    SlimeArgsType = Any  # argparse.Namespace with Slime arguments
    ActorModelType = Any  # Slime actor model group
    RolloutGeneratorType = Any  # Slime rollout generator group
else:
    # Runtime imports with graceful fallback
    try:
        import ray
        from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_group
        from slime.utils.arguments import parse_args as slime_parse_args, get_slime_extra_args_provider
        _slime_available = True
        
        # Type aliases for runtime
        RayType = type(ray)
        SlimeArgsType = Any
        ActorModelType = Any
        RolloutGeneratorType = Any
    except ImportError:
        # Graceful fallback - set everything to None but don't use these at runtime
        ray = None  # type: ignore
        create_actor_group = None  # type: ignore
        create_placement_groups = None  # type: ignore
        create_rollout_group = None  # type: ignore
        slime_parse_args = None  # type: ignore
        get_slime_extra_args_provider = None  # type: ignore
        _slime_available = False
        
        # Dummy types for fallback
        RayType = Any
        SlimeArgsType = Any
        ActorModelType = Any
        RolloutGeneratorType = Any


class SlimeTrainerAdapter(BaseTrainer):
    """
    Adapter for Slime distributed RL training framework.
    
    Bridges retrain's high-level interface with Slime's Ray-based distributed 
    training system. Handles:
    - Ray cluster management and GPU allocation
    - Configuration translation from retrain format to Slime's 994-parameter format
    - Model format conversion (HuggingFace → Megatron)
    - Distributed rollout and training coordination
    """
    
    def __init__(self,
                 model: ModelObject,
                 algorithm_config: AlgorithmConfig,
                 reward_functions: List[RewardFunction],
                 prompt_source: Optional[PromptSource] = None,
                 tokenizer: Optional[Any] = None,
                 reference_model: Optional[ModelObject] = None,
                 environment: Optional[Any] = None,
                 reward_calculator: Optional[Any] = None,
                 **trainer_specific_kwargs: Any):
        """
        Initialize Slime adapter with retrain components.
        
        Args:
            model: HuggingFace model to be trained
            algorithm_config: retrain's algorithm configuration
            reward_functions: List of reward functions (will be adapted for Slime)
            prompt_source: Source of prompts for training
            tokenizer: Model tokenizer
            reference_model: Reference model for KL divergence (optional)
            environment: retrain Environment (will be bridged to Slime rollout)
            reward_calculator: retrain RewardCalculator (will be bridged to Slime rewards)
        """
        # Store retrain components for later use
        self.environment = environment
        self.reward_calculator = reward_calculator
        
        # Slime-specific state with proper type annotations
        self.ray_initialized = False
        self.actor_model: Optional[ActorModelType] = None
        self.rollout_generator: Optional[RolloutGeneratorType] = None
        self.slime_args: Optional[SlimeArgsType] = None
        self.bridge_system: Optional[Any] = None
        
        logger.debug(f"[SlimeTrainerAdapter] Initializing with Slime backend")
        logger.debug(f"[SlimeTrainerAdapter] Received {len(reward_functions)} reward functions")
        
        # Call parent constructor which will trigger _validate_and_setup_backend
        super().__init__(
            model=model,
            algorithm_config=algorithm_config,
            reward_functions=reward_functions,
            prompt_source=prompt_source,
            tokenizer=tokenizer,
            reference_model=reference_model,
            **trainer_specific_kwargs
        )
    
    def _validate_and_setup_backend(self) -> None:
        """
        Validate dependencies and setup Slime backend.
        
        This method:
        1. Checks if Slime is available
        2. Validates input components compatibility
        3. Translates retrain config to Slime args format
        4. Sets up Ray cluster and Slime components
        """
        self._check_dependencies()
        self._validate_input_components()
        self._translate_config_to_slime_args()
        self._setup_ray_and_slime_components()
    
    def _check_dependencies(self) -> None:
        """Check if required Slime dependencies are available."""
        if not _slime_available:
            raise ImportError(
                "Slime backend is not available. Please install Slime with:\n"
                "pip install -e ./slime\n"
                "or ensure Slime is in your Python path."
            )
        
        logger.debug("[SlimeTrainerAdapter] Slime dependencies check passed")
    
    def _validate_input_components(self) -> None:
        """Validate that input components are compatible with Slime."""
        # Check that we have essential components
        if self.model is None:
            raise ValueError("Model is required for Slime training")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for Slime training")
        
        # Validate algorithm config
        if not hasattr(self.algorithm_config, 'hyperparameters'):
            raise ValueError("AlgorithmConfig must have hyperparameters for Slime")
        
        logger.debug("[SlimeTrainerAdapter] Input component validation passed")
    
    def _translate_config_to_slime_args(self) -> None:
        """
        Translate retrain's simplified config to Slime's 994-parameter format.
        
        This creates a Slime args object with smart defaults and user overrides.
        """
        # Runtime check to ensure Slime is available - this ensures get_slime_extra_args_provider is not None
        if not _slime_available or get_slime_extra_args_provider is None:  # type: ignore
            raise RuntimeError("Slime is not available for configuration translation")
            
        logger.debug("[SlimeTrainerAdapter] Translating retrain config to Slime format")
        
        # Get Slime's argument parser to create proper args object
        # Type checker knows get_slime_extra_args_provider is not None due to check above
        slime_arg_provider = get_slime_extra_args_provider()  # type: ignore
        
        # Create a mock parser to get default Slime args
        import argparse
        parser = argparse.ArgumentParser()
        slime_arg_provider(parser)
        
        # Create base Slime args with smart defaults
        slime_args_dict = self._create_slime_smart_defaults()
        
        # Apply user-provided hyperparameters
        if hasattr(self.algorithm_config, 'hyperparameters'):
            for key, value in self.algorithm_config.hyperparameters.items():
                if key.startswith('slime_'):
                    # Direct Slime parameter (e.g., slime_actor_num_nodes -> actor_num_nodes)
                    slime_key = key[6:]  # Remove 'slime_' prefix
                    slime_args_dict[slime_key] = value
                    logger.debug(f"[SlimeTrainerAdapter] Set Slime param {slime_key}={value}")
        
        # Convert dict to namespace (Slime expects argparse.Namespace)
        self.slime_args = argparse.Namespace(**slime_args_dict)
        
        logger.debug(f"[SlimeTrainerAdapter] Created Slime args with {len(slime_args_dict)} parameters")
    
    def _create_slime_smart_defaults(self) -> Dict[str, Any]:
        """
        Create smart default values for Slime's 994 parameters.
        
        These defaults are chosen to work well for most retrain use cases while
        being easily overridable by user configuration.
        """
        # Get basic system info for smart defaults
        import torch
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Model path - we'll need to convert HF model to Megatron format
        # For now, assume the model path or set up conversion
        model_name = getattr(self.model, 'name_or_path', 'unknown-model')
        
        smart_defaults = {
            # === Core Model & Data ===
            'hf_checkpoint': model_name,  # Will be converted to Megatron format
            'prompt_data': '/tmp/retrain_prompts.jsonl',  # Will be created from prompt_source
            
            # === Ray/Distributed Config ===
            'actor_num_nodes': 1,
            'actor_num_gpus_per_node': min(available_gpus, 8),
            'rollout_num_gpus': max(1, available_gpus // 2),
            'rollout_num_gpus_per_engine': 1,
            'colocate': available_gpus < 8,  # Auto-enable for small setups
            'offload': False,
            
            # === Training Hyperparameters ===
            'num_rollout': 10,  # Conservative default
            'rollout_batch_size': 2,
            'n_samples_per_prompt': 4,
            'global_batch_size': 8,
            'micro_batch_size': 1,
            
            # === Algorithm Config ===
            'advantage_estimator': 'grpo',
            'loss_type': 'policy_loss',
            'eps_clip': 0.2,
            'kl_coef': 0.0,
            'entropy_coef': 0.0,
            'normalize_advantages': True,
            
            # === Rollout/Generation ===
            'rollout_temperature': 0.7,
            'rollout_top_p': 0.9,
            'rollout_max_response_len': 256,
            'rollout_global_dataset': True,
            
            # === Optimization ===
            'lr': 1e-5,
            'weight_decay': 0.01,
            'lr_decay_style': 'cosine',
            'lr_warmup_fraction': 0.1,
            
            # === Logging & Checkpointing ===
            'save_interval': 5,
            'logging_steps': 1,
            'use_wandb': False,  # Let retrain handle logging
            
            # === System ===
            'seed': 42,
            'bf16': True,
            'tensor_model_parallel_size': 1,
            'pipeline_model_parallel_size': 1,
            'sequence_parallel': False,
            
            # === SGLang Config ===
            'sglang_mem_fraction_static': 0.7 if available_gpus < 8 else 0.9,
            'sglang_tp_size': 1,
            'sglang_dp_size': 1,
            
            # === Safety & Debug ===
            'debug_train_only': False,
            'debug_rollout_only': False,
            
            # === Required for type safety ===
            'start_rollout_id': None,
            'kl_coef': 0.0,
        }
        
        logger.debug(f"[SlimeTrainerAdapter] Smart defaults: colocate={smart_defaults['colocate']}, "
                     f"actor_gpus={smart_defaults['actor_num_gpus_per_node']}, "
                     f"rollout_gpus={smart_defaults['rollout_num_gpus']}")
        
        return smart_defaults
    
    def _setup_ray_and_slime_components(self) -> None:
        """
        Initialize Ray cluster and create Slime's actor/rollout components.
        
        This handles the distributed setup that Slime requires.
        """
        # Runtime checks to ensure all Slime components are available
        if not _slime_available:
            raise RuntimeError("Slime components are not available")
        
        if ray is None or create_placement_groups is None or create_actor_group is None or create_rollout_group is None:  # type: ignore
            raise RuntimeError("Required Slime functions are not available")
        
        if self.slime_args is None:
            raise RuntimeError("Slime args not initialized")
            
        try:
            # Initialize Ray if not already done
            if not ray.is_initialized():  # type: ignore
                logger.info("[SlimeTrainerAdapter] Initializing Ray cluster")
                ray.init(ignore_reinit_error=True)  # type: ignore
                self.ray_initialized = True
            
            # Create GPU placement groups for distributed training
            logger.debug("[SlimeTrainerAdapter] Creating Ray placement groups")
            pgs = create_placement_groups(self.slime_args)  # type: ignore
            
            # Create actor model group (for training)
            logger.debug("[SlimeTrainerAdapter] Creating actor model group")
            self.actor_model = create_actor_group(self.slime_args, pgs["actor"])  # type: ignore
            
            # Create rollout generator group (for inference/data generation)
            logger.debug("[SlimeTrainerAdapter] Creating rollout generator group")
            self.rollout_generator = create_rollout_group(self.slime_args, pgs["rollout"])  # type: ignore
            
            # Initialize retrain-Slime bridge system
            logger.debug("[SlimeTrainerAdapter] Setting up retrain-Slime bridges")
            self._setup_retrain_bridges()
            
            logger.info("[SlimeTrainerAdapter] Slime components initialized successfully")
            
        except Exception as e:
            logger.error(f"[SlimeTrainerAdapter] Failed to setup Ray/Slime components: {e}")
            if self.ray_initialized and ray is not None:  # type: ignore
                ray.shutdown()  # type: ignore
            raise RuntimeError(f"Slime setup failed: {e}") from e
    
    def _setup_retrain_bridges(self) -> None:
        """
        Set up bridge system to connect retrain components with Slime.
        
        This creates the bridge that allows retrain's Environment and RewardCalculator
        to work with Slime's distributed training system.
        """
        try:
            # Import bridge components (done here to avoid import errors if Slime not available)
            from .bridge.rollout_bridge import create_retrain_slime_bridge
            
            # Validate required components
            if self.environment is None:
                raise ValueError("Environment is required for bridge setup")
            if self.reward_calculator is None:
                raise ValueError("RewardCalculator is required for bridge setup")
            if self.slime_args is None:
                raise ValueError("Slime args not initialized")
            
            # Create the complete bridge system
            self.bridge_system = create_retrain_slime_bridge(
                environment=self.environment,
                reward_calculator=self.reward_calculator,
                tokenizer=self.tokenizer
            )
            
            # Set up custom rollout function for Slime to use retrain components
            custom_rollout_func = self.bridge_system.create_slime_rollout_function()
            
            # Store the function reference so Slime can call it
            # This would typically be set via Slime's --custom-generate-function-path
            self.slime_args.custom_rollout_function = custom_rollout_func
            
            logger.info("[SlimeTrainerAdapter] Bridge system initialized successfully")
            
        except ImportError as e:
            logger.warning(f"[SlimeTrainerAdapter] Could not import bridge components: {e}")
            logger.warning("[SlimeTrainerAdapter] Running without retrain component integration")
            self.bridge_system = None
        except Exception as e:
            logger.error(f"[SlimeTrainerAdapter] Failed to setup retrain bridges: {e}", exc_info=True)
            self.bridge_system = None
    
    async def train(self, total_training_iterations: Optional[int] = None) -> TrainingMetrics:
        """
        Execute the main training loop using Slime's distributed system.
        
        This orchestrates the rollout → training → weight update cycle that
        Slime uses for RL training.
        """
        if self.actor_model is None or self.rollout_generator is None:
            raise RuntimeError("Slime components not initialized. Call _validate_and_setup_backend first.")
        
        if self.slime_args is None:
            raise RuntimeError("Slime args not initialized")
        
        logger.info("[SlimeTrainerAdapter] Starting Slime distributed training")
        
        try:
            # Override num_rollout if total_training_iterations is provided
            if total_training_iterations is not None:
                self.slime_args.num_rollout = total_training_iterations
                logger.debug(f"[SlimeTrainerAdapter] Set num_rollout to {total_training_iterations}")
            
            # Initialize Slime training components
            await self._initialize_slime_training()
            
            # Run the main Slime training loop
            training_metrics = await self._run_slime_training_loop()
            
            logger.info("[SlimeTrainerAdapter] Training completed successfully")
            return training_metrics
            
        except Exception as e:
            logger.error(f"[SlimeTrainerAdapter] Training failed: {e}", exc_info=True)
            raise RuntimeError(f"Slime training failed: {e}") from e
    
    async def _initialize_slime_training(self) -> None:
        """Initialize Slime's distributed training setup."""
        # Runtime checks for required components
        if ray is None:  # type: ignore
            raise RuntimeError("Ray not available")
        if self.actor_model is None or self.rollout_generator is None:
            raise RuntimeError("Actor model or rollout generator not initialized")
        if self.slime_args is None:
            raise RuntimeError("Slime args not initialized")
            
        logger.debug("[SlimeTrainerAdapter] Initializing Slime training setup")
        
        # Sync model initialization across all workers
        start_rollout_ids = ray.get(  # type: ignore
            self.actor_model.async_init(
                self.slime_args, 
                role="actor", 
                with_ref=self.slime_args.kl_coef != 0
            )
        )
        
        # Set starting rollout ID
        if self.slime_args.start_rollout_id is None:
            self.slime_args.start_rollout_id = start_rollout_ids[0]
        
        # Load dataset if using global dataset
        if self.slime_args.rollout_global_dataset:
            ray.get(self.rollout_generator.data_buffer.load.remote(  # type: ignore
                self.slime_args.start_rollout_id - 1
            ))
        
        # Initialize weight update connections between actor and rollout
        ray.get(self.actor_model.async_init_weight_update_connections(self.rollout_generator))  # type: ignore
        
        # Initial weight sync
        ray.get(self.actor_model.async_update_weights())  # type: ignore
        
        logger.debug("[SlimeTrainerAdapter] Slime training initialization completed")
    
    async def _run_slime_training_loop(self) -> TrainingMetrics:
        """Execute Slime's main training loop with rollout → train → update cycle."""
        # Runtime checks
        if ray is None:  # type: ignore
            raise RuntimeError("Ray not available")
        if self.actor_model is None or self.rollout_generator is None:
            raise RuntimeError("Actor model or rollout generator not initialized")
        if self.slime_args is None:
            raise RuntimeError("Slime args not initialized")
            
        metrics = {}
        
        start_rollout = self.slime_args.start_rollout_id
        end_rollout = self.slime_args.num_rollout
        
        logger.info(f"[SlimeTrainerAdapter] Running training from rollout {start_rollout} to {end_rollout}")
        
        for rollout_id in range(start_rollout, end_rollout):
            logger.debug(f"[SlimeTrainerAdapter] Starting rollout {rollout_id}")
            
            # Generate rollout data using SGLang
            ray.get(self.rollout_generator.async_generate(rollout_id))  # type: ignore
            
            # Train on the generated data using Megatron
            ray.get(self.actor_model.async_train(rollout_id))  # type: ignore
            
            # Save checkpoint if needed
            if (self.slime_args.save_interval and 
                (rollout_id + 1) % self.slime_args.save_interval == 0):
                ray.get(self.actor_model.async_save_model(rollout_id))  # type: ignore
            
            # Update weights from training to rollout
            ray.get(self.actor_model.async_update_weights())  # type: ignore
            
            logger.debug(f"[SlimeTrainerAdapter] Completed rollout {rollout_id}")
        
        # Collect final metrics (simplified for now)
        metrics = {
            'rollouts_completed': end_rollout - start_rollout,
            'final_rollout_id': end_rollout - 1,
        }
        
        return metrics
    
    async def step(self, experience_batch: ExperienceBatch) -> TrainingMetrics:
        """
        Single step training (not the primary mode for Slime).
        
        Slime is designed for full rollout → train cycles rather than 
        external step-by-step control. This method provides compatibility
        but may have limited functionality.
        """
        logger.warning("[SlimeTrainerAdapter] step() called - Slime prefers full train() cycles")
        
        if self.slime_args is None:
            raise RuntimeError("Slime args not initialized")
        
        # For now, defer to the full training loop with num_rollout=1
        original_num_rollout = self.slime_args.num_rollout
        self.slime_args.num_rollout = 1
        
        try:
            return await self.train(total_training_iterations=1)
        finally:
            self.slime_args.num_rollout = original_num_rollout
    
    def save_checkpoint(self, checkpoint_directory: str) -> None:
        """Save model checkpoint using Slime's distributed checkpoint system."""
        if self.actor_model is None:
            raise RuntimeError("Slime components not initialized")
        if ray is None:  # type: ignore
            raise RuntimeError("Ray not available")
        
        logger.info(f"[SlimeTrainerAdapter] Saving checkpoint to {checkpoint_directory}")
        
        try:
            # Use Slime's checkpoint saving
            ray.get(self.actor_model.async_save_model_to_path(checkpoint_directory))  # type: ignore
            logger.info("[SlimeTrainerAdapter] Checkpoint saved successfully")
        except Exception as e:
            logger.error(f"[SlimeTrainerAdapter] Failed to save checkpoint: {e}")
            raise RuntimeError(f"Checkpoint save failed: {e}") from e
    
    def load_checkpoint(self, checkpoint_directory: str) -> None:
        """Load model checkpoint using Slime's distributed checkpoint system."""
        if self.slime_args is None:
            raise RuntimeError("Slime args not initialized")
        
        logger.info(f"[SlimeTrainerAdapter] Loading checkpoint from {checkpoint_directory}")
        
        # Update Slime args to point to the checkpoint
        self.slime_args.load = checkpoint_directory
        
        logger.info(f"[SlimeTrainerAdapter] Checkpoint path set to {checkpoint_directory}")
    
    def __del__(self):
        """Cleanup Ray resources when adapter is destroyed."""
        if hasattr(self, 'ray_initialized') and self.ray_initialized and ray is not None:  # type: ignore
            try:
                logger.debug("[SlimeTrainerAdapter] Shutting down Ray")
                ray.shutdown()  # type: ignore
            except Exception as e:
                logger.warning(f"[SlimeTrainerAdapter] Error during Ray shutdown: {e}") 