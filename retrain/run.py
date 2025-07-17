from typing import Any, Dict, Optional, List, Callable, Tuple, Type, Union, TYPE_CHECKING, cast
import os
import asyncio
import time

from loguru import logger # Import Loguru logger

# Import Pydantic models and error type
from .config_models import TrainingConfig, ModelConfig, EnvironmentConfig, PromptSourceConfig, RewardSetupConfig, AlgorithmConfig, RewardFunctionConfig # Import the main config model and sub-models
from pydantic import ValidationError

# Import model loaders
from .model import HuggingFaceModel, UnslothModel, Model as ModelObject, get_model
from .environment import Environment, get_environment # Use BaseEnvironment for type hint
from .prompt_source import PromptSource, get_prompt_source
from .reward.calculator import RewardCalculator
from .reward.reward import REWARD_REGISTRY, create_grpo_batch_reward_func # For TRL reward prep
from .trainer import BaseTrainer # Base class for type hinting
# from .trainer.utils import log_system_info # Utility for logging system info
from .utils.logging_utils import setup_logging, get_logger # For setting up logger per module

# Specific trainer imports - these are chosen based on config
# For GRPO with TRL backend
from .trainer.grpo.trl import GRPOTrainerAdapter as GRPOTrainerAdapterTRL

# First, check if TRL is available at a basic level to set the flag
_trl_grpo_config_available = False
try:
    from trl import GRPOConfig # Try importing the real one
    _trl_grpo_config_available = True
except ImportError:
    pass # _trl_grpo_config_available remains False

# Conditional import for TrlGRPOConfig using the flag and TYPE_CHECKING
if TYPE_CHECKING or _trl_grpo_config_available:
    try:
        from trl import GRPOConfig as TrlGRPOConfigActual
        _trl_grpo_config_available_runtime_check = True
    except ImportError:
        class TrlGRPOConfigFallback:
            def __init__(self, **kwargs): pass
        TrlGRPOConfigActual = TrlGRPOConfigFallback # type: ignore
        _trl_grpo_config_available_runtime_check = False
else:
    class TrlGRPOConfigFallback:
        def __init__(self, **kwargs): pass
    TrlGRPOConfigActual = TrlGRPOConfigFallback # type: ignore
    _trl_grpo_config_available_runtime_check = False

TrlGRPOConfigType = TrlGRPOConfigActual

# Initialize logger for this module (run.py)
logger_run = get_logger(__name__)

# Helper: Setup TRL-compatible GRPOConfig
def _get_or_create_trl_grpo_config(
    base_hyperparameters: Dict[str, Any], 
    report_to_settings: Optional[List[str]],
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    wandb_run_name: Optional[str],
    reward_weights_for_config: Optional[List[float]]
) -> TrlGRPOConfigType:
    """
    Creates a TrlGRPOConfig object.
    Merges base hyperparameters with specific arguments like W&B settings and reward_weights.
    """
    if not _trl_grpo_config_available_runtime_check: # Check the runtime flag
        raise ImportError("TRL GRPOConfig is not available. Please install TRL.")

    # Start with a copy of base hyperparameters meant for TrlGRPOConfig
    effective_hyperparameters = base_hyperparameters.copy()

    # Add reward_weights if provided
    if reward_weights_for_config is not None: # Can be an empty list if no rewards, or list of floats
        effective_hyperparameters["reward_weights"] = reward_weights_for_config

    # Ensure 'report_to' is correctly in the hyperparameters if wandb is specified
    if report_to_settings and "wandb" in report_to_settings:
        if "report_to" not in effective_hyperparameters:
            effective_hyperparameters["report_to"] = report_to_settings
        elif isinstance(effective_hyperparameters.get("report_to"), list) and "wandb" not in effective_hyperparameters["report_to"]:
            effective_hyperparameters["report_to"].append("wandb")
        elif isinstance(effective_hyperparameters.get("report_to"), str) and effective_hyperparameters["report_to"] != "wandb":
             effective_hyperparameters["report_to"] = [effective_hyperparameters["report_to"], "wandb"]
        elif effective_hyperparameters.get("report_to") is None: # if report_to was None in hyperparams but now specified
            effective_hyperparameters["report_to"] = report_to_settings
    # Removed merging of wandb_project, wandb_entity, run_name into effective_hyperparameters.
    # These will be handled by setting environment variables if needed.

    # Remove peft_config from hyperparameters if it's None, as TrlGRPOConfig doesn't expect it if None
    if "peft_config" in effective_hyperparameters and effective_hyperparameters.get("peft_config") is None:
        logger_run.debug("Removing 'peft_config: None' from hyperparameters for TrlGRPOConfig construction.")
        del effective_hyperparameters["peft_config"]

    # Handle num_iterations:
    # TRL's GRPOConfig might not always accept 'num_iterations' in its __init__
    # depending on the TRL version. However, our GRPOTrainerAdapter expects it
    # as an attribute on the config object.
    # We'll remove it before __init__ and add it back as an attribute.
    num_iterations_val = None
    if "num_iterations" in effective_hyperparameters:
        num_iterations_val = effective_hyperparameters.pop("num_iterations")
        logger_run.debug(f"Temporarily removed 'num_iterations': {num_iterations_val} from TrlGRPOConfig constructor arguments.")

    logger_run.debug(f"Attempting to create TrlGRPOConfig with effective hyperparameters: {effective_hyperparameters}")
    try:
        config_instance = TrlGRPOConfigActual(**effective_hyperparameters)
        
        # If num_iterations was extracted, set it as an attribute on the config instance.
        # This ensures it's available for the GRPOTrainerAdapter, which relies on this attribute.
        if num_iterations_val is not None:
            setattr(config_instance, "num_iterations", num_iterations_val)
            logger_run.debug(f"Set 'num_iterations': {num_iterations_val} as an attribute on the TrlGRPOConfig instance.")
            
        return config_instance
    except Exception as e:
        logger_run.error("Error creating TrlGRPOConfig with hyperparameters: %s. Error: %s", effective_hyperparameters, repr(e), exc_info=True)
        raise

# Main Trainer Setup
def _setup_trainer_instance(
    cfg: TrainingConfig, 
    model: ModelObject,
    environment: Environment,
    reward_calculator: RewardCalculator,
    prompt_source: PromptSource,
    tokenizer: Any, 
    reference_model: Optional[ModelObject] = None
) -> BaseTrainer:
    """Helper to instantiate and configure the correct trainer based on config."""
    algo_cfg = cfg.algorithm
    
    # These will be kwargs for our GRPOTrainerAdapterTRL, not for TrlGRPOConfig directly
    adapter_specific_params = {} 

    if algo_cfg.name == "grpo" and algo_cfg.backend == "trl":
        logger_run.info(f"Setting up GRPOTrainer (TRL backend) with algorithm config: {algo_cfg}")
        
        # Set WANDB environment variables if 'wandb' is in report_to and corresponding algo_cfg fields are set
        if algo_cfg.report_to and "wandb" in algo_cfg.report_to:
            if algo_cfg.wandb_project:
                os.environ["WANDB_PROJECT"] = algo_cfg.wandb_project
                logger_run.info(f"Set WANDB_PROJECT environment variable to: {algo_cfg.wandb_project}")
            if algo_cfg.wandb_entity:
                os.environ["WANDB_ENTITY"] = algo_cfg.wandb_entity
                logger_run.info(f"Set WANDB_ENTITY environment variable to: {algo_cfg.wandb_entity}")
            if algo_cfg.wandb_run_name:
                os.environ["WANDB_RUN_NAME"] = algo_cfg.wandb_run_name
                logger_run.info(f"Set WANDB_RUN_NAME environment variable to: {algo_cfg.wandb_run_name}")
            # Also ensure WANDB_DISABLED is not 'true' if we want to report
            if os.environ.get("WANDB_DISABLED", "").lower() == "true":
                logger_run.warning("WANDB_DISABLED is set to 'true', but report_to includes 'wandb'. Unsetting WANDB_DISABLED.")
                del os.environ["WANDB_DISABLED"]

        actual_trl_reward_functions: List[Callable] = []
        actual_trl_reward_weights: List[float] = []

        all_reward_configs: Dict[str, RewardFunctionConfig] = {
            **cfg.reward_setup.step_reward_configs,
            **cfg.reward_setup.rollout_reward_configs
        }
        if not all_reward_configs:
            logger_run.warning("No reward functions configured. TRL will operate with no explicit reward guidance.")

        for reward_name, reward_config_entry in all_reward_configs.items():
            base_reward_cfg_for_wrapper = {
                "name": reward_name,
                "params": reward_config_entry.params,
                "verifiers": reward_config_entry.verifiers or [],
                "verifier_penalty": reward_config_entry.verifier_penalty
            }
            logger_run.debug(f"Preparing TRL reward wrapper for: {reward_name} with config {base_reward_cfg_for_wrapper}")
            try:
                trl_compatible_reward_func = create_grpo_batch_reward_func(
                    base_reward_config=base_reward_cfg_for_wrapper
                )
                if trl_compatible_reward_func:
                    actual_trl_reward_functions.append(trl_compatible_reward_func)
                    actual_trl_reward_weights.append(reward_config_entry.weight)
                    logger_run.info(f"Created TRL-compatible reward function for '{reward_name}' with weight {reward_config_entry.weight}.")
                else:
                    logger_run.error(f"Failed to create TRL-compatible reward for '{reward_name}'. Skipping.")
            except Exception as e:
                logger_run.error(f"Exception creating TRL-compatible reward for '{reward_name}': {e}", exc_info=True)

        if not actual_trl_reward_functions:
            logger_run.warning("No TRL-compatible reward functions were created. TRL GRPOTrainer will lack specific reward guidance.")

        # Separate hyperparameters for TrlGRPOConfig vs. GRPOTrainerAdapterTRL
        trl_config_hyperparams = algo_cfg.hyperparameters.copy()
        
        # ADDED: Log the raw hyperparameters from the config for the algorithm
        logger_run.debug(f"[RUN.PY _setup_trainer_instance] Raw algo_cfg.hyperparameters: {algo_cfg.hyperparameters}")
        logger_run.debug(f"[RUN.PY _setup_trainer_instance] Copied trl_config_hyperparams (before adapter specific extraction): {trl_config_hyperparams}")

        # Known adapter-specific params that should not go into TrlGRPOConfig
        # These include both adapter-specific parameters and generation parameters
        # that TRL's GRPOConfig doesn't accept in its constructor
        known_adapter_params = [
            "sampling_params", 
            "max_tokens_per_generation", 
            "temperature_for_logprobs", 
            "max_steps_this_trainer_train_call",
            # Generation parameters that should be handled by the adapter, not TRL config
            "temperature",  # Used for generation during rollouts
            "top_p",        # Used for generation during rollouts  
            "top_k",        # Used for generation during rollouts
            # Parameters that don't exist in TRL GRPOConfig but need mapping
            "batch_size",   # Maps to per_device_train_batch_size
            "max_length",   # Maps to max_completion_length
            "num_iterations", # Handled separately as attribute
            "evaluation_strategy", # Maps to eval_strategy
            # WandB logging parameters that should be handled by the adapter
            "num_completions_to_print",
            "wandb_log_unique_prompts",
        ]
        
        # Extract generation parameters to build sampling_params for the adapter
        generation_params = {}
        adapter_specific_params = {}
        
        for param_name in known_adapter_params:
            if param_name in trl_config_hyperparams:
                param_value = trl_config_hyperparams.pop(param_name)
                
                # Handle generation parameters specially - add them to sampling_params
                if param_name in ["temperature", "top_p", "top_k"]:
                    generation_params[param_name] = param_value
                    logger_run.debug(f"Extracted generation parameter '{param_name}': {param_value} for adapter's sampling_params.")
                # Handle parameter mapping to TRL equivalents
                elif param_name == "batch_size":
                    trl_config_hyperparams["per_device_train_batch_size"] = param_value
                    logger_run.debug(f"Mapped 'batch_size': {param_value} to 'per_device_train_batch_size' for TRL config.")
                elif param_name == "max_length":
                    trl_config_hyperparams["max_completion_length"] = param_value
                    logger_run.debug(f"Mapped 'max_length': {param_value} to 'max_completion_length' for TRL config.")
                elif param_name == "evaluation_strategy":
                    trl_config_hyperparams["eval_strategy"] = param_value
                    logger_run.debug(f"Mapped 'evaluation_strategy': {param_value} to 'eval_strategy' for TRL config.")
                else:
                    # Store other adapter-specific parameters
                    adapter_specific_params[param_name] = param_value
                    logger_run.debug(f"Extracted adapter parameter '{param_name}': {param_value}.")
        
        # If we extracted generation parameters, merge them into sampling_params for the adapter
        if generation_params:
            if "sampling_params" not in adapter_specific_params:
                adapter_specific_params["sampling_params"] = {}
            adapter_specific_params["sampling_params"].update(generation_params)
            logger_run.debug(f"Merged generation parameters into adapter's sampling_params: {generation_params}")
        
        # ADDED: Align TrlGRPOConfig's max_completion_length with the adapter's generation limit
        if "max_tokens_per_generation" in adapter_specific_params:
            intended_max_length = adapter_specific_params["max_tokens_per_generation"]
            # If max_completion_length is also in trl_config_hyperparams and differs, log that we are overriding it.
            if "max_completion_length" in trl_config_hyperparams and trl_config_hyperparams["max_completion_length"] != intended_max_length:
                logger_run.info(f"Overriding 'max_completion_length' (was {trl_config_hyperparams['max_completion_length']}) "
                                f"with 'max_tokens_per_generation' ({intended_max_length}) for TrlGRPOConfig.")
            else:
                logger_run.info(f"Setting 'max_completion_length' for TrlGRPOConfig to the adapter's 'max_tokens_per_generation': {intended_max_length}.")
            # This ensures that TrlGRPOConfig uses the same maximum length as the environment generation,
            # allowing TRL to process the full generated text.
            trl_config_hyperparams["max_completion_length"] = intended_max_length
        elif "max_completion_length" in trl_config_hyperparams:
            logger_run.info(f"Using 'max_completion_length': {trl_config_hyperparams['max_completion_length']} from hyperparameters for TrlGRPOConfig, as 'max_tokens_per_generation' was not specified for the adapter.")
        else:
            # This case means neither was specified in the original hyperparameters for TrlGRPOConfig.
            # TRL's GRPOConfig will use its own default for max_completion_length.
            logger_run.warning("Neither 'max_tokens_per_generation' (for adapter) nor 'max_completion_length' (for TrlGRPOConfig source hyperparams) "
                               "was found. TrlGRPOConfig will use its default for 'max_completion_length'. "
                               "Ensure this aligns with environment's generation length.")
        
        # Ensure num_generations is compatible with batch size for GRPO
        # GRPO requires: global_batch_size % num_generations == 0
        # Our batch size is set via per_device_train_batch_size
        if "per_device_train_batch_size" in trl_config_hyperparams:
            batch_size = trl_config_hyperparams["per_device_train_batch_size"]
            # Set num_generations to be a divisor of batch_size, defaulting to 4
            if "num_generations" not in trl_config_hyperparams:
                if batch_size >= 4:
                    trl_config_hyperparams["num_generations"] = 4
                elif batch_size >= 2:
                    trl_config_hyperparams["num_generations"] = 2
                else:
                    trl_config_hyperparams["num_generations"] = 1
                logger_run.info(f"Set num_generations to {trl_config_hyperparams['num_generations']} to be compatible with batch_size {batch_size}")

        # Now, trl_config_hyperparams contains only what's intended for TrlGRPOConfig
        # AND max_completion_length should be aligned if max_tokens_per_generation was provided for the adapter.
        trl_grpo_config = _get_or_create_trl_grpo_config(
            base_hyperparameters=trl_config_hyperparams,
            report_to_settings=algo_cfg.report_to,
            wandb_project=algo_cfg.wandb_project,
            wandb_entity=algo_cfg.wandb_entity,
            wandb_run_name=algo_cfg.wandb_run_name,
            reward_weights_for_config=actual_trl_reward_weights if actual_trl_reward_functions else None # Pass None if no functions
        )
        
        # ADDED: Log the instantiated TrlGRPOConfig object's key attributes
        # logger_run.debug(f"[RUN.PY _setup_trainer_instance] Instantiated trl_grpo_config object: {trl_grpo_config}")
        if _trl_grpo_config_available_runtime_check and isinstance(trl_grpo_config, TrlGRPOConfigActual):
            logger_run.debug(f"[RUN.PY _setup_trainer_instance] trl_grpo_config.loss_type: {getattr(trl_grpo_config, 'loss_type', 'N/A')}")
            logger_run.debug(f"[RUN.PY _setup_trainer_instance] trl_grpo_config.max_completion_length: {getattr(trl_grpo_config, 'max_completion_length', 'N/A')}")
            logger_run.debug(f"[RUN.PY _setup_trainer_instance] trl_grpo_config.num_generations: {getattr(trl_grpo_config, 'num_generations', 'N/A')}")
            logger_run.debug(f"[RUN.PY _setup_trainer_instance] trl_grpo_config.beta (KL coeff): {getattr(trl_grpo_config, 'beta', 'N/A')}")
            logger_run.debug(f"[RUN.PY _setup_trainer_instance] trl_grpo_config.reward_weights: {getattr(trl_grpo_config, 'reward_weights', 'N/A')}")
        elif isinstance(trl_grpo_config, TrlGRPOConfigFallback):
             logger_run.warning("[RUN.PY _setup_trainer_instance] trl_grpo_config is a Fallback. TRL not fully available.")

        logger_run.debug(f"Num actual TRL reward functions: {len(actual_trl_reward_functions)}")
        # logger_run.debug(f"TRL reward weights in config: {getattr(trl_grpo_config, 'reward_weights', 'Not Set')}")
        logger_run.debug(f"Adapter specific kwargs for GRPOTrainerAdapterTRL: {adapter_specific_params}")

        try:
            trainer_instance = GRPOTrainerAdapterTRL(
                model=model,
                tokenizer=tokenizer,
                algorithm_config=trl_grpo_config, 
                environment=environment,
                reward_calculator=reward_calculator, 
                prompt_source=prompt_source,
                reward_functions=actual_trl_reward_functions, 
                reference_model=reference_model,
                **adapter_specific_params # Pass adapter-specific params here
            )
            logger_run.info("GRPOTrainerAdapterTRL instantiated successfully with actual reward functions.")
        except Exception as e:
            logger_run.error(f"Error instantiating GRPOTrainerAdapterTRL: {e}", exc_info=True)
            raise
    
    elif algo_cfg.name == "grpo" and algo_cfg.backend == "slime":
        logger_run.info(f"Setting up GRPOTrainer (Slime backend) with algorithm config: {algo_cfg}")
        
        # Import Slime adapter with error handling
        try:
            from .trainer.grpo.slime import SlimeTrainerAdapter
        except ImportError as e:
            raise ImportError(
                f"Failed to import Slime backend. Please ensure Slime is installed:\n"
                f"pip install -e ./slime\n"
                f"Original error: {e}"
            ) from e
        
        try:
            trainer_instance = SlimeTrainerAdapter(
                model=model,
                algorithm_config=algo_cfg,  # Pass the original AlgorithmConfig, not a converted one
                reward_functions=[],  # Slime handles rewards through its own internal system
                prompt_source=prompt_source,
                tokenizer=tokenizer,
                reference_model=reference_model,
                environment=environment,  # Slime adapter will bridge this to its rollout system
                reward_calculator=reward_calculator,  # Slime adapter will bridge this to its reward system
                **adapter_specific_params
            )
            logger_run.info("SlimeTrainerAdapter instantiated successfully.")
        except Exception as e:
            logger_run.error(f"Error instantiating SlimeTrainerAdapter: {e}", exc_info=True)
            raise
    
    else:
        raise ValueError(f"Unsupported algorithm/backend: {algo_cfg.name}/{algo_cfg.backend}")
    
    return trainer_instance

# Main Orchestration Function
async def run_async_training(cfg: TrainingConfig):
    """Asynchronous main training loop orchestration."""
    # Fix HuggingFace tokenizers parallelism warning that occurs when process forks
    # after tokenizers have been used. This is common in training environments.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # DIAGNOSTIC: Manual WANDB INIT
    if cfg.algorithm.report_to and "wandb" in cfg.algorithm.report_to:
        try:
            import wandb
            logger_run.info("Attempting manual wandb.init() for diagnostics...")
            wandb.init(
                project=cfg.algorithm.wandb_project or f"retrain-run-{cfg.experiment_name or 'default'}",
                entity=cfg.algorithm.wandb_entity, # Can be None
                name=cfg.algorithm.wandb_run_name,  # Can be None
                config=cfg.model_dump(exclude_none=True), # Log the whole config
                reinit=True, # Allow reinit if TRL also calls it
                # mode="online" # Optional: Force online mode if suspecting offline issues
            )
            logger_run.info(f"Manual wandb.init() called. Run name: {wandb.run.name if wandb.run else 'N/A'}")
        except Exception as e:
            logger_run.error(f"Error during manual wandb.init(): {e}", exc_info=True)
    # END DIAGNOSTIC

    start_time = time.time()
    setup_logging(level=cfg.logging_level) # Configure logging based on input config
    logger_run.info(f"Starting training run with experiment: {cfg.experiment_name or 'default'}")
    logger_run.debug(f"Full TrainingConfig: {cfg.model_dump_json(indent=2)}")

    # Seed everything for reproducibility if seed is provided
    if cfg.seed is not None:
        logger_run.info(f"Setting random seed to: {cfg.seed}")
        # (Implementation of actual seeding for torch, numpy, random would go here)
        # e.g., torch.manual_seed(cfg.seed), np.random.seed(cfg.seed), random.seed(cfg.seed)
        # For now, this is a placeholder for where seeding logic would be.
        # Note: TRL's TrainingArguments (which GRPOConfig inherits from) also has a `seed` param.
        # It should handle seeding for TRL components if set there.

    # Component Setup
    logger_run.info("\n--- Setting up Model ---")
    # Extract arguments for get_model from cfg.model (which is a ModelConfig instance)
    model_loader_type = cfg.model.loader
    model_identifier = cfg.model.name_or_path
    
    # Prepare model_config_overrides: these are additional parameters for the specific loader
    # (HuggingFaceModel or UnslothModel) beyond name_or_path and loader type.
    # Exclude fields already handled by get_model's signature or part of ModelConfig but not loader args.
    model_cfg_dict = cfg.model.model_dump(exclude_none=True) # Get all set fields
    
    # Fields that are part of ModelConfig but either passed directly to get_model 
    # or not directly part of the loader's internal `model_config` dict for its `load` method.
    handled_elsewhere = {"name_or_path", "loader", "peft_config"}
    model_config_for_loader = { 
        k: v for k, v in model_cfg_dict.items() if k not in handled_elsewhere
    }

    model, tokenizer = get_model(
        model_type=model_loader_type,
        model_name_or_path=model_identifier,
        model_config_overrides=model_config_for_loader,
        peft_config=cfg.model.peft_config # Pass peft_config directly
    )
    # reference_model is not explicitly handled by get_model in this iteration.
    # If a reference model is needed, it would typically be loaded separately or
    # the get_model function would need another parameter for it.
    # For now, assuming no separate reference model loading via this get_model call.
    reference_model = None # Placeholder, as get_model doesn't return it

    logger_run.info("\n--- Setting up Environment ---")
    # The get_environment function expects env_type and env_specific_config
    environment_instance = await get_environment(
        env_type=cfg.environment.type,
        env_specific_config=cfg.environment.env_specific_config
    )

    logger_run.info("\n--- Setting up Prompt Source ---")
    # get_prompt_source expects the prompt_source part of the config directly
    prompt_source_instance = get_prompt_source(config=cfg.prompt_source.model_dump(exclude_none=True))

    logger_run.info("\n--- Setting up Reward Calculator ---")
    # RewardCalculator uses the reward_setup from the main config
    # It expects step_reward_configs and rollout_reward_configs directly.
    reward_calculator_instance = RewardCalculator(
        step_reward_configs=cfg.reward_setup.step_reward_configs,      # Pass step_reward_configs from RewardSetupConfig
        rollout_reward_configs=cfg.reward_setup.rollout_reward_configs, # Pass rollout_reward_configs from RewardSetupConfig
        tokenizer=tokenizer,                                             # Pass tokenizer as before
        # reward_combination_strategy can be added if defined in RewardSetupConfig
        # For now, relying on RewardCalculator's default or internal logic if strategy is None.
    )
    
    logger_run.info("\n--- Setting up Trainer Instance ---")
    trainer_instance = _setup_trainer_instance(
        cfg=cfg, # Pass the full TrainingConfig
        model=model,
        environment=environment_instance,
        reward_calculator=reward_calculator_instance,
        prompt_source=prompt_source_instance,
        tokenizer=tokenizer,
        reference_model=reference_model # Pass the reference model
    )

    # Start Training
    logger_run.info("\n--- Starting Training Loop ---")
    # Pass total_training_iterations directly from algo_cfg.hyperparameters.
    # The GRPOTrainerAdapterTRL's train method has its own fallback logic if this is None.
    total_iterations_for_adapter_train = cfg.algorithm.hyperparameters.get(
        "max_steps_this_trainer_train_call"
    )

    training_metrics = await trainer_instance.train(
        total_training_iterations=total_iterations_for_adapter_train
    )
    
    logger_run.info("\n--- Training Finished ---")
    logger_run.info(f"Final training metrics: {training_metrics}")

    # Save Model (Optional)
    output_dir = cfg.algorithm.hyperparameters.get("output_dir")
    if output_dir:
        logger_run.info(f"Saving final model to: {output_dir}")
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if hasattr(trainer_instance, 'save_checkpoint'):
            # This is the preferred method, as implemented by our GRPOTrainerAdapterTRL
            trainer_instance.save_checkpoint(output_dir)
        # The GRPOTrainerAdapterTRL's save_checkpoint calls the underlying TRL trainer's save_model.
        # There's no need for an elif trying to call save_model on the adapter itself.
        else:
            logger_run.warning(
                "Trainer instance does not have a 'save_checkpoint' method. "
                "Model not saved by the run script."
            )
    else:
        logger_run.info("No output_dir specified in algorithm hyperparameters. Final model not saved by run script.")

    end_time = time.time()
    logger_run.info(f"Total training run time: {end_time - start_time:.2f} seconds.")
    logger_run.info(f"Training run for experiment '{cfg.experiment_name or 'default'}' completed.")
    # run_async_training should return metrics
    return training_metrics # Ensure metrics are returned

async def run(config: Any) -> Dict[str, Any]: # Changed type hint from TrainingConfig to Any for input flexibility
    """
    Main entry point to start a training run.
    Handles overall setup and initiates the asynchronous training process.
    This function is now asynchronous.

    Args:
        config: Can be a TrainingConfig Pydantic model instance or a dictionary
                that can be parsed into a TrainingConfig model.
    """
    parsed_config: TrainingConfig
    if isinstance(config, TrainingConfig):
        parsed_config = config
    elif isinstance(config, dict):
        try:
            parsed_config = TrainingConfig(**config)
            logger_run.info("Successfully parsed input dictionary into TrainingConfig model.")
        except ValidationError as e:
            logger_run.error(f"Failed to parse input dictionary into TrainingConfig: {e}", exc_info=True)
            raise
    else:
        err_msg = f"Invalid config type: {type(config)}. Must be a TrainingConfig instance or a dict."
        logger_run.error(err_msg)
        raise TypeError(err_msg)

    try:
        # Run the async training function and await its result, now with parsed_config
        metrics = await run_async_training(parsed_config)
        return {"status": "completed", "metrics": metrics}
    except ImportError as e:
        logger_run.critical(f"ImportError: {e}. Ensure all dependencies are installed.", exc_info=True)
        raise
    except ValidationError as e:
        logger_run.critical(f"Configuration validation error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger_run.critical(f"An unexpected error occurred during the training run: {e}", exc_info=True)
        raise
