from typing import Dict, Callable, Optional, List, Any, Tuple, Union
import inspect
from loguru import logger
import torch

# Import verifier components needed for the adapter
from ..verifier.verifier import VERIFIER_REGISTRY, VerifierFunction, get_boolean_verifier

# Global registry for reward functions
REWARD_REGISTRY: Dict[str, Callable[..., float]] = {}

def reward(name: Optional[str] = None):
    """
    Decorator to register a reward function.

    Functions decorated with this will be added to the global REWARD_REGISTRY.
    They can then be referenced by their registration name in configurations.

    Args:
        name (Optional[str]): The name to register the function under.
                              If None, the function's __name__ is used.

    Returns:
        Callable: The decorator function.
    """
    def decorator(func: Callable) -> Callable:
        registration_name = name if name is not None else func.__name__
        if registration_name in REWARD_REGISTRY:
            logger.warning(
                f"[RewardRegistry] Reward function '{registration_name}' is already registered. "
                f"Overwriting with {func.__module__}.{func.__name__}"
            )
        REWARD_REGISTRY[registration_name] = func
        logger.debug(f"[RewardRegistry] Registered reward function: '{registration_name}' -> {func.__module__}.{func.__name__}")
        return func
    return decorator

def get_reward_function(name: str) -> Optional[Callable[..., float]]:
    """Retrieves a reward function from the registry by its name."""
    return REWARD_REGISTRY.get(name)

def calculate_total_reward(*args, **kwargs) -> float:
    """
    Calculates the total reward by summing the scores from all registered reward functions.

    This function iterates through all callables in REWARD_REGISTRY,
    invokes each one with the provided *args and **kwargs, and sums their
    numerical return values. It assumes all registered reward functions
    can accept the same set of arguments passed to this function.

    Args:
        *args: Positional arguments to be passed to each registered reward function.
        **kwargs: Keyword arguments to be passed to each registered reward function.

    Returns:
        float: The sum of scores from all reward functions.
               Returns 0.0 if no reward functions are registered.

    Raises:
        TypeError: If a registered reward function cannot be called with the
                   provided arguments, or if it does not return a numeric value (int or float).
        Exception: Any other exception raised by an individual reward function during its execution
                   will propagate upwards.
    """
    total_score: float = 0.0
    
    if not REWARD_REGISTRY:
        logger.debug("[RewardCalculator] No reward functions registered, total reward is 0.0.")
        return 0.0

    logger.debug(f"[RewardCalculator] Calculating total reward from {len(REWARD_REGISTRY)} registered functions.")
    for func_name, reward_func in REWARD_REGISTRY.items():
        # The call to reward_func might raise TypeError if args don't match,
        # or any other exception if the function itself fails. This is intended behavior.
        score = reward_func(*args, **kwargs)
        
        if not isinstance(score, (int, float)):
            # Get module and function name for better error reporting
            module_name = getattr(reward_func, '__module__', 'unknown_module')
            function_name_attr = getattr(reward_func, '__name__', 'unknown_function')
            raise TypeError(
                f"Reward function '{func_name}' (from {module_name}.{function_name_attr}) "
                f"must return an int or float, but returned {type(score)}."
            )
        total_score += score
            
    return total_score 

# Define the expected signature for the batch reward function needed by trainers like TRL GRPO
# TRL GRPOTrainer expects rewards as List[torch.Tensor].
# The **kwargs part is where 'infos': List[Dict[str, Any]] would be passed.
BatchRewardFunction = Callable[..., List[torch.Tensor]]

def _load_reward_setup(
    base_reward_name: str, 
    verifier_names: List[str]
) -> Tuple[Optional[Callable[..., float]], List[VerifierFunction]]:
    """Loads base reward and verifier functions from their registries."""
    base_reward_fn = get_reward_function(base_reward_name)
    if base_reward_fn is None:
        logger.error(f"[RewardSetup] Base reward function '{base_reward_name}' not found in REWARD_REGISTRY. Available: {list(REWARD_REGISTRY.keys())}")
        raise KeyError(f"Base reward '{base_reward_name}' not in REWARD_REGISTRY. Available: {list(REWARD_REGISTRY.keys())}")

    loaded_verifier_fns: List[VerifierFunction] = []
    missing_verifiers = [v_name for v_name in verifier_names if v_name not in VERIFIER_REGISTRY]
    if missing_verifiers:
        logger.error(
            f"[RewardSetup] Verifiers {missing_verifiers} (for reward '{base_reward_name}') not in VERIFIER_REGISTRY. "
            f"Available: {list(VERIFIER_REGISTRY.keys())}"
        )
        raise KeyError(
            f"Verifiers {missing_verifiers} (for reward '{base_reward_name}') not in VERIFIER_REGISTRY. "
            f"Available: {list(VERIFIER_REGISTRY.keys())}"
        )
    for v_name in verifier_names: 
        verifier_func = get_boolean_verifier(v_name) 
        if verifier_func:
            loaded_verifier_fns.append(verifier_func)
        else:
            logger.warning(f"[RewardSetup] Verifier '{v_name}' could not be loaded (get_boolean_verifier returned None). Skipping.")
            
    return base_reward_fn, loaded_verifier_fns

def _process_one_sample(
    prompt: str, 
    completion: str, 
    step_info_or_example: Dict[str, Any],
    base_reward_fn: Callable[..., float],              # Pre-loaded single-instance base reward function
    verifier_fns: List[VerifierFunction],  # Pre-loaded verifier functions
    verifier_names: List[str],             # Names corresponding to verifier_fns (for logging)
    base_reward_name: str,                 # Name of the base reward (for logging)
    failure_penalty: float,                # Penalty if a verifier fails
    base_reward_fn_config_params: Dict[str, Any] # Added: params for the base_reward_fn
) -> float:
    """Applies verifiers and base reward for a single sample."""
    # 1. Apply Verifiers
    for i, verifier_func in enumerate(verifier_fns):
        try:
            # Call with positional arguments to match the VerifierFunction type hint more directly for the linter
            if not verifier_func(prompt, completion, step_info_or_example):
                logger.warning(f"[RewardProcessor] Verifier '{verifier_names[i]}' failed for base reward '{base_reward_name}'. Applying penalty: {failure_penalty}")
                return failure_penalty # Verifier returned False
        except Exception as e:
            # Verifier errored
            logger.error(f"[RewardProcessor] Verifier '{verifier_names[i]}' (for base reward '{base_reward_name}') errored: {e}. Applying penalty: {failure_penalty}", exc_info=True)
            return failure_penalty

    # 2. Calculate Base Reward (if all verifiers passed)
    try:
        # Prepare args for base_reward_fn.
        # The base reward functions (e.g., tool_usage_reward, substring_match_reward)
        # are now expected to take 'config_params' as a dictionary.
        reward_args = {
            "prompt": prompt,
            "completion": completion,
            "config_params": base_reward_fn_config_params # Pass the raw params dict as 'config_params'
        }
        
        # Add step_info if it's provided and the reward function can handle it (e.g., via **kwargs or explicit param).
        # The error messages do not indicate an issue with step_info handling itself.
        if step_info_or_example is not None:
            # Ensure step_info is only added if the base_reward_fn actually expects it.
            # Inspecting the signature or relying on robust **kwargs in reward_fn is ideal.
            # For now, we continue to add it if present, assuming functions can ignore it if not needed.
            # A more robust solution might involve inspecting signature of base_reward_fn.
            sig = inspect.signature(base_reward_fn)
            if "step_info" in sig.parameters:
                 reward_args["step_info"] = step_info_or_example
            elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()): # checks for **kwargs
                 reward_args["step_info"] = step_info_or_example # pass if **kwargs is present
            elif step_info_or_example is not None: # step_info provided but not explicitly accepted and no **kwargs
                logger.trace(f"[RewardProcessor] Step info provided for {base_reward_name} but not explicitly in its signature and no **kwargs. It will be ignored by the reward function.")


        score = base_reward_fn(**reward_args)

        if not isinstance(score, (int, float)):
            logger.error(f"[RewardProcessor] Base reward '{base_reward_name}' returned {type(score)}, not float/int. Applying 0.0.")
            return 0.0 # Default for type error from reward function
        return float(score)
    except TypeError as te:
        # Log a more specific error if the arguments didn't match the function signature
        logger.error(
            f"[RewardProcessor] Base reward function '{base_reward_name}' called with incompatible arguments. Error: {te}. "
            f"Provided args: {list(reward_args.keys())}. Check function signature."
        )
        return 0.0 # Default for type error from reward function
    except Exception as e:
        # Base reward function errored
        logger.error(f"[RewardProcessor] Base reward '{base_reward_name}' errored: {e}. Applying 0.0.", exc_info=True)
        return 0.0 # Default for execution error in reward function

def create_grpo_batch_reward_func(
    base_reward_config: Dict[str, Any],
    default_penalty: float = 0.0
) -> Optional[BatchRewardFunction]:
    """
    Creates a TRL-compatible batch-processing reward function.
    Uses a configured base reward and associated verifiers.
    """
    if not isinstance(base_reward_config, dict) or 'name' not in base_reward_config:
        logger.error("[RewardAdapterFactory] 'base_reward_config' must be a dict with a 'name' key for the base reward function.")
        # Raising ValueError here as it's a fundamental config issue, caught by retrain.run
        raise ValueError("'base_reward_config' must be a dict with a 'name' key for the base reward function.")

    base_reward_name_closure: str = base_reward_config['name']
    raw_verifiers = base_reward_config.get('verifiers')
    verifier_names_closure: List[str] = raw_verifiers if raw_verifiers is not None else []
    penalty_val_closure: float = float(base_reward_config.get('verifier_penalty', default_penalty))
    base_reward_params_closure: Dict[str, Any] = base_reward_config.get('params', {})

    try:
        loaded_base_reward_fn_closure, loaded_verifier_fns_closure = _load_reward_setup(
            base_reward_name_closure, verifier_names_closure
        )
        if loaded_base_reward_fn_closure is None:
            logger.error(f"[RewardAdapterFactory] Failed to load base reward function '{base_reward_name_closure}' in _load_reward_setup (returned None). Cannot create batch processor.")
            return None
    except KeyError as e:
        # Error already logged by _load_reward_setup
        logger.error(f"[RewardAdapterFactory] Error loading reward setup for '{base_reward_name_closure}' due to KeyError: {e}. Cannot create batch processor.")
        return None
    except Exception as e_load: # Catch any other unexpected errors during load
        logger.error(f"[RewardAdapterFactory] Unexpected error loading reward setup for '{base_reward_name_closure}': {e_load}. Cannot create batch processor.", exc_info=True)
        return None

    # Inner function (closure) - this is what gets returned and used as BatchRewardFunction
    def actual_batch_processor(
        prompts: List[str], 
        completions: List[str], 
        **kwargs  # Typically includes 'infos': List[Dict[str, Any]] from TRL
    ) -> List[torch.Tensor]:
        # Pass closure variables explicitly to batch_reward_processor_sync.
        # 'infos' is extracted from kwargs as it's part of the dynamic data from the caller (e.g., TRL).
        return batch_reward_processor_sync(
            prompts=prompts,
            completions=completions,
            infos=kwargs.get("infos"), # Pass 'infos' if available in kwargs
            # Explicitly pass the captured closure variables
            loaded_base_reward_fn=loaded_base_reward_fn_closure,
            loaded_verifier_fns=loaded_verifier_fns_closure,
            verifier_names=verifier_names_closure,
            base_reward_name=base_reward_name_closure,
            penalty_val=penalty_val_closure,
            base_reward_params=base_reward_params_closure
        )

    name_suffix = f"_verified_by_{'_and_'.join(verifier_names_closure)}" if verifier_names_closure else ""
    processor_name = f"batch_adapter_sync_for_{base_reward_name_closure}{name_suffix}"
    try:
        actual_batch_processor.__name__ = processor_name
    except (TypeError, AttributeError):
        # Setting __name__ can fail on some callable types, not critical.
        logger.trace(f"[RewardAdapterFactory] Could not set __name__ for processor {processor_name}.")
        pass 

    return actual_batch_processor

# This function is called by TRL GRPOTrainer and must be synchronous.
def batch_reward_processor_sync( 
    prompts: List[str], 
    completions: List[str], 
    *, # Make subsequent arguments keyword-only for clarity
    infos: Optional[List[Dict[str, Any]]], # Explicitly accept 'infos'
    loaded_base_reward_fn: Callable[..., float],
    loaded_verifier_fns: List[VerifierFunction],
    verifier_names: List[str],
    base_reward_name: str,
    penalty_val: float,
    base_reward_params: Dict[str, Any]
    # **kwargs removed as specific parameters are now named
) -> List[torch.Tensor]: # TRL expects a list of tensors as rewards
    
    logger.debug(f"[TRL Adapter] batch_reward_processor_sync (for {base_reward_name}): Batch size: {len(prompts)}")
    # logger.trace(f"[TRL Adapter] batch_reward_processor_sync: All kwargs received: {list(kwargs.keys())}") # kwargs removed

    # infos_list = kwargs.get("infos") # 'infos' is now a direct parameter
    infos_list = infos # Use the direct parameter

    if infos_list is None:
        logger.warning(f"[TRL Adapter] batch_reward_processor_sync (for {base_reward_name}): 'infos' (step_info) was None. Defaulting to empty dicts for each sample.")
        infos_list = [{}] * len(prompts)
    elif not isinstance(infos_list, list) or len(infos_list) != len(prompts):
        logger.warning(
            f"[TRL Adapter] batch_reward_processor_sync (for {base_reward_name}): 'infos' has unexpected type ({type(infos_list)}) or length "
            f"(got {len(infos_list) if isinstance(infos_list, list) else 'N/A'}, expected {len(prompts)}). "
            f"Defaulting to empty dicts."
        )
        infos_list = [{}] * len(prompts) # Fallback
    else:
        logger.trace(f"[TRL Adapter] batch_reward_processor_sync (for {base_reward_name}): 'infos' (step_info) found and has length {len(infos_list)}.")

    individual_reward_values: List[Union[float, Exception]] = [] # Allow Exception for error handling
    
    # These variables are from the closure of create_grpo_batch_reward_func
    # For the linter to be happier without passing them explicitly, we'd need to restructure or accept warnings.
    # This edit focuses on the direct linter errors mentioned.
    # Assuming: loaded_base_reward_fn, loaded_verifier_fns, verifier_names, base_reward_name, penalty_val, base_reward_params
    # are available from the closure as intended.

    # For _process_one_sample, we need the actual closure variables. The linter won't see them.
    # This is a known complexity with linters and closures. We assume they are correctly captured.

    for i in range(len(prompts)):
        current_step_info_for_sample = infos_list[i]
        try:
            # These variables are now passed as explicit arguments
            res_val = _process_one_sample(
                prompt=prompts[i],
                completion=completions[i],
                step_info_or_example=current_step_info_for_sample,
                base_reward_fn=loaded_base_reward_fn, 
                verifier_fns=loaded_verifier_fns,
                verifier_names=verifier_names,
                base_reward_name=base_reward_name,
                failure_penalty=penalty_val,
                base_reward_fn_config_params=base_reward_params
            )
            individual_reward_values.append(res_val)
        except KeyError as e_closure_key: # This specific KeyError for closure variables should no longer occur with direct args
             logger.error(f"[TRL Adapter] batch_reward_processor_sync (for {base_reward_name}): Missing expected variable ({e_closure_key}) for _process_one_sample. This is an internal setup error. Sample {i}", exc_info=True)
             individual_reward_values.append(ValueError(f"Internal setup error: {e_closure_key}")) # Append an error
        except Exception as e:
            logger.error(f"[TRL Adapter] batch_reward_processor_sync (for {base_reward_name}): Error processing sample {i} with _process_one_sample: {e}", exc_info=True)
            individual_reward_values.append(e) 

    final_rewards_tensors: List[torch.Tensor] = []
    for i, res_val in enumerate(individual_reward_values):
        if isinstance(res_val, Exception):
            logger.error(f"[TRL Adapter] batch_reward_processor_sync (for {base_reward_name}): Converting stored exception for sample {i} to 0.0 reward. Exception: {res_val}")
            final_rewards_tensors.append(torch.tensor(0.0, dtype=torch.float32))
        else:
            try:
                final_rewards_tensors.append(torch.tensor(float(res_val), dtype=torch.float32))
            except (ValueError, TypeError) as e_conversion:
                logger.error(f"[TRL Adapter] batch_reward_processor_sync (for {base_reward_name}): Could not convert reward value '{res_val}' to float for sample {i}. Error: {e_conversion}. Defaulting to 0.0.", exc_info=True)
                final_rewards_tensors.append(torch.tensor(0.0, dtype=torch.float32))
        
    logger.debug(f"[TRL Adapter] batch_reward_processor_sync (for {base_reward_name}): Final reward tensors: {final_rewards_tensors}")
    return final_rewards_tensors
