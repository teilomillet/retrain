from typing import Dict, Callable, Optional, List, Any, Tuple, Union
import inspect
from loguru import logger
import torch
import time
import ray

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
        registration_name = name if name is not None else getattr(func, '__name__', 'unknown')
        if registration_name in REWARD_REGISTRY:
            logger.warning(
                f"[RewardRegistry] Reward function '{registration_name}' is already registered. "
                f"Overwriting with {getattr(func, '__module__', 'unknown')}.{getattr(func, '__name__', 'unknown')}"
            )
        REWARD_REGISTRY[registration_name] = func
        logger.debug(f"[RewardRegistry] Registered reward function: '{registration_name}' -> {getattr(func, '__module__', 'unknown')}.{getattr(func, '__name__', 'unknown')}")
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
        assert loaded_base_reward_fn_closure is not None  # Already checked above
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





@ray.remote
class ReReward:
    """
    Reward Calculator Ray Actor.
    
    Manages reward computation for training data using
    multiple reward functions with parallel processing.
    
    This actor handles:
    1. Loading and managing reward functions from the registry
    2. Batch reward computation for training data
    3. Multi-component reward aggregation
    4. Integration with DataBuffer for reward workflow
    """
    
    def __init__(self, config: Any, databuffer: ray.ObjectRef):
        """
        Initialize ReReward actor with configuration and databuffer reference.
        
        Args:
            config: Training configuration object
            databuffer: Reference to the ReDataBuffer actor for data coordination
        """
        # Initialize clean logging for this Ray actor to avoid pickle issues
        # Each actor needs its own logger setup to prevent serialization conflicts
        try:
            from loguru import logger as actor_logger
            actor_logger.remove()  # Remove inherited handlers that can't be pickled
            import sys
            actor_logger.add(sys.stderr, level="INFO")  # Add clean stderr handler
        except ImportError:
            pass  # Fallback gracefully if loguru not available
            
        self.config = config
        self.databuffer = databuffer
        self.reward_functions: List[Callable] = []
        self.reward_weights: Dict[str, float] = {}
        self.reward_stats: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False
        
        # Reward computation tracking
        self.total_rewards_computed = 0
        self.reward_computation_times = []
        self.reward_value_history = []
        
        logger.info("ReReward actor initialized")
        
    async def initialize(self) -> None:
        """Initialize reward functions based on configuration."""
        logger.info("Initializing ReReward...")
        
        try:
            # Load reward configuration
            reward_config = getattr(self.config, 'rewards', {})
            
            if not reward_config:
                # Use default reward configuration
                reward_config = {
                    'functions': ['accuracy_reward'],
                    'weights': {'accuracy_reward': 1.0}
                }
                logger.info("No reward config specified, using defaults")
            
            # Load reward functions
            await self._load_reward_functions(reward_config)
            
            # Load custom reward functions if specified
            custom_rewards = reward_config.get('custom_functions', [])
            for custom_reward in custom_rewards:
                if callable(custom_reward):
                    self.reward_functions.append(custom_reward)
                    func_name = getattr(custom_reward, '__name__', 'custom_reward')
                    self.reward_weights[func_name] = reward_config.get('weights', {}).get(func_name, 1.0)
                    
            if not self.reward_functions:
                logger.warning("No reward functions loaded - using default constant reward")
                self.reward_functions = [self._default_reward_function]
                self.reward_weights['default'] = 1.0
                
            self.is_initialized = True
            logger.info(f"ReReward initialized with {len(self.reward_functions)} reward functions")
            
        except Exception as e:
            logger.error(f"Failed to initialize ReReward: {e}")
            raise
            
    async def _load_reward_functions(self, reward_config: Dict[str, Any]) -> None:
        """Load reward functions from registry and configuration."""
        function_names = reward_config.get('functions', [])
        weights = reward_config.get('weights', {})
        
        for func_name in function_names:
            try:
                # Try to get from REWARD_REGISTRY first
                if func_name in REWARD_REGISTRY:
                    reward_func = REWARD_REGISTRY[func_name]
                    self.reward_functions.append(reward_func)
                    self.reward_weights[func_name] = weights.get(func_name, 1.0)
                    
                    # Initialize stats for this reward function
                    self.reward_stats[func_name] = {
                        'calls': 0,
                        'total_value': 0.0,
                        'average_value': 0.0,
                        'min_value': float('inf'),
                        'max_value': float('-inf'),
                        'average_time': 0.0,
                        'errors': 0
                    }
                    
                    logger.info(f"Loaded reward function: {func_name}")
                    
                else:
                    # Try to load built-in reward functions
                    if func_name == 'accuracy_reward':
                        self.reward_functions.append(self._accuracy_reward)
                        self.reward_weights['accuracy_reward'] = weights.get('accuracy_reward', 1.0)
                        self.reward_stats['accuracy_reward'] = self._init_reward_stats()
                    elif func_name == 'efficiency_reward':
                        self.reward_functions.append(self._efficiency_reward)
                        self.reward_weights['efficiency_reward'] = weights.get('efficiency_reward', 0.3)
                        self.reward_stats['efficiency_reward'] = self._init_reward_stats()
                    elif func_name == 'quality_reward':
                        self.reward_functions.append(self._quality_reward)
                        self.reward_weights['quality_reward'] = weights.get('quality_reward', 0.5)
                        self.reward_stats['quality_reward'] = self._init_reward_stats()
                    else:
                        logger.warning(f"Unknown reward function: {func_name}")
                        
            except Exception as e:
                logger.error(f"Failed to load reward function {func_name}: {e}")
                
    def _init_reward_stats(self) -> Dict[str, Any]:
        """Initialize statistics dictionary for a reward function."""
        return {
            'calls': 0,
            'total_value': 0.0,
            'average_value': 0.0,
            'min_value': float('inf'),
            'max_value': float('-inf'),
            'average_time': 0.0,
            'errors': 0
        }
        
    async def compute_rewards(self, rollout_data: List[Dict[str, Any]], episode_id: int) -> List[float]:
        """
        Compute rewards for rollout data.
        
        Args:
            rollout_data: List of rollout data to process
            episode_id: Current episode identifier
            
        Returns:
            List of computed reward values
        """
        if not self.is_initialized:
            raise RuntimeError("ReReward not initialized")
            
        logger.info(f"Computing rewards for {len(rollout_data)} rollouts in episode {episode_id}")
        
        start_time = time.time()
        rewards = []
        
        for i, rollout in enumerate(rollout_data):
            try:
                reward = await self._compute_single_reward(rollout, episode_id, i)
                rewards.append(reward)
                
            except Exception as e:
                logger.error(f"Failed to compute reward for rollout {i}: {e}")
                rewards.append(0.0)  # Default reward for failed computation
                
        computation_time = time.time() - start_time
        self.reward_computation_times.append(computation_time)
        self.total_rewards_computed += len(rewards)
        
        # Update global statistics
        self.reward_value_history.extend(rewards)
        
        # Keep only recent history to prevent memory growth
        if len(self.reward_value_history) > 1000:
            self.reward_value_history = self.reward_value_history[-1000:]
            
        logger.info(f"Computed {len(rewards)} rewards in {computation_time:.2f}s")
        return rewards
        
    async def _compute_single_reward(self, rollout: Dict[str, Any], episode_id: int, rollout_idx: int) -> float:
        """Compute reward for a single rollout."""
        if not self.reward_functions:
            return 0.0
            
        total_reward = 0.0
        total_weight = 0.0
        
        for reward_func in self.reward_functions:
            func_name = getattr(reward_func, '__name__', 'unknown')
            weight = self.reward_weights.get(func_name, 1.0)
            
            start_time = time.time()
            
            try:
                # Compute individual reward
                # Handle both underscore-prefixed and regular built-in function names
                builtin_functions = ['accuracy_reward', 'efficiency_reward', 'quality_reward', 
                                   '_accuracy_reward', '_efficiency_reward', '_quality_reward']
                if func_name in builtin_functions:
                    # Built-in functions that return (value, weight) tuple
                    reward_value, func_weight = reward_func(rollout)
                    weight = func_weight  # Use function's own weight
                else:
                    # Registry functions or custom functions
                    reward_value = reward_func(rollout)
                    
                computation_time = time.time() - start_time
                
                # Update function statistics
                if func_name in self.reward_stats:
                    stats = self.reward_stats[func_name]
                    stats['calls'] += 1
                    stats['total_value'] += reward_value
                    stats['average_value'] = stats['total_value'] / stats['calls']
                    stats['min_value'] = min(stats['min_value'], reward_value)
                    stats['max_value'] = max(stats['max_value'], reward_value)
                    
                    # Update average computation time
                    old_avg = stats['average_time']
                    new_avg = ((old_avg * (stats['calls'] - 1)) + computation_time) / stats['calls']
                    stats['average_time'] = new_avg
                    
                # Add to total weighted reward with type safety
                if isinstance(reward_value, (int, float)) and isinstance(weight, (int, float)):
                    total_reward += float(reward_value) * float(weight)
                    total_weight += float(weight)
                else:
                    logger.error(f"Invalid types for reward computation: reward_value={type(reward_value)}, weight={type(weight)}")
                    # Use default values for invalid types
                    total_reward += 0.0
                    total_weight += 1.0
                
            except Exception as e:
                logger.error(f"Reward function {func_name} failed for rollout {rollout_idx}: {e}")
                
                # Update error statistics
                if func_name in self.reward_stats:
                    self.reward_stats[func_name]['errors'] += 1
                    
        # Normalize by total weight
        if total_weight > 0:
            return total_reward / total_weight
        else:
            return 0.0
            
    def _accuracy_reward(self, rollout: Dict[str, Any]) -> tuple[float, float]:
        """Built-in accuracy-based reward function."""
        # Check for success indicators in the rollout
        success = rollout.get('success', False)
        
        # Check environment success if available
        if not success and 'environment_success' in rollout:
            success = rollout['environment_success']
            
        # Check verification results if available
        if not success and 'verification_results' in rollout:
            verification = rollout['verification_results']
            if isinstance(verification, dict):
                success = verification.get('overall_passed', False)
                
        # Check conversation completion
        if not success and 'conversation_history' in rollout:
            conversation = rollout['conversation_history']
            if isinstance(conversation, list) and conversation:
                last_msg = conversation[-1]
                if isinstance(last_msg, dict):
                    success = not last_msg.get('error', False)
                    
        return (1.0 if success else 0.0, 1.0)
        
    def _efficiency_reward(self, rollout: Dict[str, Any]) -> tuple[float, float]:
        """Built-in efficiency-based reward function."""
        # Reward based on number of steps/actions taken
        steps = rollout.get('steps', 1)
        max_steps = rollout.get('max_steps', 10)
        
        # Check conversation length as proxy for steps
        if steps == 1 and 'conversation_history' in rollout:
            conversation = rollout['conversation_history']
            if isinstance(conversation, list):
                steps = len(conversation)
                
        # Check action count
        if 'executed_llm_actions' in rollout:
            actions = rollout['executed_llm_actions']
            if isinstance(actions, list):
                steps = max(steps, len(actions))
                
        # Efficiency score: fewer steps = higher reward
        if max_steps > 0:
            efficiency = max(0, (max_steps - steps) / max_steps)
        else:
            efficiency = 1.0 if steps == 1 else 0.5
            
        return (efficiency, 0.3)
        
    def _quality_reward(self, rollout: Dict[str, Any]) -> tuple[float, float]:
        """Built-in quality-based reward function."""
        quality_score = 0.5  # Default neutral quality
        
        # Use verification confidence if available
        if 'verification_results' in rollout:
            verification = rollout['verification_results']
            if isinstance(verification, dict):
                quality_score = verification.get('confidence_score', 0.5)
                
        # Use environment processing quality if available
        elif 'processing_status' in rollout:
            status = rollout['processing_status']
            if status == 'success':
                quality_score = 0.8
            elif status == 'partial':
                quality_score = 0.5
            else:
                quality_score = 0.2
                
        # Check for processing errors
        if rollout.get('processing_error') or rollout.get('verification_error'):
            quality_score *= 0.5
            
        return (quality_score, 0.5)
        
    def _default_reward_function(self, rollout: Dict[str, Any]) -> float:
        """Default reward function when no others are available."""
        # Simple constant reward to prevent training failure
        return 0.5
        
    async def create_batch_reward_function(self, base_reward_config: Dict[str, Any]) -> Optional[Callable]:
        """Create a batch reward function compatible with training libraries."""
        try:
            # Create batch processor using existing functionality
            batch_func = create_grpo_batch_reward_func(
                base_reward_config=base_reward_config,
                default_penalty=0.0
            )
            
            if batch_func:
                logger.info(f"Created batch reward function for: {base_reward_config.get('name', 'unknown')}")
                return batch_func
            else:
                logger.error("Failed to create batch reward function")
                return None
                
        except Exception as e:
            logger.error(f"Error creating batch reward function: {e}")
            return None
            
    async def get_reward_status(self) -> Dict[str, Any]:
        """Get current status of reward computation."""
        status = {
            'initialized': self.is_initialized,
            'reward_function_count': len(self.reward_functions),
            'total_rewards_computed': self.total_rewards_computed,
            'average_computation_time': 0.0,
            'average_reward_value': 0.0,
            'reward_functions': {}
        }
        
        # Calculate average computation time
        if self.reward_computation_times:
            status['average_computation_time'] = sum(self.reward_computation_times) / len(self.reward_computation_times)
            
        # Calculate average reward value
        if self.reward_value_history:
            status['average_reward_value'] = sum(self.reward_value_history) / len(self.reward_value_history)
            
        # Add individual function statistics
        for func_name, stats in self.reward_stats.items():
            status['reward_functions'][func_name] = {
                **stats,
                'weight': self.reward_weights.get(func_name, 1.0)
            }
            
        return status
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        health_status = {
            'status': 'healthy',
            'initialized': self.is_initialized,
            'function_count': len(self.reward_functions),
            'total_computed': self.total_rewards_computed,
            'timestamp': time.time()
        }
        
        # Check for performance issues
        if self.reward_computation_times:
            avg_time = sum(self.reward_computation_times) / len(self.reward_computation_times)
            if avg_time > 5.0:  # Warning if reward computation takes >5s on average
                health_status['status'] = 'warning'
                health_status['warning'] = f'Slow reward computation: {avg_time:.2f}s average'
                
        # Check for high error rates
        total_errors = sum(stats['errors'] for stats in self.reward_stats.values())
        total_calls = sum(stats['calls'] for stats in self.reward_stats.values())
        
        if total_calls > 0:
            error_rate = total_errors / total_calls
            if error_rate > 0.1:  # Warning if >10% error rate
                health_status['status'] = 'warning'
                health_status['warning'] = f'High reward error rate: {error_rate:.1%}'
                
        # Check reward value distribution
        if self.reward_value_history:
            avg_reward = sum(self.reward_value_history) / len(self.reward_value_history)
            if avg_reward <= 0.0:  # Warning if average reward is zero or negative
                health_status['status'] = 'warning'
                health_status['warning'] = f'Low average reward: {avg_reward:.3f}'
                
        return health_status
        
    async def add_reward_function(self, func_name: str, reward_func: Callable, weight: float = 1.0) -> bool:
        """
        Dynamically add a new reward function.
        
        Args:
            func_name: Name for the reward function
            reward_func: Reward function to add
            weight: Weight for this reward function
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            if func_name in [f.__name__ for f in self.reward_functions]:
                logger.warning(f"Reward function {func_name} already exists, overwriting")
                
            self.reward_functions.append(reward_func)
            self.reward_weights[func_name] = weight
            self.reward_stats[func_name] = self._init_reward_stats()
            
            logger.info(f"Added reward function: {func_name} (weight: {weight})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add reward function {func_name}: {e}")
            return False
            
    async def remove_reward_function(self, func_name: str) -> bool:
        """
        Remove a reward function.
        
        Args:
            func_name: Name of reward function to remove
            
        Returns:
            True if successfully removed, False otherwise
        """
        try:
            # Find and remove function
            for i, func in enumerate(self.reward_functions):
                if getattr(func, '__name__', '') == func_name:
                    del self.reward_functions[i]
                    if func_name in self.reward_weights:
                        del self.reward_weights[func_name]
                    if func_name in self.reward_stats:
                        del self.reward_stats[func_name]
                    logger.info(f"Removed reward function: {func_name}")
                    return True
                    
            logger.warning(f"Reward function {func_name} not found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove reward function {func_name}: {e}")
            return False
            
    async def shutdown(self) -> None:
        """Gracefully shutdown the reward actor."""
        logger.info("Shutting down ReReward...")
        
        self.reward_functions.clear()
        self.reward_weights.clear()
        self.reward_stats.clear()
        self.reward_computation_times.clear()
        self.reward_value_history.clear()
        self.is_initialized = False
        
        logger.info("ReReward shutdown complete")
