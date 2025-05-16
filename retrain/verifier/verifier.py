from typing import Callable, Dict, Any, List, Optional, Union
import inspect
import functools
from retrain.utils.logging_utils import get_logger

# Define the expected signature for verifier functions (simple boolean checks)
VerifierFunction = Callable[[str, str, Dict[str, Any]], bool]

# Global registry for verifier functions
VERIFIER_REGISTRY: Dict[str, VerifierFunction] = {}

logger = get_logger(__name__)

# Updated decorator implementation to handle both @verifier and @verifier(name=...)
def verifier(_func: Optional[VerifierFunction] = None, *, name: Optional[str] = None) -> Union[Callable[[VerifierFunction], VerifierFunction], VerifierFunction]:
    """
    Decorator to register a new verifier function (a simple boolean check).

    Can be used in two ways:
    - `@verifier`: Registers the function using its own `__name__`.
    - `@verifier(name="custom_name")`: Registers the function using the specified name.
    """
    def decorator_verifier(func: VerifierFunction) -> VerifierFunction:
        if not callable(func):
            raise TypeError(f"Object {getattr(func, '__name__', '<unknown>')} must be a callable function to be registered as a verifier.")
        
        registration_name = name if name is not None else func.__name__

        sig = inspect.signature(func)
        if len(sig.parameters) < 3:
             logger.warning(f"Verifier function '{registration_name}' (func: {func.__name__}) has fewer than 3 expected parameters (prompt, completion, example). Ensure signature compatibility.")

        if registration_name in VERIFIER_REGISTRY:
            raise ValueError(f"Verifier function with name '{registration_name}' already registered.")

        VERIFIER_REGISTRY[registration_name] = func
        logger.info(f"Registered verifier function: '{registration_name}' -> {func.__name__}")
        
        # Return the original function (decorators typically return the decorated function)
        return func

    if _func is None:
        # Decorator was called with parentheses: @verifier() or @verifier(name=...)
        # Return the actual decorator function that waits for the function to decorate.
        return decorator_verifier
    elif callable(_func):
        # Decorator was called without parentheses: @verifier
        # The decorated function (_func) was passed directly.
        if name is not None:
             # This case should ideally be caught by static analysis or raise an error earlier,
             # but this runtime check ensures correct usage.
             raise TypeError("Cannot specify 'name' when using @verifier without parentheses. Use @verifier(name='...') instead.")
        # Apply the registration logic immediately to _func
        return decorator_verifier(_func)
    else:
        # Should not happen with standard decorator usage
        raise TypeError("Invalid arguments supplied to @verifier decorator.")

def get_boolean_verifier(name: str) -> VerifierFunction:
    """Retrieves a registered boolean verifier function by name."""
    verifier_func = VERIFIER_REGISTRY.get(name)
    if verifier_func is None:
        raise KeyError(f"Boolean verifier function '{name}' not found in registry. Available: {list(VERIFIER_REGISTRY.keys())}")
    return verifier_func

def apply_verifiers_to_reward(
    original_reward_func: Callable,    # Expected signature: (prompt, completion, example, **kwargs) -> float
    verifier_names: List[str],
    penalty_on_any_failure: float = 0.0
) -> Callable: # Returns a function with the same signature as original_reward_func
    """
    Processes an original reward function by first running a list of specified boolean verifiers.

    If any verifier in the list fails (returns False or errors), the processed function
    returns the `penalty_on_any_failure`.
    Otherwise, the `original_reward_func` is executed and its result is returned.
    """
    loaded_verifier_funcs: List[VerifierFunction] = []
    missing_verifiers = []
    for name_to_load in verifier_names:
        try:
            loaded_verifier_funcs.append(get_boolean_verifier(name_to_load))
        except KeyError:
            missing_verifiers.append(name_to_load)
        except Exception as e:
            logger.error(f"Error loading boolean verifier function '{name_to_load}' for reward processing: {e}")
            raise
            
    if missing_verifiers:
        raise ValueError(f"Could not find boolean verifier functions {missing_verifiers} specified for reward function '{getattr(original_reward_func, '__name__', 'unknown')}'. Available: {list(VERIFIER_REGISTRY.keys())}")

    def reward_function_with_verifiers(prompt: str, completion: str, example: Dict[str, Any], **kwargs: Any) -> float:
        for i, verifier_func in enumerate(loaded_verifier_funcs):
            current_verifier_name = verifier_names[i]
            try:
                if not verifier_func(prompt=prompt, completion=completion, example=example): # type: ignore[call-arg]
                    return penalty_on_any_failure
            except Exception as e:
                logger.error(f"Error during verifier '{current_verifier_name}' execution: {e}. Applying penalty.")
                return penalty_on_any_failure

        try:
            result = original_reward_func(prompt=prompt, completion=completion, example=example, **kwargs)
            return float(result) 
        except Exception as e:
            logger.error(f"Error during original reward function '{getattr(original_reward_func, '__name__', 'unknown_reward_func')}' execution (after verifiers passed): {e}. Returning 0.0 as default error penalty.")
            return 0.0
    
    original_rf_name = getattr(original_reward_func, '__name__', 'reward_func')
    if verifier_names:
        verifier_name_suffix = '_and_'.join(verifier_names)
        reward_function_with_verifiers.__name__ = f"{original_rf_name}_verified_by_{verifier_name_suffix}"
    else:
        reward_function_with_verifiers.__name__ = original_rf_name 
        
    return reward_function_with_verifiers 