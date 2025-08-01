import logging
import time
import ray
from typing import Callable, Dict, Any, List, Optional
import inspect
from retrain.utils.logging_utils import get_logger

logger = logging.getLogger(__name__)

# Define the expected signature for verifier functions (simple boolean checks)
VerifierFunction = Callable[[str, str, Dict[str, Any]], bool]

# Global registry for verifier functions
VERIFIER_REGISTRY: Dict[str, VerifierFunction] = {}

logger = get_logger(__name__)

def verifier(name: Optional[str] = None) -> Callable[[VerifierFunction], VerifierFunction]:
    """
    Decorator to register a new verifier function. Must be used with parentheses.

    Usage:
    - `@verifier()`: Registers the function using its own `__name__`.
    - `@verifier(name="custom_name")`: Registers the function using a specific name.
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
        
        return func

    return decorator_verifier

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


@ray.remote
class ReVerifier:
    """
    Verifier Manager Ray Actor.
    
    Manages verification functions and processes verification
    requests for training data quality assurance.
    
    This actor handles:
    1. Loading and managing verifier functions from the registry
    2. Batch verification processing for training data
    3. Quality scoring and confidence computation
    4. Integration with DataBuffer for verification workflow
    """
    
    def __init__(self, config: Any, databuffer: ray.ObjectRef):
        """
        Initialize ReVerifier actor with configuration and databuffer reference.
        
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
        self.active_verifiers: Dict[str, VerifierFunction] = {}
        self.verifier_stats: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False
        
        # Verification tracking
        self.total_verifications = 0
        self.total_passed = 0
        self.verification_times = []
        
        logger.info("ReVerifier actor initialized")
        
    async def initialize(self) -> None:
        """Initialize verifier functions based on configuration."""
        logger.info("Initializing ReVerifier...")
        
        try:
            # Load verifiers from config
            verifier_config = getattr(self.config, 'verifiers', [])
            
            if not verifier_config:
                # Use default verifiers if none specified
                verifier_config = ['default_verifier'] if 'default_verifier' in VERIFIER_REGISTRY else []
                logger.info("No verifiers specified in config, using defaults")
            
            # Load each verifier from the registry
            for verifier_name in verifier_config:
                if verifier_name in VERIFIER_REGISTRY:
                    self.active_verifiers[verifier_name] = VERIFIER_REGISTRY[verifier_name]
                    
                    # Initialize stats for this verifier
                    self.verifier_stats[verifier_name] = {
                        'calls': 0,
                        'passed': 0,
                        'failed': 0,
                        'errors': 0,
                        'average_time': 0.0,
                        'pass_rate': 0.0
                    }
                    
                    logger.info(f"Loaded verifier: {verifier_name}")
                else:
                    logger.warning(f"Verifier not found in registry: {verifier_name}")
                    
            if not self.active_verifiers:
                logger.warning("No verifiers loaded - all verifications will pass by default")
                
            self.is_initialized = True
            logger.info(f"ReVerifier initialized with {len(self.active_verifiers)} verifiers")
            
        except Exception as e:
            logger.error(f"Failed to initialize ReVerifier: {e}")
            raise
            
    async def verify_rollouts(self, rollout_data: List[Dict[str, Any]], episode_id: int) -> List[Dict[str, Any]]:
        """
        Verify rollout data quality.
        
        Args:
            rollout_data: List of rollout data to verify
            episode_id: Current episode identifier
            
        Returns:
            List of verification results for each rollout
        """
        if not self.is_initialized:
            raise RuntimeError("ReVerifier not initialized")
            
        logger.info(f"Verifying {len(rollout_data)} rollouts for episode {episode_id}")
        
        verification_results = []
        
        for i, rollout in enumerate(rollout_data):
            try:
                verification = await self._verify_single_rollout(rollout, episode_id, i)
                verification_results.append(verification)
                
            except Exception as e:
                logger.error(f"Failed to verify rollout {i}: {e}")
                # Create error verification result
                error_verification = {
                    'rollout_id': rollout.get('id', f'rollout_{episode_id}_{i}'),
                    'episode_id': episode_id,
                    'rollout_idx': i,
                    'verifications': {},
                    'overall_passed': False,
                    'confidence_score': 0.0,
                    'verification_error': str(e),
                    'verification_status': 'error'
                }
                verification_results.append(error_verification)
                
        # Update statistics
        await self._update_verifier_stats(verification_results)
        
        logger.info(f"Completed verification of {len(verification_results)} rollouts")
        return verification_results
        
    async def _verify_single_rollout(self, rollout: Dict[str, Any], episode_id: int, rollout_idx: int) -> Dict[str, Any]:
        """Verify a single rollout."""
        start_time = time.time()
        
        verification_result = {
            'rollout_id': rollout.get('id', f'rollout_{episode_id}_{rollout_idx}'),
            'episode_id': episode_id,
            'rollout_idx': rollout_idx,
            'verifications': {},
            'overall_passed': True,
            'confidence_score': 1.0,
            'verification_time': 0.0,
            'verification_status': 'success'
        }
        
        # Extract data for verification
        prompt = rollout.get('prompt', '')
        completion = rollout.get('completion', '')
        
        # Try to extract completion from conversation history if not directly available
        if not completion and 'conversation_history' in rollout:
            conversation = rollout['conversation_history']
            if conversation and isinstance(conversation, list):
                # Get the last assistant message as completion
                for msg in reversed(conversation):
                    if isinstance(msg, dict) and msg.get('role') == 'assistant':
                        completion = msg.get('content', '')
                        break
                        
        # Use executed actions if available for more detailed verification
        executed_actions = rollout.get('executed_llm_actions', [])
        if executed_actions and not completion:
            # Extract completion from the last action
            last_action = executed_actions[-1] if executed_actions else {}
            completion = last_action.get('raw_llm_output', '')
            
        example = rollout.get('example', {})
        
        # If no verifiers are active, pass by default
        if not self.active_verifiers:
            verification_result['verifications']['default'] = {
                'passed': True,
                'message': 'No verifiers configured - passed by default'
            }
            verification_time = time.time() - start_time
            verification_result['verification_time'] = verification_time
            return verification_result
            
        # Run each verifier
        for verifier_name, verifier_func in self.active_verifiers.items():
            verifier_start = time.time()
            
            try:
                passed = verifier_func(prompt, completion, example)
                verifier_time = time.time() - verifier_start
                
                verification_result['verifications'][verifier_name] = {
                    'passed': passed,
                    'error': None,
                    'time': verifier_time
                }
                
                # Update overall result
                if not passed:
                    verification_result['overall_passed'] = False
                    verification_result['confidence_score'] *= 0.8  # Reduce confidence for failures
                    
                # Update verifier stats
                self.verifier_stats[verifier_name]['calls'] += 1
                if passed:
                    self.verifier_stats[verifier_name]['passed'] += 1
                else:
                    self.verifier_stats[verifier_name]['failed'] += 1
                    
                # Update average time
                stats = self.verifier_stats[verifier_name]
                old_avg = stats['average_time']
                new_avg = ((old_avg * (stats['calls'] - 1)) + verifier_time) / stats['calls']
                stats['average_time'] = new_avg
                
            except Exception as e:
                verifier_time = time.time() - verifier_start
                logger.error(f"Verifier {verifier_name} failed: {e}")
                
                verification_result['verifications'][verifier_name] = {
                    'passed': False,
                    'error': str(e),
                    'time': verifier_time
                }
                
                # Treat errors as failures
                verification_result['overall_passed'] = False
                verification_result['confidence_score'] *= 0.5  # Significantly reduce confidence for errors
                
                # Update error stats
                self.verifier_stats[verifier_name]['calls'] += 1
                self.verifier_stats[verifier_name]['errors'] += 1
                
        # Calculate final confidence score based on individual verifier results
        if verification_result['verifications']:
            passed_count = sum(
                1 for v in verification_result['verifications'].values() 
                if v['passed'] and not v['error']
            )
            total_count = len(verification_result['verifications'])
            
            # Base confidence on pass rate, but reduce for any errors
            base_confidence = passed_count / total_count
            error_count = sum(
                1 for v in verification_result['verifications'].values() 
                if v['error']
            )
            
            # Reduce confidence further for errors
            error_penalty = error_count * 0.2
            verification_result['confidence_score'] = max(0.0, base_confidence - error_penalty)
            
        verification_time = time.time() - start_time
        verification_result['verification_time'] = verification_time
        self.verification_times.append(verification_time)
        
        # Update global stats
        self.total_verifications += 1
        if verification_result['overall_passed']:
            self.total_passed += 1
            
        return verification_result
        
    async def _update_verifier_stats(self, verification_results: List[Dict[str, Any]]) -> None:
        """Update verifier statistics based on verification results."""
        for verifier_name in self.verifier_stats:
            stats = self.verifier_stats[verifier_name]
            
            # Update pass rate
            if stats['calls'] > 0:
                stats['pass_rate'] = stats['passed'] / stats['calls']
                
    async def get_verifier_status(self) -> Dict[str, Any]:
        """Get current status of all verifiers."""
        status = {
            'initialized': self.is_initialized,
            'active_verifier_count': len(self.active_verifiers),
            'total_verifications': self.total_verifications,
            'total_passed': self.total_passed,
            'overall_pass_rate': self.total_passed / max(1, self.total_verifications),
            'average_verification_time': sum(self.verification_times) / max(1, len(self.verification_times)),
            'verifiers': {}
        }
        
        for verifier_name, stats in self.verifier_stats.items():
            verifier_status = stats.copy()
            verifier_status['available'] = verifier_name in self.active_verifiers
            status['verifiers'][verifier_name] = verifier_status
            
        return status
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        health_status = {
            'status': 'healthy',
            'initialized': self.is_initialized,
            'verifier_count': len(self.active_verifiers),
            'total_verifications': self.total_verifications,
            'overall_pass_rate': self.total_passed / max(1, self.total_verifications),
            'timestamp': time.time()
        }
        
        # Check for performance issues
        if self.verification_times:
            avg_time = sum(self.verification_times) / len(self.verification_times)
            if avg_time > 1.0:  # Warning if verifications take >1s on average
                health_status['status'] = 'warning'
                health_status['warning'] = f'Slow verification: {avg_time:.2f}s average'
                
        # Check for high error rates
        total_errors = sum(stats['errors'] for stats in self.verifier_stats.values())
        if self.total_verifications > 0:
            error_rate = total_errors / self.total_verifications
            if error_rate > 0.1:  # Warning if >10% error rate
                health_status['status'] = 'warning'
                health_status['warning'] = f'High verification error rate: {error_rate:.1%}'
                
        return health_status
        
    async def add_verifier(self, verifier_name: str, verifier_func: VerifierFunction) -> bool:
        """
        Dynamically add a new verifier function.
        
        Args:
            verifier_name: Name for the verifier
            verifier_func: Verifier function to add
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            if verifier_name in self.active_verifiers:
                logger.warning(f"Verifier {verifier_name} already exists, overwriting")
                
            self.active_verifiers[verifier_name] = verifier_func
            self.verifier_stats[verifier_name] = {
                'calls': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'average_time': 0.0,
                'pass_rate': 0.0
            }
            
            logger.info(f"Added verifier: {verifier_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add verifier {verifier_name}: {e}")
            return False
            
    async def remove_verifier(self, verifier_name: str) -> bool:
        """
        Remove a verifier function.
        
        Args:
            verifier_name: Name of verifier to remove
            
        Returns:
            True if successfully removed, False otherwise
        """
        try:
            if verifier_name in self.active_verifiers:
                del self.active_verifiers[verifier_name]
                if verifier_name in self.verifier_stats:
                    del self.verifier_stats[verifier_name]
                logger.info(f"Removed verifier: {verifier_name}")
                return True
            else:
                logger.warning(f"Verifier {verifier_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove verifier {verifier_name}: {e}")
            return False
            
    async def shutdown(self) -> None:
        """Gracefully shutdown the verifier actor."""
        logger.info("Shutting down ReVerifier...")
        
        self.active_verifiers.clear()
        self.verifier_stats.clear()
        self.verification_times.clear()
        self.is_initialized = False
        
        logger.info("ReVerifier shutdown complete") 