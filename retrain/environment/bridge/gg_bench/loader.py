import importlib.util
from pathlib import Path

# Attempt to import gg_bench components. This assumes gg_bench is an installed package
# or accessible in the Python path.
# If these imports fail, retrain won't be able to use gg-bench environments.
try:
    from gg_bench.utils.env_wrappers import TimeoutEnv, MetadataEnv, AlternatingAgentEnv
    # This type hint is for clarity, not strictly necessary if only used internally
    # It represents the fully wrapped synchronous gg-bench environment
    from gymnasium import Env as GGBenchSyncEnv # gg_bench envs are Gym-like
except ImportError as e:
    # Allow the module to load but make it clear that gg-bench functionality will fail
    # A more sophisticated approach might involve a configuration flag in retrain
    # to enable/disable gg-bench integration.
    print(f"Warning: Failed to import gg_bench components. GG-Bench integration will not work. Error: {e}")
    # Define placeholders if imports fail to prevent NameErrors later, though functions using them will fail.
    class GGBenchSyncEnv: 
        pass # type: ignore
    class TimeoutEnv: 
        pass # type: ignore
    class MetadataEnv: 
        pass # type: ignore
    class AlternatingAgentEnv: 
        pass # type: ignore

from retrain.environment.environment import Environment
from .wrapper import AsyncGGBenchBridgeWrapper # Relative import


def load_gg_bench_env(
    game_id: str, 
    gg_bench_envs_dir: Path, 
    # gg_bench_root_path: Path # Deprecated in favor of more specific envs_dir
) -> Environment:
    """
    Loads a specific gg-bench game environment by its ID, applies standard
    gg-bench wrappers, and then adapts it for retrain's asynchronous interface.

    Args:
        game_id: The identifier string for the gg-bench game (e.g., "1", "999").
        gg_bench_envs_dir: Path to the directory containing the generated gg-bench
                           environment .py files (e.g., .../gg-bench/gg_bench/data/envs/).

    Returns:
        An instance of Environment (specifically AsyncGGBenchBridgeWrapper) that
        is compatible with the retrain library.

    Raises:
        FileNotFoundError: If the specified game environment .py file does not exist.
        ImportError: If gg_bench library components cannot be imported or if the
                     environment .py file cannot be loaded as a module.
        AttributeError: If the loaded game module does not contain a 'CustomEnv' class.
    """
    if not isinstance(TimeoutEnv, type) or TimeoutEnv.__name__ == 'TimeoutEnv': # Check if placeholder
        if 'TimeoutEnv' in globals() and globals()['TimeoutEnv'].__name__ == 'TimeoutEnv' and not hasattr(globals()['TimeoutEnv'],'__module__'): # more robust check for placeholder
             raise ImportError("gg_bench components not properly imported. Cannot load gg-bench environment.")


    env_file_path = gg_bench_envs_dir / f"env_{game_id}.py"

    if not env_file_path.exists():
        raise FileNotFoundError(f"GG-Bench environment file not found: {env_file_path}")

    # Dynamically load the CustomEnv module from the game_id.py file
    module_name = f"gg_bench.data.envs.env_{game_id}" # A unique name for the module
    spec = importlib.util.spec_from_file_location(module_name, env_file_path)
    
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for gg-bench environment: {env_file_path}")
    
    gg_bench_game_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gg_bench_game_module)

    if not hasattr(gg_bench_game_module, "CustomEnv"):
        raise AttributeError(f"'CustomEnv' class not found in {env_file_path}")

    # Instantiate the base gg-bench environment
    # The type hint GGBenchSyncEnv assumes it's a gymnasium.Env compatible interface
    sync_env_instance: GGBenchSyncEnv = gg_bench_game_module.CustomEnv()

    # Apply gg-bench's standard wrappers in the recommended order
    sync_env_instance = TimeoutEnv(sync_env_instance) # type: ignore
    sync_env_instance = MetadataEnv(sync_env_instance) # type: ignore
    sync_env_instance = AlternatingAgentEnv(sync_env_instance) # type: ignore
    
    # Cast to a more specific type if necessary or beneficial for type checking, 
    # though Any or a Protocol would also work with the wrapper.
    # For now, the wrapper takes SyncEnv (TypeVar), which is flexible.

    # Adapt the fully wrapped synchronous environment for retrain
    adapted_env_for_retrain = AsyncGGBenchBridgeWrapper(sync_env_instance)

    return adapted_env_for_retrain 