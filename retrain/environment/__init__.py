from typing import Any, Dict, Optional

from .environment import Environment
from .types import LLMAction, EnvironmentObservation, ToolCallAction, ToolResultObservation
from .env_fastmcp import FastMCPEnv
from .env_smolagent import SmolAgentEnv
from .env_spider2 import Spider2Env

__all__ = [
    "Environment",
    "LLMAction",
    "EnvironmentObservation",
    "ToolCallAction",
    "ToolResultObservation",
    "FastMCPEnv",
    "SmolAgentEnv",
    "Spider2Env",
    "get_environment",
]

SUPPORTED_ENV_TYPES = {
    "fastmcp_env": FastMCPEnv,
    "smol_agent": SmolAgentEnv,
    "spider2": Spider2Env,
}

async def get_environment(
    env_type: str,
    env_specific_config: Optional[Dict[str, Any]] = None,
) -> Environment:
    """
    Factory function to instantiate, configure, and set up an environment.

    Args:
        env_type: The type of environment to create (e.g., "fastmcp_env", "smol_agent").
        env_specific_config: A dictionary containing configuration specific to
                             the environment type being instantiated.
                             For example, for 'fastmcp_env', this might include 'server_url',
                             'tool_registry_keys', etc. For 'smol_agent', it might
                             contain 'tools' configurations.

    Returns:
        An initialized and set-up instance of the requested Environment.

    Raises:
        ValueError: If the env_type is not supported.
        ImportError: If an environment class cannot be imported (should be caught earlier).
        Exception: For errors during environment instantiation or setup.
    """
    if env_type not in SUPPORTED_ENV_TYPES:
        raise ValueError(
            f"Unsupported environment type: '{env_type}'. "
            f"Supported types are: {list(SUPPORTED_ENV_TYPES.keys())}"
        )

    EnvClass = SUPPORTED_ENV_TYPES[env_type]
    
    if env_specific_config is None:
        env_specific_config = {}

    print(f"[get_environment] Instantiating environment of type '{env_type}' with config: {env_specific_config}")
    
    if env_type == "smol_agent":
        environment_instance = EnvClass(env_specific_config=env_specific_config)
    else:
        environment_instance = EnvClass(**env_specific_config)
        
    print(f"[get_environment] Setting up environment instance: {environment_instance}")
    await environment_instance.setup()
    print(f"[get_environment] Environment setup complete for: {environment_instance}")

    return environment_instance
