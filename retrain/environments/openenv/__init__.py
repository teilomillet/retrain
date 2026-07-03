"""Native OpenEnv (gym contract) environment provider.

Selected with ``[environment] provider = "openenv"``; the ``id`` is the base
URL of a running OpenEnv server. Rollouts run through the same multi-turn
engine as verifiers environments — no verifiers install required.
"""

from retrain.environments.openenv.client import OpenEnvClient, StepResult
from retrain.environments.openenv.environment import (
    EpisodeRewardRubric,
    OpenEnvEnvironment,
)
from retrain.environments.openenv.load import (
    examples_from_environment,
    load_environment,
)

__all__ = [
    "EpisodeRewardRubric",
    "OpenEnvClient",
    "OpenEnvEnvironment",
    "StepResult",
    "examples_from_environment",
    "load_environment",
]
