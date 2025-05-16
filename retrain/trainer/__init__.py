"""Exposes trainer components like BaseTrainer and specific trainer implementations."""

from .trainer import (
    BaseTrainer,
    AlgorithmConfig,
    ModelObject,
    ExperienceBatch,
    TrainingMetrics,
    PromptSource,
    RewardFunction
)
# Import specific trainer adapters as they are created, e.g.:
# from .grpo.trl import GRPOTrainer as GRPOTrainerTRL

__all__ = [
    "BaseTrainer",
    "AlgorithmConfig",
    "ModelObject",
    "ExperienceBatch",
    "TrainingMetrics",
    "PromptSource",
    "RewardFunction",
    # Add specific trainer classes to __all__ when imported, e.g., "GRPOTrainerTRL"
]
