"""Rollout execution for the training loop."""

from __future__ import annotations

from retrain.training.rollout.multi import run_multiturn
from retrain.training.rollout.single import run_singleturn
from retrain.training.rollout.state import (
    RolloutAccumulator,
    has_nonzero_advantage,
    prepare_echo_step_plan,
)

__all__ = [
    "RolloutAccumulator",
    "has_nonzero_advantage",
    "prepare_echo_step_plan",
    "run_multiturn",
    "run_singleturn",
]
