"""Pluggable training runner protocol and built-in implementations."""

from __future__ import annotations

from retrain.training.runner.command import CommandRunner
from retrain.training.runner.optimizer_replay import OptimizerReplayRunner
from retrain.training.runner.protocol import TrainingRunner
from retrain.training.runner.result import (
    TrainingRunResult,
    build_run_result,
    failed_run_result,
)
from retrain.training.runner.retain import RetainRunner
from retrain.training.runner.sft import SftRunner

__all__ = [
    "CommandRunner",
    "OptimizerReplayRunner",
    "RetainRunner",
    "SftRunner",
    "TrainingRunResult",
    "TrainingRunner",
    "build_run_result",
    "failed_run_result",
]
