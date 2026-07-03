"""Training runner protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from retrain.config import TrainConfig
from retrain.training.runner.result import TrainingRunResult


@runtime_checkable
class TrainingRunner(Protocol):
    """Protocol every training runner must satisfy."""

    def run(self, config: TrainConfig) -> TrainingRunResult:
        """Run a full training job and return a structured result."""
        ...
