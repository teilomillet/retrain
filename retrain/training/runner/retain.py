"""Built-in retrain loop runner."""

from __future__ import annotations

from retrain.config import TrainConfig
from retrain.training.runner.result import (
    TrainingRunResult,
    build_run_result,
    failed_run_result,
)


class RetainRunner:
    """Built-in runner: delegates to ``retrain.training.trainer.train()``."""

    def run(self, config: TrainConfig) -> TrainingRunResult:
        from retrain.training.trainer import train

        try:
            policy_ref = train(config) or ""
        except Exception as exc:
            return failed_run_result(
                config,
                failure_status=f"exception:{type(exc).__name__}",
                error_message=str(exc),
            )
        if not policy_ref:
            return failed_run_result(
                config,
                failure_status="missing_policy_ref",
                error_message="Training completed without returning a policy reference.",
            )
        return build_run_result(config, policy_ref=policy_ref)
