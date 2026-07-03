"""Training-run result objects and artifact collection."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from retrain.config import TrainConfig
from retrain.metrics.scan import scan_metrics_file


def _snapshot_metadata_value(value: object) -> object:
    """Copy JSON-shaped metadata without paying generic deepcopy everywhere."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {key: _snapshot_metadata_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_metadata_value(item) for item in value)
    return deepcopy(value)


@dataclass(frozen=True)
class TrainingRunResult:
    """Structured result for a single training/update run."""

    policy_ref: str = ""
    run_id: str = ""
    status: str = "succeeded"
    failure_status: str = ""
    error_message: str = ""
    metrics: dict[str, object] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"

    @property
    def checkpoint_path(self) -> str:
        """Alias for callers that think in checkpoint paths rather than policy refs."""
        return self.policy_ref

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation for run metadata."""
        return {
            "policy_ref": self.policy_ref,
            "run_id": self.run_id,
            "status": self.status,
            "failure_status": self.failure_status,
            "error_message": self.error_message,
            "metrics": {
                key: _snapshot_metadata_value(value)
                for key, value in self.metrics.items()
            },
            "artifacts": dict(self.artifacts),
        }


def build_run_result(
    config: TrainConfig,
    *,
    policy_ref: str = "",
    status: str = "succeeded",
    failure_status: str = "",
    error_message: str = "",
) -> TrainingRunResult:
    """Collect metrics/artifacts for a completed run."""
    log_dir = Path(config.log_dir)
    metrics_path = log_dir / "metrics.jsonl"
    metrics: dict[str, object] = {}
    if metrics_path.is_file():
        metrics = scan_metrics_file(metrics_path).last or {}

    artifacts: dict[str, str] = {}
    if log_dir.is_dir():
        for path in sorted(log_dir.iterdir()):
            if path.is_file():
                artifacts[path.name] = str(path)
    if policy_ref:
        artifacts.setdefault("policy_ref", policy_ref)

    return TrainingRunResult(
        policy_ref=policy_ref,
        run_id=log_dir.name.strip() or "run",
        status=status,
        failure_status=failure_status,
        error_message=error_message,
        metrics=metrics,
        artifacts=artifacts,
    )


def failed_run_result(
    config: TrainConfig,
    *,
    failure_status: str,
    error_message: str = "",
) -> TrainingRunResult:
    """Build a failed result while preserving partial artifacts/metrics."""
    return build_run_result(
        config,
        status="failed",
        failure_status=failure_status,
        error_message=error_message,
    )
