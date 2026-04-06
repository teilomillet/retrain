"""Pluggable training runner protocol and built-in implementations.

The trainer registry (parallel to the backend registry) controls *what*
runs the training loop.  ``trainer = "retrain"`` uses the built-in loop.
``trainer = "command"`` wraps an arbitrary shell command.  Dotted paths
load third-party plugins.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Protocol, runtime_checkable

from retrain.config import TrainConfig


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
        return asdict(self)


def _run_id_for(config: TrainConfig) -> str:
    log_dir = Path(config.log_dir)
    name = log_dir.name.strip()
    if name:
        return name
    return "run"


def _latest_metrics(log_dir: Path) -> dict[str, object]:
    metrics_path = log_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return {}

    last = ""
    with open(metrics_path) as f:
        for line in f:
            if line.strip():
                last = line
    if not last:
        return {}

    try:
        payload = json.loads(last)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _artifact_map(log_dir: Path, policy_ref: str = "") -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if log_dir.is_dir():
        for path in sorted(log_dir.iterdir()):
            if path.is_file():
                artifacts[path.name] = str(path)
    if policy_ref:
        artifacts.setdefault("policy_ref", policy_ref)
    return artifacts


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
    return TrainingRunResult(
        policy_ref=policy_ref,
        run_id=_run_id_for(config),
        status=status,
        failure_status=failure_status,
        error_message=error_message,
        metrics=_latest_metrics(log_dir),
        artifacts=_artifact_map(log_dir, policy_ref=policy_ref),
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


@runtime_checkable
class TrainingRunner(Protocol):
    """Protocol every training runner must satisfy."""

    def run(self, config: TrainConfig) -> TrainingRunResult:
        """Run a full training job and return a structured result."""
        ...


class RetainRunner:
    """Built-in runner — delegates to ``retrain.trainer.train()``."""

    def run(self, config: TrainConfig) -> TrainingRunResult:
        from retrain.trainer import train

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


class CommandRunner:
    """Runs an external shell command as the training loop.

    The command template may contain placeholders:
        {config_path}  — path to a JSON file with all TrainConfig fields
        {log_dir}      — config.log_dir
        {adapter_path} — config.adapter_path
        {model}        — config.model

    The command is expected to:
    - Read config from {config_path} (or use its own config)
    - Write metrics to {log_dir}/metrics.jsonl (so ``retrain status`` works)
    - Save adapter to {adapter_path} (so downstream tools find it)
    """

    def __init__(self, command_template: str) -> None:
        self.command_template = command_template

    def run(self, config: TrainConfig) -> TrainingRunResult:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Export config as JSON for the external command
        config_dict: dict[str, object] = {}
        for f in fields(TrainConfig):
            config_dict[f.name] = getattr(config, f.name)
        config_path = log_dir / "retrain_config.json"
        config_path.write_text(json.dumps(config_dict, indent=2, default=str))

        # Substitute placeholders
        cmd = self.command_template.format(
            config_path=str(config_path),
            log_dir=str(log_dir),
            adapter_path=config.adapter_path,
            model=config.model,
        )

        result = subprocess.run(
            cmd,
            shell=True,
            cwd=Path.cwd(),
        )

        if result.returncode != 0:
            return failed_run_result(
                config,
                failure_status=f"exit_code:{result.returncode}",
                error_message=f"Trainer command exited with status {result.returncode}.",
            )

        if Path(config.adapter_path).exists():
            return build_run_result(config, policy_ref=config.adapter_path)
        return failed_run_result(
            config,
            failure_status="missing_policy_ref",
            error_message=(
                "Trainer command completed successfully but no adapter/policy "
                f"was found at {config.adapter_path}."
            ),
        )
