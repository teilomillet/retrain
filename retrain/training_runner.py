"""Pluggable training runner protocol and built-in implementations.

The trainer registry (parallel to the backend registry) controls *what*
runs the training loop.  ``trainer = "retrain"`` uses the built-in loop.
``trainer = "command"`` wraps an arbitrary shell command.  Dotted paths
load third-party plugins.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Protocol, runtime_checkable

from retrain.config import TrainConfig


@runtime_checkable
class TrainingRunner(Protocol):
    """Protocol every training runner must satisfy."""

    def run(self, config: TrainConfig) -> str | None:
        """Run a full training job. Returns adapter path or None."""
        ...


class RetainRunner:
    """Built-in runner — delegates to ``retrain.trainer.train()``."""

    def run(self, config: TrainConfig) -> str | None:
        from retrain.trainer import train

        return train(config)


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

    def run(self, config: TrainConfig) -> str | None:
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
            return None

        if Path(config.adapter_path).exists():
            return config.adapter_path
        return None
