"""Shell-command training runner."""

from __future__ import annotations

import json
import subprocess
from dataclasses import fields
from pathlib import Path

from retrain.config import TrainConfig
from retrain.training.runner.result import (
    TrainingRunResult,
    build_run_result,
    failed_run_result,
)


class CommandRunner:
    """Runs an external shell command as the training loop.

    The command template may contain placeholders:
        {config_path}  - path to a JSON file with all TrainConfig fields
        {log_dir}      - config.log_dir
        {adapter_path} - config.adapter_path
        {model}        - config.model

    The command is expected to:
    - Read config from {config_path} or use its own config.
    - Write metrics to {log_dir}/metrics.jsonl so ``retrain status`` works.
    - Save adapter to {adapter_path} so downstream tools can find it.
    """

    def __init__(self, command_template: str) -> None:
        self.command_template = command_template

    def run(self, config: TrainConfig) -> TrainingRunResult:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        config_dict: dict[str, object] = {}
        for field in fields(TrainConfig):
            config_dict[field.name] = getattr(config, field.name)
        config_path = log_dir / "retrain_config.json"
        config_path.write_text(json.dumps(config_dict, indent=2, default=str))

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
