"""Pluggable training runner protocol and built-in implementations.

The trainer registry (parallel to the backend registry) controls *what*
runs the training loop.  ``trainer = "retrain"`` uses the built-in loop.
``trainer = "command"`` wraps an arbitrary shell command.  Dotted paths
load third-party plugins.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Protocol, runtime_checkable

from retrain.backends import collect_runtime_metrics, run_sft_train_step
from retrain.config import TrainConfig
from retrain.metrics_scan import scan_metrics_file
from retrain.process_metrics import process_max_rss_mb


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


def _snapshot_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {key: _snapshot_metadata_value(value) for key, value in metrics.items()}


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
            "metrics": _snapshot_metrics(self.metrics),
            "artifacts": dict(self.artifacts),
        }


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
    return scan_metrics_file(metrics_path).last or {}


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


class SftRunner:
    """Standalone supervised fine-tuning runner using retrain backends."""

    def run(self, config: TrainConfig) -> TrainingRunResult:
        try:
            return self._run(config)
        except Exception as exc:
            return failed_run_result(
                config,
                failure_status=f"exception:{type(exc).__name__}",
                error_message=str(exc),
            )

    def _run(self, config: TrainConfig) -> TrainingRunResult:
        from transformers import AutoTokenizer

        from retrain.backends.catalog import (
            _effective_sft_loss_fn,
            backend_capability_source,
            resolve_backend_capabilities,
        )
        from retrain.io.log import JsonlLogger
        from retrain.registry import get_registry
        from retrain.sft import (
            build_sft_tokenized_batch,
            build_sft_artifact_manifest,
            build_sft_example_order,
            load_sft_jsonl,
            select_sft_batch_indices,
            tokenize_sft_dataset,
            write_sft_artifact_manifest,
        )
        from retrain.trainer_state import save_trainer_state

        if not config.sft_data_path:
            raise ValueError(
                "trainer='sft' requires [training] sft_data_path to point at a JSONL dataset."
            )

        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        emergence_dir = log_dir / "emergence"
        emergence_dir.mkdir(parents=True, exist_ok=True)
        metrics_logger = JsonlLogger(str(log_dir / "metrics.jsonl"))
        steps_logger = JsonlLogger(str(emergence_dir / "steps.jsonl"))

        examples = load_sft_jsonl(config.sft_data_path)
        if not examples:
            raise RuntimeError("SFT dataset is empty — cannot fine-tune with zero examples.")

        if config.seed >= 0:
            import random

            random.seed(config.seed)
            try:
                import numpy as np

                np.random.seed(config.seed)
            except ImportError:
                pass
            try:
                import torch

                torch.manual_seed(config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(config.seed)
            except ImportError:
                pass

        print("retrain SFT")
        print(f"  model         : {config.model}")
        print(f"  backend       : {config.backend}")
        print(f"  examples      : {len(examples)}")
        print(f"  max_steps     : {config.max_steps}")
        print(f"  adapter_path  : {config.adapter_path}")

        helper = get_registry("backend").create(config.backend, config)
        loss_fn = _effective_sft_loss_fn(config)
        setattr(helper, "sft_loss_fn", loss_fn)

        backend_caps = resolve_backend_capabilities(
            config.backend,
            config.backend_options,
        )
        print(
            "Backend capabilities: "
            f"backend={config.backend}, "
            f"source={backend_capability_source(config.backend, config.backend_options)}, "
            f"reports_sync_loss={backend_caps.reports_sync_loss}, "
            f"preserves_token_advantages={backend_caps.preserves_token_advantages}, "
            f"supports_checkpoint_resume={backend_caps.supports_checkpoint_resume}, "
            f"resume_runtime_dependent={backend_caps.resume_runtime_dependent}"
        )
        print(f"SFT loss: {loss_fn}")

        if config.resume_from:
            load_state = getattr(helper, "load_state", None)
            if not callable(load_state):
                raise RuntimeError(
                    "trainer='sft' resume_from requires a backend with load_state()."
                )
            print(f"Loading SFT initial adapter from {config.resume_from} ...")
            load_state(config.resume_from)

        print(f"Loading tokenizer for {config.model} ...")
        tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)

        batch_size = config.sft_batch_size if config.sft_batch_size > 0 else config.batch_size
        batch_size = min(max(1, batch_size), len(examples))
        max_tokens = config.sft_max_tokens if config.sft_max_tokens > 0 else config.max_tokens
        lr = config.sft_lr if config.sft_lr > 0 else config.lr
        print("Tokenizing SFT dataset ...")
        tokenized_examples = tokenize_sft_dataset(
            tokenizer,
            examples,
            max_tokens=max_tokens,
        )
        token_lengths = [example.total_tokens for example in tokenized_examples]
        order = build_sft_example_order(
            len(tokenized_examples),
            config.seed,
            lengths=token_lengths,
            batch_order=config.sft_batch_order,
            length_bucket_size=config.sft_length_bucket_size,
        )
        print(
            "SFT batching: "
            f"order={config.sft_batch_order}, "
            f"length_bucket_size={config.sft_length_bucket_size or len(tokenized_examples)}"
        )

        policy_ref = ""
        last_metrics: dict[str, int | float | str] = {}
        try:
            for step in range(config.max_steps):
                step_start = time.perf_counter()
                indices = select_sft_batch_indices(
                    order,
                    batch_size=batch_size,
                    step=step,
                )
                batch = [tokenized_examples[idx] for idx in indices]
                tokenized = build_sft_tokenized_batch(batch)

                loss = run_sft_train_step(
                    helper,
                    tokenized.tokens,
                    tokenized.advantages,
                    lr,
                    config.weight_decay,
                )

                elapsed = time.perf_counter() - step_start
                metrics: dict[str, int | float | str] = {
                    "step": step,
                    "phase": "sft",
                    "trainer": "sft",
                    "backend": config.backend,
                    "loss": loss,
                    "sft_loss_fn": loss_fn,
                    "sft_batch_order": config.sft_batch_order,
                    "sft_length_bucket_size": int(config.sft_length_bucket_size),
                    "lr": lr,
                    "datums": len(batch),
                    "tokens": tokenized.total_tokens,
                    "supervised_tokens": tokenized.supervised_tokens,
                    "sft_unique_examples_seen": min(
                        len(examples),
                        (step + 1) * batch_size,
                    ),
                    "sft_dataset_coverage": min(
                        1.0,
                        ((step + 1) * batch_size) / max(len(examples), 1),
                    ),
                    "time_s": round(elapsed, 2),
                }
                rss_mb = process_max_rss_mb()
                if rss_mb is not None:
                    metrics["process_max_rss_mb"] = round(rss_mb, 3)
                for key, value in collect_runtime_metrics(helper).items():
                    metrics[f"backend/{key}"] = value

                metrics_logger.log(metrics)
                steps_logger.log(metrics)
                last_metrics = dict(metrics)
                print(
                    f"Step {step} [SFT] | loss={loss:.4f} | "
                    f"datums={len(batch)} | tokens={tokenized.total_tokens} | "
                    f"supervised={tokenized.supervised_tokens} | time={elapsed:.1f}s",
                    flush=True,
                )

                if config.save_every > 0 and (step + 1) % config.save_every == 0:
                    checkpoint_name = f"checkpoint_step_{step + 1}"
                    policy_ref = helper.save_adapter(
                        config.adapter_path,
                        checkpoint_name,
                    )
                    save_trainer_state(
                        log_dir,
                        step=step,
                        example_idx=(step + 1) * batch_size,
                        total_correct=0,
                        total_completions=0,
                        current_batch_size=batch_size,
                        current_group_size=1,
                        checkpoint_name=checkpoint_name,
                        checkpoint_path=policy_ref,
                        sepa_state={},
                    )
                    print(f"Saved checkpoint: {checkpoint_name}")

            policy_ref = helper.save_adapter(
                config.adapter_path,
                "final",
            )
            save_trainer_state(
                log_dir,
                step=config.max_steps - 1,
                example_idx=config.max_steps * batch_size,
                total_correct=0,
                total_completions=0,
                current_batch_size=batch_size,
                current_group_size=1,
                checkpoint_name="final",
                checkpoint_path=policy_ref,
                sepa_state={},
            )
            manifest = build_sft_artifact_manifest(
                config,
                policy_ref=policy_ref,
                examples_count=len(examples),
                batch_size=batch_size,
                max_tokens=max_tokens,
                loss_fn=loss_fn,
                latest_metrics=last_metrics,
            )
            manifest_paths = write_sft_artifact_manifest(
                log_dir,
                policy_ref,
                manifest,
            )
            print(f"SFT manifest: {manifest_paths['log_manifest']}")
        finally:
            shutdown = getattr(helper, "shutdown", None)
            if callable(shutdown):
                shutdown()

        print(f"SFT complete. Adapter: {policy_ref}")
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
