"""Local one-step runner for exact captured optimizer batches."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import cast

import orjson

from retrain.backends import collect_runtime_metrics
from retrain.config import TrainConfig
from retrain.config.snapshot import config_snapshot
from retrain.io.log import JsonlLogger
from retrain.process.metrics import max_rss_mb
from retrain.training.echo import run_rl_echo_train_step
from retrain.training.optimizer_batch import (
    adapter_provenance,
    load_optimizer_batch_capture,
    resolve_initial_adapter,
    restore_torch_rng_state,
    sha256_json,
    validate_replay_contract,
)
from retrain.training.runner.result import (
    TrainingRunResult,
    build_run_result,
    guarded_run_result,
)


class OptimizerReplayRunner:
    """Replay one verified local optimizer batch without rollout side effects."""

    def run(self, config: TrainConfig) -> TrainingRunResult:
        return guarded_run_result(config, lambda: self._run(config))

    def _run(self, config: TrainConfig) -> TrainingRunResult:
        from retrain.registry.builtin import get_registry

        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        _ensure_fresh_outputs(config, log_dir)
        capture = load_optimizer_batch_capture(
            config.optimizer_batch_replay_path,
            expected_manifest_sha256=(config.optimizer_batch_expected_manifest_sha256),
        )
        expected_logical_sha = (
            config.optimizer_batch_expected_logical_sha256.strip().lower()
        )
        if capture.logical_batch_sha256 != expected_logical_sha:
            raise ValueError(
                "optimizer-batch replay pin mismatch: expected "
                f"{expected_logical_sha}, got {capture.logical_batch_sha256}."
            )

        initial_adapter = resolve_initial_adapter(config)
        contract = validate_replay_contract(
            capture.manifest,
            config,
            initial_adapter,
        )
        helper = get_registry("backend").create(config.backend, config)
        try:
            # Model and optimizer construction may consume RNG. Restore only
            # after loading the exact source adapter and immediately before the
            # measured update.
            helper.load_state(initial_adapter.adapter_dir)
            restore_torch_rng_state(capture.batch.torch_rng)

            batch = capture.batch
            train_started = time.perf_counter()
            loss, echo_loss, echo_joint = run_rl_echo_train_step(
                helper,
                batch.tokens,
                batch.old_logprobs,
                batch.advantages,
                batch.echo_advantages or [],
                batch.echo_full_observation_counts or [],
                echo_loss_fn=config.echo_loss_fn,
                lr=config.lr,
                weight_decay=config.weight_decay,
                echo_rollout_denominator=(batch.echo_rollout_denominator or 0),
            )
            train_time = time.perf_counter() - train_started
            final_path = helper.save_adapter(config.adapter_path, "final")
            final_adapter = adapter_provenance(final_path)
            runtime = collect_runtime_metrics(helper)
        finally:
            shutdown = getattr(helper, "shutdown", None)
            if callable(shutdown):
                shutdown()

        effective_sha = runtime.get(
            "optimizer/local_effective_rows_sha256",
            "",
        )
        if not isinstance(effective_sha, str) or not effective_sha:
            raise RuntimeError(
                "local optimizer replay did not report an effective-row digest."
            )

        total_tokens = sum(len(row) for row in capture.batch.tokens)
        nonzero_advantages = sum(
            value != 0.0 for row in capture.batch.advantages for value in row
        )
        nonzero_echo = sum(
            value != 0.0
            for row in (capture.batch.echo_advantages or [])
            for value in row
        )
        source_config = _source_config(capture.manifest)
        metrics: dict[str, object] = {
            "step": 0,
            "trainer": "optimizer_replay",
            "loss": loss,
            "reported_loss": loss,
            "echo/loss": echo_loss,
            "echo/joint_optimizer_step": int(echo_joint),
            "num_datums": len(capture.batch.tokens),
            "tokens_per_step": total_tokens,
            "tokens_per_second": (total_tokens / train_time if train_time > 0 else 0.0),
            "rl/optimizer_nonzero_advantage_action_tokens": nonzero_advantages,
            "echo/kept_tokens": nonzero_echo,
            "step_time_s": train_time,
            "sample_time_s": 0.0,
            "train_time_s": train_time,
            "train_time_semantics": "synchronous_optimizer_update",
            "sample_share": 0.0,
            "train_share": 1.0,
            "optimizer/logical_batch_sha256": capture.logical_batch_sha256,
            "optimizer/batch_sha256": capture.logical_batch_sha256,
            "optimizer_batch/mode": "replay",
            "optimizer_batch/source_manifest": str(capture.manifest_path),
            "optimizer_batch/payload_sha256": capture.payload_sha256,
            "optimizer_batch/manifest_sha256": capture.manifest_sha256,
            "optimizer_batch/source_config_sha256": source_config["sha256"],
            "optimizer_batch/replay_config_sha256": (contract.current_config_sha256),
            "optimizer_batch/source_optimizer_contract_sha256": (
                source_config["optimizer_contract_sha256"]
            ),
            "optimizer_batch/replay_optimizer_contract_sha256": (
                contract.current_optimizer_contract_sha256
            ),
            "optimizer_batch/initial_adapter_sha256": (initial_adapter.weight_sha256),
            "optimizer_batch/final_adapter_sha256": final_adapter.weight_sha256,
            "optimizer_batch/allowed_config_differences": _json_list(
                contract.allowed_differences
            ),
            "optimizer_batch/observed_config_differences": _json_list(
                contract.observed_differences
            ),
            "optimizer_batch/dataset_skipped": 1,
            "optimizer_batch/environment_skipped": 1,
            "optimizer_batch/sampling_skipped": 1,
            "optimizer_batch/rollout_skipped": 1,
        }
        metrics.update(runtime)
        rss = max_rss_mb()
        if rss is not None:
            metrics["process_max_rss_mb"] = round(rss, 3)

        with JsonlLogger(str(log_dir / "metrics.jsonl")) as logger:
            logger.log(metrics)
        replay_manifest: dict[str, object] = {
            "format": "retrain.optimizer-batch-replay.v1",
            "source": {
                "manifest": str(capture.manifest_path),
                "manifest_sha256": capture.manifest_sha256,
                "payload": str(capture.payload_path),
                "payload_sha256": capture.payload_sha256,
                "logical_batch_sha256": capture.logical_batch_sha256,
                "config_sha256": source_config["sha256"],
                "optimizer_contract_sha256": source_config["optimizer_contract_sha256"],
            },
            "replay": {
                "config": config_snapshot(config),
                "config_sha256": contract.current_config_sha256,
                "optimizer_contract_sha256": (
                    contract.current_optimizer_contract_sha256
                ),
                "allowed_config_differences": list(contract.allowed_differences),
                "observed_config_differences": list(contract.observed_differences),
                "initial_adapter": initial_adapter.to_dict(),
                "final_adapter": final_adapter.to_dict(),
            },
            "metrics": metrics,
        }
        _write_json_atomic(
            log_dir / "optimizer_batch_replay_manifest.json",
            replay_manifest,
        )
        return build_run_result(config, policy_ref=final_path)


def _source_config(manifest: dict[str, object]) -> dict[str, object]:
    value = manifest.get("config")
    if not isinstance(value, dict):
        raise ValueError("optimizer-batch manifest config must be an object.")
    return cast(dict[str, object], value)


def _ensure_fresh_outputs(config: TrainConfig, log_dir: Path) -> None:
    candidates = (
        log_dir / "metrics.jsonl",
        log_dir / "optimizer_batch_replay_manifest.json",
        Path(config.adapter_path).expanduser() / "final",
    )
    existing = [str(path) for path in candidates if path.exists()]
    if existing:
        raise FileExistsError(
            "optimizer-batch replay requires fresh output paths; refusing to "
            f"append or overwrite {existing}."
        )


def _json_list(values: tuple[str, ...]) -> str:
    return orjson.dumps(list(values)).decode("utf-8")


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    data = orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
