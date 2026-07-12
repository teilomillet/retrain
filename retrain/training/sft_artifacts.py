"""SFT adapter manifests and reproducibility artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from retrain.config.snapshot import config_snapshot
from retrain.training.sft_audit import SFT_AUDIT_SCHEMA
from retrain.training.sft_data import SftDataProvenance

if TYPE_CHECKING:
    from retrain.config import TrainConfig

SFT_DATA_SNAPSHOT_MAX_BYTES = 16 * 1024 * 1024


def build_sft_artifact_manifest(
    config: "TrainConfig",
    *,
    policy_ref: str,
    examples_count: int,
    batch_size: int,
    max_tokens: int,
    loss_fn: str,
    data_provenance: SftDataProvenance | None = None,
    snapshot_artifacts: Mapping[str, str] | None = None,
    latest_metrics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a self-describing manifest for an SFT LoRA adapter."""
    adapter_dir = Path(policy_ref)
    adapter_files: list[str] = []
    if adapter_dir.is_dir():
        adapter_files = sorted(
            path.name for path in adapter_dir.iterdir() if path.is_file()
        )

    sft_payload: dict[str, object] = {
        "data_path": config.sft_data_path,
        "examples": int(examples_count),
        "batch_size": int(batch_size),
        "max_tokens": int(max_tokens),
        "max_steps": int(config.max_steps),
        "lr": float(config.sft_lr if config.sft_lr > 0 else config.lr),
        "loss_fn": loss_fn,
        "batch_order": config.sft_batch_order,
        "length_bucket_size": int(config.sft_length_bucket_size),
        "reshuffle_each_epoch": bool(config.sft_reshuffle_each_epoch),
        "seed": int(config.seed),
        "epoch_seed_rule": (
            "seed_plus_epoch" if config.sft_reshuffle_each_epoch else "fixed_seed"
        ),
        "schedule_hash_algorithm": "sha256",
        "schedule_hash_encoding": "uint64_be_concatenation",
    }
    if data_provenance is not None:
        sft_payload.update(
            {
                "configured_data_path": config.sft_data_path,
                "data_path": data_provenance.data_path,
                "data_sha256": data_provenance.data_sha256,
                "data_rows": data_provenance.data_rows,
                "data_bytes": data_provenance.data_bytes,
                "data_path_status": data_provenance.data_path_status,
                "data_root": data_provenance.data_root,
                "git_root": data_provenance.git_root,
                "git_tracked": data_provenance.git_tracked,
                "data_warnings": list(data_provenance.data_warnings),
            }
        )
    if config.sft_audit_path:
        sft_payload.update(
            {
                "audit_path": config.sft_audit_path,
                "audit_sha256": config.sft_audit_sha256.strip().lower(),
                "audit_schema": SFT_AUDIT_SCHEMA,
            }
        )
    if config.sft_token_audit_path:
        sft_payload.update(
            {
                "token_audit_path": config.sft_token_audit_path,
                "token_audit_sha256": (config.sft_token_audit_sha256.strip().lower()),
            }
        )

    return {
        "schema_version": 1,
        "kind": "retrain_sft_adapter",
        "trainer": "sft",
        "backend": config.backend,
        "base_model": config.model,
        "base_model_revision": config.model_revision,
        "base_model_local_files_only": config.model_local_files_only,
        "adapter_path": str(adapter_dir),
        "adapter_root": config.adapter_path,
        "log_dir": config.log_dir,
        "resume": {
            "from": config.log_dir,
            "checkpoint_path": str(adapter_dir),
        },
        "sft": sft_payload,
        "reproducibility": {
            "artifacts": dict(snapshot_artifacts or {}),
        },
        "lora": {
            "rank": int(config.lora_rank),
            "alpha": int(
                config.lora_alpha if config.lora_alpha else config.lora_rank * 2
            ),
            "dropout": float(config.lora_dropout),
        },
        "huggingface": {
            "format": "peft_lora_adapter",
            "adapter_files": adapter_files,
            "load_snippet": (
                "from transformers import AutoModelForCausalLM\n"
                "from peft import PeftModel\n"
                "base = AutoModelForCausalLM.from_pretrained(\n"
                f"    {config.model!r}, revision={config.model_revision or None!r}, "
                f"local_files_only={config.model_local_files_only!r},\n"
                ")\n"
                f"model = PeftModel.from_pretrained(base, {str(adapter_dir)!r})"
            ),
            "publish_hint": (
                "Use `huggingface-cli upload <repo-id> "
                f"{str(adapter_dir)} .` or load the directory with PEFT."
            ),
        },
        "resource_metrics": dict(latest_metrics or {}),
        "ergonomics": {
            "no_rl_rollouts": True,
            "no_reward_model": True,
            "no_environment_bridge": True,
            "resume_into_retrain": True,
        },
    }


def write_sft_artifact_manifest(
    log_dir: str | Path,
    policy_ref: str,
    manifest: Mapping[str, object],
) -> dict[str, str]:
    """Write the SFT manifest beside logs and inside the adapter directory."""
    payload = json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n"
    log_path = Path(log_dir) / "sft_manifest.json"
    log_path.write_text(payload)

    paths = {"log_manifest": str(log_path)}
    adapter_dir = Path(policy_ref)
    if adapter_dir.is_dir():
        adapter_path = adapter_dir / "retrain_sft_manifest.json"
        adapter_path.write_text(payload)
        paths["adapter_manifest"] = str(adapter_path)
    return paths


def write_sft_run_snapshot_artifacts(
    log_dir: str | Path,
    config: "TrainConfig",
    provenance: SftDataProvenance,
    *,
    snapshot_max_bytes: int = SFT_DATA_SNAPSHOT_MAX_BYTES,
) -> dict[str, str]:
    """Write reproducibility artifacts beside an SFT run's logs."""

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    config_path = log_path / "resolved_config.json"
    config_payload = {
        "schema_version": 1,
        "kind": "retrain_resolved_config",
        "config": config_snapshot(config),
    }
    config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    paths["resolved_config.json"] = str(config_path)

    source_path = Path(provenance.data_path)
    snapshot_path = log_path / "sft_data.snapshot.jsonl"
    copied = False
    reason = ""
    if provenance.data_bytes <= snapshot_max_bytes:
        if source_path.resolve() != snapshot_path.resolve():
            shutil.copyfile(source_path, snapshot_path)
        copied = True
        copied_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        if copied_sha256 != provenance.data_sha256:
            raise RuntimeError(
                "SFT data snapshot hash mismatch: "
                f"expected {provenance.data_sha256}, got {copied_sha256}"
            )
        paths["sft_data.snapshot.jsonl"] = str(snapshot_path)
    else:
        reason = (
            f"data_bytes {provenance.data_bytes} exceeds snapshot_max_bytes "
            f"{snapshot_max_bytes}"
        )

    recoverability_path = log_path / "sft_data_recoverability.json"
    recoverability = {
        "schema_version": 1,
        "kind": "retrain_sft_data_recoverability",
        "source_path": provenance.data_path,
        "source_sha256": provenance.data_sha256,
        "source_rows": provenance.data_rows,
        "source_bytes": provenance.data_bytes,
        "source_path_status": provenance.data_path_status,
        "source_git_root": provenance.git_root,
        "source_git_tracked": provenance.git_tracked,
        "snapshot_max_bytes": int(snapshot_max_bytes),
        "copied": copied,
        "snapshot_path": str(snapshot_path) if copied else "",
        "recoverable": copied,
        "reason": reason,
    }
    recoverability_path.write_text(
        json.dumps(recoverability, indent=2, sort_keys=True) + "\n"
    )
    paths["sft_data_recoverability.json"] = str(recoverability_path)
    return paths
