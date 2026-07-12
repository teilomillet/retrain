"""Dry-run preview for a single training run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

from retrain.commands.backends.capability import payload as capability_payload
from retrain.commands.backends.capability import summary as capability_summary
from retrain.training.resume import contract_for_capability_payload

if TYPE_CHECKING:
    from retrain.config import TrainConfig


def _should_load_sft_provenance(config: "TrainConfig") -> bool:
    sft_data_path = str(getattr(config, "sft_data_path", ""))
    if not sft_data_path:
        return False
    has_data_pins = bool(
        getattr(config, "sft_data_sha256", "")
        or int(getattr(config, "sft_data_rows", 0) or 0) > 0
        or getattr(config, "sft_audit_path", "")
        or getattr(config, "sft_audit_sha256", "")
        or getattr(config, "sft_token_audit_path", "")
        or getattr(config, "sft_token_audit_sha256", "")
    )
    if getattr(config, "trainer", "") == "sft" or has_data_pins:
        return True
    if int(getattr(config, "sft_warmup_steps", 0) or 0) > 0:
        return Path(sft_data_path).exists()
    return False


def _sft_provenance_info(config: "TrainConfig") -> dict[str, object] | None:
    if not _should_load_sft_provenance(config):
        return None

    from retrain.training.sft_data import load_sft_dataset, verify_sft_data_contract

    dataset = load_sft_dataset(str(getattr(config, "sft_data_path")))
    audit = verify_sft_data_contract(config, dataset.provenance)
    provenance = dataset.provenance
    return {
        "data_path": provenance.data_path,
        "data_sha256": provenance.data_sha256,
        "data_rows": provenance.data_rows,
        "data_bytes": provenance.data_bytes,
        "data_path_status": provenance.data_path_status,
        "data_root": provenance.data_root,
        "git_root": provenance.git_root,
        "git_tracked": provenance.git_tracked,
        "data_warnings": list(provenance.data_warnings),
        "pinned_sha256": str(getattr(config, "sft_data_sha256", "")).strip(),
        "pinned_rows": int(getattr(config, "sft_data_rows", 0) or 0),
        "audit": audit,
    }


def _rl_condition(config: "TrainConfig") -> str:
    """Return the same human-readable RL condition used by training."""
    condition = (
        config.algorithm_mode
        if config.algorithm_mode
        else f"{config.advantage_mode}+{config.transform_mode}"
    )
    if config.echo_enabled:
        return f"{condition}+echo"
    return condition


def _environment_preview(config: "TrainConfig") -> dict[str, object]:
    """Extract the environment settings that define rollout identity."""
    from retrain.environments.load import parse_args

    args = parse_args(config.environment_args) if config.environment_provider else {}
    return {
        "environment_provider": config.environment_provider,
        "environment_id": config.environment_id,
        "environment_max_turns": config.environment_max_turns,
        "environment_renderer": args.get("renderer"),
        "environment_expected_task_source": args.get("expected_task_source"),
        "environment_expected_task_ids": args.get("expected_task_ids"),
    }


def _optimizer_batch_preview(config: "TrainConfig") -> dict[str, object] | None:
    if config.trainer != "optimizer_replay":
        return None
    from retrain.training.optimizer_batch import (
        load_optimizer_batch_capture,
        resolve_initial_adapter,
        validate_replay_contract,
    )

    capture = load_optimizer_batch_capture(
        config.optimizer_batch_replay_path,
        expected_manifest_sha256=(config.optimizer_batch_expected_manifest_sha256),
    )
    if (
        capture.logical_batch_sha256
        != config.optimizer_batch_expected_logical_sha256.strip().lower()
    ):
        raise ValueError(
            "optimizer-batch replay pin does not match the captured logical digest."
        )
    initial_adapter = resolve_initial_adapter(config)
    contract = validate_replay_contract(
        capture.manifest,
        config,
        initial_adapter,
    )
    return {
        "manifest": str(capture.manifest_path),
        "manifest_sha256": capture.manifest_sha256,
        "payload_sha256": capture.payload_sha256,
        "logical_batch_sha256": capture.logical_batch_sha256,
        "rows": len(capture.batch.tokens),
        "tokens": sum(len(row) for row in capture.batch.tokens),
        "initial_adapter_sha256": initial_adapter.weight_sha256,
        "allowed_config_differences": list(contract.allowed_differences),
        "observed_config_differences": list(contract.observed_differences),
        "skips_environment_and_sampling": True,
    }


def _optimizer_batch_capture_preflight(
    config: "TrainConfig",
) -> dict[str, object] | None:
    if not config.optimizer_batch_capture:
        return None
    from retrain.training.optimizer_batch import preflight_optimizer_batch_capture

    adapter = preflight_optimizer_batch_capture(config)
    return {
        "resume_step": -1,
        "initial_adapter_dir": adapter.adapter_dir,
        "initial_adapter_sha256": adapter.weight_sha256,
        "single_device": config.devices,
        "inference_engine": config.inference_engine,
    }


def explain_single(config_path: str | None, fmt: str) -> None:
    """Explain what a single training run would do."""
    import warnings

    from retrain.config import load_config
    from retrain.registry.health import check_environment

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = load_config(config_path)

    optimizer_batch_capture_preflight = _optimizer_batch_capture_preflight(config)
    optimizer_batch_preview = _optimizer_batch_preview(config)
    if config.trainer == "sft":
        condition = "sft"
        datums_per_step = (
            config.sft_batch_size if config.sft_batch_size > 0 else config.batch_size
        )
    elif optimizer_batch_preview is not None:
        condition = "optimizer_replay"
        datums_per_step = cast(int, optimizer_batch_preview["rows"])
    else:
        condition = _rl_condition(config)
        datums_per_step = config.batch_size * config.group_size
    total_datums = datums_per_step * config.max_steps
    sft_loss_fn = config.sft_loss_fn
    if sft_loss_fn == "auto":
        sft_loss_fn = (
            "cross_entropy" if config.trainer == "sft" else "importance_sampling"
        )
    lora_alpha = config.lora_alpha if config.lora_alpha else config.lora_rank * 2
    data_info = config.data_source
    if config.environment_provider:
        data_info = f"{config.environment_provider}:{config.environment_id}"
    if config.trainer == "sft":
        data_info = config.sft_data_path
    elif optimizer_batch_preview is not None:
        data_info = f"optimizer_batch:{optimizer_batch_preview['manifest']}"
    sft_provenance = _sft_provenance_info(config)
    backend_capabilities = capability_payload(
        config.backend,
        config.backend_options,
    )
    resume_contract = contract_for_capability_payload(backend_capabilities)
    environment_preview = _environment_preview(config)

    info: dict = {
        "mode": "single",
        "config": config_path or "retrain.toml",
        "model": config.model,
        "model_revision": config.model_revision,
        "model_local_files_only": config.model_local_files_only,
        "trainer": config.trainer,
        "backend": config.backend,
        "backend_options": dict(config.backend_options),
        "backend_capabilities": backend_capabilities,
        "resume_mode": resume_contract.mode,
        "resume_warning": resume_contract.warning,
        "condition": condition,
        "algorithm_mode": config.algorithm_mode,
        "advantage_mode": config.advantage_mode,
        "transform_mode": config.transform_mode,
        "max_steps": config.max_steps,
        "batch_size": config.batch_size,
        "group_size": config.group_size,
        "datums_per_step": datums_per_step,
        "total_datums": total_datums,
        "sft_warmup_steps": config.sft_warmup_steps,
        "sft_data_path": config.sft_data_path,
        "sft_data_sha256": config.sft_data_sha256,
        "sft_data_rows": config.sft_data_rows,
        "sft_audit_path": config.sft_audit_path,
        "sft_audit_sha256": config.sft_audit_sha256,
        "sft_token_audit_path": config.sft_token_audit_path,
        "sft_token_audit_sha256": config.sft_token_audit_sha256,
        "sft_data_provenance": sft_provenance,
        "sft_batch_size": config.sft_batch_size,
        "sft_max_tokens": config.sft_max_tokens,
        "sft_loss_fn": sft_loss_fn,
        "sft_batch_order": config.sft_batch_order,
        "sft_length_bucket_size": config.sft_length_bucket_size,
        "sft_reshuffle_each_epoch": config.sft_reshuffle_each_epoch,
        "echo_enabled": config.echo_enabled,
        "echo_weight": config.echo_weight,
        "echo_loss_fn": config.echo_loss_fn,
        "echo_target_retention": config.echo_target_retention,
        "echo_max_tokens_per_step": (
            config.echo_max_tokens_per_step
            if config.echo_target_retention == "bounded"
            else None
        ),
        "echo_max_token_ratio": (
            config.echo_max_token_ratio
            if config.echo_target_retention == "bounded"
            else None
        ),
        "echo_entropy_floor": config.echo_entropy_floor,
        "echo_min_prompt_overlap": config.echo_min_prompt_overlap,
        "echo_require_live_observation_bridge": (
            config.echo_require_live_observation_bridge
        ),
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "lr": config.lr,
        "seed": config.seed,
        "lora_rank": config.lora_rank,
        "lora_alpha": lora_alpha,
        "data": data_info,
        "reward_type": config.reward_type,
        "log_dir": config.log_dir,
        "adapter_path": config.adapter_path,
        "wandb_project": config.wandb_project or "(disabled)",
        "checkpoint_artifacts": config.checkpoint_artifacts,
        "optimizer_batch": optimizer_batch_preview,
        "optimizer_batch_capture_preflight": optimizer_batch_capture_preflight,
        "warnings": [str(w.message) for w in caught],
        **environment_preview,
    }

    # Dependency warnings
    dep_warnings = []
    results = check_environment(config=config)
    for name, import_name, hint, available in results:
        if not available:
            dep_warnings.append(f"{name} requires {import_name} ({hint})")
    if dep_warnings:
        info["dep_warnings"] = dep_warnings

    if fmt == "json":
        print(json.dumps(info, indent=2))
        return

    print("retrain explain — dry-run preview")
    print(f"  config        : {info['config']}")
    print(f"  model         : {config.model}")
    print(f"  trainer       : {config.trainer}")
    print(f"  backend       : {config.backend}")
    print(f"  backend caps  : {capability_summary(backend_capabilities)}")
    print(f"  resume mode   : {resume_contract.mode}")
    if resume_contract.warning:
        print(f"  resume warning: {resume_contract.warning}")
    if not backend_capabilities["reports_sync_loss"]:
        print("  note          : loss is reported as placeholder by backend design")
    print(f"  condition     : {condition}")
    if config.echo_enabled:
        print(
            f"  echo          : weight={config.echo_weight} loss={config.echo_loss_fn}"
        )
        if config.echo_target_retention == "all":
            print("  echo targets  : retention=all caps=disabled entropy_floor=0.0")
        else:
            print(
                "  echo targets  : retention=bounded "
                f"tokens/step={config.echo_max_tokens_per_step} "
                f"ratio={config.echo_max_token_ratio} "
                f"entropy_floor={config.echo_entropy_floor}"
            )
        print(
            f"  echo bridge   : required={config.echo_require_live_observation_bridge}"
        )
    if optimizer_batch_preview is not None:
        print(
            "  optimizer batch: "
            f"rows={optimizer_batch_preview['rows']} "
            f"tokens={optimizer_batch_preview['tokens']}"
        )
        print(f"  batch logical : {optimizer_batch_preview['logical_batch_sha256']}")
        print(
            "  batch config  : allowed="
            f"{optimizer_batch_preview['allowed_config_differences']} "
            "observed="
            f"{optimizer_batch_preview['observed_config_differences']}"
        )
        print("  replay I/O    : dataset/environment/sampling skipped")
    if optimizer_batch_capture_preflight is not None:
        print(
            "  optimizer batch capture: "
            "resume_step=-1 "
            f"adapter_sha256={optimizer_batch_capture_preflight['initial_adapter_sha256']}"
        )
    print(f"  steps         : {config.max_steps}")
    print(f"  batch_size    : {config.batch_size}")
    if config.trainer == "sft":
        print(f"  sft_batch     : {datums_per_step}")
    elif optimizer_batch_preview is None:
        print(f"  group_size    : {config.group_size}")
    else:
        print("  group_size    : (not used by optimizer replay)")
    print(f"  datums/step   : {datums_per_step}")
    print(f"  total datums  : {total_datums}")
    print(f"  max_tokens    : {config.max_tokens}")
    if config.model_revision:
        print(
            "  model source  : "
            f"revision={config.model_revision} "
            f"local_files_only={config.model_local_files_only}"
        )
    print(f"  temperature   : {config.temperature}")
    print(f"  lr            : {config.lr}")
    if config.trainer == "sft" or config.sft_warmup_steps > 0:
        print(
            "  sft           : "
            f"steps={config.max_steps if config.trainer == 'sft' else config.sft_warmup_steps} "
            f"batch={config.sft_batch_size or '(default)'} "
            f"loss={sft_loss_fn} "
            f"order={config.sft_batch_order} "
            f"reshuffle_each_epoch={config.sft_reshuffle_each_epoch}"
        )
        if config.sft_data_path:
            print(f"  sft_data      : {config.sft_data_path}")
        if sft_provenance is not None:
            print(f"  sft_resolved  : {sft_provenance['data_path']}")
            print(f"  sft_sha256    : {sft_provenance['data_sha256']}")
            print(
                "  sft_rows      : "
                f"{sft_provenance['data_rows']} "
                f"bytes={sft_provenance['data_bytes']}"
            )
            print(
                "  sft_tracking  : "
                f"{sft_provenance['data_path_status']} "
                f"git_tracked={sft_provenance['git_tracked']}"
            )
            audit_info = sft_provenance.get("audit")
            if isinstance(audit_info, Mapping):
                audit_map = cast(Mapping[str, object], audit_info)
                print(
                    "  sft_audit     : "
                    f"{audit_map.get('status')} "
                    f"mode={audit_map.get('corpus_mode')} "
                    f"sha256={audit_map.get('audit_sha256')}"
                )
            if config.sft_token_audit_path:
                print(
                    "  sft_token_audit: pass "
                    f"sha256={config.sft_token_audit_sha256.strip().lower()}"
                )
            for warning in cast(list[str], sft_provenance["data_warnings"]):
                print(f"  sft_warning   : {warning}")
    print(f"  seed          : {config.seed}")
    print(f"  lora          : rank={config.lora_rank} alpha={lora_alpha}")
    print(f"  data          : {data_info}")
    if config.environment_provider:
        renderer = environment_preview["environment_renderer"] or "(default)"
        expected_source = (
            environment_preview["environment_expected_task_source"] or "(unguarded)"
        )
        expected_ids = environment_preview["environment_expected_task_ids"]
        expected_ids_text = (
            json.dumps(expected_ids, separators=(",", ":"))
            if expected_ids is not None
            else "(unguarded)"
        )
        print(f"  env max_turns : {config.environment_max_turns}")
        print(f"  env renderer  : {renderer}")
        print(f"  task guard    : source={expected_source} ids={expected_ids_text}")
    print(f"  reward        : {config.reward_type}")
    print(f"  log_dir       : {config.log_dir}")
    print(f"  adapter_path  : {config.adapter_path}")
    print(f"  wandb         : {info['wandb_project']}")
    print(f"  ckpt artifacts: {config.checkpoint_artifacts}")
    if caught:
        print("\nWarnings:")
        for w in caught:
            print(f"  - {w.message}")
    if dep_warnings:
        print("\nMissing dependencies:")
        for dw in dep_warnings:
            print(f"  - {dw}")
