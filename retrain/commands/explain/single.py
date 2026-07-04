"""Dry-run preview for a single training run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from retrain.commands.backends.capability import payload as capability_payload
from retrain.commands.backends.capability import summary as capability_summary

if TYPE_CHECKING:
    from retrain.config import TrainConfig


def _should_load_sft_provenance(config: "TrainConfig") -> bool:
    sft_data_path = str(getattr(config, "sft_data_path", ""))
    if not sft_data_path:
        return False
    has_data_pins = bool(
        getattr(config, "sft_data_sha256", "")
        or int(getattr(config, "sft_data_rows", 0) or 0) > 0
    )
    if getattr(config, "trainer", "") == "sft" or has_data_pins:
        return True
    if int(getattr(config, "sft_warmup_steps", 0) or 0) > 0:
        return Path(sft_data_path).exists()
    return False


def _sft_provenance_info(config: "TrainConfig") -> dict[str, object] | None:
    if not _should_load_sft_provenance(config):
        return None

    from retrain.training.sft import load_sft_dataset, verify_sft_data_contract

    dataset = load_sft_dataset(str(getattr(config, "sft_data_path")))
    verify_sft_data_contract(config, dataset.provenance)
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
    }


def explain_single(config_path: str | None, fmt: str) -> None:
    """Explain what a single training run would do."""
    import warnings

    from retrain.config import load_config
    from retrain.registry.health import check_environment

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = load_config(config_path)

    if config.trainer == "sft":
        condition = "sft"
        datums_per_step = config.sft_batch_size if config.sft_batch_size > 0 else config.batch_size
    else:
        condition = (
            config.algorithm_mode
            if config.algorithm_mode
            else f"{config.advantage_mode}+{config.transform_mode}"
        )
        datums_per_step = config.batch_size * config.group_size
    total_datums = datums_per_step * config.max_steps
    sft_loss_fn = config.sft_loss_fn
    if sft_loss_fn == "auto":
        sft_loss_fn = "cross_entropy" if config.trainer == "sft" else "importance_sampling"
    lora_alpha = config.lora_alpha if config.lora_alpha else config.lora_rank * 2
    data_info = config.data_source
    if config.environment_provider:
        data_info = f"{config.environment_provider}:{config.environment_id}"
    if config.trainer == "sft":
        data_info = config.sft_data_path
    sft_provenance = _sft_provenance_info(config)
    backend_capabilities = capability_payload(
        config.backend,
        config.backend_options,
    )

    info: dict = {
        "mode": "single",
        "config": config_path or "retrain.toml",
        "model": config.model,
        "trainer": config.trainer,
        "backend": config.backend,
        "backend_options": dict(config.backend_options),
        "backend_capabilities": backend_capabilities,
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
        "sft_data_provenance": sft_provenance,
        "sft_batch_size": config.sft_batch_size,
        "sft_max_tokens": config.sft_max_tokens,
        "sft_loss_fn": sft_loss_fn,
        "sft_batch_order": config.sft_batch_order,
        "sft_length_bucket_size": config.sft_length_bucket_size,
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
        "warnings": [str(w.message) for w in caught],
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
    if not backend_capabilities["reports_sync_loss"]:
        print("  note          : loss is reported as placeholder by backend design")
    print(f"  condition     : {condition}")
    print(f"  steps         : {config.max_steps}")
    print(f"  batch_size    : {config.batch_size}")
    if config.trainer == "sft":
        print(f"  sft_batch     : {datums_per_step}")
    else:
        print(f"  group_size    : {config.group_size}")
    print(f"  datums/step   : {datums_per_step}")
    print(f"  total datums  : {total_datums}")
    print(f"  max_tokens    : {config.max_tokens}")
    print(f"  temperature   : {config.temperature}")
    print(f"  lr            : {config.lr}")
    if config.trainer == "sft" or config.sft_warmup_steps > 0:
        print(
            "  sft           : "
            f"steps={config.max_steps if config.trainer == 'sft' else config.sft_warmup_steps} "
            f"batch={config.sft_batch_size or '(default)'} "
            f"loss={sft_loss_fn} "
            f"order={config.sft_batch_order}"
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
            for warning in cast(list[str], sft_provenance["data_warnings"]):
                print(f"  sft_warning   : {warning}")
    print(f"  seed          : {config.seed}")
    print(f"  lora          : rank={config.lora_rank} alpha={lora_alpha}")
    print(f"  data          : {data_info}")
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
