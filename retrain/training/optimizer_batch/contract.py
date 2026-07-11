"""Fail-closed config contracts for deterministic optimizer replay."""

from __future__ import annotations

import hashlib
from typing import cast

import orjson

from retrain.config import TrainConfig
from retrain.config.snapshot import config_snapshot, sanitize_config_value
from retrain.training.optimizer_batch.types import (
    AdapterProvenance,
    ReplayContract,
)


_ALLOWED_V1_DIFFERENCE = "backend.options.gradient_checkpointing"


def canonical_json_bytes(value: object) -> bytes:
    """Encode JSON-shaped provenance with stable key ordering."""

    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def optimizer_contract(config: TrainConfig) -> dict[str, object]:
    """Fields that affect the local model, loss, or optimizer update.

    Sampling, environment, output, and logging settings are intentionally not
    part of this contract because replay does not execute those paths. Every
    normalized backend option is included; v1 permits only an explicitly
    declared gradient-checkpointing difference.
    """

    contract = {
        "seed": config.seed,
        "model": {
            "name": config.model,
            "base_url": config.base_url,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
        },
        "backend": {
            "name": config.backend,
            "devices": config.devices,
            "inference_engine": config.inference_engine,
            "inference_url": config.inference_url,
            "attention_kernel": config.attention_kernel,
            "prefix_caching": config.prefix_caching,
            "options": dict(config.backend_options),
        },
        "optimizer": {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "beta1": config.optim_beta1,
            "beta2": config.optim_beta2,
            "eps": config.optim_eps,
            "grad_clip_norm": config.grad_clip_norm,
        },
        "policy_loss": {
            "clip_eps": config.clip_eps,
            "clip_eps_high": config.clip_eps_high,
            "clip_ratio_c": config.clip_ratio_c,
            "mode": config.policy_loss_mode,
            "kl_cov_percent": config.kl_cov_percent,
            "kl_cov_coef": config.kl_cov_coef,
            "clip_cov_ratio": config.clip_cov_ratio,
            "clip_cov_min": config.clip_cov_min,
            "clip_cov_max": config.clip_cov_max,
        },
        "echo": {
            "enabled": config.echo_enabled,
            "loss_fn": config.echo_loss_fn,
            "weight": config.echo_weight,
        },
    }
    return cast(
        dict[str, object],
        sanitize_config_value(contract, redact_ambiguous_keys=False),
    )


def validate_replay_contract(
    manifest: dict[str, object],
    config: TrainConfig,
    initial_adapter: AdapterProvenance,
) -> ReplayContract:
    """Validate source config and adapter identity before model allocation."""

    source_config = _require_object(manifest, "config")
    source_contract = _require_object(source_config, "optimizer_contract")
    source_contract_sha = _require_string(
        source_config,
        "optimizer_contract_sha256",
    )
    if sha256_json(source_contract) != source_contract_sha:
        raise ValueError("optimizer-batch manifest optimizer contract hash mismatch.")

    current_contract = optimizer_contract(config)
    observed = tuple(sorted(_diff_paths(source_contract, current_contract)))
    allowed = tuple(sorted(config.optimizer_batch_allow_config_differences))
    if set(allowed) - {_ALLOWED_V1_DIFFERENCE}:
        raise ValueError(
            "optimizer-batch replay v1 only permits an explicitly declared "
            f"{_ALLOWED_V1_DIFFERENCE!r} difference."
        )
    if observed != allowed:
        undeclared = sorted(set(observed) - set(allowed))
        unused = sorted(set(allowed) - set(observed))
        raise ValueError(
            "optimizer-batch replay config contract mismatch: "
            f"observed={list(observed)}, allowed={list(allowed)}, "
            f"undeclared={undeclared}, unused={unused}."
        )

    source_adapter = _require_object(manifest, "initial_adapter")
    expected_adapter_sha = _require_string(source_adapter, "weight_sha256")
    if initial_adapter.weight_sha256 != expected_adapter_sha:
        raise ValueError(
            "optimizer-batch initial adapter hash mismatch: expected "
            f"{expected_adapter_sha}, got {initial_adapter.weight_sha256}."
        )

    snapshot = config_snapshot(config)
    return ReplayContract(
        current_config_sha256=sha256_json(snapshot),
        current_optimizer_contract_sha256=sha256_json(current_contract),
        allowed_differences=allowed,
        observed_differences=observed,
        initial_adapter=initial_adapter,
    )


def _diff_paths(
    left: object,
    right: object,
    *,
    prefix: str = "",
) -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        left_map = cast(dict[str, object], left)
        right_map = cast(dict[str, object], right)
        paths: set[str] = set()
        for key in set(left_map) | set(right_map):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left_map or key not in right_map:
                paths.add(path)
                continue
            paths.update(_diff_paths(left_map[key], right_map[key], prefix=path))
        return paths
    return {prefix} if left != right else set()


def _require_object(parent: dict[str, object], key: str) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"optimizer-batch manifest field {key!r} must be an object.")
    return cast(dict[str, object], value)


def _require_string(parent: dict[str, object], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"optimizer-batch manifest field {key!r} must be a non-empty string."
        )
    return value
