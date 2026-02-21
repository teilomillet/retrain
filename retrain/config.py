"""TrainConfig dataclass and TOML config loader.

Loads training configuration from a TOML file. All defaults match
src/config.mojo exactly.
"""

from __future__ import annotations

import sys
import tomllib
import typing
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class TrainConfig:
    """All training hyperparameters."""

    # Algorithm selection
    advantage_mode: str = "maxrl"
    transform_mode: str = "gtpo_sepa"

    # Backend selection
    backend: str = "local"
    devices: str = "gpu:0"
    adapter_path: str = "/tmp/retrain_adapter"

    # Model
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    base_url: str = ""
    lora_rank: int = 32

    # Training
    max_steps: int = 500
    batch_size: int = 8
    group_size: int = 16
    max_tokens: int = 2048
    temperature: float = 0.7
    lr: float = 4e-5
    weight_decay: float = 0.0
    max_examples: int = 0
    save_every: int = 20

    # Algorithm hyperparameters
    gtpo_beta: float = 0.1
    hicra_alpha: float = 0.2

    # SEPA
    sepa_steps: int = 500
    sepa_schedule: str = "linear"
    sepa_delay_steps: int = 50
    sepa_correct_rate_gate: float = 0.1

    # Strategic grams (JSON string, empty = use defaults)
    strategic_grams: str = ""

    # Back pressure
    bp_enabled: bool = False
    bp_warmup_steps: int = 10
    bp_ema_decay: float = 0.9
    bp_throttle_margin: float = 0.85
    bp_increase_margin: float = 0.5
    bp_min_batch_size: int = 1
    bp_max_batch_size: int = 64
    bp_peak_gflops: float = 0.0
    bp_peak_bw_gb_s: float = 0.0

    # Inference engine
    inference_engine: str = "pytorch"
    inference_url: str = ""
    attention_kernel: str = "default"
    inference_dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    prefix_caching: bool = True

    # Reward / verifier
    reward_type: str = "match"
    reward_judge_model: str = ""
    reward_custom_module: str = ""
    reward_custom_function: str = "score"

    # Logging
    log_dir: str = "logs/train"
    wandb_project: str = ""
    wandb_run_name: str = ""


# TOML section -> config field mapping
_TOML_MAP: dict[str, dict[str, str]] = {
    "algorithm": {
        "advantage_mode": "advantage_mode",
        "transform_mode": "transform_mode",
    },
    "backend": {
        "backend": "backend",
        "devices": "devices",
        "adapter_path": "adapter_path",
    },
    "model": {
        "model": "model",
        "base_url": "base_url",
        "lora_rank": "lora_rank",
    },
    "training": {
        "max_steps": "max_steps",
        "batch_size": "batch_size",
        "group_size": "group_size",
        "max_tokens": "max_tokens",
        "temperature": "temperature",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "max_examples": "max_examples",
        "save_every": "save_every",
    },
    "gtpo": {"beta": "gtpo_beta"},
    "hicra": {"alpha": "hicra_alpha"},
    "sepa": {
        "steps": "sepa_steps",
        "schedule": "sepa_schedule",
        "delay_steps": "sepa_delay_steps",
        "correct_rate_gate": "sepa_correct_rate_gate",
    },
    "inference": {
        "engine": "inference_engine",
        "url": "inference_url",
        "attention_kernel": "attention_kernel",
        "dtype": "inference_dtype",
        "kv_cache_dtype": "kv_cache_dtype",
        "prefix_caching": "prefix_caching",
    },
    "backpressure": {
        "enabled": "bp_enabled",
        "warmup_steps": "bp_warmup_steps",
        "ema_decay": "bp_ema_decay",
        "throttle_margin": "bp_throttle_margin",
        "increase_margin": "bp_increase_margin",
        "min_batch_size": "bp_min_batch_size",
        "max_batch_size": "bp_max_batch_size",
        "peak_gflops": "bp_peak_gflops",
        "peak_bw_gb_s": "bp_peak_bw_gb_s",
    },
    "reward": {
        "type": "reward_type",
        "judge_model": "reward_judge_model",
        "custom_module": "reward_custom_module",
        "custom_function": "reward_custom_function",
    },
    "logging": {
        "log_dir": "log_dir",
        "wandb_project": "wandb_project",
        "wandb_run_name": "wandb_run_name",
        "strategic_grams": "strategic_grams",
    },
}

# Resolve annotations to actual types (works with `from __future__ import annotations`)
_FIELD_TYPES: dict[str, type] = typing.get_type_hints(TrainConfig)


def load_config(path: str | None = None) -> TrainConfig:
    """Load config from a TOML file.

    If path is None, looks for retrain.toml in cwd.
    Returns TrainConfig with TOML values overlaid on defaults.

    Matches Mojo behavior: empty-string TOML values are ignored
    for string fields (keeps the default).
    """
    config = TrainConfig()

    if path is None:
        if Path("retrain.toml").is_file():
            path = "retrain.toml"
        else:
            return config

    with open(path, "rb") as f:
        data = tomllib.load(f)

    for section, mapping in _TOML_MAP.items():
        sec = data.get(section)
        if sec is None:
            continue
        for toml_key, field_name in mapping.items():
            if toml_key not in sec:
                continue
            val = sec[toml_key]
            ftype = _FIELD_TYPES[field_name]
            if ftype is bool:
                setattr(config, field_name, bool(val))
            elif ftype is int:
                setattr(config, field_name, int(val))
            elif ftype is float:
                setattr(config, field_name, float(val))
            else:
                # Match Mojo: ignore empty-string values for string fields
                s = str(val)
                if s:
                    setattr(config, field_name, s)

    return config
