"""TrainConfig dataclass and TOML config loader.

Loads training configuration from a TOML file. All defaults match
src/config.mojo exactly.
"""

from __future__ import annotations

import sys
import tomllib
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path

from retrain.advantages import (
    get_builtin_transform_modes,
    is_valid_transform_mode_name,
)

_VALID_ADVANTAGE_MODES = {"grpo", "maxrl"}
_VALID_TRANSFORM_MODES = set(get_builtin_transform_modes())


@dataclass
class SqueezeConfig:
    """Configuration for LoRA-Squeeze rank analysis and compression."""

    adapter_path: str = ""
    source_rank: int = 0  # 0 = fallback to [model].lora_rank
    target_ranks: list[int] = field(default_factory=list)  # [] = auto power-of-2
    min_variance_retention: float = 0.95
    output_path: str = ""
    compress_to: int = 0  # 0 = use recommended rank
    device: str = "cpu"


def load_squeeze_config(path: str) -> SqueezeConfig:
    """Load squeeze config from a TOML file with a [squeeze] section.

    Falls back to [model].lora_rank for source_rank when not specified.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    sq = data.get("squeeze", {})
    if not sq:
        raise ValueError(f"No [squeeze] section in {path}")

    adapter_path = sq.get("adapter_path", "")
    if not adapter_path:
        raise ValueError("[squeeze].adapter_path is required")

    source_rank = int(sq.get("source_rank", 0))
    if source_rank == 0:
        # Fallback to [model].lora_rank
        model_sec = data.get("model", {})
        source_rank = int(model_sec.get("lora_rank", 0))

    target_ranks_raw = sq.get("target_ranks", [])
    target_ranks = [int(r) for r in target_ranks_raw]

    return SqueezeConfig(
        adapter_path=adapter_path,
        source_rank=source_rank,
        target_ranks=target_ranks,
        min_variance_retention=float(sq.get("min_variance_retention", 0.95)),
        output_path=str(sq.get("output_path", "")),
        compress_to=int(sq.get("compress_to", 0)),
        device=str(sq.get("device", "cpu")),
    )


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
    seed: int = -1
    max_steps: int = 500
    batch_size: int = 8
    group_size: int = 16
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    lr: float = 4e-5
    weight_decay: float = 0.0
    max_examples: int = 0
    save_every: int = 20

    # Optimizer
    optim_beta1: float = 0.9
    optim_beta2: float = 0.95
    optim_eps: float = 1e-8

    # LoRA
    lora_alpha: int = 0  # 0 = auto = rank * 2
    lora_dropout: float = 0.0

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

    # Planning detector
    planning_detector: str = "regex"
    planning_model: str = "all-MiniLM-L6-v2"
    planning_threshold: float = 0.02

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

    # Data source
    data_source: str = "math"

    # Reward / verifier
    reward_type: str = "match"
    reward_judge_model: str = ""
    reward_custom_module: str = ""
    reward_custom_function: str = "score"

    # Resume
    resume_from: str = ""

    # Logging
    log_dir: str = "logs/train"
    wandb_project: str = ""
    wandb_run_name: str = ""
    wandb_entity: str = ""
    wandb_group: str = ""
    wandb_tags: str = ""

    def __post_init__(self) -> None:
        if self.advantage_mode not in _VALID_ADVANTAGE_MODES:
            raise ValueError(
                f"Invalid advantage_mode '{self.advantage_mode}'. "
                f"Must be one of: {sorted(_VALID_ADVANTAGE_MODES)}"
            )
        if self.transform_mode not in _VALID_TRANSFORM_MODES:
            if not is_valid_transform_mode_name(self.transform_mode):
                raise ValueError(
                    f"Invalid transform_mode '{self.transform_mode}'. "
                    f"Must be one of: {sorted(_VALID_TRANSFORM_MODES)} "
                    "or a dotted plugin path (e.g. 'my_module.make_transform_spec')."
                )


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
        "seed": "seed",
        "max_steps": "max_steps",
        "batch_size": "batch_size",
        "group_size": "group_size",
        "max_tokens": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "max_examples": "max_examples",
        "save_every": "save_every",
    },
    "optimizer": {
        "beta1": "optim_beta1",
        "beta2": "optim_beta2",
        "eps": "optim_eps",
    },
    "lora": {
        "alpha": "lora_alpha",
        "dropout": "lora_dropout",
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
    "planning": {
        "detector": "planning_detector",
        "model": "planning_model",
        "threshold": "planning_threshold",
    },
    "data": {
        "source": "data_source",
    },
    "reward": {
        "type": "reward_type",
        "judge_model": "reward_judge_model",
        "custom_module": "reward_custom_module",
        "custom_function": "reward_custom_function",
    },
    "resume": {
        "from": "resume_from",
    },
    "logging": {
        "log_dir": "log_dir",
        "wandb_project": "wandb_project",
        "wandb_run_name": "wandb_run_name",
        "wandb_entity": "wandb_entity",
        "wandb_group": "wandb_group",
        "wandb_tags": "wandb_tags",
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

    # Re-validate after TOML overrides
    config.__post_init__()
    return config
