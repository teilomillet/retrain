"""TrainConfig dataclass and TOML config loader.

Loads training configuration from a TOML file. All defaults match
src/config.mojo exactly.
"""

from __future__ import annotations

import difflib
import json
import sys
import tomllib
import typing
import warnings
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path

from retrain.advantages import (
    get_builtin_transform_modes,
    is_valid_transform_mode_name,
)

_VALID_ADVANTAGE_MODES = {"grpo", "maxrl"}
_VALID_TRANSFORM_MODES = set(get_builtin_transform_modes())
_VALID_ENVIRONMENT_PROVIDERS = {"", "verifiers"}


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
    prime_rl_transport: str = "filesystem"
    prime_rl_zmq_host: str = "localhost"
    prime_rl_zmq_port: int = 5555
    prime_rl_zmq_hwm: int = 10
    prime_rl_strict_advantages: bool = True
    prime_rl_sync_wait_s: int = 30
    prime_rl_sync_poll_s: float = 0.2

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
    entropy_mask_rho: float = 0.0

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

    # Environment bridge (optional; e.g. verifiers envs from Prime Intellect Hub)
    environment_provider: str = ""
    environment_id: str = ""
    environment_args: str = ""
    environment_max_turns: int = -1
    environment_auto_install: bool = False

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
        # --- Hard errors (batched) ---
        errors: list[str] = []

        if self.batch_size <= 0:
            errors.append("batch_size must be > 0. Try: batch_size = 4")
        if self.group_size <= 0:
            errors.append("group_size must be > 0. Try: group_size = 16")
        if self.max_steps <= 0:
            errors.append("max_steps must be > 0. Try: max_steps = 100")
        if self.max_tokens <= 0:
            errors.append("max_tokens must be > 0. Try: max_tokens = 2048")
        if self.lora_rank <= 0:
            errors.append("lora_rank must be > 0. Try: lora_rank = 32")
        if self.lr <= 0:
            errors.append("lr must be > 0. Try: lr = 4e-5")
        if self.temperature < 0:
            errors.append("temperature must be >= 0. Try: temperature = 0.7")
        if self.top_p <= 0 or self.top_p > 1:
            errors.append("top_p must be in (0, 1]. Try: top_p = 0.95")
        if self.entropy_mask_rho < 0.0 or self.entropy_mask_rho > 1.0:
            errors.append(
                "entropy_mask_rho must be in [0.0, 1.0]. Try: entropy_mask_rho = 0.2"
            )
        if self.prime_rl_transport not in {"filesystem", "zmq"}:
            errors.append(
                "prime_rl_transport must be 'filesystem' or 'zmq'. "
                "Try: prime_rl_transport = 'filesystem'"
            )
        if self.prime_rl_zmq_port <= 0 or self.prime_rl_zmq_port > 65535:
            errors.append(
                "prime_rl_zmq_port must be in [1, 65535]. "
                "Try: prime_rl_zmq_port = 5555"
            )
        if self.prime_rl_zmq_hwm <= 0:
            errors.append("prime_rl_zmq_hwm must be > 0. Try: prime_rl_zmq_hwm = 10")
        if self.prime_rl_sync_wait_s < 0:
            errors.append(
                "prime_rl_sync_wait_s must be >= 0. Try: prime_rl_sync_wait_s = 30"
            )
        if self.prime_rl_sync_poll_s <= 0:
            errors.append(
                "prime_rl_sync_poll_s must be > 0. Try: prime_rl_sync_poll_s = 0.2"
            )

        if self.advantage_mode not in _VALID_ADVANTAGE_MODES:
            errors.append(
                f"Invalid advantage_mode '{self.advantage_mode}'. "
                f"Must be one of: {sorted(_VALID_ADVANTAGE_MODES)}"
            )
        if self.transform_mode not in _VALID_TRANSFORM_MODES:
            if not is_valid_transform_mode_name(self.transform_mode):
                errors.append(
                    f"Invalid transform_mode '{self.transform_mode}'. "
                    f"Must be one of: {sorted(_VALID_TRANSFORM_MODES)} "
                    "or a dotted plugin path (e.g. 'my_module.make_transform_spec')."
                )
        if self.environment_provider not in _VALID_ENVIRONMENT_PROVIDERS:
            errors.append(
                f"Invalid environment_provider '{self.environment_provider}'. "
                f"Must be one of: {sorted(_VALID_ENVIRONMENT_PROVIDERS)}"
            )
        if self.environment_provider and not self.environment_id:
            errors.append(
                "environment_id is required when environment_provider is set."
            )
        if self.environment_provider and self.environment_args:
            try:
                parsed_args = json.loads(self.environment_args)
            except json.JSONDecodeError:
                errors.append(
                    "environment_args must be valid JSON when "
                    "environment_provider is set. "
                    "For TOML, prefer: [environment] args = { ... }"
                )
            else:
                if not isinstance(parsed_args, dict):
                    errors.append(
                        "environment_args must decode to a JSON object "
                        "when environment_provider is set."
                    )

        if errors:
            raise ValueError("\n".join(errors))

        # --- Warnings (non-fatal) ---
        if self.adapter_path.startswith("/tmp"):
            warnings.warn(
                "adapter_path starts with /tmp — checkpoints may be lost on reboot.",
                stacklevel=2,
            )
        if self.temperature > 2.0:
            warnings.warn(
                f"temperature={self.temperature} is unusually high.",
                stacklevel=2,
            )
        if self.save_every == 0:
            warnings.warn(
                "save_every=0 disables periodic checkpoints.",
                stacklevel=2,
            )
        if self.weight_decay < 0:
            warnings.warn(
                f"weight_decay={self.weight_decay} is negative — this is unusual.",
                stacklevel=2,
            )

    @property
    def post_process_params(self) -> dict[str, float]:
        """Build the params dict passed to TransformSpec.post_process hooks.

        Collects algorithm hyperparameters that post-process hooks may need.
        Hooks pick the keys they care about; unknown keys are ignored.
        """
        return {
            "entropy_mask_rho": self.entropy_mask_rho,
        }


# TOML section -> config field mapping
_TOML_MAP: dict[str, dict[str, str]] = {
    "algorithm": {
        "advantage_mode": "advantage_mode",
        "transform_mode": "transform_mode",
        "entropy_mask_rho": "entropy_mask_rho",
    },
    "backend": {
        "backend": "backend",
        "devices": "devices",
        "adapter_path": "adapter_path",
        "prime_rl_transport": "prime_rl_transport",
        "prime_rl_zmq_host": "prime_rl_zmq_host",
        "prime_rl_zmq_port": "prime_rl_zmq_port",
        "prime_rl_zmq_hwm": "prime_rl_zmq_hwm",
        "prime_rl_strict_advantages": "prime_rl_strict_advantages",
        "prime_rl_sync_wait_s": "prime_rl_sync_wait_s",
        "prime_rl_sync_poll_s": "prime_rl_sync_poll_s",
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
    "environment": {
        "provider": "environment_provider",
        "id": "environment_id",
        "args": "environment_args",
        "max_turns": "environment_max_turns",
        "auto_install": "environment_auto_install",
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


def _coerce_value(field_name: str, raw: str) -> object:
    """Coerce a CLI string value to the type expected by *field_name*."""
    ftype = _FIELD_TYPES[field_name]
    if ftype is bool:
        return raw.lower() in ("1", "true", "yes")
    if ftype is int:
        return int(raw)
    if ftype is float:
        return float(raw)
    return raw


# Build CLI flag map: --kebab-case → snake_case field name
_CLI_FLAG_MAP: dict[str, str] = {}
for _f in fields(TrainConfig):
    _CLI_FLAG_MAP["--" + _f.name.replace("_", "-")] = _f.name
# Explicit alias
_CLI_FLAG_MAP["--resume"] = "resume_from"


def parse_cli_overrides(argv: list[str]) -> tuple[str | None, dict[str, str]]:
    """Parse CLI args into (config_path, overrides).

    Supports ``--kebab-case value`` and ``--kebab-case=value``.
    The first positional argument (not starting with ``--``) is the config path.
    Unknown flags produce a helpful error with close-match suggestions.
    """
    config_path: str | None = None
    overrides: dict[str, str] = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith("--"):
            if config_path is None:
                config_path = arg
            i += 1
            continue

        # Handle --flag=value
        if "=" in arg:
            flag, value = arg.split("=", 1)
        else:
            flag = arg
            value = None

        if flag not in _CLI_FLAG_MAP:
            close = difflib.get_close_matches(flag, _CLI_FLAG_MAP.keys(), n=1, cutoff=0.6)
            hint = f" Did you mean: {close[0]}?" if close else ""
            print(f"Unknown flag: {flag}.{hint}", file=sys.stderr)
            sys.exit(1)

        if value is None:
            i += 1
            if i >= len(argv):
                print(f"Flag {flag} requires a value.", file=sys.stderr)
                sys.exit(1)
            value = argv[i]

        overrides[_CLI_FLAG_MAP[flag]] = value
        i += 1

    return config_path, overrides


def load_config(path: str | None = None, overrides: dict[str, str] | None = None) -> TrainConfig:
    """Load config from a TOML file.

    If path is None, looks for retrain.toml in cwd.
    Returns TrainConfig with TOML values overlaid on defaults.

    Matches Mojo behavior: empty-string TOML values are ignored
    for string fields (keeps the default).

    *overrides* (from CLI flags) are applied after TOML loading
    but before validation.
    """
    config = TrainConfig.__new__(TrainConfig)
    # Initialise with defaults (skip __post_init__ until the end)
    for f in fields(TrainConfig):
        if f.default is not MISSING:
            setattr(config, f.name, f.default)
        elif f.default_factory is not MISSING:
            setattr(config, f.name, f.default_factory())

    if path is None:
        if Path("retrain.toml").is_file():
            path = "retrain.toml"

    if path is not None:
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
                    if field_name == "environment_args" and isinstance(
                        val, (dict, list, tuple)
                    ):
                        setattr(config, field_name, json.dumps(val))
                        continue
                    # Match Mojo: ignore empty-string values for string fields
                    s = str(val)
                    if s:
                        setattr(config, field_name, s)

    # Apply CLI overrides
    if overrides:
        for field_name, raw_value in overrides.items():
            setattr(config, field_name, _coerce_value(field_name, raw_value))

    # Validate after all overrides
    config.__post_init__()
    return config
