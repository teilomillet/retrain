"""TrainConfig dataclass and TOML config loader.

Loads training configuration from a TOML file. All defaults match
src/config.mojo exactly.
"""

from __future__ import annotations

import difflib
import json
import re
import sys
import tomllib
import typing
import warnings
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path

from retrain.advantages import (
    canonicalize_uncertainty_kind,
    get_builtin_algorithm_modes,
    get_builtin_advantage_modes,
    get_builtin_transform_modes,
    is_valid_algorithm_mode_name,
    is_valid_advantage_mode_name,
    is_valid_transform_mode_name,
)
from retrain.backend_definitions import normalize_backend_options
from retrain.plugin_resolver import set_plugin_runtime

_VALID_ENVIRONMENT_PROVIDERS = {"", "verifiers"}
_DEFAULT_ADAPTER_PATH = "/tmp/retrain_adapter"


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
    algorithm_mode: str = ""
    advantage_mode: str = "maxrl"
    transform_mode: str = "gtpo_sepa"
    algorithm_params: dict[str, object] = field(default_factory=dict)
    advantage_params: dict[str, object] = field(default_factory=dict)
    transform_params: dict[str, object] = field(default_factory=dict)

    # Backend selection
    backend: str = "local"
    devices: str = "gpu:0"
    adapter_path: str = _DEFAULT_ADAPTER_PATH
    backend_options: dict[str, object] = field(default_factory=dict)

    # Model
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    base_url: str = ""
    lora_rank: int = 32

    # Training
    seed: int = -1
    max_steps: int = 500
    batch_size: int = 8
    group_size: int = 16
    max_tokens: int = 10240
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
    uncertainty_kind: str = "surprisal"
    surprisal_mask_rho: float = 0.0

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

    # Tinker throttle (cross-process concurrency limiter)
    tinker_throttle_dir: str = ""
    tinker_max_concurrent: int = 4

    # Resume
    resume_from: str = ""

    # Logging
    log_dir: str = "logs/train"
    wandb_project: str = ""
    wandb_run_name: str = ""
    wandb_entity: str = ""
    wandb_group: str = ""
    wandb_tags: str = ""

    # Plugin loading
    plugins_search_paths: list[str] = field(default_factory=lambda: ["plugins"])
    plugins_strict: bool = True

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
        if self.surprisal_mask_rho < 0.0 or self.surprisal_mask_rho > 1.0:
            errors.append(
                "surprisal_mask_rho must be in [0.0, 1.0]. Try: surprisal_mask_rho = 0.2"
            )
        try:
            self.uncertainty_kind = canonicalize_uncertainty_kind(
                self.uncertainty_kind
            )
        except ValueError as exc:
            errors.append(str(exc))

        valid_algorithm_modes = set(get_builtin_algorithm_modes())
        if self.algorithm_mode:
            if self.algorithm_mode not in valid_algorithm_modes:
                if not is_valid_algorithm_mode_name(self.algorithm_mode):
                    errors.append(
                        f"Invalid algorithm_mode '{self.algorithm_mode}'. "
                        f"Must be one of: {sorted(valid_algorithm_modes)} "
                        "or a dotted plugin path (e.g. 'my_module.my_algorithm')."
                    )

        valid_advantage_modes = set(get_builtin_advantage_modes())
        if self.advantage_mode not in valid_advantage_modes:
            if not is_valid_advantage_mode_name(self.advantage_mode):
                errors.append(
                    f"Invalid advantage_mode '{self.advantage_mode}'. "
                    f"Must be one of: {sorted(valid_advantage_modes)} "
                    "or a dotted plugin path (e.g. 'my_module.my_advantage')."
                )

        valid_transform_modes = set(get_builtin_transform_modes())
        if self.transform_mode not in valid_transform_modes:
            if not is_valid_transform_mode_name(self.transform_mode):
                errors.append(
                    f"Invalid transform_mode '{self.transform_mode}'. "
                    f"Must be one of: {sorted(valid_transform_modes)} "
                    "or a dotted plugin path (e.g. 'my_module.make_transform_spec')."
                )

        for field_name in ("algorithm_params", "advantage_params", "transform_params"):
            value = getattr(self, field_name)
            if not isinstance(value, dict):
                errors.append(
                    f"{field_name} must be a mapping table."
                )

        if not isinstance(self.plugins_search_paths, list) or not all(
            isinstance(p, str) and p.strip() for p in self.plugins_search_paths
        ):
            errors.append(
                "plugins_search_paths must be a non-empty list of module prefixes."
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

        if not isinstance(self.backend_options, dict):
            errors.append(
                "backend_options must be a mapping. "
                "Use [backend.options] in TOML or --backend-opt key=value."
            )
        else:
            try:
                self.backend_options = normalize_backend_options(
                    self.backend,
                    self.backend_options,
                )
            except ValueError as exc:
                errors.append(str(exc))

        if (
            self.backend == "prime_rl"
            and isinstance(self.backend_options, dict)
            and self.backend_options.get("strict_advantages") is False
        ):
            errors.append(
                "backend.options.strict_advantages=false is disallowed for backend='prime_rl' "
                "to prevent silent token-advantage aggregation. "
                "Use strict_advantages=true."
            )

        if errors:
            raise ValueError("\n".join(errors))

        # Keep plugin runtime config synchronized for dotted-path resolution.
        set_plugin_runtime(self.plugins_search_paths, self.plugins_strict)

        # --- Warnings (non-fatal) ---
        if (
            self.adapter_path.startswith("/tmp")
            and self.adapter_path != _DEFAULT_ADAPTER_PATH
        ):
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
    def post_process_params(self) -> dict[str, object]:
        """Build the params dict passed to TransformSpec.post_process hooks.

        Collects algorithm hyperparameters that post-process hooks may need.
        Hooks pick the keys they care about; unknown keys are ignored.
        """
        params = dict(self.transform_params)
        params.setdefault("uncertainty_kind", self.uncertainty_kind)
        params.setdefault("surprisal_mask_rho", self.surprisal_mask_rho)
        params.setdefault("entropy_mask_rho", self.surprisal_mask_rho)
        return params

    @property
    def effective_advantage_params(self) -> dict[str, object]:
        """Params passed to advantage plugins/callables."""
        params = dict(self.advantage_params)
        params.setdefault("algorithm_mode", self.algorithm_mode)
        params.setdefault("advantage_mode", self.advantage_mode)
        params.setdefault("transform_mode", self.transform_mode)
        return params

    @property
    def effective_algorithm_params(self) -> dict[str, object]:
        """Params passed to algorithm plugins/callables."""
        params = dict(self.algorithm_params)
        params.setdefault("advantage_mode", self.advantage_mode)
        params.setdefault("transform_mode", self.transform_mode)
        params.setdefault("advantage_params", dict(self.advantage_params))
        raw_transform_params = params.get("transform_params")
        if isinstance(raw_transform_params, dict):
            transform_params = dict(raw_transform_params)
        elif isinstance(raw_transform_params, typing.Mapping):
            transform_params = dict(raw_transform_params)
        else:
            transform_params = dict(self.transform_params)
        transform_params.setdefault("uncertainty_kind", self.uncertainty_kind)
        params["transform_params"] = transform_params
        params.setdefault("uncertainty_kind", self.uncertainty_kind)
        params.setdefault("surprisal_mask_rho", self.surprisal_mask_rho)
        params.setdefault("entropy_mask_rho", self.surprisal_mask_rho)
        params.setdefault("gtpo_beta", self.gtpo_beta)
        params.setdefault("hicra_alpha", self.hicra_alpha)
        return params


# TOML section -> config field mapping
_TOML_MAP: dict[str, dict[str, str]] = {
    "algorithm": {
        "algorithm_mode": "algorithm_mode",
        "advantage_mode": "advantage_mode",
        "transform_mode": "transform_mode",
        "uncertainty_kind": "uncertainty_kind",
        "surprisal_mask_rho": "surprisal_mask_rho",
        "entropy_mask_rho": "surprisal_mask_rho",
    },
    "backend": {
        "backend": "backend",
        "devices": "devices",
        "adapter_path": "adapter_path",
        "throttle_dir": "tinker_throttle_dir",
        "max_concurrent": "tinker_max_concurrent",
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
    "plugins": {
        "strict": "plugins_strict",
    },
}

# Resolve annotations to actual types (works with `from __future__ import annotations`)
_FIELD_TYPES: dict[str, type] = typing.get_type_hints(TrainConfig)


_LEGACY_PRIME_RL_KEYS: dict[str, str] = {
    "prime_rl_transport": "transport",
    "prime_rl_zmq_host": "zmq_host",
    "prime_rl_zmq_port": "zmq_port",
    "prime_rl_zmq_hwm": "zmq_hwm",
    "prime_rl_strict_advantages": "strict_advantages",
    "prime_rl_sync_wait_s": "sync_wait_s",
    "prime_rl_sync_poll_s": "sync_poll_s",
}

_TOML_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$")
_TOML_KEY_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=")


@dataclass(frozen=True)
class BackendConfigMigrationResult:
    """Result payload for legacy [backend] PRIME-RL key migration."""

    changed: bool
    legacy_keys: tuple[str, ...]
    merged_backend_options: dict[str, object]
    output_text: str


def detect_legacy_prime_rl_backend_keys(
    backend_sec: object,
) -> dict[str, object]:
    """Detect legacy PRIME-RL keys still present in [backend]."""
    if not isinstance(backend_sec, dict):
        return {}
    return {
        key: backend_sec[key]
        for key in _LEGACY_PRIME_RL_KEYS
        if key in backend_sec
    }


def _toml_literal(value: object) -> str:
    """Render a python value as a TOML literal for migration hints."""
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


def _migration_error_for_legacy_prime_rl_keys(backend_sec: dict[str, object]) -> ValueError:
    """Build a concrete rewrite message for legacy PRIME-RL backend keys."""
    lines = [
        "Legacy PRIME-RL keys are no longer supported in [backend].",
        "Move them under [backend.options].",
        "",
        "Rewrite this section as:",
        "[backend]",
        f'backend = {_toml_literal(backend_sec.get("backend", "prime_rl"))}',
    ]
    if "devices" in backend_sec:
        lines.append(f'devices = {_toml_literal(backend_sec["devices"])}')
    if "adapter_path" in backend_sec:
        lines.append(f'adapter_path = {_toml_literal(backend_sec["adapter_path"])}')
    lines.append("")
    lines.append("[backend.options]")
    for old_key, new_key in _LEGACY_PRIME_RL_KEYS.items():
        if old_key in backend_sec:
            lines.append(f"{new_key} = {_toml_literal(backend_sec[old_key])}")
    return ValueError("\n".join(lines))


def _section_bounds(lines: list[str], section: str) -> tuple[int | None, int]:
    """Return (start,end) bounds for one top-level TOML section."""
    start: int | None = None
    end = len(lines)
    for idx, line in enumerate(lines):
        m = _TOML_SECTION_RE.match(line)
        if not m:
            continue
        if m.group(1).strip() == section:
            start = idx
            break
    if start is None:
        return None, end
    for idx in range(start + 1, len(lines)):
        if _TOML_SECTION_RE.match(lines[idx]):
            end = idx
            break
    return start, end


def _line_key(line: str) -> str | None:
    m = _TOML_KEY_RE.match(line)
    if not m:
        return None
    return m.group(1)


def _ordered_backend_option_keys(options: dict[str, object]) -> list[str]:
    """Stable key order for synthesized [backend.options] blocks."""
    ordered: list[str] = []
    seen: set[str] = set()
    for new_key in _LEGACY_PRIME_RL_KEYS.values():
        if new_key in options:
            ordered.append(new_key)
            seen.add(new_key)
    for key in sorted(options):
        if key not in seen:
            ordered.append(key)
    return ordered


def migrate_legacy_backend_keys_toml_text(toml_text: str) -> BackendConfigMigrationResult:
    """Migrate legacy PRIME-RL [backend] keys to [backend.options] textually.

    Existing keys under [backend.options] win over migrated legacy values.
    """
    data = tomllib.loads(toml_text)
    backend_sec = data.get("backend")
    legacy_values = detect_legacy_prime_rl_backend_keys(backend_sec)
    legacy_keys = tuple(sorted(legacy_values))
    if not legacy_values:
        return BackendConfigMigrationResult(
            changed=False,
            legacy_keys=legacy_keys,
            merged_backend_options={},
            output_text=toml_text,
        )

    if not isinstance(backend_sec, dict):
        return BackendConfigMigrationResult(
            changed=False,
            legacy_keys=legacy_keys,
            merged_backend_options={},
            output_text=toml_text,
        )

    existing_opts_raw = backend_sec.get("options", {})
    if existing_opts_raw is None:
        existing_options: dict[str, object] = {}
    elif isinstance(existing_opts_raw, dict):
        existing_options = dict(existing_opts_raw)
    else:
        raise ValueError(
            "Invalid [backend].options value. "
            "Use a TOML table, e.g. [backend.options] transport = \"filesystem\"."
        )

    migrated_options: dict[str, object] = {}
    for old_key, new_key in _LEGACY_PRIME_RL_KEYS.items():
        if old_key in legacy_values:
            migrated_options[new_key] = legacy_values[old_key]

    merged_options = dict(migrated_options)
    merged_options.update(existing_options)

    had_trailing_newline = toml_text.endswith("\n")
    lines = toml_text.splitlines()

    backend_start, backend_end = _section_bounds(lines, "backend")
    if backend_start is None:
        return BackendConfigMigrationResult(
            changed=False,
            legacy_keys=legacy_keys,
            merged_backend_options=merged_options,
            output_text=toml_text,
        )

    options_start, _ = _section_bounds(lines, "backend.options")
    creating_options_section = options_start is None

    stripped_lines: list[str] = []
    for idx, line in enumerate(lines):
        if backend_start < idx < backend_end:
            key = _line_key(line)
            if key in legacy_values:
                continue
            # Avoid duplicate definition conflict when we create [backend.options].
            if creating_options_section and key == "options":
                continue
        stripped_lines.append(line)

    lines = stripped_lines
    backend_start, backend_end = _section_bounds(lines, "backend")
    options_start, options_end = _section_bounds(lines, "backend.options")

    if options_start is not None:
        existing_option_keys = set(existing_options)
        to_add = [k for k in _ordered_backend_option_keys(migrated_options) if k not in existing_option_keys]
        if to_add:
            insertion = options_end
            for key in to_add:
                lines.insert(insertion, f"{key} = {_toml_literal(merged_options[key])}")
                insertion += 1
    elif backend_start is not None and merged_options:
        insertion = backend_end
        block: list[str] = [
            "",
            "[backend.options]",
        ]
        for key in _ordered_backend_option_keys(merged_options):
            block.append(f"{key} = {_toml_literal(merged_options[key])}")
        lines[insertion:insertion] = block

    migrated_text = "\n".join(lines)
    if had_trailing_newline:
        migrated_text += "\n"

    changed = migrated_text != toml_text
    return BackendConfigMigrationResult(
        changed=changed,
        legacy_keys=legacy_keys,
        merged_backend_options=merged_options,
        output_text=migrated_text,
    )


def _extract_backend_options(backend_sec: object) -> dict[str, object] | None:
    """Extract [backend.options] table when present."""
    if backend_sec is None:
        return None
    if not isinstance(backend_sec, dict):
        return None

    legacy_hits = sorted(detect_legacy_prime_rl_backend_keys(backend_sec))
    if legacy_hits:
        raise _migration_error_for_legacy_prime_rl_keys(backend_sec)

    raw_options = backend_sec.get("options")
    if raw_options is None:
        return None
    if not isinstance(raw_options, dict):
        raise ValueError(
            "Invalid [backend].options value. "
            "Use a TOML table, e.g. [backend.options] transport = \"filesystem\"."
        )
    return dict(raw_options)


def _coerce_value(field_name: str, raw: object) -> object:
    """Coerce a CLI string value to the type expected by *field_name*."""
    if field_name in (
        "backend_options",
        "algorithm_params",
        "advantage_params",
        "transform_params",
    ):
        if not isinstance(raw, dict):
            raise ValueError(
                f"{field_name} override must be a mapping of key=value options."
            )
        return dict(raw)

    if field_name == "plugins_search_paths":
        if isinstance(raw, list):
            return [str(v) for v in raw]
        if isinstance(raw, str):
            if not raw.strip():
                return []
            if raw.strip().startswith("["):
                loaded = json.loads(raw)
                if not isinstance(loaded, list):
                    raise ValueError(
                        "plugins_search_paths JSON override must decode to a list."
                    )
                return [str(v) for v in loaded]
            return [p.strip() for p in raw.split(",") if p.strip()]
        raise ValueError("plugins_search_paths override must be list or comma string.")

    ftype = _FIELD_TYPES[field_name]
    if ftype is bool:
        if isinstance(raw, bool):
            return raw
        if not isinstance(raw, str):
            raise ValueError(f"Expected string for {field_name}, got {type(raw).__name__}")
        return raw.lower() in ("1", "true", "yes")
    if ftype is int:
        if isinstance(raw, int):
            return raw
        return int(raw)
    if ftype is float:
        if isinstance(raw, float):
            return raw
        return float(raw)
    return raw


# Build CLI flag map: --kebab-case → snake_case field name
_CLI_FLAG_MAP: dict[str, str] = {}
for _f in fields(TrainConfig):
    if _f.name in (
        "backend_options",
        "algorithm_params",
        "advantage_params",
        "transform_params",
    ):
        continue
    _CLI_FLAG_MAP["--" + _f.name.replace("_", "-")] = _f.name
# Explicit alias
_CLI_FLAG_MAP["--resume"] = "resume_from"


def _parse_backend_opt(raw_value: str) -> tuple[str, str]:
    """Parse one backend option override from CLI key=value format."""
    if "=" not in raw_value:
        raise ValueError(
            "Flag --backend-opt requires key=value (example: --backend-opt transport=zmq)."
        )
    key, value = raw_value.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("Flag --backend-opt requires a non-empty key (key=value).")
    return key, value


def _parse_param_opt(raw_value: str, flag_name: str) -> tuple[str, object]:
    """Parse one repeatable parameter override from key=value format."""
    if "=" not in raw_value:
        raise ValueError(
            f"Flag {flag_name} requires key=value "
            f"(example: {flag_name} alpha=0.2)."
        )
    key, value = raw_value.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Flag {flag_name} requires a non-empty key (key=value).")

    v = value.strip()
    if not v:
        return key, ""
    if v.lower() in {"true", "false"}:
        return key, v.lower() == "true"
    if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
        try:
            return key, json.loads(v)
        except json.JSONDecodeError:
            pass
    try:
        if "." in v:
            return key, float(v)
        return key, int(v)
    except ValueError:
        return key, v


def parse_cli_overrides(argv: list[str]) -> tuple[str | None, dict[str, object]]:
    """Parse CLI args into (config_path, overrides).

    Supports ``--kebab-case value`` and ``--kebab-case=value``.
    The first positional argument (not starting with ``--``) is the config path.
    Unknown flags produce a helpful error with close-match suggestions.
    """
    config_path: str | None = None
    overrides: dict[str, object] = {}
    backend_opt_overrides: dict[str, object] = {}
    algorithm_param_overrides: dict[str, object] = {}
    advantage_param_overrides: dict[str, object] = {}
    transform_param_overrides: dict[str, object] = {}
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

        if flag == "--backend-opt":
            if value is None:
                i += 1
                if i >= len(argv):
                    print("Flag --backend-opt requires a value.", file=sys.stderr)
                    sys.exit(1)
                value = argv[i]
            try:
                key, opt_value = _parse_backend_opt(value)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                sys.exit(1)
            backend_opt_overrides[key] = opt_value
            i += 1
            continue

        if flag in ("--algorithm-param", "--advantage-param", "--transform-param"):
            if value is None:
                i += 1
                if i >= len(argv):
                    print(f"Flag {flag} requires a value.", file=sys.stderr)
                    sys.exit(1)
                value = argv[i]
            try:
                key, param_value = _parse_param_opt(value, flag)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                sys.exit(1)
            if flag == "--algorithm-param":
                algorithm_param_overrides[key] = param_value
            elif flag == "--advantage-param":
                advantage_param_overrides[key] = param_value
            else:
                transform_param_overrides[key] = param_value
            i += 1
            continue

        if flag not in _CLI_FLAG_MAP:
            close = difflib.get_close_matches(
                flag,
                list(_CLI_FLAG_MAP.keys())
                + [
                    "--backend-opt",
                    "--algorithm-param",
                    "--advantage-param",
                    "--transform-param",
                ],
                n=1,
                cutoff=0.6,
            )
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

    if backend_opt_overrides:
        overrides["backend_options"] = backend_opt_overrides
    if algorithm_param_overrides:
        overrides["algorithm_params"] = algorithm_param_overrides
    if advantage_param_overrides:
        overrides["advantage_params"] = advantage_param_overrides
    if transform_param_overrides:
        overrides["transform_params"] = transform_param_overrides

    return config_path, overrides


def load_config(
    path: str | None = None,
    overrides: dict[str, object] | None = None,
) -> TrainConfig:
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

        backend_options = _extract_backend_options(data.get("backend"))
        if backend_options is not None:
            setattr(config, "backend_options", backend_options)

        plugins_sec = data.get("plugins")
        if isinstance(plugins_sec, dict) and "search_paths" in plugins_sec:
            raw_paths = plugins_sec["search_paths"]
            if isinstance(raw_paths, list):
                setattr(config, "plugins_search_paths", [str(v) for v in raw_paths])
            elif isinstance(raw_paths, str):
                setattr(
                    config,
                    "plugins_search_paths",
                    [p.strip() for p in raw_paths.split(",") if p.strip()],
                )
            else:
                raise ValueError(
                    "Invalid [plugins].search_paths value. "
                    "Use a TOML list of strings."
                )

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

        algorithm_sec = data.get("algorithm")
        if isinstance(algorithm_sec, dict):
            for key, field_name in (
                ("params", "algorithm_params"),
                ("advantage_params", "advantage_params"),
                ("transform_params", "transform_params"),
            ):
                if key not in algorithm_sec:
                    continue
                raw_map = algorithm_sec[key]
                if not isinstance(raw_map, dict):
                    raise ValueError(
                        f"Invalid [algorithm].{key} value. "
                        "Use a TOML table."
                    )
                setattr(config, field_name, dict(raw_map))

    # Apply CLI overrides
    if overrides:
        for field_name, raw_value in overrides.items():
            if field_name in (
                "backend_options",
                "algorithm_params",
                "advantage_params",
                "transform_params",
            ):
                merged = dict(getattr(config, "backend_options", {}))
                if field_name != "backend_options":
                    merged = dict(getattr(config, field_name, {}))
                merged.update(_coerce_value(field_name, raw_value))
                setattr(config, field_name, merged)
                continue
            setattr(config, field_name, _coerce_value(field_name, raw_value))

    # Validate after all overrides
    config.__post_init__()
    return config
