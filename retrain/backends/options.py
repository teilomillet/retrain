"""Backend option schemas, validation, and coercion."""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast


OptionValidator = Callable[[object], str | None]


@dataclass(frozen=True)
class BackendOptionSpec:
    """Schema entry for one backend option."""

    value_type: type
    default: object
    choices: tuple[object, ...] | None = None
    validator: OptionValidator | None = None


def local_option_schema() -> dict[str, BackendOptionSpec]:
    return {
        "train_microbatch_size": BackendOptionSpec(
            value_type=int,
            default=0,
            validator=_validate_non_negative_int,
        ),
        "train_sft_microbatch_token_budget": BackendOptionSpec(
            value_type=int,
            default=0,
            validator=_validate_non_negative_int,
        ),
        "train_logprob_chunk_size": BackendOptionSpec(
            value_type=int,
            default=0,
            validator=_validate_non_negative_int,
        ),
        "liger_kernel": BackendOptionSpec(value_type=bool, default=True),
        "liger_fused_linear_ce": BackendOptionSpec(value_type=bool, default=True),
        "cuda_empty_cache": BackendOptionSpec(value_type=bool, default=True),
        "cuda_expandable_segments": BackendOptionSpec(
            value_type=str,
            default="auto",
        ),
        "sample_use_cache": BackendOptionSpec(value_type=bool, default=True),
        "sample_kv_quantization": BackendOptionSpec(
            value_type=str,
            default="off",
            choices=("off", "oscar"),
        ),
        "sample_oscar_repo": BackendOptionSpec(value_type=str, default=""),
        "sample_oscar_bits": BackendOptionSpec(
            value_type=int,
            default=2,
            choices=(2, 4),
        ),
        "sample_oscar_quant_mode": BackendOptionSpec(
            value_type=str,
            default="k-channel",
        ),
        "sample_oscar_group_size": BackendOptionSpec(
            value_type=int,
            default=0,
            validator=_validate_non_negative_int,
        ),
        "sample_oscar_kv_rotation": BackendOptionSpec(
            value_type=str,
            default="hadamard",
        ),
        "sample_oscar_kv_norm": BackendOptionSpec(
            value_type=str,
            default="1",
        ),
        "sample_oscar_residual_block_size": BackendOptionSpec(
            value_type=int,
            default=128,
            validator=_validate_positive_int,
        ),
        "sample_oscar_attn_implementation": BackendOptionSpec(
            value_type=str,
            default="sdpa",
        ),
        "gradient_checkpointing": BackendOptionSpec(value_type=bool, default=True),
        "gradient_checkpointing_use_reentrant": BackendOptionSpec(
            value_type=str,
            default="auto",
            choices=("auto", "true", "false"),
        ),
        "gradient_checkpointing_skip_last_n": BackendOptionSpec(
            value_type=int,
            default=0,
            validator=_validate_non_negative_int,
        ),
        "cudnn_causal_conv1d_shim": BackendOptionSpec(
            value_type=bool,
            default=False,
        ),
        "qwen35_gated_delta_kernel": BackendOptionSpec(
            value_type=str,
            default="auto",
            choices=("auto", "off", "torch", "flash_qla"),
        ),
        "train_selective_suffix_logits": BackendOptionSpec(
            value_type=bool,
            default=False,
        ),
        "train_save_on_cpu": BackendOptionSpec(value_type=bool, default=False),
        "train_save_on_cpu_pin_memory": BackendOptionSpec(
            value_type=bool,
            default=True,
        ),
        "train_save_on_cpu_min_numel": BackendOptionSpec(
            value_type=int,
            default=0,
            validator=_validate_non_negative_int,
        ),
        "train_supervised_context_tokens": BackendOptionSpec(
            value_type=int,
            default=0,
            validator=_validate_non_negative_int,
        ),
        "train_unsloth_fused_ce": BackendOptionSpec(
            value_type=str,
            default="off",
            choices=("off", "auto", "require"),
        ),
        "train_unsloth_fused_ce_target_gb": BackendOptionSpec(
            value_type=float,
            default=0.0,
            validator=_validate_non_negative_float,
        ),
        "train_unsloth_fused_ce_torch_compile": BackendOptionSpec(
            value_type=bool,
            default=True,
        ),
        "train_compile_selective_ce": BackendOptionSpec(
            value_type=str,
            default="off",
            choices=("off", "auto", "require"),
        ),
        "train_compile_selective_ce_min_tokens": BackendOptionSpec(
            value_type=int,
            default=128,
            validator=_validate_non_negative_int,
        ),
        "lora_target_modules": BackendOptionSpec(
            value_type=str,
            default="",
        ),
        "lora_layers_to_transform": BackendOptionSpec(
            value_type=str,
            default="",
        ),
        "lora_layers_pattern": BackendOptionSpec(
            value_type=str,
            default="layers",
        ),
        "lora_detach_input": BackendOptionSpec(
            value_type=bool,
            default=False,
        ),
        "lora_fast_linear": BackendOptionSpec(
            value_type=bool,
            default=False,
        ),
        "lora_freeze_a": BackendOptionSpec(
            value_type=bool,
            default=False,
        ),
        "trust_remote_code": BackendOptionSpec(value_type=bool, default=False),
    }


def unsloth_option_schema() -> dict[str, BackendOptionSpec]:
    return {
        "max_seq_length": BackendOptionSpec(
            value_type=int,
            default=0,
            validator=_validate_non_negative_int,
        ),
        "load_in_4bit": BackendOptionSpec(value_type=bool, default=True),
        "load_in_8bit": BackendOptionSpec(value_type=bool, default=False),
        "load_in_16bit": BackendOptionSpec(value_type=bool, default=False),
        "fast_inference": BackendOptionSpec(value_type=bool, default=False),
        "gpu_memory_utilization": BackendOptionSpec(
            value_type=float,
            default=0.5,
            validator=_validate_utilization_fraction,
        ),
        "float8_kv_cache": BackendOptionSpec(value_type=bool, default=False),
        "max_lora_rank": BackendOptionSpec(
            value_type=int,
            default=64,
            validator=_validate_positive_int,
        ),
        "use_gradient_checkpointing": BackendOptionSpec(
            value_type=str,
            default="unsloth",
        ),
        "device_map": BackendOptionSpec(value_type=str, default="retrain"),
        "trust_remote_code": BackendOptionSpec(value_type=bool, default=False),
        "use_exact_model_name": BackendOptionSpec(value_type=bool, default=False),
        "offload_embedding": BackendOptionSpec(value_type=bool, default=False),
        "unsloth_tiled_mlp": BackendOptionSpec(value_type=bool, default=False),
        "unsloth_tiled_mlp_mode": BackendOptionSpec(value_type=str, default=""),
        "text_only": BackendOptionSpec(value_type=bool, default=False),
        "use_rslora": BackendOptionSpec(value_type=bool, default=False),
        "random_state": BackendOptionSpec(value_type=int, default=3407),
        "qwen35_gated_delta_chunk_size": BackendOptionSpec(
            value_type=str,
            default="auto",
        ),
        "qwen35_gated_delta_kernel": BackendOptionSpec(
            value_type=str,
            default="auto",
            choices=("auto", "off", "torch", "flash_qla"),
        ),
        "train_microbatch_size": BackendOptionSpec(
            value_type=int,
            default=0,
            validator=_validate_non_negative_int,
        ),
        "train_logprob_chunk_size": BackendOptionSpec(
            value_type=int,
            default=0,
            validator=_validate_non_negative_int,
        ),
        "train_selective_suffix_logits": BackendOptionSpec(
            value_type=bool,
            default=True,
        ),
        "train_save_on_cpu": BackendOptionSpec(value_type=bool, default=False),
        "train_save_on_cpu_pin_memory": BackendOptionSpec(
            value_type=bool,
            default=True,
        ),
        "train_save_on_cpu_min_numel": BackendOptionSpec(
            value_type=int,
            default=0,
            validator=_validate_non_negative_int,
        ),
        "train_supervised_context_tokens": BackendOptionSpec(
            value_type=int,
            default=0,
            validator=_validate_non_negative_int,
        ),
        "train_unsloth_fused_ce": BackendOptionSpec(
            value_type=str,
            default="auto",
            choices=("off", "auto", "require"),
        ),
        "train_unsloth_fused_ce_target_gb": BackendOptionSpec(
            value_type=float,
            default=0.0,
            validator=_validate_non_negative_float,
        ),
        "train_unsloth_fused_ce_torch_compile": BackendOptionSpec(
            value_type=bool,
            default=True,
        ),
        "train_compile_selective_ce": BackendOptionSpec(
            value_type=str,
            default="off",
            choices=("off", "auto", "require"),
        ),
        "train_compile_selective_ce_min_tokens": BackendOptionSpec(
            value_type=int,
            default=128,
            validator=_validate_non_negative_int,
        ),
        "lora_target_modules": BackendOptionSpec(
            value_type=str,
            default="",
        ),
        "liger_kernel": BackendOptionSpec(value_type=bool, default=False),
        "liger_fused_linear_ce": BackendOptionSpec(value_type=bool, default=True),
        "cuda_empty_cache": BackendOptionSpec(value_type=bool, default=True),
        "sample_use_cache": BackendOptionSpec(value_type=bool, default=True),
        "gradient_checkpointing": BackendOptionSpec(value_type=bool, default=True),
    }


def prime_rl_option_schema() -> dict[str, BackendOptionSpec]:
    return {
        "transport": BackendOptionSpec(
            value_type=str,
            default="filesystem",
            choices=("filesystem", "zmq"),
        ),
        "zmq_host": BackendOptionSpec(value_type=str, default="localhost"),
        "zmq_port": BackendOptionSpec(
            value_type=int,
            default=5555,
            validator=_validate_port,
        ),
        "zmq_hwm": BackendOptionSpec(
            value_type=int,
            default=10,
            validator=_validate_positive_int,
        ),
        "strict_advantages": BackendOptionSpec(value_type=bool, default=True),
        "sync_wait_s": BackendOptionSpec(
            value_type=int,
            default=30,
            validator=_validate_non_negative_int,
        ),
        "sync_poll_s": BackendOptionSpec(
            value_type=float,
            default=0.2,
            validator=_validate_positive_float,
        ),
    }


def coerce_plugin_option_schema(
    backend_name: str,
    raw: object,
) -> dict[str, BackendOptionSpec] | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        schema: dict[str, BackendOptionSpec] = {}
        for key, spec in raw.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Invalid backend option schema for '{backend_name}': non-string key {key!r}."
                )
            schema[key] = _coerce_plugin_option_spec(key, spec)
        return schema
    raise ValueError(
        f"Invalid backend option schema for '{backend_name}'. Expected mapping."
    )


def normalize_option_schema(
    backend: str,
    options: Mapping[str, object],
    schema: Mapping[str, BackendOptionSpec],
) -> dict[str, object]:
    unknown = sorted(k for k in options if k not in schema)
    if unknown:
        bad = unknown[0]
        close = difflib.get_close_matches(bad, schema.keys(), n=1, cutoff=0.6)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(
            f"Unknown [backend.options] key '{bad}' for backend '{backend}'.{hint}"
        )

    normalized = {k: spec.default for k, spec in schema.items()}
    for key, raw in options.items():
        spec = schema[key]
        normalized[key] = coerce_option_value(backend, key, raw, spec)
    return normalized


def schema_to_payload(
    schema: Mapping[str, BackendOptionSpec],
) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for key, spec in schema.items():
        payload[key] = {
            "type": option_type_name(spec.value_type),
            "default": spec.default,
            "choices": list(spec.choices) if spec.choices else None,
            "has_validator": spec.validator is not None,
        }
    return payload


def option_type_name(value_type: type) -> str:
    if value_type is bool:
        return "bool"
    if value_type is int:
        return "int"
    if value_type is float:
        return "float"
    if value_type is str:
        return "str"
    return getattr(value_type, "__name__", str(value_type))


def coerce_option_value(
    backend: str,
    key: str,
    raw: object,
    spec: BackendOptionSpec,
) -> object:
    value: object

    if spec.value_type is bool:
        if isinstance(raw, bool):
            value = raw
        elif isinstance(raw, str):
            s = raw.strip().lower()
            if s in {"1", "true", "yes", "on"}:
                value = True
            elif s in {"0", "false", "no", "off"}:
                value = False
            else:
                raise ValueError(
                    f"Invalid [backend.options] {key}={raw!r} for backend '{backend}': "
                    "expected a boolean."
                )
        else:
            raise ValueError(
                f"Invalid [backend.options] {key}={raw!r} for backend '{backend}': "
                "expected a boolean."
            )
    elif spec.value_type is int:
        if isinstance(raw, bool):
            raise ValueError(
                f"Invalid [backend.options] {key}={raw!r} for backend '{backend}': "
                "expected an integer."
            )
        try:
            value = int(cast(str | int | float, raw))
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid [backend.options] {key}={raw!r} for backend '{backend}': "
                "expected an integer."
            ) from None
    elif spec.value_type is float:
        try:
            value = float(cast(str | int | float, raw))
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid [backend.options] {key}={raw!r} for backend '{backend}': "
                "expected a float."
            ) from None
    elif spec.value_type is str:
        value = str(raw)
    else:
        value = raw

    if spec.choices and value not in spec.choices:
        raise ValueError(
            f"Invalid [backend.options] {key}={value!r} for backend '{backend}': "
            f"must be one of {list(spec.choices)}."
        )

    if spec.validator:
        err = spec.validator(value)
        if err:
            raise ValueError(
                f"Invalid [backend.options] {key}={value!r} for backend '{backend}': {err}"
            )

    return value


def _coerce_option_type(raw_type: object) -> type:
    if isinstance(raw_type, type):
        return raw_type
    if isinstance(raw_type, str):
        key = raw_type.strip().lower()
        if key in {"bool", "boolean"}:
            return bool
        if key in {"int", "integer"}:
            return int
        if key == "float":
            return float
        if key in {"str", "string"}:
            return str
    raise ValueError(
        "Invalid backend option type. "
        "Expected one of bool/int/float/str (or corresponding python type)."
    )


def _coerce_plugin_option_spec(key: str, raw: object) -> BackendOptionSpec:
    if isinstance(raw, BackendOptionSpec):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"Invalid backend option schema entry for '{key}'. "
            "Expected BackendOptionSpec or mapping."
        )
    raw_map = cast(Mapping[str, object], raw)
    raw_type = raw_map.get("value_type", raw_map.get("type", str))
    value_type = _coerce_option_type(raw_type)
    default = raw_map.get("default")
    choices_raw = raw_map.get("choices")
    choices: tuple[object, ...] | None = None
    if choices_raw is not None:
        if not isinstance(choices_raw, (list, tuple)):
            raise ValueError(
                f"Invalid backend option schema choices for '{key}': expected list/tuple."
            )
        choices = tuple(choices_raw)
    validator_raw = raw_map.get("validator")
    if validator_raw is not None and not callable(validator_raw):
        raise ValueError(
            f"Invalid backend option validator for '{key}': expected callable."
        )
    validator = cast(OptionValidator | None, validator_raw)
    return BackendOptionSpec(
        value_type=value_type,
        default=default,
        choices=choices,
        validator=validator,
    )


def _validate_port(value: object) -> str | None:
    v = cast(int, value)
    if v <= 0 or v > 65535:
        return "must be in [1, 65535]. Try: 5555"
    return None


def _validate_positive_int(value: object) -> str | None:
    if cast(int, value) <= 0:
        return "must be > 0"
    return None


def _validate_non_negative_int(value: object) -> str | None:
    if cast(int, value) < 0:
        return "must be >= 0"
    return None


def _validate_positive_float(value: object) -> str | None:
    if cast(float, value) <= 0:
        return "must be > 0"
    return None


def _validate_non_negative_float(value: object) -> str | None:
    if cast(float, value) < 0:
        return "must be >= 0"
    return None


def _validate_utilization_fraction(value: object) -> str | None:
    v = cast(float, value)
    if v <= 0 or v > 1:
        return "must be in (0, 1]"
    return None
