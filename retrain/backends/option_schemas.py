"""Built-in backend option schema declarations."""

from __future__ import annotations

from collections.abc import Callable
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
        "strict_deterministic": BackendOptionSpec(
            value_type=bool,
            default=False,
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
