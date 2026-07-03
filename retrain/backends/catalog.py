"""Built-in backend definitions and backend-option normalization."""

from __future__ import annotations

import difflib
import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from retrain.backends import TrainHelper
    from retrain.config import TrainConfig


BackendFactory = Callable[["TrainConfig"], "TrainHelper"]
OptionValidator = Callable[[object], str | None]


class PrimeRLOptions(TypedDict):
    transport: str
    zmq_host: str
    zmq_port: int
    zmq_hwm: int
    strict_advantages: bool
    sync_wait_s: int
    sync_poll_s: float


@dataclass(frozen=True)
class BackendOptionSpec:
    """Schema entry for one built-in backend option."""

    value_type: type
    default: object
    choices: tuple[object, ...] | None = None
    validator: OptionValidator | None = None


@dataclass(frozen=True)
class BackendCapabilities:
    """Core runtime capability metadata exposed by a backend."""

    reports_sync_loss: bool
    preserves_token_advantages: bool
    supports_checkpoint_resume: bool
    resume_runtime_dependent: bool
    supports_echo_shared_forward: bool = False


@dataclass(frozen=True)
class BackendDefinition:
    """Single source of truth for a built-in backend."""

    name: str
    factory: BackendFactory
    dependency_import: str
    dependency_hint: str
    capabilities: BackendCapabilities
    option_schema: dict[str, BackendOptionSpec] = field(default_factory=dict)


def _backend_option_int(
    options: Mapping[str, object],
    key: str,
    default: int = 0,
) -> int:
    raw = options.get(key, default)
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int | float | str):
        return int(raw)
    return default


def _backend_option_float(
    options: Mapping[str, object],
    key: str,
    default: float = 0.0,
) -> float:
    raw = options.get(key, default)
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, int | float | str):
        return float(raw)
    return default


def _effective_sft_loss_fn(config: "TrainConfig") -> str:
    """Resolve SFT loss without changing legacy warmup defaults."""
    if config.sft_loss_fn != "auto":
        return config.sft_loss_fn
    if config.trainer == "sft":
        return "cross_entropy"
    return "importance_sampling"


def _create_local(config: "TrainConfig") -> "TrainHelper":
    try:
        from retrain.backends.local import LocalTrainHelper
    except ImportError:
        raise RuntimeError(
            "Backend 'local' requires PyTorch.\n"
            "Install it with: pip install retrain[local]"
        ) from None
    helper = LocalTrainHelper(
        config.model,
        config.adapter_path,
        config.devices,
        config.lora_rank,
        config.inference_engine,
        config.inference_url,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        optim_beta1=config.optim_beta1,
        optim_beta2=config.optim_beta2,
        optim_eps=config.optim_eps,
        clip_eps=config.clip_eps,
        clip_eps_high=config.clip_eps_high,
        policy_loss_mode=config.policy_loss_mode,
        kl_cov_percent=config.kl_cov_percent,
        kl_cov_coef=config.kl_cov_coef,
        clip_cov_ratio=config.clip_cov_ratio,
        clip_cov_min=config.clip_cov_min,
        clip_cov_max=config.clip_cov_max,
        train_microbatch_size=_backend_option_int(
            config.backend_options,
            "train_microbatch_size",
        ),
        train_sft_microbatch_token_budget=_backend_option_int(
            config.backend_options,
            "train_sft_microbatch_token_budget",
        ),
        train_logprob_chunk_size=_backend_option_int(
            config.backend_options,
            "train_logprob_chunk_size",
        ),
        liger_kernel=bool(config.backend_options.get("liger_kernel", True)),
        liger_fused_linear_ce=bool(
            config.backend_options.get("liger_fused_linear_ce", True)
        ),
        cuda_empty_cache=bool(config.backend_options.get("cuda_empty_cache", True)),
        cuda_expandable_segments=str(
            config.backend_options.get("cuda_expandable_segments", "auto")
        ),
        sample_use_cache=bool(config.backend_options.get("sample_use_cache", True)),
        gradient_checkpointing=bool(
            config.backend_options.get("gradient_checkpointing", True)
        ),
        gradient_checkpointing_use_reentrant=str(
            config.backend_options.get("gradient_checkpointing_use_reentrant", "auto")
        ),
        gradient_checkpointing_skip_last_n=_backend_option_int(
            config.backend_options,
            "gradient_checkpointing_skip_last_n",
        ),
        cudnn_causal_conv1d_shim=bool(
            config.backend_options.get("cudnn_causal_conv1d_shim", False)
        ),
        attention_kernel=config.attention_kernel,
        prefix_caching=config.prefix_caching,
        train_selective_suffix_logits=bool(
            config.backend_options.get("train_selective_suffix_logits", False)
        ),
        train_save_on_cpu=bool(config.backend_options.get("train_save_on_cpu", False)),
        train_save_on_cpu_pin_memory=bool(
            config.backend_options.get("train_save_on_cpu_pin_memory", True)
        ),
        train_save_on_cpu_min_numel=_backend_option_int(
            config.backend_options,
            "train_save_on_cpu_min_numel",
        ),
        train_supervised_context_tokens=_backend_option_int(
            config.backend_options,
            "train_supervised_context_tokens",
        ),
        train_unsloth_fused_ce=str(
            config.backend_options.get("train_unsloth_fused_ce", "off")
        ),
        train_unsloth_fused_ce_target_gb=_backend_option_float(
            config.backend_options,
            "train_unsloth_fused_ce_target_gb",
            0.0,
        ),
        train_unsloth_fused_ce_torch_compile=bool(
            config.backend_options.get("train_unsloth_fused_ce_torch_compile", True)
        ),
        train_compile_selective_ce=str(
            config.backend_options.get("train_compile_selective_ce", "off")
        ),
        train_compile_selective_ce_min_tokens=_backend_option_int(
            config.backend_options,
            "train_compile_selective_ce_min_tokens",
            128,
        ),
        lora_target_modules=str(
            config.backend_options.get("lora_target_modules", "")
        ),
        lora_layers_to_transform=str(
            config.backend_options.get("lora_layers_to_transform", "")
        ),
        lora_layers_pattern=str(
            config.backend_options.get("lora_layers_pattern", "layers")
        ),
        lora_detach_input=bool(
            config.backend_options.get("lora_detach_input", False)
        ),
        lora_fast_linear=bool(
            config.backend_options.get("lora_fast_linear", False)
        ),
        lora_freeze_a=bool(
            config.backend_options.get("lora_freeze_a", False)
        ),
        qwen35_gated_delta_kernel=str(
            config.backend_options.get("qwen35_gated_delta_kernel", "auto")
        ),
        trust_remote_code=bool(config.backend_options.get("trust_remote_code", False)),
    )
    setattr(helper, "sft_loss_fn", _effective_sft_loss_fn(config))
    return helper


def _create_unsloth(config: "TrainConfig") -> "TrainHelper":
    try:
        from retrain.backends.unsloth import UnslothTrainHelper
    except ImportError:
        raise RuntimeError(
            "Backend 'unsloth' requires PyTorch and Unsloth Core.\n"
            "Install Unsloth with: uv pip install unsloth --torch-backend=auto"
        ) from None

    options = config.backend_options
    max_seq_length = _backend_option_int(options, "max_seq_length")
    if max_seq_length <= 0:
        max_seq_length = max(2048, int(config.max_tokens))
    helper = UnslothTrainHelper(
        config.model,
        config.adapter_path,
        config.devices,
        config.lora_rank,
        config.inference_engine,
        config.inference_url,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        optim_beta1=config.optim_beta1,
        optim_beta2=config.optim_beta2,
        optim_eps=config.optim_eps,
        clip_eps=config.clip_eps,
        clip_eps_high=config.clip_eps_high,
        policy_loss_mode=config.policy_loss_mode,
        kl_cov_percent=config.kl_cov_percent,
        kl_cov_coef=config.kl_cov_coef,
        clip_cov_ratio=config.clip_cov_ratio,
        clip_cov_min=config.clip_cov_min,
        clip_cov_max=config.clip_cov_max,
        train_microbatch_size=_backend_option_int(
            options,
            "train_microbatch_size",
        ),
        train_logprob_chunk_size=_backend_option_int(
            options,
            "train_logprob_chunk_size",
        ),
        train_selective_suffix_logits=bool(
            options.get("train_selective_suffix_logits", True)
        ),
        train_save_on_cpu=bool(options.get("train_save_on_cpu", False)),
        train_save_on_cpu_pin_memory=bool(
            options.get("train_save_on_cpu_pin_memory", True)
        ),
        train_save_on_cpu_min_numel=_backend_option_int(
            options,
            "train_save_on_cpu_min_numel",
        ),
        train_supervised_context_tokens=_backend_option_int(
            options,
            "train_supervised_context_tokens",
        ),
        train_unsloth_fused_ce=str(
            options.get("train_unsloth_fused_ce", "auto")
        ),
        train_unsloth_fused_ce_target_gb=_backend_option_float(
            options,
            "train_unsloth_fused_ce_target_gb",
            0.0,
        ),
        train_unsloth_fused_ce_torch_compile=bool(
            options.get("train_unsloth_fused_ce_torch_compile", True)
        ),
        train_compile_selective_ce=str(
            options.get("train_compile_selective_ce", "off")
        ),
        train_compile_selective_ce_min_tokens=_backend_option_int(
            options,
            "train_compile_selective_ce_min_tokens",
            128,
        ),
        lora_target_modules=str(options.get("lora_target_modules", "")),
        liger_kernel=bool(options.get("liger_kernel", False)),
        liger_fused_linear_ce=bool(options.get("liger_fused_linear_ce", True)),
        cuda_empty_cache=bool(options.get("cuda_empty_cache", True)),
        sample_use_cache=bool(options.get("sample_use_cache", True)),
        gradient_checkpointing=bool(options.get("gradient_checkpointing", True)),
        attention_kernel=config.attention_kernel,
        prefix_caching=config.prefix_caching,
        max_seq_length=max_seq_length,
        load_in_4bit=bool(options.get("load_in_4bit", True)),
        load_in_8bit=bool(options.get("load_in_8bit", False)),
        load_in_16bit=bool(options.get("load_in_16bit", False)),
        fast_inference=bool(options.get("fast_inference", False)),
        gpu_memory_utilization=_backend_option_float(
            options,
            "gpu_memory_utilization",
            0.5,
        ),
        float8_kv_cache=bool(options.get("float8_kv_cache", False)),
        max_lora_rank=_backend_option_int(options, "max_lora_rank", 64),
        use_gradient_checkpointing=str(
            options.get("use_gradient_checkpointing", "unsloth")
        ),
        device_map=str(options.get("device_map", "retrain")),
        trust_remote_code=bool(options.get("trust_remote_code", False)),
        use_exact_model_name=bool(options.get("use_exact_model_name", False)),
        offload_embedding=bool(options.get("offload_embedding", False)),
        unsloth_tiled_mlp=bool(options.get("unsloth_tiled_mlp", False)),
        unsloth_tiled_mlp_mode=str(options.get("unsloth_tiled_mlp_mode", "")),
        text_only=bool(options.get("text_only", False)),
        use_rslora=bool(options.get("use_rslora", False)),
        random_state=_backend_option_int(options, "random_state", 3407),
        qwen35_gated_delta_chunk_size=str(
            options.get("qwen35_gated_delta_chunk_size", "auto")
        ),
        qwen35_gated_delta_kernel=str(
            options.get("qwen35_gated_delta_kernel", "auto")
        ),
    )
    setattr(helper, "sft_loss_fn", _effective_sft_loss_fn(config))
    return helper


def _create_tinker(config: "TrainConfig") -> "TrainHelper":
    try:
        from retrain.backends.tinker import TinkerTrainHelper
    except ImportError:
        raise RuntimeError(
            "Backend 'tinker' requires the tinker SDK.\n"
            "Install it with: pip install retrain[tinker]"
        ) from None

    tinker_url = config.inference_url or config.base_url
    helper = TinkerTrainHelper(
        config.model,
        tinker_url,
        config.lora_rank,
        optim_beta1=config.optim_beta1,
        optim_beta2=config.optim_beta2,
        optim_eps=config.optim_eps,
        throttle_dir=config.tinker_throttle_dir,
        max_concurrent=config.tinker_max_concurrent,
        clip_eps=config.clip_eps,
        clip_eps_high=config.clip_eps_high,
        grad_clip_norm=config.grad_clip_norm,
        clip_ratio_c=config.clip_ratio_c,
        sample_log_dir=str(Path(config.log_dir).resolve()),
    )
    setattr(helper, "sft_loss_fn", _effective_sft_loss_fn(config))
    return helper


def _normalize_prime_rl_options(raw_options: Mapping[str, object]) -> PrimeRLOptions:
    options = normalize_backend_options("prime_rl", raw_options)
    return {
        "transport": cast(str, options["transport"]),
        "zmq_host": cast(str, options["zmq_host"]),
        "zmq_port": cast(int, options["zmq_port"]),
        "zmq_hwm": cast(int, options["zmq_hwm"]),
        "strict_advantages": cast(bool, options["strict_advantages"]),
        "sync_wait_s": cast(int, options["sync_wait_s"]),
        "sync_poll_s": cast(float, options["sync_poll_s"]),
    }


def _create_prime_rl(config: "TrainConfig") -> "TrainHelper":
    try:
        from retrain.backends.prime import PrimeRLTrainHelper
    except ImportError:
        raise RuntimeError(
            "Backend 'prime_rl' requires PRIME-RL.\n"
            "Install it with: pip install prime-rl"
        ) from None

    options = _normalize_prime_rl_options(config.backend_options)
    inference_url = config.inference_url or config.base_url or "http://localhost:8000"

    return PrimeRLTrainHelper(
        model_name=config.model,
        output_dir=config.adapter_path,
        inference_url=inference_url,
        transport_type=options["transport"],
        zmq_host=options["zmq_host"],
        zmq_port=options["zmq_port"],
        zmq_hwm=options["zmq_hwm"],
        strict_advantages=options["strict_advantages"],
        sync_wait_s=options["sync_wait_s"],
        sync_poll_s=options["sync_poll_s"],
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


_BUILTIN_BACKENDS: dict[str, BackendDefinition] = {
    "local": BackendDefinition(
        name="local",
        factory=_create_local,
        dependency_import="torch",
        dependency_hint="pip install retrain[local]",
        capabilities=BackendCapabilities(
            reports_sync_loss=True,
            preserves_token_advantages=True,
            supports_checkpoint_resume=True,
            resume_runtime_dependent=False,
            supports_echo_shared_forward=True,
        ),
        option_schema={
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
        },
    ),
    "unsloth": BackendDefinition(
        name="unsloth",
        factory=_create_unsloth,
        dependency_import="unsloth",
        dependency_hint="uv pip install unsloth --torch-backend=auto",
        capabilities=BackendCapabilities(
            reports_sync_loss=True,
            preserves_token_advantages=True,
            supports_checkpoint_resume=True,
            resume_runtime_dependent=False,
            supports_echo_shared_forward=True,
        ),
        option_schema={
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
        },
    ),
    "tinker": BackendDefinition(
        name="tinker",
        factory=_create_tinker,
        dependency_import="tinker",
        dependency_hint="pip install retrain[tinker]",
        capabilities=BackendCapabilities(
            reports_sync_loss=True,
            preserves_token_advantages=True,
            supports_checkpoint_resume=True,
            resume_runtime_dependent=True,
            supports_echo_shared_forward=False,
        ),
        option_schema={},
    ),
    "prime_rl": BackendDefinition(
        name="prime_rl",
        factory=_create_prime_rl,
        dependency_import="prime_rl",
        dependency_hint="pip install prime-rl",
        capabilities=BackendCapabilities(
            reports_sync_loss=False,
            preserves_token_advantages=False,
            supports_checkpoint_resume=True,
            resume_runtime_dependent=False,
            supports_echo_shared_forward=False,
        ),
        option_schema={
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
        },
    ),
}

_PLUGIN_DEFAULT_CAPABILITIES = BackendCapabilities(
    reports_sync_loss=True,
    preserves_token_advantages=False,
    supports_checkpoint_resume=True,
    resume_runtime_dependent=False,
)
_PLUGIN_CAPABILITIES_HOOKS = (
    "retrain_backend_capabilities",
    "RETRAIN_BACKEND_CAPABILITIES",
)
_PLUGIN_OPTION_SCHEMA_HOOKS = (
    "retrain_backend_option_schema",
    "RETRAIN_BACKEND_OPTION_SCHEMA",
)


def get_builtin_backend_definitions() -> dict[str, BackendDefinition]:
    """Return all built-in backend definitions keyed by backend name."""
    return dict(_BUILTIN_BACKENDS)


def get_backend_dependency_map() -> dict[str, tuple[str, str]]:
    """Return dependency metadata for built-in backends."""
    return {
        name: (definition.dependency_import, definition.dependency_hint)
        for name, definition in _BUILTIN_BACKENDS.items()
    }


def _coerce_backend_capabilities(raw: object) -> BackendCapabilities | None:
    if isinstance(raw, BackendCapabilities):
        return raw
    if isinstance(raw, Mapping):
        payload = cast(Mapping[str, object], raw)
        try:
            return BackendCapabilities(
                reports_sync_loss=bool(payload["reports_sync_loss"]),
                preserves_token_advantages=bool(payload["preserves_token_advantages"]),
                supports_checkpoint_resume=bool(payload["supports_checkpoint_resume"]),
                resume_runtime_dependent=bool(payload["resume_runtime_dependent"]),
                supports_echo_shared_forward=bool(
                    payload.get("supports_echo_shared_forward", False)
                ),
            )
        except KeyError:
            return None
    return None


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


def _coerce_plugin_option_schema(
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


def _resolve_hook_value(raw: object, backend_options: Mapping[str, object]) -> object:
    if not callable(raw):
        return raw
    callback = cast(Callable[..., object], raw)
    try:
        return callback(dict(backend_options))
    except TypeError:
        return callback()


def _import_plugin_target(backend_name: str) -> tuple[object, object] | tuple[None, None]:
    module_path, _, attr_name = backend_name.rpartition(".")
    if not module_path or not attr_name:
        return None, None
    try:
        module = importlib.import_module(module_path)
    except Exception:
        return None, None
    target = getattr(module, attr_name, None)
    if target is None:
        return None, None
    return module, target


def _plugin_hook_value(
    backend_name: str,
    hook_names: tuple[str, ...],
    backend_options: Mapping[str, object],
) -> object | None:
    module, target = _import_plugin_target(backend_name)
    if module is None or target is None:
        return None
    for hook_name in hook_names:
        for holder in (target, module):
            raw = getattr(holder, hook_name, None)
            if raw is None:
                continue
            return _resolve_hook_value(raw, backend_options)
    return None


def resolve_backend_capabilities(
    backend_name: str,
    backend_options: Mapping[str, object] | None = None,
) -> BackendCapabilities:
    """Resolve capability metadata for built-in and dotted plugin backends."""
    options = {} if backend_options is None else dict(backend_options)
    definition = _BUILTIN_BACKENDS.get(backend_name)
    if definition is not None:
        return definition.capabilities
    if "." in backend_name:
        raw = _plugin_hook_value(
            backend_name,
            _PLUGIN_CAPABILITIES_HOOKS,
            options,
        )
        caps = _coerce_backend_capabilities(raw)
        if caps is not None:
            return caps
        return _PLUGIN_DEFAULT_CAPABILITIES
    return _PLUGIN_DEFAULT_CAPABILITIES


def backend_capability_source(
    backend_name: str,
    backend_options: Mapping[str, object] | None = None,
) -> str:
    """Human-readable source for capability resolution diagnostics."""
    options = {} if backend_options is None else dict(backend_options)
    if backend_name in _BUILTIN_BACKENDS:
        return "builtin"
    if "." in backend_name:
        raw = _plugin_hook_value(
            backend_name,
            _PLUGIN_CAPABILITIES_HOOKS,
            options,
        )
        if _coerce_backend_capabilities(raw) is not None:
            return "plugin/hook"
    return "plugin/default"


def _option_type_name(value_type: type) -> str:
    if value_type is bool:
        return "bool"
    if value_type is int:
        return "int"
    if value_type is float:
        return "float"
    if value_type is str:
        return "str"
    return getattr(value_type, "__name__", str(value_type))


def _capabilities_to_payload(caps: BackendCapabilities) -> dict[str, object]:
    return {
        "reports_sync_loss": caps.reports_sync_loss,
        "preserves_token_advantages": caps.preserves_token_advantages,
        "supports_checkpoint_resume": caps.supports_checkpoint_resume,
        "resume_runtime_dependent": caps.resume_runtime_dependent,
        "supports_echo_shared_forward": caps.supports_echo_shared_forward,
    }


def _schema_to_payload(
    schema: Mapping[str, BackendOptionSpec],
) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for key, spec in schema.items():
        payload[key] = {
            "type": _option_type_name(spec.value_type),
            "default": spec.default,
            "choices": list(spec.choices) if spec.choices else None,
            "has_validator": spec.validator is not None,
        }
    return payload


def describe_backends_catalog() -> dict[str, object]:
    """Machine-readable catalog for built-in backends + plugin hook contract."""
    builtins: list[dict[str, object]] = []
    for name, definition in sorted(_BUILTIN_BACKENDS.items()):
        builtins.append(
            {
                "name": name,
                "dependency": {
                    "import": definition.dependency_import,
                    "hint": definition.dependency_hint,
                },
                "capabilities": _capabilities_to_payload(definition.capabilities),
                "option_schema": _schema_to_payload(definition.option_schema),
            }
        )
    return {
        "builtins": builtins,
        "plugin": {
            "dotted_path_supported": True,
            "default_capabilities": _capabilities_to_payload(
                _PLUGIN_DEFAULT_CAPABILITIES
            ),
            "capability_hooks": list(_PLUGIN_CAPABILITIES_HOOKS),
            "option_schema_hooks": list(_PLUGIN_OPTION_SCHEMA_HOOKS),
            "option_schema_format": (
                "mapping[str, BackendOptionSpec | "
                "{type|value_type, default, choices?, validator?}]"
            ),
        },
    }


def _coerce_option_value(backend: str, key: str, raw: object, spec: BackendOptionSpec) -> object:
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


def normalize_backend_options(
    backend: str,
    raw_options: Mapping[str, object] | None,
) -> dict[str, object]:
    """Normalize backend options for built-in backends; pass through plugins."""
    options = {} if raw_options is None else dict(raw_options)
    if not isinstance(options, dict):
        raise ValueError("backend_options must be a mapping.")

    if backend not in _BUILTIN_BACKENDS:
        if "." in backend:
            raw_schema = _plugin_hook_value(
                backend,
                _PLUGIN_OPTION_SCHEMA_HOOKS,
                options,
            )
            schema = _coerce_plugin_option_schema(backend, raw_schema)
            if not schema:
                return options
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
                normalized[key] = _coerce_option_value(backend, key, raw, spec)
            return normalized
        return options

    schema = _BUILTIN_BACKENDS[backend].option_schema
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
        normalized[key] = _coerce_option_value(backend, key, raw, spec)

    return normalized
