"""Constructors for built-in training backends."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from retrain.backends.options import normalize_option_schema, prime_rl_option_schema
from retrain.training.sft import effective_sft_loss_fn

if TYPE_CHECKING:
    from retrain.backends import TrainHelper
    from retrain.config import TrainConfig


class PrimeRLOptions(TypedDict):
    transport: str
    zmq_host: str
    zmq_port: int
    zmq_hwm: int
    strict_advantages: bool
    sync_wait_s: int
    sync_poll_s: float


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


def create_local(config: "TrainConfig") -> "TrainHelper":
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
    setattr(helper, "sft_loss_fn", effective_sft_loss_fn(config))
    return helper


def create_unsloth(config: "TrainConfig") -> "TrainHelper":
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
    setattr(helper, "sft_loss_fn", effective_sft_loss_fn(config))
    return helper


def create_tinker(config: "TrainConfig") -> "TrainHelper":
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
    setattr(helper, "sft_loss_fn", effective_sft_loss_fn(config))
    return helper


def _normalize_prime_rl_options(raw_options: Mapping[str, object]) -> PrimeRLOptions:
    options = normalize_option_schema("prime_rl", raw_options, prime_rl_option_schema())
    return {
        "transport": cast(str, options["transport"]),
        "zmq_host": cast(str, options["zmq_host"]),
        "zmq_port": cast(int, options["zmq_port"]),
        "zmq_hwm": cast(int, options["zmq_hwm"]),
        "strict_advantages": cast(bool, options["strict_advantages"]),
        "sync_wait_s": cast(int, options["sync_wait_s"]),
        "sync_poll_s": cast(float, options["sync_poll_s"]),
    }


def create_prime_rl(config: "TrainConfig") -> "TrainHelper":
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
