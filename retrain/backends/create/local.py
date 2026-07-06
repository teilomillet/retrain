"""Local backend constructor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from retrain.backends.create.values import backend_option_float, backend_option_int
from retrain.training.sft import effective_sft_loss_fn

if TYPE_CHECKING:
    from retrain.backends import TrainHelper
    from retrain.config import TrainConfig


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
        train_microbatch_size=backend_option_int(
            config.backend_options,
            "train_microbatch_size",
        ),
        train_sft_microbatch_token_budget=backend_option_int(
            config.backend_options,
            "train_sft_microbatch_token_budget",
        ),
        train_logprob_chunk_size=backend_option_int(
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
        gradient_checkpointing_skip_last_n=backend_option_int(
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
        train_save_on_cpu_min_numel=backend_option_int(
            config.backend_options,
            "train_save_on_cpu_min_numel",
        ),
        train_supervised_context_tokens=backend_option_int(
            config.backend_options,
            "train_supervised_context_tokens",
        ),
        train_unsloth_fused_ce=str(
            config.backend_options.get("train_unsloth_fused_ce", "off")
        ),
        train_unsloth_fused_ce_target_gb=backend_option_float(
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
        train_compile_selective_ce_min_tokens=backend_option_int(
            config.backend_options,
            "train_compile_selective_ce_min_tokens",
            128,
        ),
        lora_target_modules=str(config.backend_options.get("lora_target_modules", "")),
        lora_layers_to_transform=str(
            config.backend_options.get("lora_layers_to_transform", "")
        ),
        lora_layers_pattern=str(
            config.backend_options.get("lora_layers_pattern", "layers")
        ),
        lora_detach_input=bool(config.backend_options.get("lora_detach_input", False)),
        lora_fast_linear=bool(config.backend_options.get("lora_fast_linear", False)),
        lora_freeze_a=bool(config.backend_options.get("lora_freeze_a", False)),
        qwen35_gated_delta_kernel=str(
            config.backend_options.get("qwen35_gated_delta_kernel", "auto")
        ),
        sample_kv_quantization=str(
            config.backend_options.get("sample_kv_quantization", "off")
        ),
        sample_oscar_repo=str(config.backend_options.get("sample_oscar_repo", "")),
        sample_oscar_bits=backend_option_int(
            config.backend_options,
            "sample_oscar_bits",
            2,
        ),
        sample_oscar_quant_mode=str(
            config.backend_options.get("sample_oscar_quant_mode", "k-channel")
        ),
        sample_oscar_group_size=backend_option_int(
            config.backend_options,
            "sample_oscar_group_size",
            0,
        ),
        sample_oscar_kv_rotation=str(
            config.backend_options.get("sample_oscar_kv_rotation", "hadamard")
        ),
        sample_oscar_kv_norm=str(
            config.backend_options.get("sample_oscar_kv_norm", "1")
        ),
        sample_oscar_residual_block_size=backend_option_int(
            config.backend_options,
            "sample_oscar_residual_block_size",
            128,
        ),
        sample_oscar_attn_implementation=str(
            config.backend_options.get("sample_oscar_attn_implementation", "sdpa")
        ),
        trust_remote_code=bool(config.backend_options.get("trust_remote_code", False)),
    )
    setattr(helper, "sft_loss_fn", effective_sft_loss_fn(config))
    return helper
