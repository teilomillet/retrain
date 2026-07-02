"""Python helper for LocalBackend: PyTorch/PEFT training + pluggable inference.

This helper provides:
- PyTorch model with PEFT LoRA for training (gradient computation)
- Pluggable InferenceEngine for sampling (PyTorch fallback or server-based)
- Adapter save/load for weight synchronization

GPU split mode (multi-GPU): separate devices for inference and training.
- engine on first device (sampling)
- train_model on last device (gradient updates)
- checkpoint() syncs LoRA weights train -> engine

The LocalBackend in Mojo calls into this module via Python interop.
"""

import gc
import os
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from types import MethodType
from typing import cast

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from retrain.accelerators import (
    accelerator_status,
    apply_liger_kernel_if_available,
    from_pretrained_attention_kwargs,
    install_cudnn_causal_conv1d_shim,
    module_available,
)
from retrain.gemma4_text import (
    forward_hidden_states_and_lm_head,
    forward_logits,
    parse_lora_target_module_suffixes,
    resolve_lora_target_modules,
)
from retrain.inference_engine import create_engine


def _masked_mean(values, mask):
    denom = mask.float().sum().clamp(min=1)
    return (values * mask.float()).sum() / denom


def _config_value(config, key):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _infer_transformer_layer_count(model):
    config = getattr(model, "config", None)
    candidates = [config]
    for nested_key in ("text_config", "language_config", "llm_config"):
        nested = _config_value(config, nested_key)
        if nested is not None:
            candidates.append(nested)

    for candidate in candidates:
        for key in ("num_hidden_layers", "n_layer", "num_layers"):
            value = _config_value(candidate, key)
            if value is None:
                continue
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count > 0:
                return count
    return 0


class _FastLoRALinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, lora_a, lora_b, scaling):
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        compute_dtype = x_flat.dtype
        lora_a_compute = lora_a.to(compute_dtype)
        lora_b_compute = lora_b.to(compute_dtype)

        output = torch.matmul(x_flat, weight.to(compute_dtype).T)
        if bias is not None:
            output = output + bias.to(output.dtype)
        lora_hidden = torch.matmul(x_flat, lora_a_compute.T)
        output.addmm_(lora_hidden, lora_b_compute.T, alpha=float(scaling))

        ctx.save_for_backward(x, weight, lora_a, lora_b)
        ctx.scaling = float(scaling)
        return output.reshape(*original_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, lora_a, lora_b = ctx.saved_tensors
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        x_flat = x.reshape(-1, x.shape[-1])
        compute_dtype = x_flat.dtype
        lora_a_compute = lora_a.to(compute_dtype)
        lora_b_compute = lora_b.to(compute_dtype)
        scaling = ctx.scaling

        grad_x = None
        grad_lora_a = None
        grad_lora_b = None
        grad_lora_hidden = grad_flat.matmul(lora_b_compute)

        if ctx.needs_input_grad[0]:
            grad_x = grad_flat.matmul(weight.to(compute_dtype))
            grad_x = grad_x.addmm(grad_lora_hidden, lora_a_compute, alpha=scaling)
            grad_x = grad_x.reshape_as(x)
        if ctx.needs_input_grad[3]:
            grad_lora_a = grad_lora_hidden.T.matmul(x_flat)
            grad_lora_a = (grad_lora_a * scaling).to(lora_a.dtype)
        if ctx.needs_input_grad[4]:
            lora_hidden = x_flat.matmul(lora_a_compute.T)
            grad_lora_b = grad_flat.T.matmul(lora_hidden)
            grad_lora_b = (grad_lora_b * scaling).to(lora_b.dtype)

        return grad_x, None, None, grad_lora_a, grad_lora_b, None


def _mapping_value(container, key, default=None):
    if container is None:
        return default
    try:
        if key in container:
            return container[key]
    except TypeError:
        pass
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def _fast_lora_linear_forward(module, x, *args, **kwargs):
    original_forward = getattr(module, "_retrain_fast_lora_original_forward")
    if args or kwargs:
        return original_forward(x, *args, **kwargs)
    check_args = getattr(module, "_check_forward_args", None)
    if callable(check_args):
        check_args(x, *args, **kwargs)
    if getattr(module, "disable_adapters", False) or getattr(module, "merged", False):
        return original_forward(x, *args, **kwargs)
    active_adapters = list(getattr(module, "active_adapters", []) or [])
    if len(active_adapters) != 1:
        return original_forward(x, *args, **kwargs)
    adapter = active_adapters[0]
    lora_a_layers = getattr(module, "lora_A", {})
    lora_b_layers = getattr(module, "lora_B", {})
    if adapter not in lora_a_layers or adapter not in lora_b_layers:
        return original_forward(x, *args, **kwargs)
    use_dora = getattr(module, "use_dora", {})
    if bool(_mapping_value(use_dora, adapter, False)):
        return original_forward(x, *args, **kwargs)
    dropout = _mapping_value(getattr(module, "lora_dropout", {}), adapter)
    if dropout is None:
        return original_forward(x, *args, **kwargs)
    dropout_p = float(getattr(dropout, "p", 0.0))
    if dropout_p != 0.0:
        return original_forward(x, *args, **kwargs)
    base_layer = getattr(module, "base_layer", None)
    weight = getattr(base_layer, "weight", None)
    if not torch.is_tensor(weight):
        return original_forward(x, *args, **kwargs)
    bias = getattr(base_layer, "bias", None)
    lora_a = lora_a_layers[adapter].weight
    lora_b = lora_b_layers[adapter].weight
    scaling = _mapping_value(getattr(module, "scaling", {}), adapter)
    if scaling is None:
        return original_forward(x, *args, **kwargs)
    return _FastLoRALinearFunction.apply(x, weight, bias, lora_a, lora_b, scaling)


def _parse_lora_layers_to_transform(spec, layer_count=0):
    raw = str(spec or "").strip()
    if not raw or raw.lower() in {"all", "default"}:
        return None

    normalized = raw.replace(" ", "").lower()
    if normalized.startswith(("last:", "first:")):
        mode, raw_count = normalized.split(":", 1)
        try:
            count = int(raw_count)
        except ValueError:
            raise ValueError(
                f"Invalid lora_layers_to_transform={raw!r}: expected {mode}:N."
            ) from None
        if count <= 0:
            raise ValueError(
                f"Invalid lora_layers_to_transform={raw!r}: N must be > 0."
            )
        if layer_count <= 0:
            raise ValueError(
                f"Invalid lora_layers_to_transform={raw!r}: model layer count is unknown."
            )
        if count > layer_count:
            raise ValueError(
                f"Invalid lora_layers_to_transform={raw!r}: N exceeds {layer_count} layers."
            )
        if mode == "first":
            return list(range(count))
        return list(range(layer_count - count, layer_count))

    layers: list[int] = []
    for part in normalized.split(","):
        if not part:
            raise ValueError(
                f"Invalid lora_layers_to_transform={raw!r}: empty layer entry."
            )
        if "-" in part:
            bounds = part.split("-", 1)
            if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                raise ValueError(
                    f"Invalid lora_layers_to_transform={raw!r}: bad range {part!r}."
                )
            try:
                start, end = (int(bounds[0]), int(bounds[1]))
            except ValueError:
                raise ValueError(
                    f"Invalid lora_layers_to_transform={raw!r}: bad range {part!r}."
                ) from None
            if start > end:
                raise ValueError(
                    f"Invalid lora_layers_to_transform={raw!r}: range {part!r} is reversed."
                )
            layers.extend(range(start, end + 1))
        else:
            try:
                layers.append(int(part))
            except ValueError:
                raise ValueError(
                    f"Invalid lora_layers_to_transform={raw!r}: bad layer {part!r}."
                ) from None

    if not layers:
        raise ValueError(f"Invalid lora_layers_to_transform={raw!r}: no layers selected.")
    if any(layer < 0 for layer in layers):
        raise ValueError(
            f"Invalid lora_layers_to_transform={raw!r}: layer ids must be >= 0."
        )
    if len(set(layers)) != len(layers):
        raise ValueError(
            f"Invalid lora_layers_to_transform={raw!r}: duplicate layer id."
        )
    if layer_count > 0 and max(layers) >= layer_count:
        raise ValueError(
            f"Invalid lora_layers_to_transform={raw!r}: layer id exceeds {layer_count - 1}."
        )
    return layers


def _selected_linear_ce_logprobs_no_bias(hidden, weight, target_ids):
    logits = hidden @ weight.T
    return -F.cross_entropy(logits.float(), target_ids, reduction="none")


def _selected_linear_ce_logprobs_with_bias(hidden, weight, bias, target_ids):
    logits = (hidden @ weight.T) + bias
    return -F.cross_entropy(logits.float(), target_ids, reduction="none")


def _apply_static_range_quantization(x, scale, bits=8):
    scale = scale.to(x.dtype)
    max_value = 2 ** (bits - 1) - 1
    min_value = -max_value - 1
    calibrated = scale != 0
    safe_scale = torch.where(calibrated, scale, torch.ones_like(scale))
    x_q = (
        torch.clamp(
            torch.round(x / safe_scale),
            float(min_value),
            float(max_value),
        )
        * safe_scale
    )
    return torch.where(calibrated, x_q, x)


def _unpack_quantized_weight(weight, num_bits, original_width):
    if num_bits == 2:
        packed = weight.to(torch.uint8)
        v0 = (packed & 0x03).to(torch.int8) - 2
        v1 = ((packed >> 2) & 0x03).to(torch.int8) - 2
        v2 = ((packed >> 4) & 0x03).to(torch.int8) - 2
        v3 = (packed >> 6).to(torch.int8) - 2
        return torch.stack([v0, v1, v2, v3], dim=-1).reshape(
            *packed.shape[:-1],
            -1,
        )[..., :original_width]
    if num_bits == 4:
        packed = weight.to(torch.uint8)
        low = (packed & 0x0F).to(torch.int8) - 8
        high = (packed >> 4).to(torch.int8) - 8
        return torch.stack([low, high], dim=-1).reshape(
            *packed.shape[:-1],
            -1,
        )[..., :original_width]
    if num_bits == 8:
        return weight
    return None


def _packed_quantized_linear_target_logprobs(
    hidden,
    lm_head,
    target_ids,
    *,
    vocab_chunk_size=8192,
):
    """Exact target logprobs for packed Gemma QAT-style QuantizedLinear heads."""
    num_bits = getattr(lm_head, "num_bits", None)
    if num_bits not in (2, 4, 8):
        return None

    weight = getattr(lm_head, "weight", None)
    weight_scale = getattr(lm_head, "weight_scale", None)
    in_features = getattr(lm_head, "in_features", None)
    out_features = getattr(lm_head, "out_features", None)
    if (
        weight is None
        or weight_scale is None
        or in_features is None
        or out_features is None
        or int(hidden.shape[-1]) != int(in_features)
    ):
        return None

    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_target_ids = target_ids.reshape(-1).long()
    if flat_hidden.shape[0] != flat_target_ids.numel():
        return None
    if flat_target_ids.numel() == 0:
        return flat_hidden.new_empty(target_ids.shape)

    invalid_targets = (flat_target_ids < 0) | (flat_target_ids >= int(out_features))
    if bool(invalid_targets.any().item()):
        raise ValueError("target id is outside the quantized LM head vocabulary.")

    input_scale = getattr(lm_head, "input_activation_scale", None)
    if input_scale is not None:
        flat_hidden = _apply_static_range_quantization(flat_hidden, input_scale)

    bias = getattr(lm_head, "bias", None)
    output_scale = getattr(lm_head, "output_activation_scale", None)
    vocab_chunk_size = max(1, int(vocab_chunk_size))
    target_logits = flat_hidden.new_empty(flat_target_ids.shape, dtype=torch.float32)
    log_denominator = None

    for start in range(0, int(out_features), vocab_chunk_size):
        stop = min(start + vocab_chunk_size, int(out_features))
        int_weight = _unpack_quantized_weight(
            weight[start:stop],
            int(num_bits),
            int(in_features),
        )
        if int_weight is None:
            return None
        dequant_weight = int_weight.to(flat_hidden.dtype) * weight_scale[
            start:stop
        ].to(flat_hidden.dtype)
        chunk_bias = None if bias is None else bias[start:stop].to(flat_hidden.dtype)
        logits = F.linear(flat_hidden, dequant_weight, chunk_bias)
        if output_scale is not None:
            logits = _apply_static_range_quantization(logits, output_scale)
        logits = logits.float()

        chunk_denominator = torch.logsumexp(logits, dim=-1)
        if log_denominator is None:
            log_denominator = chunk_denominator
        else:
            log_denominator = torch.logaddexp(log_denominator, chunk_denominator)

        in_chunk = (flat_target_ids >= start) & (flat_target_ids < stop)
        if bool(in_chunk.any().item()):
            local_target_ids = flat_target_ids[in_chunk] - start
            selected_logits = logits[in_chunk].gather(
                1,
                local_target_ids.unsqueeze(1),
            ).squeeze(1)
            target_logits[in_chunk] = selected_logits

    assert log_denominator is not None
    return (target_logits - log_denominator).reshape_as(target_ids)


def _compute_policy_loss(
    old_logprobs,
    new_logprobs,
    adv,
    mask,
    clip_eps,
    clip_eps_high,
    policy_loss_mode="standard",
    kl_cov_percent=0.2,
    kl_cov_coef=1.0,
    clip_cov_ratio=0.0002,
    clip_cov_min=1.0,
    clip_cov_max=5.0,
):
    """Compute policy loss, optionally with covariance-aware entropy control.

    Args:
        old_logprobs: Log probability under rollout policy.
        new_logprobs: Log probability under current policy.
        adv: Per-token advantages.
        mask: Attention mask (1 for real tokens, 0 for padding).
        clip_eps: Lower clipping epsilon. 0 = no clipping.
        clip_eps_high: Upper clipping epsilon. 0 = use clip_eps (symmetric).
        policy_loss_mode: ``standard``, ``kl_cov``, or ``clip_cov``.

    Returns:
        ``(masked_loss, clip_fraction, cov_fraction, abs_kl)`` tuple.
    """
    logprob_delta = new_logprobs - old_logprobs
    ratio = torch.exp(logprob_delta)
    abs_kl = _masked_mean(logprob_delta.abs(), mask)

    def _standard_loss():
        if clip_eps > 0:
            eps_high = clip_eps_high if clip_eps_high > 0 else clip_eps
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + eps_high)
            surr1 = ratio * adv
            surr2 = clipped_ratio * adv
            per_token_loss = -torch.min(surr1, surr2)
            with torch.no_grad():
                clipped = (
                    (ratio < 1.0 - clip_eps) | (ratio > 1.0 + eps_high)
                ).float()
                frac = (clipped * mask).sum().item() / mask.sum().clamp(min=1).item()
        else:
            per_token_loss = -(ratio * adv)
            frac = 0.0
        return per_token_loss, frac

    if policy_loss_mode == "kl_cov":
        per_token_loss = -(ratio * adv)
        valid = mask > 0
        valid_count = int(valid.sum().item())
        cov_selected = torch.zeros_like(mask, dtype=torch.bool)
        if valid_count > 0 and kl_cov_percent > 0:
            with torch.no_grad():
                valid_adv = adv[valid]
                valid_logp = new_logprobs[valid]
                cov_values = (
                    (valid_adv - valid_adv.mean())
                    * (valid_logp - valid_logp.mean())
                )
                k_tokens = max(1, int(valid_count * min(kl_cov_percent, 100.0) / 100.0))
                selected_flat = torch.topk(cov_values, k_tokens, largest=True).indices
                valid_indices = torch.nonzero(valid.reshape(-1), as_tuple=True)[0]
                selected_indices = valid_indices[selected_flat]
                cov_selected = cov_selected.reshape(-1)
                cov_selected[selected_indices] = True
                cov_selected = cov_selected.reshape_as(mask)
        if cov_selected.any():
            per_token_loss = torch.where(
                cov_selected,
                per_token_loss + kl_cov_coef * logprob_delta.abs(),
                per_token_loss,
            )
        cov_fraction = (cov_selected.float() * mask.float()).sum() / mask.float().sum().clamp(min=1)
        masked_loss = (per_token_loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
        return masked_loss, 0.0, float(cov_fraction.detach().item()), float(abs_kl.detach().item())

    if policy_loss_mode == "clip_cov":
        cov_selected = torch.zeros_like(mask, dtype=torch.bool)
        eps_low = clip_eps if clip_eps > 0 else 1.0
        eps_high = clip_eps_high if clip_eps_high > 0 else eps_low
        pg_loss_unclipped = -(ratio * adv)
        pg_loss_clipped = -(
            torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
        )
        clip_by_origin = (pg_loss_clipped > pg_loss_unclipped) & (mask > 0)
        per_token_loss = torch.maximum(pg_loss_unclipped, pg_loss_clipped)
        with torch.no_grad():
            cov_all = (
                (adv - _masked_mean(adv, mask))
                * (new_logprobs - _masked_mean(new_logprobs, mask))
            )
            eligible = (
                (mask > 0)
                & ~clip_by_origin
                & (cov_all > clip_cov_min)
                & (cov_all < clip_cov_max)
            )
            eligible_idx = torch.nonzero(eligible)
            clip_num = max(int(float(clip_cov_ratio) * mask.sum().item()), 1)
            if len(eligible_idx) > 0:
                perm = torch.randperm(len(eligible_idx), device=eligible_idx.device)
                selected = eligible_idx[perm[: min(clip_num, len(eligible_idx))]]
                cov_selected[selected[:, 0], selected[:, 1]] = True
        per_token_loss = torch.where(
            cov_selected,
            torch.zeros_like(per_token_loss),
            per_token_loss,
        )
        cov_fraction = (cov_selected.float() * mask.float()).sum() / mask.float().sum().clamp(min=1)
        masked_loss = (per_token_loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
        return masked_loss, float(cov_fraction.detach().item()), float(cov_fraction.detach().item()), float(abs_kl.detach().item())

    if policy_loss_mode != "standard":
        raise ValueError(
            "policy_loss_mode must be 'standard', 'kl_cov', or 'clip_cov'."
        )

    per_token_loss, frac = _standard_loss()
    masked_loss = (per_token_loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
    return masked_loss, frac, 0.0, float(abs_kl.detach().item())


def _parse_device(device_str):
    """Convert a device spec like 'gpu:0' to a torch device string like 'cuda:0'."""
    device_str = device_str.strip()
    if device_str.startswith("gpu:"):
        return device_str.replace("gpu:", "cuda:")
    elif device_str == "cpu":
        return "cpu"
    else:
        return "cuda:0"


def _pad_to_width(tensor, width, value):
    """Right-pad a batch-major tensor to ``width`` columns."""
    if tensor.shape[1] >= width:
        return tensor
    pad = tensor.new_full((tensor.shape[0], width - tensor.shape[1]), value)
    return torch.cat([tensor, pad], dim=1)


def _is_cuda_device(device) -> bool:
    return isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()


def _timer_start(device):
    if _is_cuda_device(device):
        with torch.cuda.device(torch.device(device)):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        return ("cuda", device, event)
    return ("cpu", device, time.perf_counter())


def _timer_stop(start) -> float:
    kind, device, marker = start
    if kind == "cuda":
        with torch.cuda.device(torch.device(device)):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
        return marker.elapsed_time(end) / 1000.0
    return time.perf_counter() - marker


def _reset_cuda_peak(device) -> None:
    if _is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def _cuda_peak_metrics(prefix: str, device) -> dict[str, float]:
    if not _is_cuda_device(device):
        return {}
    torch_device = torch.device(device)
    return {
        f"{prefix}_peak_memory_allocated_mb": (
            torch.cuda.max_memory_allocated(torch_device) / (1024.0 * 1024.0)
        ),
        f"{prefix}_peak_memory_reserved_mb": (
            torch.cuda.max_memory_reserved(torch_device) / (1024.0 * 1024.0)
        ),
    }


class LocalTrainHelper:
    """Local GPU helper: pluggable inference engine + PyTorch/PEFT training."""

    def __init__(self, model_name, adapter_path, devices, lora_rank=32,
                 engine_type="pytorch", inference_url="",
                 lora_alpha=0, lora_dropout=0.0,
                 optim_beta1=0.9, optim_beta2=0.95, optim_eps=1e-8,
                 clip_eps=0.0, clip_eps_high=0.0,
                 policy_loss_mode="standard",
                 kl_cov_percent=0.2,
                 kl_cov_coef=1.0,
                 clip_cov_ratio=0.0002,
                 clip_cov_min=1.0,
                 clip_cov_max=5.0,
                 train_microbatch_size=0,
                 train_sft_microbatch_token_budget=0,
                 train_logprob_chunk_size=0,
                 liger_kernel=True,
                 liger_fused_linear_ce=True,
                 cuda_empty_cache=False,
                 sample_use_cache=True,
                 gradient_checkpointing=True,
                 gradient_checkpointing_use_reentrant="auto",
                 gradient_checkpointing_skip_last_n=0,
                 cudnn_causal_conv1d_shim=False,
                 attention_kernel="default",
                 prefix_caching=True,
                 train_selective_suffix_logits=False,
                 train_save_on_cpu=False,
                 train_save_on_cpu_pin_memory=True,
                 train_save_on_cpu_min_numel=0,
                 train_supervised_context_tokens=0,
                 train_unsloth_fused_ce="off",
                 train_unsloth_fused_ce_target_gb=0.0,
                 train_unsloth_fused_ce_torch_compile=True,
                 train_compile_selective_ce="off",
                 train_compile_selective_ce_min_tokens=128,
                 lora_target_modules="",
                 lora_layers_to_transform="",
                 lora_layers_pattern="layers",
                 lora_detach_input=False,
                 lora_fast_linear=False,
                 trust_remote_code=False):
        self.adapter_path = adapter_path
        self.model_name = model_name
        self.engine_type = engine_type
        self.trust_remote_code = bool(trust_remote_code)
        self.clip_eps = clip_eps
        self.clip_eps_high = clip_eps_high
        self._clip_fraction = 0.0
        self.policy_loss_mode = policy_loss_mode
        self.kl_cov_percent = float(kl_cov_percent)
        self.kl_cov_coef = float(kl_cov_coef)
        self.clip_cov_ratio = float(clip_cov_ratio)
        self.clip_cov_min = float(clip_cov_min)
        self.clip_cov_max = float(clip_cov_max)
        self._policy_cov_fraction = 0.0
        self._policy_abs_kl = 0.0
        self.train_microbatch_size = max(0, int(train_microbatch_size))
        self.train_sft_microbatch_token_budget = max(
            0,
            int(train_sft_microbatch_token_budget),
        )
        self.train_logprob_chunk_size = max(0, int(train_logprob_chunk_size))
        self.liger_kernel = bool(liger_kernel)
        self.liger_fused_linear_ce = bool(liger_fused_linear_ce)
        self.attention_kernel = str(attention_kernel or "default")
        self.cuda_empty_cache = bool(cuda_empty_cache)
        self.sample_use_cache = bool(sample_use_cache)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.gradient_checkpointing_use_reentrant = str(
            gradient_checkpointing_use_reentrant or "auto"
        ).lower()
        self.gradient_checkpointing_skip_last_n = max(
            0,
            int(gradient_checkpointing_skip_last_n),
        )
        self._gradient_checkpointing_layer_metrics = {
            "total": 0,
            "enabled": 0,
            "skipped_last_n": 0,
        }
        self.cudnn_causal_conv1d_shim = bool(cudnn_causal_conv1d_shim)
        self.prefix_caching = bool(prefix_caching)
        self.train_selective_suffix_logits = bool(train_selective_suffix_logits)
        self.train_save_on_cpu = bool(train_save_on_cpu)
        self.train_save_on_cpu_pin_memory = bool(train_save_on_cpu_pin_memory)
        self.train_save_on_cpu_min_numel = max(0, int(train_save_on_cpu_min_numel))
        self.train_supervised_context_tokens = max(
            0,
            int(train_supervised_context_tokens),
        )
        self.train_unsloth_fused_ce = self._normalize_unsloth_fused_ce_mode(
            train_unsloth_fused_ce
        )
        self.train_unsloth_fused_ce_target_gb = max(
            0.0,
            float(train_unsloth_fused_ce_target_gb),
        )
        self.train_unsloth_fused_ce_torch_compile = bool(
            train_unsloth_fused_ce_torch_compile
        )
        self.train_compile_selective_ce = self._normalize_compile_mode(
            train_compile_selective_ce,
            option_name="train_compile_selective_ce",
        )
        self.train_compile_selective_ce_min_tokens = max(
            0,
            int(train_compile_selective_ce_min_tokens),
        )
        self.lora_target_module_suffixes = parse_lora_target_module_suffixes(
            lora_target_modules
        )
        self.lora_layers_to_transform_spec = str(lora_layers_to_transform or "")
        self.lora_layers_pattern = str(lora_layers_pattern or "layers")
        self.lora_detach_input = bool(lora_detach_input)
        self.lora_fast_linear = bool(lora_fast_linear)
        self._lora_layers_to_transform: list[int] | None = None
        self._lora_detach_input_hook_handles = []
        self._lora_detach_input_hook_count = 0
        self._lora_fast_linear_patch_count = 0
        self._lora_model_metrics: dict[str, float | int | str] = {}
        self._last_sample_metrics: dict[str, float | int] = {}
        self._last_train_metrics: dict[str, float | int | str] = {}
        self._last_sync_metrics: dict[str, float | int] = {}
        self._last_context_crop_metrics: dict[str, float | int] = {}
        self._train_logits_to_keep_supported: bool | None = None
        self._selective_logprob_path_counts: dict[str, int] = {}
        self._loss_path_counts: dict[str, int] = {}
        self._unsloth_fused_ce_available: bool | None = None
        self._unsloth_fused_ce_unavailable_reason = ""
        self._unsloth_fused_ce_fallback_reason = ""
        self._last_unsloth_fused_ce_effective_target_gb = 0.0
        self._compiled_selective_ce_fallback_reason = ""
        self._compiled_selective_ce_available: bool | None = None

        # Parse all devices from comma-separated spec
        raw_devices = [d.strip() for d in devices.split(",") if d.strip()]
        parsed_devices = [_parse_device(d) for d in raw_devices]

        # Determine split mode: need >1 device and CUDA available
        cuda_devices = [d for d in parsed_devices if d.startswith("cuda")]
        self.split_mode = len(cuda_devices) > 1 and torch.cuda.is_available()

        # Non-PyTorch engines manage their own inference independently.
        # Server engines use a remote process; MAX engine uses its own in-process model.
        self._server_engine = engine_type in (
            "vllm",
            "sglang",
            "trtllm",
            "mlx",
            "openai",
        )
        self._external_engine = engine_type in (
            "max",
            "vllm",
            "sglang",
            "trtllm",
            "mlx",
            "openai",
        )

        if self._external_engine:
            # Server handles inference — all local GPUs for training
            device = parsed_devices[-1] if parsed_devices else "cuda:0"
            if device.startswith("cuda") and not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                device = "cpu"
            self.infer_device = device  # not used for sampling, but kept for compat
            self.train_device = device
            self.split_mode = False
        elif self.split_mode:
            self.infer_device = cuda_devices[0]   # first GPU for inference
            self.train_device = cuda_devices[-1]   # last GPU for training
        else:
            # Single-model mode: use first device (with CUDA fallback)
            device = parsed_devices[0] if parsed_devices else "cuda:0"
            if device.startswith("cuda") and not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                device = "cpu"
            self.infer_device = device
            self.train_device = device

        self.use_amp = self.train_device != "cpu"
        dtype = torch.bfloat16 if self.train_device != "cpu" else torch.float32
        self.amp_dtype = dtype

        self._accelerator_metrics = cast(
            dict[str, object],
            install_cudnn_causal_conv1d_shim(
                enabled=self.cudnn_causal_conv1d_shim,
            ),
        )
        self._accelerator_metrics.update(
            cast(
                dict[str, object],
                dict(accelerator_status()),
            )
        )
        self._accelerator_metrics.update(
            apply_liger_kernel_if_available(
                model_name,
                enabled=self.liger_kernel,
            )
        )

        # Create train model
        print(f"Loading train model: {model_name} on {self.train_device}...")
        self.train_model, peft_config = self._load_train_model(
            model_name,
            dtype,
            lora_rank,
            lora_alpha,
            lora_dropout,
        )
        self._configure_lora_detached_input()
        self._configure_lora_fast_linear()
        self._move_train_model_to_device()
        self._record_lora_model_metrics()
        if hasattr(self.train_model, "print_trainable_parameters"):
            self.train_model.print_trainable_parameters()

        self._configure_gradient_checkpointing()

        # Async pipeline state — initialized before first _sync_lora_weights call
        self._train_future = None       # Future from last submitted training
        self._pending_loss = 0.0        # Loss from last completed training
        self._weight_snapshot = None    # LoRA weight snapshot for safe cross-thread sync
        self._weights_dirty = False     # Track if weights changed since last server sync

        # Create inference engine
        if self._external_engine:
            # External engine (MAX in-process or server-based) — manages its own model
            self.engine = create_engine(
                engine_type=engine_type,
                model_name=model_name,
                device=self.infer_device,
                peft_config=peft_config,
                dtype=dtype,
                inference_url=inference_url,
                sample_use_cache=self.sample_use_cache,
                prefix_caching=self.prefix_caching,
                attention_kernel=self.attention_kernel,
                liger_kernel=self.liger_kernel,
            )
        elif self.split_mode:
            self._train_executor = ThreadPoolExecutor(max_workers=1)

            # PyTorch engine on separate inference device
            self.engine = create_engine(
                engine_type="pytorch",
                model_name=model_name,
                device=self.infer_device,
                peft_config=peft_config,
                dtype=dtype,
                sample_use_cache=self.sample_use_cache,
                prefix_caching=self.prefix_caching,
                attention_kernel=self.attention_kernel,
                liger_kernel=self.liger_kernel,
            )
            # Initial sync: copy LoRA weights from train to infer
            self._do_initial_sync()
        else:
            # Single-model mode: engine wraps the train model directly
            # We use a thin wrapper that points to train_model
            self.engine = create_engine(
                engine_type="pytorch",
                model_name=model_name,
                device=self.infer_device,
                peft_config=None,
                dtype=dtype,
                existing_model=self.train_model,
                sample_use_cache=self.sample_use_cache,
                prefix_caching=self.prefix_caching,
                attention_kernel=self.attention_kernel,
                liger_kernel=self.liger_kernel,
            )

        # Optimizer (only for train_model)
        self.optimizer = torch.optim.AdamW(
            self.train_model.parameters(),
            lr=4e-5,
            betas=(optim_beta1, optim_beta2),
            eps=optim_eps,
            weight_decay=0.0,
        )

        # BF16 autocast does not need loss scaling. Enabling GradScaler for
        # BF16 can silently skip every optimizer step after scaled-grad overflow.
        self.scaler = torch.amp.GradScaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        print(f"LocalTrainHelper ready (engine={engine_type}, split_mode={self.split_mode}).")

    def _autocast_context(self):
        device_type = self.train_device.split(":")[0]
        return torch.amp.autocast(
            device_type=device_type,
            enabled=self.use_amp,
            dtype=self.amp_dtype if self.use_amp else None,
        )

    def _build_peft_config(self, base_model, lora_rank, lora_alpha, lora_dropout):
        effective_alpha = lora_alpha if lora_alpha > 0 else lora_rank * 2
        layer_count = _infer_transformer_layer_count(base_model)
        selected_layers = _parse_lora_layers_to_transform(
            self.lora_layers_to_transform_spec,
            layer_count,
        )
        self._lora_layers_to_transform = selected_layers
        layer_kwargs = {}
        if selected_layers is not None:
            layer_kwargs = {
                "layers_to_transform": selected_layers,
                "layers_pattern": self.lora_layers_pattern,
            }
        target_module_suffixes = getattr(
            self,
            "lora_target_module_suffixes",
            parse_lora_target_module_suffixes(""),
        )
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=effective_alpha,
            lora_dropout=lora_dropout,
            target_modules=resolve_lora_target_modules(
                base_model,
                target_module_suffixes,
            ),
            **layer_kwargs,
        )

    def _record_lora_model_metrics(self) -> None:
        model = getattr(self, "train_model", None)
        named_parameters = getattr(model, "named_parameters", None)
        if not callable(named_parameters):
            self._lora_model_metrics = {}
            return
        lora_param_count = 0
        lora_tensor_count = 0
        trainable_param_count = 0
        for name, param in named_parameters():
            numel = int(param.numel())
            if getattr(param, "requires_grad", False):
                trainable_param_count += numel
            if "lora_" in name:
                lora_param_count += numel
                lora_tensor_count += 1
        selected_layers = self._lora_layers_to_transform
        target_module_suffixes = getattr(
            self,
            "lora_target_module_suffixes",
            parse_lora_target_module_suffixes(""),
        )
        self._lora_model_metrics = {
            "local_lora_layer_selection_enabled": int(selected_layers is not None),
            "local_lora_selected_layer_count": (
                0 if selected_layers is None else len(selected_layers)
            ),
            "local_lora_selected_layers": (
                "" if selected_layers is None else ",".join(map(str, selected_layers))
            ),
            "local_lora_layers_pattern": self.lora_layers_pattern,
            "local_lora_target_modules": ",".join(target_module_suffixes),
            "local_lora_target_module_count": len(target_module_suffixes),
            "local_lora_parameter_count": lora_param_count,
            "local_lora_parameter_tensor_count": lora_tensor_count,
            "local_lora_trainable_parameter_count": trainable_param_count,
            "local_lora_detach_input_enabled": int(self.lora_detach_input),
            "local_lora_detach_input_hook_count": self._lora_detach_input_hook_count,
            "local_lora_fast_linear_enabled": int(self.lora_fast_linear),
            "local_lora_fast_linear_patch_count": self._lora_fast_linear_patch_count,
        }

    def _load_train_model(self, model_name, dtype, lora_rank, lora_alpha, lora_dropout):
        model_kwargs = from_pretrained_attention_kwargs(self.attention_kernel)
        base_train = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
            **model_kwargs,
        )
        peft_config = self._build_peft_config(
            base_train,
            lora_rank,
            lora_alpha,
            lora_dropout,
        )
        return get_peft_model(base_train, peft_config), peft_config

    @staticmethod
    def _detach_first_tensor_input(_module, inputs):
        if not inputs:
            return inputs
        first = inputs[0]
        if torch.is_tensor(first):
            return (first.detach(), *inputs[1:])
        return inputs

    def _configure_lora_detached_input(self):
        self._lora_detach_input_hook_handles = []
        self._lora_detach_input_hook_count = 0
        if not self.lora_detach_input:
            return
        named_modules = getattr(self.train_model, "named_modules", None)
        if not callable(named_modules):
            return
        for name, module in named_modules():
            if ".lora_A." not in f".{name}.":
                continue
            if not torch.is_tensor(getattr(module, "weight", None)):
                continue
            register = getattr(module, "register_forward_pre_hook", None)
            if not callable(register):
                continue
            handle = register(self._detach_first_tensor_input)
            self._lora_detach_input_hook_handles.append(handle)
        self._lora_detach_input_hook_count = len(self._lora_detach_input_hook_handles)

    def _configure_lora_fast_linear(self):
        self._lora_fast_linear_patch_count = 0
        if not self.lora_fast_linear:
            return
        if self.lora_detach_input:
            print("lora_fast_linear disabled because lora_detach_input is enabled")
            return
        named_modules = getattr(self.train_model, "named_modules", None)
        if not callable(named_modules):
            return
        for _name, module in named_modules():
            if not hasattr(module, "base_layer"):
                continue
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            if hasattr(module, "_retrain_fast_lora_original_forward"):
                continue
            module._retrain_fast_lora_original_forward = module.forward
            module.forward = MethodType(_fast_lora_linear_forward, module)
            self._lora_fast_linear_patch_count += 1

    def _move_train_model_to_device(self):
        self.train_model.to(self.train_device)

    def _enable_gradient_checkpointing(self, model):
        mode = getattr(self, "gradient_checkpointing_use_reentrant", "auto")
        if mode in ("true", "false"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": mode == "true"}
            )
        else:
            model.gradient_checkpointing_enable()

    @staticmethod
    def _gradient_checkpointing_layers(model):
        modules = getattr(model, "modules", None)
        if not callable(modules):
            return []
        return [
            module
            for module in modules()
            if module is not model
            and hasattr(module, "gradient_checkpointing")
            and isinstance(getattr(module, "gradient_checkpointing"), bool)
        ]

    def _apply_gradient_checkpointing_layer_policy(self, model) -> None:
        layers = self._gradient_checkpointing_layers(model)
        skip_last_n = min(
            int(getattr(self, "gradient_checkpointing_skip_last_n", 0)),
            len(layers),
        )
        if skip_last_n:
            for module in layers[-skip_last_n:]:
                module.gradient_checkpointing = False
        enabled = sum(
            int(bool(getattr(module, "gradient_checkpointing", False)))
            for module in layers
        )
        self._gradient_checkpointing_layer_metrics = {
            "total": len(layers),
            "enabled": enabled,
            "skipped_last_n": skip_last_n,
        }

    def _configure_gradient_checkpointing(self):
        # Gradient checkpointing trades compute for VRAM; benchmarks must be
        # able to toggle it because it changes both memory fit and throughput.
        if self.gradient_checkpointing and hasattr(
            self.train_model,
            "gradient_checkpointing_enable",
        ):
            self._enable_gradient_checkpointing(self.train_model)
            self._apply_gradient_checkpointing_layer_policy(self.train_model)
        elif hasattr(self.train_model, "gradient_checkpointing_disable"):
            self.train_model.gradient_checkpointing_disable()
            self._gradient_checkpointing_layer_metrics = {
                "total": len(self._gradient_checkpointing_layers(self.train_model)),
                "enabled": 0,
                "skipped_last_n": 0,
            }

    def _liger_fused_linear_ce_loss(self):
        if not (
            getattr(self, "liger_fused_linear_ce", False)
            and module_available("liger_kernel")
        ):
            return None
        loss = getattr(self, "_liger_fused_ce_loss", None)
        if loss is not None:
            return loss
        try:
            from liger_kernel.transformers import (  # type: ignore[unresolved-import]
                LigerFusedLinearCrossEntropyLoss,
            )

            loss = LigerFusedLinearCrossEntropyLoss(reduction="none")
        except Exception:  # noqa: BLE001 - optional accelerator path.
            return None
        self._liger_fused_ce_loss = loss
        return loss

    def _do_initial_sync(self):
        """Copy LoRA weights from train_model to engine at init time.

        Used once during __init__ before any training has occurred (no snapshot
        exists yet). After this, all syncs go through _sync_lora_weights which
        reads from the weight snapshot.
        """
        train_state = dict(self.train_model.named_parameters())
        lora_dict = {}
        for name, param in train_state.items():
            if "lora_" in name:
                lora_dict[name] = param.data
        self.engine.sync_from_state_dict(lora_dict)

    def _sync_lora_weights(self):
        """Copy LoRA weights from snapshot to engine.

        In split mode: copies from _weight_snapshot (created after each completed
        training step) rather than live train_model weights. This is safe to call
        while training is running on GPU 1 — reads snapshot, not live weights.

        For external engines (MAX, server-based): saves adapter to disk and
        tells engine to reload.

        No-op when split_mode is False (engine shares train_model) or when
        no training has completed yet (_weight_snapshot is None).
        """
        sync_start = time.perf_counter()
        save_s = 0.0
        reload_s = 0.0
        copied = 0
        try:
            if self._external_engine:
                if self._weights_dirty and self._weight_snapshot is not None:
                    # Save adapter to disk, then tell engine to reload
                    save_dir = os.path.join(self.adapter_path, "_live_adapter")
                    os.makedirs(save_dir, exist_ok=True)
                    save_start = time.perf_counter()
                    self.train_model.save_pretrained(save_dir)
                    save_s = time.perf_counter() - save_start
                    reload_start = time.perf_counter()
                    self.engine.reload_weights(save_dir)
                    reload_s = time.perf_counter() - reload_start
                    self._weights_dirty = False
                    copied = 1
                return

            if not self.split_mode:
                return
            if self._weight_snapshot is None:
                return

            self.engine.sync_from_state_dict(self._weight_snapshot)
            copied = 1
        finally:
            self._last_sync_metrics = {
                "local_adapter_sync_s": time.perf_counter() - sync_start,
                "local_adapter_save_s": save_s,
                "local_adapter_reload_s": reload_s,
                "local_adapter_sync_copied": copied,
            }

    def checkpoint(self, name):
        """Prepare for sampling by syncing LoRA weights from snapshot -> engine.

        Non-blocking: if _train_future is done, collects loss into _pending_loss.
        Syncs from latest weight snapshot. Does NOT block-wait for in-flight
        training — this enables overlap between sample(N) and train(N-1).
        """
        if self._train_future is not None and self._train_future.done():
            self._pending_loss = self._train_future.result()
            self._train_future = None
        self._sync_lora_weights()
        if hasattr(self.engine, "clear_prefix_cache"):
            self.engine.clear_prefix_cache()

    @contextmanager
    def _shared_model_sampling_cache_context(self):
        toggled = False
        config = None
        previous_use_cache = None
        if (
            getattr(self, "gradient_checkpointing", False)
            and getattr(self, "sample_use_cache", True)
            and not getattr(self, "_external_engine", False)
            and not getattr(self, "split_mode", False)
        ):
            model = self.train_model
            config = getattr(model, "config", None)
            previous_use_cache = getattr(config, "use_cache", None)
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
                toggled = True
            if config is not None and previous_use_cache is not None:
                config.use_cache = True
        self._last_sample_gc_disabled_for_cache = int(toggled)
        try:
            yield
        finally:
            if toggled:
                model = self.train_model
                if hasattr(model, "gradient_checkpointing_enable"):
                    self._enable_gradient_checkpointing(model)
                if config is not None and previous_use_cache is not None:
                    config.use_cache = previous_use_cache

    def sample(self, prompt_ids_list, num_samples, max_tokens, temperature, top_p):
        """Generate completions with per-token logprobs.

        Delegates to the configured InferenceEngine, then converts
        SampleResult objects to (token_ids, logprobs) tuples for
        backward compatibility with the Mojo caller.

        Args:
            prompt_ids_list: List of lists of token IDs (one per prompt).
            num_samples: Number of completions per prompt.
            max_tokens: Maximum new tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            List of lists of (token_ids, logprobs) tuples.
        """
        sample_start = time.perf_counter()
        _reset_cuda_peak(getattr(self, "infer_device", getattr(self, "train_device", "cpu")))
        try:
            with self._shared_model_sampling_cache_context():
                engine_results = self.engine.generate(
                    prompt_ids_list, num_samples, max_tokens, temperature, top_p
                )
        finally:
            self._empty_cuda_cache_if_requested()
        self._record_sample_metrics(sample_start, prompt_ids_list, num_samples, engine_results)

        # Convert SampleResult -> tuples for Mojo interop
        results = []
        for group in engine_results:
            converted = []
            for sr in group:
                converted.append((sr.token_ids, sr.logprobs))
            results.append(converted)
        return results

    def sample_with_entropy(self, prompt_ids_list, num_samples, max_tokens,
                            temperature, top_p):
        """Generate completions with per-token logprobs and Shannon entropy.

        Like sample(), but requests per-token entropy from the engine.
        Returns 3-tuples (token_ids, logprobs, token_entropies).

        Args:
            prompt_ids_list: List of lists of token IDs (one per prompt).
            num_samples: Number of completions per prompt.
            max_tokens: Maximum new tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            List of lists of (token_ids, logprobs, token_entropies) tuples.
        """
        sample_start = time.perf_counter()
        _reset_cuda_peak(getattr(self, "infer_device", getattr(self, "train_device", "cpu")))
        try:
            with self._shared_model_sampling_cache_context():
                engine_results = self.engine.generate(
                    prompt_ids_list, num_samples, max_tokens, temperature, top_p,
                    compute_entropy=True,
                )
        finally:
            self._empty_cuda_cache_if_requested()
        self._record_sample_metrics(sample_start, prompt_ids_list, num_samples, engine_results)

        results = []
        for group in engine_results:
            converted = []
            for sr in group:
                converted.append((sr.token_ids, sr.logprobs, sr.token_entropies))
            results.append(converted)
        return results

    def runtime_metrics(self):
        """Expose optional engine-level runtime counters to the trainer."""
        metrics = {
            "local_gradient_checkpointing_enabled": int(
                getattr(self, "gradient_checkpointing", False)
            ),
            "local_gradient_checkpointing_use_reentrant": str(
                getattr(self, "gradient_checkpointing_use_reentrant", "auto")
            ),
            "local_gradient_checkpointing_skip_last_n": int(
                getattr(self, "gradient_checkpointing_skip_last_n", 0)
            ),
            "local_gradient_checkpointing_layer_count": int(
                getattr(self, "_gradient_checkpointing_layer_metrics", {}).get(
                    "total",
                    0,
                )
            ),
            "local_gradient_checkpointing_enabled_layers": int(
                getattr(self, "_gradient_checkpointing_layer_metrics", {}).get(
                    "enabled",
                    0,
                )
            ),
            "local_gradient_checkpointing_skipped_last_layers": int(
                getattr(self, "_gradient_checkpointing_layer_metrics", {}).get(
                    "skipped_last_n",
                    0,
                )
            ),
            "local_train_amp_dtype": str(
                getattr(self, "amp_dtype", "")
            ).replace("torch.", ""),
            "local_train_grad_scaler_enabled": int(
                bool(getattr(self, "scaler", None))
                and bool(getattr(self.scaler, "is_enabled", lambda: False)())
            ),
            "local_sample_use_cache": int(getattr(self, "sample_use_cache", True)),
            "local_train_microbatch_size": int(
                getattr(self, "train_microbatch_size", 0)
            ),
            "local_train_sft_microbatch_token_budget": int(
                getattr(self, "train_sft_microbatch_token_budget", 0)
            ),
            "local_train_selective_suffix_logits": int(
                getattr(self, "train_selective_suffix_logits", False)
            ),
            "local_train_save_on_cpu": int(
                getattr(self, "train_save_on_cpu", False)
            ),
            "local_train_save_on_cpu_pin_memory": int(
                getattr(self, "train_save_on_cpu_pin_memory", True)
            ),
            "local_train_save_on_cpu_min_numel": int(
                getattr(self, "train_save_on_cpu_min_numel", 0)
            ),
            "local_train_supervised_context_tokens": int(
                getattr(self, "train_supervised_context_tokens", 0)
            ),
            "local_train_logprob_chunk_size": int(
                getattr(self, "train_logprob_chunk_size", 0)
            ),
            "local_train_unsloth_fused_ce_mode": str(
                getattr(self, "train_unsloth_fused_ce", "off")
            ),
            "local_train_unsloth_fused_ce_target_gb": float(
                getattr(self, "train_unsloth_fused_ce_target_gb", 0.0)
            ),
            "local_train_unsloth_fused_ce_effective_target_gb": float(
                getattr(self, "_last_unsloth_fused_ce_effective_target_gb", 0.0)
            ),
            "local_train_unsloth_fused_ce_torch_compile": int(
                getattr(self, "train_unsloth_fused_ce_torch_compile", True)
            ),
            "local_train_compile_selective_ce_mode": str(
                getattr(self, "train_compile_selective_ce", "off")
            ),
            "local_train_compile_selective_ce_min_tokens": int(
                getattr(self, "train_compile_selective_ce_min_tokens", 0)
            ),
            "local_train_compile_selective_ce_available": (
                self._compiled_selective_ce_available_metric()
            ),
            "local_train_compile_selective_ce_fallback_reason": str(
                getattr(self, "_compiled_selective_ce_fallback_reason", "")
            ),
            "local_train_unsloth_fused_ce_available": (
                self._unsloth_fused_ce_available_metric()
            ),
            "local_train_unsloth_fused_ce_unavailable_reason": str(
                getattr(self, "_unsloth_fused_ce_unavailable_reason", "")
            ),
            "local_train_unsloth_fused_ce_fallback_reason": str(
                getattr(self, "_unsloth_fused_ce_fallback_reason", "")
            ),
            "local_train_unsloth_fused_ce_attempts": int(
                getattr(self, "_unsloth_fused_ce_attempts", 0)
            ),
            "local_train_unsloth_fused_ce_batches": int(
                getattr(self, "_loss_path_counts", {}).get(
                    "unsloth_fused_ce",
                    0,
                )
            ),
            "local_train_liger_fused_ce_batches": int(
                getattr(self, "_loss_path_counts", {}).get("liger_fused_ce", 0)
            ),
            "local_train_dense_logprob_batches": int(
                getattr(self, "_loss_path_counts", {}).get("dense_logprob", 0)
            ),
            "local_train_chunked_logprob_batches": int(
                getattr(self, "_loss_path_counts", {}).get("chunked_logprob", 0)
            ),
            "local_train_packed_quantized_lm_head_batches": int(
                getattr(self, "_loss_path_counts", {}).get(
                    "packed_quantized_lm_head",
                    0,
                )
            ),
            "local_train_logits_to_keep_supported": (
                self._logits_to_keep_supported_metric()
            ),
            "local_train_selective_suffix_logprob_batches": int(
                getattr(self, "_selective_logprob_path_counts", {}).get(
                    "suffix",
                    0,
                )
            ),
            "local_train_selective_sparse_suffix_skips": int(
                getattr(self, "_selective_logprob_path_counts", {}).get(
                    "sparse_suffix_skip",
                    0,
                )
            ),
            "local_train_selective_hidden_logprob_batches": int(
                getattr(self, "_selective_logprob_path_counts", {}).get(
                    "hidden",
                    0,
                )
            ),
            "local_train_selective_compiled_ce_batches": int(
                getattr(self, "_selective_logprob_path_counts", {}).get(
                    "compiled_ce",
                    0,
                )
            ),
            "local_train_selective_packed_quantized_lm_head_batches": int(
                getattr(self, "_selective_logprob_path_counts", {}).get(
                    "packed_quantized_lm_head",
                    0,
                )
            ),
            "local_train_selective_fallback_logprob_batches": int(
                getattr(self, "_selective_logprob_path_counts", {}).get(
                    "fallback",
                    0,
                )
            ),
            "local_liger_kernel_enabled": int(
                getattr(self, "liger_kernel", False)
            ),
            "local_liger_fused_linear_ce_enabled": int(
                getattr(self, "liger_fused_linear_ce", False)
            ),
            "local_prefix_caching": int(getattr(self, "prefix_caching", True)),
        }
        metrics.update(getattr(self, "_lora_model_metrics", {}))
        metrics.update(getattr(self, "_accelerator_metrics", {}))
        engine = getattr(self, "engine", None)
        if hasattr(engine, "performance_counters"):
            counters = engine.performance_counters()
            if isinstance(counters, dict):
                metrics.update(counters)
        metrics.update(getattr(self, "_last_context_crop_metrics", {}))
        metrics.update(getattr(self, "_last_sample_metrics", {}))
        metrics.update(getattr(self, "_last_train_metrics", {}))
        metrics.update(getattr(self, "_last_sync_metrics", {}))
        return metrics

    def _record_sample_metrics(self, start_s, prompt_ids_list, num_samples, engine_results):
        wall_s = time.perf_counter() - start_s
        prompt_tokens = sum(len(prompt) for prompt in prompt_ids_list) * int(num_samples)
        generated_tokens = sum(
            len(result.token_ids)
            for group in engine_results
            for result in group
        )
        metrics: dict[str, float | int] = {
            "local_sample_wall_s": wall_s,
            "local_sample_prompt_tokens": prompt_tokens,
            "local_sample_generated_tokens": generated_tokens,
            "local_sample_generation_tokens_per_s": (
                generated_tokens / wall_s if wall_s > 0 else 0.0
            ),
            "local_sample_gc_disabled_for_cache": int(
                getattr(self, "_last_sample_gc_disabled_for_cache", 0)
            ),
        }
        metrics.update(
            _cuda_peak_metrics(
                "local_sample_gpu",
                getattr(self, "infer_device", getattr(self, "train_device", "cpu")),
            )
        )
        self._last_sample_metrics = metrics

    def _snapshot_lora_weights_if_needed(self) -> float:
        if not (self.split_mode or self._external_engine):
            return 0.0
        timer = _timer_start(self.train_device)
        snapshot = {}
        for name, param in self.train_model.named_parameters():
            if "lora_" in name:
                snapshot[name] = param.data.clone()
        self._weight_snapshot = snapshot
        self._weights_dirty = True
        return _timer_stop(timer)

    def _record_train_metrics(
        self,
        *,
        kind: str,
        wall_start_s: float,
        forward_s: float,
        backward_s: float,
        optimizer_s: float,
        snapshot_s: float,
        microbatches: int,
        total_tokens: float,
        batch_size: int,
    ) -> None:
        wall_s = time.perf_counter() - wall_start_s
        metrics: dict[str, float | int | str] = {
            "local_train_kind": kind,
            "local_train_wall_s": wall_s,
            "local_train_forward_s": forward_s,
            "local_train_backward_s": backward_s,
            "local_train_optimizer_s": optimizer_s,
            "local_train_snapshot_s": snapshot_s,
            "local_train_microbatches": microbatches,
            "local_train_tokens": total_tokens,
            "local_train_batch_size": batch_size,
            "local_train_tokens_per_s": total_tokens / wall_s if wall_s > 0 else 0.0,
        }
        metrics.update(_cuda_peak_metrics("local_train_gpu", self.train_device))
        self._last_train_metrics = metrics

    def _empty_cuda_cache_if_requested(self):
        if self.cuda_empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _clear_inference_prefix_cache(self):
        engine = getattr(self, "engine", None)
        if engine is not None and hasattr(engine, "clear_prefix_cache"):
            engine.clear_prefix_cache()

    def shutdown(self) -> None:
        """Release model, optimizer, executor, and CUDA allocator state."""
        future = getattr(self, "_train_future", None)
        if future is not None:
            try:
                future.result()
            except Exception:  # noqa: BLE001 - cleanup should not mask failures.
                pass

        executor = getattr(self, "_train_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=True)

        engine = getattr(self, "engine", None)
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:  # noqa: BLE001 - best-effort cleanup.
                pass

        for name in (
            "engine",
            "train_model",
            "optimizer",
            "scaler",
            "_weight_snapshot",
            "_train_future",
        ):
            if hasattr(self, name):
                try:
                    delattr(self, name)
                except Exception:  # noqa: BLE001 - best-effort cleanup.
                    pass

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001 - CUDA may be recovering from OOM.
                pass
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:  # noqa: BLE001 - cleanup must not raise.
                pass

    def _record_selective_logprob_path(self, path: str) -> None:
        counts = getattr(self, "_selective_logprob_path_counts", None)
        if not isinstance(counts, dict):
            counts = {}
        counts[path] = int(counts.get(path, 0)) + 1
        self._selective_logprob_path_counts = counts

    def _record_loss_path(self, path: str) -> None:
        counts = getattr(self, "_loss_path_counts", None)
        if not isinstance(counts, dict):
            counts = {}
        counts[path] = int(counts.get(path, 0)) + 1
        self._loss_path_counts = counts

    @staticmethod
    def _normalize_compile_mode(raw, *, option_name: str) -> str:
        if isinstance(raw, bool):
            return "auto" if raw else "off"
        text = str(raw or "off").strip().lower()
        aliases = {
            "0": "off",
            "false": "off",
            "no": "off",
            "none": "off",
            "disabled": "off",
            "1": "auto",
            "true": "auto",
            "yes": "auto",
            "on": "auto",
            "required": "require",
        }
        text = aliases.get(text, text)
        if text not in {"off", "auto", "require"}:
            raise ValueError(f"{option_name} must be 'off', 'auto', or 'require'.")
        return text

    @staticmethod
    def _normalize_unsloth_fused_ce_mode(raw) -> str:
        if isinstance(raw, bool):
            return "require" if raw else "off"
        text = str(raw or "off").strip().lower()
        aliases = {
            "0": "off",
            "false": "off",
            "no": "off",
            "none": "off",
            "disabled": "off",
            "1": "require",
            "true": "require",
            "yes": "require",
            "on": "require",
            "required": "require",
        }
        text = aliases.get(text, text)
        if text not in {"off", "auto", "require"}:
            raise ValueError(
                "train_unsloth_fused_ce must be 'off', 'auto', or 'require'."
            )
        return text

    def _logits_to_keep_supported_metric(self) -> int:
        supported = getattr(self, "_train_logits_to_keep_supported", None)
        if supported is None:
            return -1
        return int(bool(supported))

    def _unsloth_fused_ce_available_metric(self) -> int:
        available = getattr(self, "_unsloth_fused_ce_available", None)
        if available is None:
            return -1
        return int(bool(available))

    def _compiled_selective_ce_available_metric(self) -> int:
        available = getattr(self, "_compiled_selective_ce_available", None)
        if available is None:
            return -1
        return int(bool(available))

    def _reject_compiled_selective_ce(self, reason: str):
        self._compiled_selective_ce_fallback_reason = reason
        if getattr(self, "train_compile_selective_ce", "off") == "require":
            raise RuntimeError(
                "train_compile_selective_ce=require but the compiled selected "
                f"CE path cannot be used: {reason}"
            )
        return None

    def _compiled_selective_ce_logprobs(self, selected_hidden, lm_head, target_ids):
        mode = getattr(self, "train_compile_selective_ce", "off")
        if mode == "off":
            return None
        if int(selected_hidden.shape[0]) < int(
            getattr(self, "train_compile_selective_ce_min_tokens", 0)
        ):
            return self._reject_compiled_selective_ce("below_min_tokens")
        if selected_hidden.device.type != "cuda":
            return self._reject_compiled_selective_ce("non_cuda")
        if not hasattr(torch, "compile"):
            self._compiled_selective_ce_available = False
            return self._reject_compiled_selective_ce("torch_compile_unavailable")

        if type(lm_head) is not torch.nn.Linear:
            return self._reject_compiled_selective_ce("lm_head_not_plain_linear")
        weight = getattr(lm_head, "weight", None)
        if weight is None:
            return self._reject_compiled_selective_ce("lm_head_weight_unavailable")
        bias = getattr(lm_head, "bias", None)

        try:
            if bias is None:
                compiled = getattr(
                    self,
                    "_compiled_selective_ce_no_bias",
                    None,
                )
                if compiled is None:
                    compiled = torch.compile(
                        _selected_linear_ce_logprobs_no_bias,
                        mode="reduce-overhead",
                        fullgraph=True,
                    )
                    self._compiled_selective_ce_no_bias = compiled
                selected_logprobs = compiled(selected_hidden, weight, target_ids)
            else:
                compiled = getattr(
                    self,
                    "_compiled_selective_ce_with_bias",
                    None,
                )
                if compiled is None:
                    compiled = torch.compile(
                        _selected_linear_ce_logprobs_with_bias,
                        mode="reduce-overhead",
                        fullgraph=True,
                    )
                    self._compiled_selective_ce_with_bias = compiled
                selected_logprobs = compiled(
                    selected_hidden,
                    weight,
                    bias,
                    target_ids,
                )
        except Exception as exc:  # noqa: BLE001 - optional compiler path.
            self._compiled_selective_ce_available = False
            return self._reject_compiled_selective_ce(type(exc).__name__)

        self._compiled_selective_ce_available = True
        self._compiled_selective_ce_fallback_reason = ""
        return selected_logprobs

    def _unsloth_fused_ce_loss(self):
        if bool(getattr(self, "_unsloth_fused_ce_runtime_disabled", False)):
            return None
        cached = getattr(self, "_unsloth_fused_ce_loss_fn", None)
        if cached is not None:
            return cached
        try:
            try:
                import unsloth  # type: ignore[import-not-found]  # noqa: F401
            except Exception:
                pass
            from unsloth_zoo.loss_utils import (  # type: ignore[import-not-found]
                HAS_CUT_CROSS_ENTROPY,
            )
            from unsloth_zoo.loss_utils import (  # type: ignore[import-not-found]
                unsloth_fused_ce_loss,
            )
        except Exception as exc:  # noqa: BLE001 - optional accelerator path.
            self._unsloth_fused_ce_available = False
            self._unsloth_fused_ce_unavailable_reason = type(exc).__name__
            return None
        if not HAS_CUT_CROSS_ENTROPY:
            self._unsloth_fused_ce_available = False
            self._unsloth_fused_ce_unavailable_reason = "cut_cross_entropy_unavailable"
            return None
        self._unsloth_fused_ce_available = True
        self._unsloth_fused_ce_unavailable_reason = ""
        self._unsloth_fused_ce_loss_fn = unsloth_fused_ce_loss
        return unsloth_fused_ce_loss

    def _effective_unsloth_fused_ce_target_gb(self) -> float:
        configured = float(getattr(self, "train_unsloth_fused_ce_target_gb", 0.0))
        if configured > 0:
            return configured
        device = str(getattr(self, "train_device", ""))
        if not device.startswith("cuda") or not torch.cuda.is_available():
            return 0.0
        try:
            total_gb = torch.cuda.get_device_properties(device).total_memory / (
                1024**3
            )
        except Exception:  # noqa: BLE001 - diagnostic/tuning fallback only.
            return 0.0
        if total_gb <= 16:
            return 0.25
        if total_gb <= 24:
            return 0.5
        return 0.0

    @staticmethod
    def _is_cuda_oom_exception(exc: BaseException) -> bool:
        message = str(exc).lower()
        return "cuda" in message and "out of memory" in message

    def _disable_unsloth_fused_ce_after_runtime_failure(
        self,
        exc: BaseException,
    ) -> str:
        reason = f"runtime_{type(exc).__name__}"
        self._unsloth_fused_ce_available = False
        self._unsloth_fused_ce_unavailable_reason = reason
        self._unsloth_fused_ce_fallback_reason = reason
        self._unsloth_fused_ce_runtime_disabled = True
        try:
            torch._dynamo.reset()
        except Exception:  # noqa: BLE001 - best-effort compiler cleanup.
            pass
        return reason

    @staticmethod
    def _constant_positive_weight(weights, target_mask):
        selected = weights[target_mask]
        if selected.numel() == 0:
            return None
        first = selected[:1]
        if torch.allclose(selected, first.expand_as(selected)):
            return first.squeeze()
        return None

    def _maybe_compute_unsloth_fused_sft_loss(
        self,
        input_ids,
        attention_mask,
        weights,
        target_mask,
        token_count,
    ):
        mode = getattr(self, "train_unsloth_fused_ce", "off")
        if mode == "off":
            return None

        def reject(reason: str):
            self._unsloth_fused_ce_fallback_reason = reason
            if mode == "require":
                raise RuntimeError(
                    "train_unsloth_fused_ce=require but the fused CE path "
                    f"cannot be used: {reason}"
                )
            return None

        if target_mask is None or not bool(target_mask.any().item()):
            return reject("no_supervised_tokens")

        supervised_tokens = target_mask.float().sum()
        shifted_tokens = attention_mask[:, 1:].float().sum().clamp(min=1)
        if (
            int(target_mask.shape[1]) > 2048
            and float((supervised_tokens / shifted_tokens).detach().item()) < 0.25
        ):
            return reject("sparse_supervised_tokens")

        weight_scale = self._constant_positive_weight(weights, target_mask)
        if weight_scale is None:
            return reject("non_constant_token_weights")
        if bool(getattr(self, "train_save_on_cpu", False)):
            return reject("saved_tensor_hooks_incompatible")

        loss_fn = self._unsloth_fused_ce_loss()
        if loss_fn is None:
            reason = getattr(
                self,
                "_unsloth_fused_ce_unavailable_reason",
                "unavailable",
            )
            return reject(reason)
        self._unsloth_fused_ce_attempts = (
            int(getattr(self, "_unsloth_fused_ce_attempts", 0)) + 1
        )

        try:
            hidden_and_head = forward_hidden_states_and_lm_head(
                self.train_model,
                input_ids,
                attention_mask,
            )
        except Exception as exc:  # noqa: BLE001 - optional accelerator path.
            if self._is_cuda_oom_exception(exc):
                raise
            reason = self._disable_unsloth_fused_ce_after_runtime_failure(exc)
            return reject(reason)
        if hidden_and_head is None:
            return reject("hidden_states_unavailable")

        hidden_states, lm_head = hidden_and_head
        weight = getattr(lm_head, "weight", None)
        if weight is None:
            return reject("lm_head_weight_unavailable")
        bias = getattr(lm_head, "bias", None)

        labels = input_ids.clone()
        ignore_index = -100
        labels[:, 0] = ignore_index
        labels[:, 1:] = torch.where(
            target_mask,
            labels[:, 1:],
            torch.full_like(labels[:, 1:], ignore_index),
        )
        label_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        label_mask[:, 1:] = target_mask
        target_gb = self._effective_unsloth_fused_ce_target_gb()
        self._last_unsloth_fused_ce_effective_target_gb = target_gb
        try:
            loss = loss_fn(
                None,
                hidden_states,
                weight,
                bias,
                labels,
                mask=label_mask,
                n_items=token_count,
                target_gb=target_gb if target_gb > 0 else None,
                torch_compile=bool(
                    getattr(self, "train_unsloth_fused_ce_torch_compile", True)
                ),
                shift_labels=True,
                ignore_index=ignore_index,
            )
        except Exception as exc:  # noqa: BLE001 - optional accelerator path.
            if self._is_cuda_oom_exception(exc):
                raise
            reason = self._disable_unsloth_fused_ce_after_runtime_failure(exc)
            return reject(reason)
        self._unsloth_fused_ce_fallback_reason = ""
        self._record_loss_path("unsloth_fused_ce")
        return loss * weight_scale.to(loss.device)

    def _supports_train_logits_to_keep(self, input_ids, attention_mask) -> bool:
        cached = getattr(self, "_train_logits_to_keep_supported", None)
        if cached is not None:
            return bool(cached)

        seq_len = int(input_ids.shape[1])
        probe_len = min(seq_len, 4)
        if probe_len < 2:
            self._train_logits_to_keep_supported = False
            return False

        logits_to_keep = min(2, probe_len - 1)
        probe_ids = input_ids[:1, -probe_len:]
        probe_mask = (
            attention_mask[:1, -probe_len:] if attention_mask is not None else None
        )
        was_training = bool(getattr(self.train_model, "training", False))
        try:
            self.train_model.eval()
            with torch.no_grad():
                outputs = self.train_model(
                    input_ids=probe_ids,
                    attention_mask=probe_mask,
                    use_cache=False,
                    logits_to_keep=logits_to_keep,
                )
        except TypeError:
            supported = False
        except Exception:  # noqa: BLE001 - fall back to the hidden-state path.
            supported = False
        else:
            logits = getattr(outputs, "logits", None)
            supported = (
                logits is not None
                and len(getattr(logits, "shape", ())) >= 2
                and int(logits.shape[1]) == logits_to_keep
            )
        finally:
            self.train_model.train(was_training)

        self._train_logits_to_keep_supported = supported
        return supported

    def _selective_suffix_token_logprobs(self, input_ids, attention_mask, target_mask):
        if not getattr(self, "train_selective_suffix_logits", False):
            return None
        if target_mask is None or not bool(target_mask.any().item()):
            return None
        if int(getattr(self, "train_logprob_chunk_size", 0)) > 0:
            return None

        selected = torch.nonzero(target_mask, as_tuple=False)
        if selected.numel() == 0:
            return None

        seq_len = int(input_ids.shape[1])
        min_target_pos = int(selected[:, 1].min().item())
        logits_to_keep = seq_len - min_target_pos
        if logits_to_keep <= 1 or logits_to_keep >= seq_len:
            return None
        selected_tokens = int(selected.shape[0])
        suffix_target_slots = max(1, logits_to_keep - 1)
        # Long ECHO/tool traces often supervise a small number of tokens inside
        # a large suffix. In that shape, logits_to_keep would still materialize
        # [long_suffix, vocab]; the hidden path keeps the same transformer
        # forward and computes the LM head only for selected target positions.
        if (
            suffix_target_slots > 2048
            and selected_tokens / suffix_target_slots < 0.25
        ):
            self._record_selective_logprob_path("sparse_suffix_skip")
            return None
        if not self._supports_train_logits_to_keep(input_ids, attention_mask):
            return None

        try:
            outputs = self.train_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=logits_to_keep,
            )
        except TypeError:
            return None

        logits = getattr(outputs, "logits", None)
        if logits is None:
            return None
        if int(logits.shape[1]) != logits_to_keep:
            self._train_logits_to_keep_supported = False
            return None

        logits_start = seq_len - logits_to_keep
        target_suffix = input_ids[:, logits_start + 1 :]
        usable_logits = logits[:, : target_suffix.shape[1], :]
        suffix_logprobs = F.log_softmax(usable_logits.float(), dim=-1)
        suffix_logprobs = suffix_logprobs.gather(
            2,
            target_suffix.unsqueeze(2),
        ).squeeze(2)
        full = suffix_logprobs.new_zeros((input_ids.shape[0], seq_len - 1))
        full[:, logits_start : logits_start + suffix_logprobs.shape[1]] = (
            suffix_logprobs
        )
        self._record_selective_logprob_path("suffix")
        return full

    def _selective_hidden_token_logprobs(self, input_ids, attention_mask, target_mask):
        if target_mask is None or not bool(target_mask.any().item()):
            return None

        selected = torch.nonzero(target_mask, as_tuple=False)
        if selected.numel() == 0:
            return None

        hidden_and_head = forward_hidden_states_and_lm_head(
            self.train_model,
            input_ids,
            attention_mask,
        )
        if hidden_and_head is None:
            return None

        hidden_states, lm_head = hidden_and_head
        shifted_len = int(input_ids.shape[1]) - 1
        if int(hidden_states.shape[1]) < shifted_len:
            return None

        target_ids = input_ids[selected[:, 0], selected[:, 1] + 1]
        selected_hidden = hidden_states[selected[:, 0], selected[:, 1], :]
        chunk_size = int(getattr(self, "train_logprob_chunk_size", 0))
        if chunk_size <= 0:
            chunk_size = 256

        selected_logprobs = self._compiled_selective_ce_logprobs(
            selected_hidden,
            lm_head,
            target_ids,
        )
        used_compiled_ce = selected_logprobs is not None
        used_packed_quantized = False
        if selected_logprobs is None:
            selected_logprobs = _packed_quantized_linear_target_logprobs(
                selected_hidden,
                lm_head,
                target_ids,
            )
            used_packed_quantized = selected_logprobs is not None
            if selected_logprobs is None:
                chunks = []
                for start in range(0, selected_hidden.shape[0], chunk_size):
                    stop = min(start + chunk_size, selected_hidden.shape[0])
                    logits = lm_head(selected_hidden[start:stop])
                    # cross_entropy is exactly LogSoftmax + NLLLoss for class-index
                    # targets, and avoids materializing the full selected-token
                    # log-probability matrix before gathering one class per row.
                    chunks.append(
                        -F.cross_entropy(
                            logits.float(),
                            target_ids[start:stop],
                            reduction="none",
                        )
                    )
                if not chunks:
                    return None
                selected_logprobs = torch.cat(chunks, dim=0)
        if selected_logprobs.numel() == 0:
            return None
        full = selected_logprobs.new_zeros((input_ids.shape[0], shifted_len))
        full[selected[:, 0], selected[:, 1]] = selected_logprobs
        self._record_selective_logprob_path("hidden")
        if used_compiled_ce:
            self._record_selective_logprob_path("compiled_ce")
        if used_packed_quantized:
            self._record_selective_logprob_path("packed_quantized_lm_head")
        return full

    def _shifted_token_logprobs(self, input_ids, attention_mask, target_mask=None):
        """Compute next-token logprobs, optionally chunking the LM head."""
        selective = self._selective_suffix_token_logprobs(
            input_ids,
            attention_mask,
            target_mask,
        )
        if selective is not None:
            return selective

        if (
            getattr(self, "train_selective_suffix_logits", False)
            and target_mask is not None
        ):
            selective = self._selective_hidden_token_logprobs(
                input_ids,
                attention_mask,
                target_mask,
            )
            if selective is not None:
                return selective
            self._record_selective_logprob_path("fallback")

        liger_loss = self._liger_fused_linear_ce_loss()
        if liger_loss is not None:
            hidden_and_head = forward_hidden_states_and_lm_head(
                self.train_model,
                input_ids,
                attention_mask,
            )
            if hidden_and_head is not None:
                hidden_states, lm_head = hidden_and_head
                shifted_hidden = hidden_states[:, :-1, :]
                target_ids = input_ids[:, 1:]
                flat_hidden = shifted_hidden.reshape(-1, shifted_hidden.shape[-1])
                flat_target_ids = target_ids.reshape(-1)
                packed_logprobs = _packed_quantized_linear_target_logprobs(
                    flat_hidden,
                    lm_head,
                    flat_target_ids,
                )
                if packed_logprobs is not None:
                    self._record_loss_path("packed_quantized_lm_head")
                    return packed_logprobs.reshape_as(target_ids)
                weight = getattr(lm_head, "weight")
                try:
                    nll = liger_loss(weight, flat_hidden, flat_target_ids)
                except TypeError:
                    nll = liger_loss(flat_hidden, weight, flat_target_ids)
                self._record_loss_path("liger_fused_ce")
                return -nll.reshape_as(target_ids)

        chunk_size = int(getattr(self, "train_logprob_chunk_size", 0))
        if chunk_size <= 0:
            logits = forward_logits(self.train_model, input_ids, attention_mask)[:, :-1]
            new_logprobs = F.log_softmax(logits.float(), dim=-1)
            target_ids = input_ids[:, 1:]
            self._record_loss_path("dense_logprob")
            return new_logprobs.gather(2, target_ids.unsqueeze(2)).squeeze(2)

        hidden_and_head = forward_hidden_states_and_lm_head(
            self.train_model,
            input_ids,
            attention_mask,
        )
        if hidden_and_head is None:
            logits = forward_logits(self.train_model, input_ids, attention_mask)[:, :-1]
            new_logprobs = F.log_softmax(logits.float(), dim=-1)
            target_ids = input_ids[:, 1:]
            self._record_loss_path("dense_logprob")
            return new_logprobs.gather(2, target_ids.unsqueeze(2)).squeeze(2)

        hidden_states, lm_head = hidden_and_head
        shifted_hidden = hidden_states[:, :-1, :]
        target_ids = input_ids[:, 1:]
        packed_logprobs = _packed_quantized_linear_target_logprobs(
            shifted_hidden,
            lm_head,
            target_ids,
        )
        if packed_logprobs is not None:
            self._record_loss_path("packed_quantized_lm_head")
            return packed_logprobs

        chunks = []
        for start in range(0, shifted_hidden.shape[1], chunk_size):
            stop = min(start + chunk_size, shifted_hidden.shape[1])
            logits = lm_head(shifted_hidden[:, start:stop, :])
            logprobs = F.log_softmax(logits.float(), dim=-1)
            chunks.append(
                logprobs.gather(
                    2,
                    target_ids[:, start:stop].unsqueeze(2),
                ).squeeze(2)
            )
        if not chunks:
            return shifted_hidden.new_empty((shifted_hidden.shape[0], 0))
        self._record_loss_path("chunked_logprob")
        return torch.cat(chunks, dim=1)

    @contextmanager
    def _saved_tensors_context(self):
        if (
            getattr(self, "train_save_on_cpu", False)
            and self.train_device.startswith("cuda")
            and torch.cuda.is_available()
        ):
            min_numel = int(getattr(self, "train_save_on_cpu_min_numel", 0))
            if min_numel > 0:
                pin_memory = bool(getattr(self, "train_save_on_cpu_pin_memory", True))

                def pack(tensor):
                    if not tensor.is_cuda or tensor.numel() < min_numel:
                        return tensor
                    if not pin_memory:
                        return tensor.device, tensor.to("cpu")
                    cpu_tensor = torch.empty_like(
                        tensor,
                        device="cpu",
                        pin_memory=True,
                    )
                    cpu_tensor.copy_(tensor, non_blocking=True)
                    return tensor.device, cpu_tensor

                def unpack(packed):
                    if isinstance(packed, tuple) and len(packed) == 2:
                        device, cpu_tensor = packed
                        return cpu_tensor.to(device, non_blocking=pin_memory)
                    return packed

                with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                    yield
                return

            with torch.autograd.graph.save_on_cpu(
                pin_memory=getattr(self, "train_save_on_cpu_pin_memory", True)
            ):
                yield
        else:
            yield

    def _compute_train_loss(self, input_ids, old_logprobs, advantages, attention_mask):
        """Compute masked policy loss for one already-padded microbatch."""
        with self._autocast_context():
            old_lp = old_logprobs[:, 1:]  # [N, max_len-1]
            adv = advantages[:, 1:]       # [N, max_len-1]
            mask = attention_mask[:, 1:]   # [N, max_len-1] — exclude padding
            loss_mask = mask
            target_mask = None
            if getattr(self, "train_selective_suffix_logits", False):
                target_mask = (mask > 0) & (adv != 0)
                loss_mask = target_mask.to(mask.dtype)

            new_logprobs = self._shifted_token_logprobs(
                input_ids,
                attention_mask,
                target_mask=target_mask,
            )

            masked_loss, clip_frac, cov_frac, abs_kl = _compute_policy_loss(
                old_lp,
                new_logprobs,
                adv,
                loss_mask,
                self.clip_eps,
                self.clip_eps_high,
                getattr(self, "policy_loss_mode", "standard"),
                getattr(self, "kl_cov_percent", 0.2),
                getattr(self, "kl_cov_coef", 1.0),
                getattr(self, "clip_cov_ratio", 0.0002),
                getattr(self, "clip_cov_min", 1.0),
                getattr(self, "clip_cov_max", 5.0),
            )
            token_count = loss_mask.sum().clamp(min=1)

        return masked_loss, token_count, clip_frac, cov_frac, abs_kl

    def _compute_sft_loss(self, input_ids, advantages, attention_mask):
        """Compute weighted next-token cross-entropy for SFT/ECHO datums."""
        with self._autocast_context():
            weights = torch.clamp(advantages[:, 1:], min=0.0)
            weights = weights * attention_mask[:, 1:].float()
            sft_target_mask = weights > 0
            target_mask = None
            if getattr(self, "train_selective_suffix_logits", False):
                target_mask = sft_target_mask

            fused_loss = self._maybe_compute_unsloth_fused_sft_loss(
                input_ids,
                attention_mask,
                weights,
                sft_target_mask,
                sft_target_mask.float().sum().clamp(min=1),
            )
            if fused_loss is not None:
                token_count = sft_target_mask.float().sum().clamp(min=1)
                return fused_loss, token_count

            new_logprobs = self._shifted_token_logprobs(
                input_ids,
                attention_mask,
                target_mask=target_mask,
            )

            token_mask = (weights > 0).float()
            token_count = token_mask.sum().clamp(min=1)
            loss = (-new_logprobs * weights).sum() / token_count

        return loss, token_count

    def _do_train_impl(self, input_ids, old_logprobs, advantages, attention_mask):
        """Execute training forward/backward/step on pre-prepared tensors.

        After the optimizer step, clones LoRA params into _weight_snapshot
        for safe cross-thread syncing. Runs on background thread in split mode.

        Returns:
            Scalar loss value.
        """
        self.train_model.train()
        self.optimizer.zero_grad()
        wall_start = time.perf_counter()
        _reset_cuda_peak(self.train_device)
        forward_s = 0.0
        backward_s = 0.0
        optimizer_s = 0.0
        snapshot_s = 0.0
        microbatches = 0

        try:
            batch_size = int(input_ids.shape[0])
            microbatch_size = self.train_microbatch_size or batch_size
            if getattr(self, "train_selective_suffix_logits", False):
                total_tokens = (
                    ((attention_mask[:, 1:] > 0) & (advantages[:, 1:] != 0))
                    .sum()
                    .clamp(min=1)
                )
            else:
                total_tokens = attention_mask[:, 1:].sum().clamp(min=1)
            total_tokens_value = float(total_tokens.item())
            loss_sum = 0.0
            clip_count = 0.0
            cov_count = 0.0
            abs_kl_sum = 0.0

            for start in range(0, batch_size, microbatch_size):
                stop = min(start + microbatch_size, batch_size)
                microbatches += 1
                timer = _timer_start(self.train_device)
                with self._saved_tensors_context():
                    masked_loss, token_count, clip_frac, cov_frac, abs_kl = (
                        self._compute_train_loss(
                            input_ids[start:stop],
                            old_logprobs[start:stop],
                            advantages[start:stop],
                            attention_mask[start:stop],
                        )
                    )
                    forward_s += _timer_stop(timer)
                    token_count_value = float(token_count.item())
                    scaled_loss = masked_loss * (token_count / total_tokens)
                    timer = _timer_start(self.train_device)
                    self.scaler.scale(scaled_loss).backward()
                    backward_s += _timer_stop(timer)
                loss_sum += float(masked_loss.detach().item()) * token_count_value
                clip_count += clip_frac * token_count_value
                cov_count += cov_frac * token_count_value
                abs_kl_sum += abs_kl * token_count_value

            timer = _timer_start(self.train_device)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_s += _timer_stop(timer)

            self._clip_fraction = clip_count / total_tokens_value
            self._policy_cov_fraction = cov_count / total_tokens_value
            self._policy_abs_kl = abs_kl_sum / total_tokens_value
            loss_val = loss_sum / total_tokens_value

            # Snapshot LoRA weights for safe cross-thread sync
            snapshot_s = self._snapshot_lora_weights_if_needed()
            self._record_train_metrics(
                kind="rl",
                wall_start_s=wall_start,
                forward_s=forward_s,
                backward_s=backward_s,
                optimizer_s=optimizer_s,
                snapshot_s=snapshot_s,
                microbatches=microbatches,
                total_tokens=total_tokens_value,
                batch_size=batch_size,
            )

            return loss_val
        finally:
            self.optimizer.zero_grad()
            self._empty_cuda_cache_if_requested()

    def _do_sft_impl(self, input_ids, advantages, attention_mask):
        """Execute weighted cross-entropy SFT/ECHO update synchronously."""
        self.train_model.train()
        self.optimizer.zero_grad()
        wall_start = time.perf_counter()
        _reset_cuda_peak(self.train_device)
        forward_s = 0.0
        backward_s = 0.0
        optimizer_s = 0.0
        snapshot_s = 0.0
        microbatches = 0

        try:
            batch_size = int(input_ids.shape[0])
            microbatch_size = self.train_microbatch_size or batch_size
            weights = torch.clamp(advantages[:, 1:], min=0.0)
            total_tokens = (weights * attention_mask[:, 1:].float() > 0).sum().clamp(min=1)
            total_tokens_value = float(total_tokens.item())
            loss_sum = 0.0

            for start in range(0, batch_size, microbatch_size):
                stop = min(start + microbatch_size, batch_size)
                microbatches += 1
                retried_fused_ce_runtime_failure = False
                while True:
                    fused_ce_attempts_before = int(
                        getattr(self, "_unsloth_fused_ce_attempts", 0)
                    )
                    loss_counts = getattr(self, "_loss_path_counts", {})
                    fused_ce_batches_before = int(
                        loss_counts.get("unsloth_fused_ce", 0)
                    )
                    timer = _timer_start(self.train_device)
                    try:
                        with self._saved_tensors_context():
                            masked_loss, token_count = self._compute_sft_loss(
                                input_ids[start:stop],
                                advantages[start:stop],
                                attention_mask[start:stop],
                            )
                            forward_s += _timer_stop(timer)
                            token_count_value = float(token_count.item())
                            scaled_loss = masked_loss * (token_count / total_tokens)
                            timer = _timer_start(self.train_device)
                            self.scaler.scale(scaled_loss).backward()
                            backward_s += _timer_stop(timer)
                    except Exception as exc:
                        fused_ce_attempted = (
                            int(getattr(self, "_unsloth_fused_ce_attempts", 0))
                            > fused_ce_attempts_before
                        )
                        if (
                            self._is_cuda_oom_exception(exc)
                            or not fused_ce_attempted
                            or getattr(self, "train_unsloth_fused_ce", "off") != "auto"
                            or retried_fused_ce_runtime_failure
                        ):
                            raise
                        loss_counts = getattr(self, "_loss_path_counts", {})
                        if (
                            int(loss_counts.get("unsloth_fused_ce", 0))
                            > fused_ce_batches_before
                        ):
                            loss_counts["unsloth_fused_ce"] = fused_ce_batches_before
                        self._disable_unsloth_fused_ce_after_runtime_failure(exc)
                        self.optimizer.zero_grad()
                        retried_fused_ce_runtime_failure = True
                        continue
                    break
                loss_sum += float(masked_loss.detach().item()) * token_count_value

            timer = _timer_start(self.train_device)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_s += _timer_stop(timer)
            loss_val = loss_sum / total_tokens_value

            snapshot_s = self._snapshot_lora_weights_if_needed()
            self._record_train_metrics(
                kind="sft",
                wall_start_s=wall_start,
                forward_s=forward_s,
                backward_s=backward_s,
                optimizer_s=optimizer_s,
                snapshot_s=snapshot_s,
                microbatches=microbatches,
                total_tokens=total_tokens_value,
                batch_size=batch_size,
            )

            return loss_val
        finally:
            self.optimizer.zero_grad()
            self._empty_cuda_cache_if_requested()

    @staticmethod
    def _sft_microbatch_ranges(lengths, microbatch_size, token_budget=0):
        """Return microbatch [start, stop) ranges under count/token limits."""
        batch_size = len(lengths)
        if batch_size <= 0:
            return []
        max_count = int(microbatch_size or batch_size)
        max_count = max(1, min(max_count, batch_size))
        token_budget = max(0, int(token_budget or 0))

        ranges = []
        start = 0
        while start < batch_size:
            stop = start
            max_len = 0
            while stop < batch_size and stop - start < max_count:
                candidate_max = max(max_len, int(lengths[stop]))
                candidate_count = stop - start + 1
                candidate_padded = candidate_count * candidate_max
                if (
                    token_budget > 0
                    and stop > start
                    and candidate_padded > token_budget
                ):
                    break
                max_len = candidate_max
                stop += 1
            if stop == start:
                stop += 1
            ranges.append((start, stop))
            start = stop
        return ranges

    @staticmethod
    def _microbatch_padding_stats(lengths, microbatch_size, token_budget=0):
        """Return global and microbatch-local padded token counts."""
        if not lengths:
            return 0, 0
        batch_size = len(lengths)
        global_padded = batch_size * max(lengths)
        microbatch_padded = 0
        for start, stop in LocalTrainHelper._sft_microbatch_ranges(
            lengths,
            microbatch_size,
            token_budget,
        ):
            chunk = lengths[start:stop]
            if chunk:
                microbatch_padded += len(chunk) * max(chunk)
        return global_padded, microbatch_padded

    def _pad_sft_microbatch(self, all_tokens, all_advantages, start, stop):
        token_tensors = [
            torch.tensor(t, dtype=torch.long) for t in all_tokens[start:stop]
        ]
        adv_tensors = [
            torch.tensor(a, dtype=torch.float32) for a in all_advantages[start:stop]
        ]

        input_ids = pad_sequence(
            token_tensors,
            batch_first=True,
            padding_value=0,
        ).to(self.train_device)
        advantages = pad_sequence(
            adv_tensors,
            batch_first=True,
            padding_value=0.0,
        ).to(self.train_device)

        lengths = torch.tensor(
            [len(t) for t in all_tokens[start:stop]],
            device=self.train_device,
        )
        max_len = input_ids.shape[1]
        attention_mask = (
            torch.arange(max_len, device=self.train_device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        return input_ids, advantages, attention_mask

    def _do_sft_sequence_impl(self, all_tokens, all_advantages):
        """Execute one SFT update while padding only each microbatch.

        This preserves the logical batch and optimizer-step semantics of
        ``_do_sft_impl`` but avoids allocating ``[batch, global_max_len]`` on the
        training device before microbatching. It targets long-context SFT rows
        where a single outlier otherwise forces every row to the same width.
        """
        self.train_model.train()
        self.optimizer.zero_grad()
        wall_start = time.perf_counter()
        _reset_cuda_peak(self.train_device)
        forward_s = 0.0
        backward_s = 0.0
        optimizer_s = 0.0
        snapshot_s = 0.0
        microbatches = 0
        batch_size = len(all_tokens)
        microbatch_size = self.train_microbatch_size or batch_size
        token_budget = getattr(self, "train_sft_microbatch_token_budget", 0)
        lengths = [len(row) for row in all_tokens]
        global_padded, microbatch_padded = self._microbatch_padding_stats(
            lengths,
            microbatch_size,
            token_budget,
        )
        microbatch_ranges = self._sft_microbatch_ranges(
            lengths,
            microbatch_size,
            token_budget,
        )

        try:
            total_tokens_value = float(
                sum(
                    1
                    for row in all_advantages
                    for value in row[1:]
                    if value > 0.0
                )
                or 1
            )
            loss_sum = 0.0

            for start, stop in microbatch_ranges:
                microbatches += 1
                input_ids, advantages, attention_mask = self._pad_sft_microbatch(
                    all_tokens,
                    all_advantages,
                    start,
                    stop,
                )
                retried_fused_ce_runtime_failure = False
                while True:
                    fused_ce_attempts_before = int(
                        getattr(self, "_unsloth_fused_ce_attempts", 0)
                    )
                    loss_counts = getattr(self, "_loss_path_counts", {})
                    fused_ce_batches_before = int(
                        loss_counts.get("unsloth_fused_ce", 0)
                    )
                    timer = _timer_start(self.train_device)
                    try:
                        with self._saved_tensors_context():
                            masked_loss, token_count = self._compute_sft_loss(
                                input_ids,
                                advantages,
                                attention_mask,
                            )
                            forward_s += _timer_stop(timer)
                            token_count_value = float(token_count.item())
                            scaled_loss = masked_loss * (
                                token_count / total_tokens_value
                            )
                            timer = _timer_start(self.train_device)
                            self.scaler.scale(scaled_loss).backward()
                            backward_s += _timer_stop(timer)
                    except Exception as exc:
                        fused_ce_attempted = (
                            int(getattr(self, "_unsloth_fused_ce_attempts", 0))
                            > fused_ce_attempts_before
                        )
                        if (
                            self._is_cuda_oom_exception(exc)
                            or not fused_ce_attempted
                            or getattr(self, "train_unsloth_fused_ce", "off") != "auto"
                            or retried_fused_ce_runtime_failure
                        ):
                            raise
                        loss_counts = getattr(self, "_loss_path_counts", {})
                        if (
                            int(loss_counts.get("unsloth_fused_ce", 0))
                            > fused_ce_batches_before
                        ):
                            loss_counts["unsloth_fused_ce"] = fused_ce_batches_before
                        self._disable_unsloth_fused_ce_after_runtime_failure(exc)
                        self.optimizer.zero_grad()
                        retried_fused_ce_runtime_failure = True
                        continue
                    break
                loss_sum += float(masked_loss.detach().item()) * token_count_value

            timer = _timer_start(self.train_device)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_s += _timer_stop(timer)
            loss_val = loss_sum / total_tokens_value

            snapshot_s = self._snapshot_lora_weights_if_needed()
            self._record_train_metrics(
                kind="sft",
                wall_start_s=wall_start,
                forward_s=forward_s,
                backward_s=backward_s,
                optimizer_s=optimizer_s,
                snapshot_s=snapshot_s,
                microbatches=microbatches,
                total_tokens=total_tokens_value,
                batch_size=batch_size,
            )
            metrics = getattr(self, "_last_train_metrics", {})
            if isinstance(metrics, dict):
                avoided = max(0, global_padded - microbatch_padded)
                metrics.update(
                    {
                        "local_train_microbatch_local_padding": 1,
                        "local_train_global_padded_tokens": global_padded,
                        "local_train_microbatch_padded_tokens": microbatch_padded,
                        "local_train_padding_tokens_avoided": avoided,
                        "local_train_padding_avoidance_fraction": (
                            avoided / global_padded if global_padded else 0.0
                        ),
                        "local_train_sft_microbatch_token_budget": int(token_budget),
                    }
                )
                if microbatches:
                    metrics["local_train_sft_avg_microbatch_examples"] = (
                        batch_size / microbatches
                    )
            return loss_val
        finally:
            self.optimizer.zero_grad()
            self._empty_cuda_cache_if_requested()

    def _do_hybrid_mask_impl(
        self,
        input_ids,
        old_logprobs,
        advantages,
        attention_mask,
        echo_advantages,
        echo_full_observation_counts,
        echo_loss_fn,
    ):
        """Execute RL + ECHO on the same rollout rows in one optimizer step."""
        self.train_model.train()
        self.optimizer.zero_grad()
        wall_start = time.perf_counter()
        _reset_cuda_peak(self.train_device)
        forward_s = 0.0
        backward_s = 0.0
        optimizer_s = 0.0
        snapshot_s = 0.0
        microbatches = 0

        try:
            batch_size = int(input_ids.shape[0])
            microbatch_size = self.train_microbatch_size or batch_size
            if getattr(self, "train_selective_suffix_logits", False):
                total_tokens = (
                    ((attention_mask[:, 1:] > 0) & (advantages[:, 1:] != 0))
                    .sum()
                    .clamp(min=1)
                )
            else:
                total_tokens = attention_mask[:, 1:].sum().clamp(min=1)
            total_tokens_value = float(total_tokens.item())
            loss_sum = 0.0
            clip_count = 0.0
            cov_count = 0.0
            abs_kl_sum = 0.0
            echo_loss_sum = 0.0

            for start in range(0, batch_size, microbatch_size):
                stop = min(start + microbatch_size, batch_size)
                microbatches += 1
                mb_input_ids = input_ids[start:stop]
                mb_old_logprobs = old_logprobs[start:stop]
                mb_advantages = advantages[start:stop]
                mb_attention_mask = attention_mask[start:stop]
                mb_echo_advantages = echo_advantages[start:stop]
                mb_echo_counts = echo_full_observation_counts[start:stop]

                timer = _timer_start(self.train_device)
                with self._saved_tensors_context():
                    with self._autocast_context():
                        old_lp = mb_old_logprobs[:, 1:]
                        adv = mb_advantages[:, 1:]
                        mask = mb_attention_mask[:, 1:]
                        echo_weights = torch.clamp(
                            mb_echo_advantages[:, 1:],
                            min=0.0,
                        )
                        echo_weights = echo_weights * mask.float()
                        loss_mask = mask
                        target_mask = None
                        if getattr(self, "train_selective_suffix_logits", False):
                            target_mask = (mask > 0) & (
                                (adv != 0) | (echo_weights > 0.0)
                            )
                            loss_mask = ((mask > 0) & (adv != 0)).to(mask.dtype)
                        token_logprobs = self._shifted_token_logprobs(
                            mb_input_ids,
                            mb_attention_mask,
                            target_mask=target_mask,
                        )
                        masked_loss, clip_frac, cov_frac, abs_kl = (
                            _compute_policy_loss(
                                old_lp,
                                token_logprobs,
                                adv,
                                loss_mask,
                                self.clip_eps,
                                self.clip_eps_high,
                                getattr(self, "policy_loss_mode", "standard"),
                                getattr(self, "kl_cov_percent", 0.2),
                                getattr(self, "kl_cov_coef", 1.0),
                                getattr(self, "clip_cov_ratio", 0.0002),
                                getattr(self, "clip_cov_min", 1.0),
                                getattr(self, "clip_cov_max", 5.0),
                            )
                        )
                        token_count = loss_mask.sum().clamp(min=1)
                        token_count_value = float(token_count.item())
                        scaled_loss = masked_loss * (token_count / total_tokens)

                        if echo_loss_fn != "cross_entropy":
                            raise ValueError(
                                "echo_loss_fn must be 'cross_entropy' for "
                                "paper-faithful ECHO."
                            )
                        echo_selected = (echo_weights > 0.0).float()
                        if echo_selected.sum() > 0:
                            denom = mb_echo_counts.float().clamp(min=1e-3).unsqueeze(1)
                            echo_loss = (
                                (-token_logprobs * echo_weights) / denom
                            ).sum()
                        else:
                            echo_loss = token_logprobs.sum() * 0.0

                        scaled_loss = scaled_loss + echo_loss / max(batch_size, 1)
                    forward_s += _timer_stop(timer)

                    timer = _timer_start(self.train_device)
                    self.scaler.scale(scaled_loss).backward()
                    backward_s += _timer_stop(timer)
                loss_sum += float(masked_loss.detach().item()) * token_count_value
                clip_count += clip_frac * token_count_value
                cov_count += cov_frac * token_count_value
                abs_kl_sum += abs_kl * token_count_value
                echo_loss_sum += float(echo_loss.detach().item())

            timer = _timer_start(self.train_device)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_s += _timer_stop(timer)

            self._clip_fraction = clip_count / total_tokens_value
            self._policy_cov_fraction = cov_count / total_tokens_value
            self._policy_abs_kl = abs_kl_sum / total_tokens_value
            loss_val = loss_sum / total_tokens_value
            echo_loss_val = echo_loss_sum / max(batch_size, 1)

            snapshot_s = self._snapshot_lora_weights_if_needed()
            self._record_train_metrics(
                kind="hybrid_echo",
                wall_start_s=wall_start,
                forward_s=forward_s,
                backward_s=backward_s,
                optimizer_s=optimizer_s,
                snapshot_s=snapshot_s,
                microbatches=microbatches,
                total_tokens=total_tokens_value,
                batch_size=batch_size,
            )

            return loss_val, echo_loss_val
        finally:
            self.optimizer.zero_grad()
            self._empty_cuda_cache_if_requested()

    @staticmethod
    def _first_supervised_token_index(*rows) -> int | None:
        earliest: int | None = None
        for row in rows:
            if row is None:
                continue
            for idx, value in enumerate(row[1:], start=1):
                if float(value) != 0.0:
                    earliest = idx if earliest is None else min(earliest, idx)
                    break
        return earliest

    def _maybe_crop_supervised_context(
        self,
        all_tokens,
        all_logprobs=None,
        all_advantages=None,
        echo_advantages=None,
    ):
        context_tokens = int(getattr(self, "train_supervised_context_tokens", 0))
        enabled = (
            context_tokens > 0
            and getattr(self, "train_selective_suffix_logits", False)
        )
        original_lengths = [len(row) for row in all_tokens]
        if not enabled or not original_lengths:
            self._last_context_crop_metrics = {
                "local_train_context_rows_cropped": 0,
                "local_train_context_tokens_removed": 0,
                "local_train_context_original_max_tokens": max(original_lengths, default=0),
                "local_train_context_cropped_max_tokens": max(original_lengths, default=0),
            }
            return all_tokens, all_logprobs, all_advantages, echo_advantages

        cropped_tokens = []
        cropped_logprobs = [] if all_logprobs is not None else None
        cropped_advantages = [] if all_advantages is not None else None
        cropped_echo = [] if echo_advantages is not None else None
        rows_cropped = 0
        tokens_removed = 0

        for idx, tokens in enumerate(all_tokens):
            logprobs = all_logprobs[idx] if all_logprobs is not None else None
            advantages = all_advantages[idx] if all_advantages is not None else None
            echo = echo_advantages[idx] if echo_advantages is not None else None
            earliest = self._first_supervised_token_index(advantages, echo)
            start = 0
            if earliest is not None:
                start = max(0, int(earliest) - context_tokens)
                if start >= earliest:
                    start = max(0, int(earliest) - 1)
            if start > 0:
                rows_cropped += 1
                tokens_removed += start
            cropped_tokens.append(tokens[start:])
            if cropped_logprobs is not None:
                assert logprobs is not None
                cropped_logprobs.append(logprobs[start:])
            if cropped_advantages is not None:
                assert advantages is not None
                cropped_advantages.append(advantages[start:])
            if cropped_echo is not None:
                assert echo is not None
                cropped_echo.append(echo[start:])

        cropped_lengths = [len(row) for row in cropped_tokens]
        self._last_context_crop_metrics = {
            "local_train_context_rows_cropped": rows_cropped,
            "local_train_context_tokens_removed": tokens_removed,
            "local_train_context_original_max_tokens": max(original_lengths, default=0),
            "local_train_context_cropped_max_tokens": max(cropped_lengths, default=0),
        }
        return cropped_tokens, cropped_logprobs, cropped_advantages, cropped_echo

    def train_step(self, all_tokens, all_logprobs, all_advantages, lr, weight_decay):
        """Run one training step with importance sampling loss.

        Split mode (async): waits for any previous training future (back pressure),
        prepares tensors on train_device, submits _do_train_impl to the executor,
        and returns _pending_loss (loss from the previously completed step).
        Step 0 returns 0.0 since no training has completed yet.

        Single mode: unchanged synchronous path returning current step's loss.

        Args:
            all_tokens: List of full token sequences (prompt + completion).
            all_logprobs: List of per-token logprobs (0-padded for prompt).
            all_advantages: List of per-token advantages (0-padded for prompt).
            lr: Learning rate.
            weight_decay: Weight decay coefficient.

        Returns:
            Mean loss (from previous step in split mode, current step otherwise).
        """
        n = len(all_tokens)
        if n == 0:
            return 0.0

        all_tokens, all_logprobs, all_advantages, _ = (
            self._maybe_crop_supervised_context(
                all_tokens,
                all_logprobs,
                all_advantages,
            )
        )

        self._clear_inference_prefix_cache()

        # Update optimizer hyperparameters
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = weight_decay

        # Pad all sequences into a single batch
        token_tensors = [torch.tensor(t, dtype=torch.long) for t in all_tokens]
        lp_tensors = [torch.tensor(lp, dtype=torch.float32) for lp in all_logprobs]
        adv_tensors = [torch.tensor(a, dtype=torch.float32) for a in all_advantages]

        # pad_sequence pads to max length in batch (batch_first=True -> [N, max_len])
        input_ids = pad_sequence(token_tensors, batch_first=True, padding_value=0).to(self.train_device)
        old_logprobs = pad_sequence(lp_tensors, batch_first=True, padding_value=0.0).to(self.train_device)
        advantages = pad_sequence(adv_tensors, batch_first=True, padding_value=0.0).to(self.train_device)

        # Build attention mask: 1 where real tokens, 0 where padding
        lengths = torch.tensor([len(t) for t in all_tokens], device=self.train_device)
        max_len = input_ids.shape[1]
        attention_mask = torch.arange(max_len, device=self.train_device).unsqueeze(0) < lengths.unsqueeze(1)

        if self.split_mode:
            # Async path: wait for previous training to finish (back pressure)
            if self._train_future is not None:
                self._pending_loss = self._train_future.result()
                self._train_future = None

            # Submit training to background thread
            self._train_future = self._train_executor.submit(
                self._do_train_impl, input_ids, old_logprobs, advantages, attention_mask
            )

            # Return loss from the previously completed training step
            return self._pending_loss
        else:
            # Synchronous path: run training inline, return current loss
            return self._do_train_impl(input_ids, old_logprobs, advantages, attention_mask)

    def sft_train_step(self, all_tokens, all_advantages, lr, weight_decay):
        """Run one SFT/ECHO update.

        ``importance_sampling`` preserves the historical warmup behavior.
        ``cross_entropy`` gives the direct next-token world-modeling objective
        ECHO needs for prompt-side environment/tool tokens.
        """
        loss_fn = getattr(self, "sft_loss_fn", "importance_sampling")
        if loss_fn == "importance_sampling":
            all_logprobs = [[0.0] * len(tokens) for tokens in all_tokens]
            return self.train_step(
                all_tokens,
                all_logprobs,
                all_advantages,
                lr,
                weight_decay,
            )
        if loss_fn != "cross_entropy":
            raise ValueError(
                "sft_loss_fn must be 'importance_sampling' or 'cross_entropy'."
            )

        n = len(all_tokens)
        if n == 0:
            return 0.0

        all_tokens, _, all_advantages, _ = self._maybe_crop_supervised_context(
            all_tokens,
            all_advantages=all_advantages,
        )

        self._clear_inference_prefix_cache()

        if self._train_future is not None:
            self._pending_loss = self._train_future.result()
            self._train_future = None

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = weight_decay

        microbatch_size = self.train_microbatch_size or n
        token_budget = getattr(self, "train_sft_microbatch_token_budget", 0)
        if microbatch_size < n or token_budget > 0:
            return self._do_sft_sequence_impl(all_tokens, all_advantages)

        token_tensors = [torch.tensor(t, dtype=torch.long) for t in all_tokens]
        adv_tensors = [torch.tensor(a, dtype=torch.float32) for a in all_advantages]

        input_ids = pad_sequence(token_tensors, batch_first=True, padding_value=0).to(self.train_device)
        advantages = pad_sequence(adv_tensors, batch_first=True, padding_value=0.0).to(self.train_device)

        lengths = torch.tensor([len(t) for t in all_tokens], device=self.train_device)
        max_len = input_ids.shape[1]
        attention_mask = torch.arange(max_len, device=self.train_device).unsqueeze(0) < lengths.unsqueeze(1)

        return self._do_sft_impl(input_ids, advantages, attention_mask)

    def train_step_with_echo_masks(
        self,
        all_tokens,
        all_logprobs,
        all_advantages,
        echo_advantages,
        echo_full_observation_counts,
        echo_loss_fn,
        lr,
        weight_decay,
    ):
        """Run RL + ECHO masks over the same rollout rows."""
        if not echo_advantages:
            return self.train_step(
                all_tokens,
                all_logprobs,
                all_advantages,
                lr,
                weight_decay,
            ), 0.0

        if echo_loss_fn != "cross_entropy":
            raise ValueError(
                "echo_loss_fn must be 'cross_entropy' for paper-faithful ECHO."
            )
        if len(echo_advantages) != len(all_tokens):
            raise ValueError("echo_advantages must have one row per training datum.")
        if len(echo_full_observation_counts) != len(all_tokens):
            raise ValueError(
                "echo_full_observation_counts must have one value per training datum."
            )

        all_tokens, all_logprobs, all_advantages, echo_advantages = (
            self._maybe_crop_supervised_context(
                all_tokens,
                all_logprobs,
                all_advantages,
                echo_advantages,
            )
        )

        self._clear_inference_prefix_cache()

        if self._train_future is not None:
            self._pending_loss = self._train_future.result()
            self._train_future = None

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = weight_decay

        token_tensors = [torch.tensor(t, dtype=torch.long) for t in all_tokens]
        lp_tensors = [torch.tensor(lp, dtype=torch.float32) for lp in all_logprobs]
        adv_tensors = [torch.tensor(a, dtype=torch.float32) for a in all_advantages]
        echo_adv_tensors = [
            torch.tensor(a, dtype=torch.float32) for a in echo_advantages
        ]

        input_ids = pad_sequence(token_tensors, batch_first=True, padding_value=0).to(self.train_device)
        old_logprobs = pad_sequence(lp_tensors, batch_first=True, padding_value=0.0).to(self.train_device)
        advantages = pad_sequence(adv_tensors, batch_first=True, padding_value=0.0).to(self.train_device)
        echo_advantages_tensor = pad_sequence(echo_adv_tensors, batch_first=True, padding_value=0.0).to(self.train_device)
        echo_counts = torch.tensor(
            [float(c) for c in echo_full_observation_counts],
            dtype=torch.float32,
            device=self.train_device,
        )

        lengths = torch.tensor([len(t) for t in all_tokens], device=self.train_device)
        max_len = input_ids.shape[1]
        attention_mask = torch.arange(max_len, device=self.train_device).unsqueeze(0) < lengths.unsqueeze(1)

        return self._do_hybrid_mask_impl(
            input_ids,
            old_logprobs,
            advantages,
            attention_mask,
            echo_advantages_tensor,
            echo_counts,
            echo_loss_fn,
        )

    def load_state(self, name):
        """Load adapter weights from a saved checkpoint or adapter directory.

        Restores LoRA adapter weights into the train model and re-syncs
        to the inference engine. Optimizer state (Adam momentum/variance)
        is NOT restored — training will re-warm the optimizer.

        Args:
            name: Checkpoint name under adapter_path, or a direct path to a
                PEFT adapter directory containing adapter_model.*.
        """
        candidate_dir = os.fspath(name)
        direct_weights = (
            os.path.isfile(os.path.join(candidate_dir, "adapter_model.safetensors"))
            or os.path.isfile(os.path.join(candidate_dir, "adapter_model.bin"))
        )
        save_dir = candidate_dir if direct_weights else os.path.join(self.adapter_path, candidate_dir)
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(
                f"Adapter checkpoint not found: {save_dir}. "
                f"Cannot resume from checkpoint '{name}'."
            )

        from peft import set_peft_model_state_dict

        safetensors_path = os.path.join(save_dir, "adapter_model.safetensors")
        bin_path = os.path.join(save_dir, "adapter_model.bin")

        if os.path.isfile(safetensors_path):
            from safetensors.torch import load_file
            adapter_state = load_file(safetensors_path, device=str(self.train_device))
        elif os.path.isfile(bin_path):
            adapter_state = torch.load(bin_path, map_location=self.train_device, weights_only=True)
        else:
            raise FileNotFoundError(
                f"No adapter weights in {save_dir}. "
                f"Expected adapter_model.safetensors or adapter_model.bin."
            )

        set_peft_model_state_dict(self.train_model, adapter_state)

        # Re-sync weights to inference engine
        if self.split_mode or self._external_engine:
            snapshot = {}
            for pname, param in self.train_model.named_parameters():
                if "lora_" in pname:
                    snapshot[pname] = param.data.clone()
            self._weight_snapshot = snapshot
            self._weights_dirty = True

        print(f"Loaded adapter checkpoint: {save_dir} (optimizer state not restored)")

    def save_adapter(self, path, name) -> str:
        """Save LoRA adapter to disk.

        Flushes any pending async training before saving to ensure
        the saved weights include the latest completed training step.

        Args:
            path: Base directory for adapter storage.
            name: Checkpoint name (creates subdirectory).

        Returns:
            Path to the saved adapter directory.
        """
        # Flush pending training before saving
        if self._train_future is not None:
            self._pending_loss = self._train_future.result()
            self._train_future = None

        save_dir = os.path.join(path, name)
        os.makedirs(save_dir, exist_ok=True)
        self.train_model.save_pretrained(save_dir)
        print(f"Adapter saved to {save_dir}")
        return save_dir
