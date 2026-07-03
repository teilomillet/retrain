"""Local-backend CE accelerator policy and helpers."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Protocol, cast

import torch

from retrain.backends.local import metrics as local_metrics
from retrain.kernels.logprobs import (
    selected_linear_ce_logprobs_no_bias as _selected_linear_ce_logprobs_no_bias,
    selected_linear_ce_logprobs_with_bias as _selected_linear_ce_logprobs_with_bias,
)
from retrain.models.gemma4 import forward_hidden_states_and_lm_head


LossCallable = Callable[..., torch.Tensor]
HiddenAndHead = tuple[torch.Tensor, torch.nn.Module]
HiddenAndHeadFn = Callable[..., object | None]


class LossOwner(Protocol):
    train_model: object
    train_device: str
    train_compile_selective_ce: str
    train_compile_selective_ce_min_tokens: int
    train_unsloth_fused_ce: str
    train_unsloth_fused_ce_target_gb: float
    train_save_on_cpu: bool
    train_unsloth_fused_ce_torch_compile: bool
    _compiled_selective_ce_available: bool | None
    _compiled_selective_ce_fallback_reason: str
    _compiled_selective_ce_no_bias: LossCallable | None
    _compiled_selective_ce_with_bias: LossCallable | None
    _unsloth_fused_ce_available: bool | None
    _unsloth_fused_ce_unavailable_reason: str
    _unsloth_fused_ce_fallback_reason: str
    _unsloth_fused_ce_runtime_disabled: bool
    _unsloth_fused_ce_loss_fn: LossCallable | None
    _unsloth_fused_ce_attempts: int
    _last_unsloth_fused_ce_effective_target_gb: float

    def _reject_compiled_selective_ce(self, reason: str): ...

    def _constant_positive_weight(
        self,
        weights: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor | None: ...

    def _unsloth_fused_ce_loss(self): ...

    def _is_cuda_oom_exception(self, exc: BaseException) -> bool: ...

    def _disable_unsloth_fused_ce_after_runtime_failure(
        self,
        exc: BaseException,
    ) -> str: ...

    def _effective_unsloth_fused_ce_target_gb(self) -> float: ...


def normalize_compile_mode(raw: object, *, option_name: str) -> str:
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


def normalize_unsloth_fused_ce_mode(raw: object) -> str:
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
        raise ValueError("train_unsloth_fused_ce must be 'off', 'auto', or 'require'.")
    return text


def reject_compiled_selective_ce(owner: object, reason: str):
    state = cast(LossOwner, owner)
    state._compiled_selective_ce_fallback_reason = reason
    if getattr(state, "train_compile_selective_ce", "off") == "require":
        raise RuntimeError(
            "train_compile_selective_ce=require but the compiled selected "
            f"CE path cannot be used: {reason}"
        )
    return None


def compiled_selective_ce_logprobs(
    owner: object,
    selected_hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    target_ids: torch.Tensor,
):
    state = cast(LossOwner, owner)
    mode = getattr(state, "train_compile_selective_ce", "off")
    if mode == "off":
        return None
    if int(selected_hidden.shape[0]) < int(
        getattr(state, "train_compile_selective_ce_min_tokens", 0)
    ):
        return state._reject_compiled_selective_ce("below_min_tokens")
    if selected_hidden.device.type != "cuda":
        return state._reject_compiled_selective_ce("non_cuda")
    if not hasattr(torch, "compile"):
        state._compiled_selective_ce_available = False
        return state._reject_compiled_selective_ce("torch_compile_unavailable")

    if type(lm_head) is not torch.nn.Linear:
        return state._reject_compiled_selective_ce("lm_head_not_plain_linear")
    weight = getattr(lm_head, "weight", None)
    if weight is None:
        return state._reject_compiled_selective_ce("lm_head_weight_unavailable")
    bias = getattr(lm_head, "bias", None)

    try:
        if bias is None:
            compiled = getattr(state, "_compiled_selective_ce_no_bias", None)
            if compiled is None:
                compiled = cast(
                    LossCallable,
                    torch.compile(
                        _selected_linear_ce_logprobs_no_bias,
                        mode="reduce-overhead",
                        fullgraph=True,
                    ),
                )
                state._compiled_selective_ce_no_bias = compiled
            selected_logprobs = compiled(selected_hidden, weight, target_ids)
        else:
            compiled = getattr(state, "_compiled_selective_ce_with_bias", None)
            if compiled is None:
                compiled = cast(
                    LossCallable,
                    torch.compile(
                        _selected_linear_ce_logprobs_with_bias,
                        mode="reduce-overhead",
                        fullgraph=True,
                    ),
                )
                state._compiled_selective_ce_with_bias = compiled
            selected_logprobs = compiled(
                selected_hidden,
                weight,
                bias,
                target_ids,
            )
    except Exception as exc:  # Optional compiler path.
        state._compiled_selective_ce_available = False
        return state._reject_compiled_selective_ce(type(exc).__name__)

    state._compiled_selective_ce_available = True
    state._compiled_selective_ce_fallback_reason = ""
    return selected_logprobs


def unsloth_fused_ce_loss(owner: object):
    state = cast(LossOwner, owner)
    if bool(getattr(state, "_unsloth_fused_ce_runtime_disabled", False)):
        return None
    cached = getattr(state, "_unsloth_fused_ce_loss_fn", None)
    if cached is not None:
        return cached
    try:
        try:
            importlib.import_module("unsloth")
        except Exception:
            pass
        from unsloth_zoo.loss_utils import (  # type: ignore[import-not-found]
            HAS_CUT_CROSS_ENTROPY,
        )
        from unsloth_zoo.loss_utils import (  # type: ignore[import-not-found]
            unsloth_fused_ce_loss,
        )
    except Exception as exc:  # Optional accelerator path.
        state._unsloth_fused_ce_available = False
        state._unsloth_fused_ce_unavailable_reason = type(exc).__name__
        return None
    if not HAS_CUT_CROSS_ENTROPY:
        state._unsloth_fused_ce_available = False
        state._unsloth_fused_ce_unavailable_reason = "cut_cross_entropy_unavailable"
        return None
    state._unsloth_fused_ce_available = True
    state._unsloth_fused_ce_unavailable_reason = ""
    loss_fn = cast(LossCallable, unsloth_fused_ce_loss)
    state._unsloth_fused_ce_loss_fn = loss_fn
    return loss_fn


def effective_unsloth_fused_ce_target_gb(owner: object) -> float:
    state = cast(LossOwner, owner)
    configured = float(getattr(state, "train_unsloth_fused_ce_target_gb", 0.0))
    if configured > 0:
        return configured
    device = str(getattr(state, "train_device", ""))
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return 0.0
    try:
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    except Exception:  # Diagnostic/tuning fallback only.
        return 0.0
    if total_gb <= 16:
        return 0.25
    if total_gb <= 24:
        return 0.5
    return 0.0


def is_cuda_oom_exception(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def disable_unsloth_fused_ce_after_runtime_failure(
    owner: object,
    exc: BaseException,
) -> str:
    state = cast(LossOwner, owner)
    reason = f"runtime_{type(exc).__name__}"
    state._unsloth_fused_ce_available = False
    state._unsloth_fused_ce_unavailable_reason = reason
    state._unsloth_fused_ce_fallback_reason = reason
    state._unsloth_fused_ce_runtime_disabled = True
    try:
        torch._dynamo.reset()
    except Exception:  # Best-effort compiler cleanup.
        pass
    return reason


def constant_positive_weight(weights: torch.Tensor, target_mask: torch.Tensor):
    selected = weights[target_mask]
    if selected.numel() == 0:
        return None
    first = selected[:1]
    if torch.allclose(selected, first.expand_as(selected)):
        return first.squeeze()
    return None


def maybe_compute_unsloth_fused_sft_loss(
    owner: object,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    weights: torch.Tensor,
    target_mask: torch.Tensor | None,
    token_count: torch.Tensor,
    *,
    hidden_and_head_fn: HiddenAndHeadFn = forward_hidden_states_and_lm_head,
):
    state = cast(LossOwner, owner)
    mode = getattr(state, "train_unsloth_fused_ce", "off")
    if mode == "off":
        return None

    def reject(reason: str):
        state._unsloth_fused_ce_fallback_reason = reason
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

    weight_scale = state._constant_positive_weight(weights, target_mask)
    if weight_scale is None:
        return reject("non_constant_token_weights")
    if bool(getattr(state, "train_save_on_cpu", False)):
        return reject("saved_tensor_hooks_incompatible")

    loss_fn = state._unsloth_fused_ce_loss()
    if loss_fn is None:
        reason = getattr(
            state,
            "_unsloth_fused_ce_unavailable_reason",
            "unavailable",
        )
        return reject(reason)
    state._unsloth_fused_ce_attempts = (
        int(getattr(state, "_unsloth_fused_ce_attempts", 0)) + 1
    )

    try:
        hidden_and_head = cast(
            HiddenAndHead | None,
            hidden_and_head_fn(
                state.train_model,
                input_ids,
                attention_mask,
            ),
        )
    except Exception as exc:  # Optional accelerator path.
        if state._is_cuda_oom_exception(exc):
            raise
        reason = state._disable_unsloth_fused_ce_after_runtime_failure(exc)
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
    target_gb = state._effective_unsloth_fused_ce_target_gb()
    state._last_unsloth_fused_ce_effective_target_gb = target_gb
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
                getattr(state, "train_unsloth_fused_ce_torch_compile", True)
            ),
            shift_labels=True,
            ignore_index=ignore_index,
        )
    except Exception as exc:  # Optional accelerator path.
        if state._is_cuda_oom_exception(exc):
            raise
        reason = state._disable_unsloth_fused_ce_after_runtime_failure(exc)
        return reject(reason)
    state._unsloth_fused_ce_fallback_reason = ""
    local_metrics.record_loss_path(state, "unsloth_fused_ce")
    return loss * weight_scale.to(loss.device)
