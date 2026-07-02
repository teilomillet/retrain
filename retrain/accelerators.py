"""Optional CUDA accelerator integration helpers."""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from functools import wraps

_ACCELERATOR_MODULES = {
    "fla": "fla",
    "flash_qla": "flash_qla",
    "tilelang": "tilelang",
    "causal_conv1d": "causal_conv1d",
    "cudnn": "cudnn",
    "liger_kernel": "liger_kernel",
    "flash_attn": "flash_attn",
}

_LIGER_PATCHERS = {
    "qwen3_5": "apply_liger_kernel_to_qwen3_5",
    "qwen3": "apply_liger_kernel_to_qwen3",
    "qwen2": "apply_liger_kernel_to_qwen2",
    "qwen2_5_vl": "apply_liger_kernel_to_qwen2_5_vl",
    "qwen2_vl": "apply_liger_kernel_to_qwen2_vl",
    "gemma4": "apply_liger_kernel_to_gemma4",
    "gemma3": "apply_liger_kernel_to_gemma3",
    "llama": "apply_liger_kernel_to_llama",
}


def module_available(module_name: str) -> bool:
    """Return whether an optional accelerator module is importable."""
    try:
        importlib.import_module(module_name)
        return True
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def accelerator_status() -> dict[str, int]:
    """Return import availability flags for optional accelerator packages."""
    return {
        f"accelerator_{name}_available": int(module_available(module))
        for name, module in _ACCELERATOR_MODULES.items()
    }


def _module_global(module: object, name: str):
    try:
        return getattr(sys.modules[type(module).__module__], name)
    except (KeyError, AttributeError):
        return None


def _cuda_capability(device: str | None) -> tuple[int, int]:
    if not device or not str(device).startswith("cuda"):
        return (0, 0)
    try:
        import torch

        if not torch.cuda.is_available():
            return (0, 0)
        return tuple(torch.cuda.get_device_capability(torch.device(device)))
    except Exception:  # noqa: BLE001 - diagnostic only.
        return (0, 0)


def _flash_qla_supported(device: str | None) -> bool:
    major, minor = _cuda_capability(device)
    return (major, minor) in {(9, 0), (10, 0)}


def patch_qwen35_gated_delta_kernel(
    model: object,
    *,
    mode: str,
    device: str | None,
) -> dict[str, object]:
    """Patch Qwen3.5 GatedDeltaNet to an explicit kernel implementation.

    ``auto``/``off`` intentionally preserve the model's installed default.  The
    FlashQLA path is explicit-only because its public API expects q/k
    normalization before the kernel, while Transformers' FLA call currently
    passes ``use_qk_l2norm_in_kernel=True`` into the kernel.
    """
    raw_mode = str(mode or "auto").strip().lower()
    if raw_mode in {"", "default", "none", "false", "0"}:
        raw_mode = "off"
    report: dict[str, object] = {
        "qwen35_gated_delta_kernel_requested": raw_mode,
        "qwen35_gated_delta_kernel_active": "default",
        "qwen35_gated_delta_kernel_patched_modules": 0,
        "qwen35_gated_delta_flash_qla_available": int(module_available("flash_qla")),
        "qwen35_gated_delta_flash_qla_supported_device": int(
            _flash_qla_supported(device)
        ),
        "qwen35_gated_delta_torch_fallback": 0,
        "qwen35_gated_delta_kernel_error": "",
    }
    if raw_mode in {"auto", "off"}:
        return report
    if raw_mode not in {"torch", "flash_qla"}:
        raise ValueError(
            "qwen35_gated_delta_kernel must be 'auto', 'off', 'torch', or 'flash_qla'."
        )

    flash_qla_rule = None
    flash_qla_l2norm = None
    if raw_mode == "flash_qla":
        if not module_available("flash_qla"):
            raise RuntimeError("qwen35_gated_delta_kernel='flash_qla' requires flash-qla.")
        if not _flash_qla_supported(device):
            major, minor = _cuda_capability(device)
            raise RuntimeError(
                "qwen35_gated_delta_kernel='flash_qla' requires CUDA SM90 or SM100; "
                f"got SM{major}{minor}."
            )
        flash_qla = importlib.import_module("flash_qla")
        flash_qla_utils = importlib.import_module("flash_qla.utils")
        flash_qla_rule = getattr(flash_qla, "chunk_gated_delta_rule", None)
        flash_qla_l2norm = getattr(flash_qla_utils, "l2norm", None)
        if not callable(flash_qla_rule) or not callable(flash_qla_l2norm):
            raise RuntimeError(
                "flash_qla must expose chunk_gated_delta_rule and flash_qla.utils.l2norm."
            )

    patched = 0
    for module in model.modules():
        rule = getattr(module, "chunk_gated_delta_rule", None)
        if not callable(rule):
            continue
        if not (
            "Qwen3_5GatedDeltaNet" in type(module).__name__
            or callable(_module_global(module, "torch_chunk_gated_delta_rule"))
        ):
            continue
        if raw_mode == "torch":
            torch_rule = _module_global(module, "torch_chunk_gated_delta_rule")
            if not callable(torch_rule):
                continue
            module.chunk_gated_delta_rule = torch_rule
        else:
            original_rule = rule

            @wraps(original_rule)
            def flash_qla_chunked_rule(
                q,
                k,
                v,
                *args,
                __flash_qla_rule=flash_qla_rule,
                __flash_qla_l2norm=flash_qla_l2norm,
                **kwargs,
            ):
                _ = args
                if bool(kwargs.pop("use_qk_l2norm_in_kernel", False)):
                    q = __flash_qla_l2norm(q.contiguous())
                    k = __flash_qla_l2norm(k.contiguous())
                return __flash_qla_rule(
                    q=q.contiguous(),
                    k=k.contiguous(),
                    v=v.contiguous(),
                    g=kwargs.get("g"),
                    beta=kwargs.get("beta"),
                    scale=kwargs.get("scale"),
                    initial_state=kwargs.get("initial_state"),
                    output_final_state=bool(kwargs.get("output_final_state", True)),
                    cu_seqlens=kwargs.get("cu_seqlens"),
                )

            module.chunk_gated_delta_rule = flash_qla_chunked_rule
        patched += 1

    if patched == 0:
        report["qwen35_gated_delta_kernel_active"] = "not_found"
        return report
    report["qwen35_gated_delta_kernel_active"] = raw_mode
    report["qwen35_gated_delta_kernel_patched_modules"] = patched
    report["qwen35_gated_delta_torch_fallback"] = int(raw_mode == "torch")
    return report


def install_cudnn_causal_conv1d_shim(*, enabled: bool) -> dict[str, object]:
    """Install an opt-in ``causal_conv1d`` shim backed by cuDNN frontend.

    Transformers checks for an importable ``causal_conv1d`` module before
    enabling the Qwen3.5 GatedDelta fast path. Dao-AILab's package is the normal
    implementation, but some CUDA 13 hosts cannot build it from source. NVIDIA's
    cuDNN frontend now exposes the same causal depthwise-conv primitive, so this
    shim provides the small API subset needed by Qwen prefill/training.
    """
    report: dict[str, object] = {
        "cudnn_causal_conv1d_shim_requested": int(enabled),
        "cudnn_causal_conv1d_shim_installed": 0,
        "cudnn_causal_conv1d_shim_error": "",
        "cudnn_causal_conv1d_frontend_version": "",
    }
    if not enabled:
        return report
    if module_available("causal_conv1d"):
        report["cudnn_causal_conv1d_shim_error"] = "causal_conv1d_already_available"
        return report

    try:
        import torch
        import torch.nn.functional as F
        import cudnn  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - optional accelerator path.
        report["cudnn_causal_conv1d_shim_error"] = f"{type(exc).__name__}: {exc}"
        return report

    report["cudnn_causal_conv1d_frontend_version"] = str(
        getattr(cudnn, "__version__", "")
    )
    causal_conv = getattr(getattr(cudnn, "ops", None), "causal_conv1d", None)
    if not callable(causal_conv):
        report["cudnn_causal_conv1d_shim_error"] = "cudnn_ops_missing_causal_conv1d"
        return report

    def _activation_name(raw):
        if raw is None:
            return "identity"
        if raw in ("silu", "swish"):
            return "silu"
        raise NotImplementedError("activation must be None, silu, or swish")

    def causal_conv1d_fn(
        x,
        weight,
        bias=None,
        seq_idx=None,
        initial_states=None,
        return_final_states=False,
        final_states_out=None,
        activation=None,
    ):
        if seq_idx is not None:
            raise NotImplementedError("cuDNN causal_conv1d shim does not support seq_idx")
        dtype_in = x.dtype
        width = int(weight.shape[1])
        seqlen = int(x.shape[-1])
        if initial_states is not None:
            x_for_conv = torch.cat([initial_states, x], dim=-1)
            out = F.conv1d(
                x_for_conv.to(weight.dtype),
                weight.unsqueeze(1),
                bias,
                padding=0,
                groups=int(weight.shape[0]),
            )[..., -seqlen:]
            if activation in ("silu", "swish"):
                out = F.silu(out)
            elif activation is not None:
                raise NotImplementedError("activation must be None, silu, or swish")
            out = out.to(dtype=dtype_in)
        else:
            out = causal_conv(
                x.contiguous(),
                weight.contiguous(),
                bias=None if bias is None else bias.contiguous(),
                activation=_activation_name(activation),
            )
        if not return_final_states:
            return out

        if initial_states is None:
            state_source = x
        else:
            state_source = torch.cat([initial_states, x], dim=-1)
        final_states = F.pad(
            state_source,
            (max(0, width - 1 - int(state_source.shape[-1])), 0),
        )[..., -(width - 1) :]
        final_states = final_states.to(dtype=dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
            final_states = final_states_out
        return out, final_states

    def causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias=None,
        activation=None,
        cache_seqlens=None,
        conv_state_indices=None,
    ):
        if conv_state_indices is not None:
            raise NotImplementedError(
                "cuDNN causal_conv1d shim does not support conv_state_indices"
            )
        if activation not in (None, "silu", "swish"):
            raise NotImplementedError("activation must be None, silu, or swish")
        dtype_in = x.dtype
        unsqueeze = x.dim() == 2
        if unsqueeze:
            x = x.unsqueeze(-1)
        batch, dim, seqlen = x.shape
        width = int(weight.shape[1])
        state_len = int(conv_state.shape[-1])
        if cache_seqlens is not None:
            width_idx = (
                torch.arange(
                    -(width - 1),
                    0,
                    dtype=torch.long,
                    device=x.device,
                ).unsqueeze(0)
                + cache_seqlens.unsqueeze(1)
            )
            width_idx = torch.remainder(width_idx, state_len).unsqueeze(1)
            width_idx = width_idx.expand(batch, dim, -1)
            prev = conv_state.gather(2, width_idx)
            copy_idx = (
                torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0)
                + cache_seqlens.unsqueeze(1)
            )
            copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1)
            copy_idx = copy_idx.expand(batch, dim, -1)
            conv_state.scatter_(2, copy_idx, x)
        else:
            prev = conv_state
            x_new = torch.cat([conv_state, x], dim=-1)
            conv_state.copy_(x_new[:, :, -state_len:])
        x_for_conv = torch.cat([prev, x], dim=-1).to(weight.dtype)
        out = F.conv1d(
            x_for_conv,
            weight.unsqueeze(1),
            bias,
            padding=0,
            groups=dim,
        )[:, :, -seqlen:]
        if activation in ("silu", "swish"):
            out = F.silu(out)
        out = out.to(dtype=dtype_in)
        return out.squeeze(-1) if unsqueeze else out

    module = types.ModuleType("causal_conv1d")
    module.__version__ = "cudnn-shim"
    module.__spec__ = importlib.machinery.ModuleSpec("causal_conv1d", loader=None)
    module.causal_conv1d_fn = causal_conv1d_fn
    module.causal_conv1d_update = causal_conv1d_update
    sys.modules["causal_conv1d"] = module
    report["cudnn_causal_conv1d_shim_installed"] = 1
    return report


def from_pretrained_attention_kwargs(attention_kernel: str) -> dict[str, str]:
    """Map retrain attention kernel names to Transformers load kwargs."""
    kernel = (attention_kernel or "default").strip().lower()
    if kernel in {"", "default", "auto"}:
        return {}
    if kernel in {"flash", "flash_attention_2", "flash-attn", "flash_attn"}:
        return {"attn_implementation": "flash_attention_2"}
    if kernel in {"sdpa", "torch_sdpa", "torch-sdpa"}:
        return {"attn_implementation": "sdpa"}
    if kernel in {"eager", "torch", "vanilla"}:
        return {"attn_implementation": "eager"}
    return {}


def apply_liger_kernel_if_available(
    model_name: str,
    *,
    enabled: bool,
) -> dict[str, object]:
    """Apply a model-specific Liger kernel patch when available.

    The patch must run before ``AutoModelForCausalLM.from_pretrained``.
    """
    report: dict[str, object] = {
        "liger_kernel_requested": int(enabled),
        "liger_kernel_available": int(module_available("liger_kernel")),
        "liger_kernel_applied": 0,
        "liger_kernel_model_type": "",
        "liger_kernel_patcher": "",
        "liger_kernel_error": "",
    }
    if not enabled:
        return report
    if not module_available("liger_kernel"):
        return report

    try:
        from transformers import AutoConfig

        liger_transformers = importlib.import_module("liger_kernel.transformers")

        config = AutoConfig.from_pretrained(model_name)
        model_type = str(getattr(config, "model_type", "") or "")
        patcher_name = _LIGER_PATCHERS.get(model_type, "")
        report["liger_kernel_model_type"] = model_type
        report["liger_kernel_patcher"] = patcher_name
        if not patcher_name:
            return report
        patcher = getattr(liger_transformers, patcher_name, None)
        if not callable(patcher):
            return report
        patcher()
        report["liger_kernel_applied"] = 1
        return report
    except Exception as exc:  # noqa: BLE001 - optional accelerator path.
        report["liger_kernel_error"] = f"{type(exc).__name__}: {exc}"
        return report
