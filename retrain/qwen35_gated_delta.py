"""Qwen3.5 GatedDelta kernel selection helpers."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import torch

from retrain.accelerators import module_available

_FLASH_QLA_SUPPORTED_CAPABILITIES = {(9, 0), (10, 0)}


class _L2NormFn(Protocol):
    def __call__(self, tensor: "torch.Tensor") -> "torch.Tensor": ...


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
    return _cuda_capability(device) in _FLASH_QLA_SUPPORTED_CAPABILITIES


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

    flash_qla_rule: Callable[..., object] | None = None
    flash_qla_l2norm: _L2NormFn | None = None
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
        raw_rule = getattr(flash_qla, "chunk_gated_delta_rule", None)
        raw_l2norm = getattr(flash_qla_utils, "l2norm", None)
        if not callable(raw_rule) or not callable(raw_l2norm):
            raise RuntimeError(
                "flash_qla must expose chunk_gated_delta_rule and flash_qla.utils.l2norm."
            )
        flash_qla_rule = cast(Callable[..., object], raw_rule)
        flash_qla_l2norm = cast(_L2NormFn, raw_l2norm)

    patched = 0
    modules = getattr(model, "modules", None)
    if not callable(modules):
        report["qwen35_gated_delta_kernel_active"] = "not_found"
        return report
    for module in modules():
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
            rule_fn = flash_qla_rule
            l2norm_fn = flash_qla_l2norm
            assert rule_fn is not None
            assert l2norm_fn is not None

            @wraps(original_rule)
            def flash_qla_chunked_rule(
                q,
                k,
                v,
                *args,
                _flash_qla_rule=rule_fn,
                _flash_qla_l2norm=l2norm_fn,
                **kwargs,
            ):
                _ = args
                if bool(kwargs.pop("use_qk_l2norm_in_kernel", False)):
                    q = _flash_qla_l2norm(q.contiguous())
                    k = _flash_qla_l2norm(k.contiguous())
                return _flash_qla_rule(
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
