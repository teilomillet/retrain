"""Optional CUDA accelerator integration helpers."""

from __future__ import annotations

import importlib

_ACCELERATOR_MODULES = {
    "fla": "fla",
    "causal_conv1d": "causal_conv1d",
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
