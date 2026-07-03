"""GPU memory policy for the local training backend."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from retrain.backends.torch import is_cuda_device


AllocatorMetrics = dict[str, int]


def normalize_expandable_segments_mode(raw: object) -> str:
    """Normalize the CUDA expandable-segments option."""
    if isinstance(raw, bool):
        return "on" if raw else "off"
    text = str(raw or "auto").strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "none": "off",
        "disabled": "off",
        "1": "on",
        "true": "on",
        "yes": "on",
        "enabled": "on",
    }
    text = aliases.get(text, text)
    if text not in {"off", "auto", "on"}:
        raise ValueError("cuda_expandable_segments must be 'off', 'auto', or 'on'.")
    return text


def configure_cuda_allocator(
    *,
    mode: str,
    train_device: str,
    gradient_checkpointing_skip_last_n: int,
) -> AllocatorMetrics:
    """Enable expandable CUDA segments when this run's memory policy needs them."""
    metrics = {"enabled": 0, "env_preset": 0, "set_failed": 0}
    if mode == "off" or not is_cuda_device(train_device):
        return metrics
    for env_name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        env_conf = os.environ.get(env_name, "")
        if "expandable_segments" in env_conf:
            metrics["env_preset"] = 1
            metrics["enabled"] = int("expandable_segments:true" in env_conf.lower())
            return metrics
    if mode == "auto" and gradient_checkpointing_skip_last_n <= 0:
        return metrics
    try:
        setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
        if setter is None:
            memory = getattr(torch.cuda, "memory")
            setter = getattr(memory, "_set_allocator_settings")
        setter("expandable_segments:True")
    except Exception as exc:  # Allocator tuning must not kill runs.
        metrics["set_failed"] = 1
        print(f"cuda_expandable_segments setup failed: {exc}")
        return metrics
    metrics["enabled"] = 1
    return metrics


def empty_cuda_cache_if_requested(enabled: bool) -> None:
    """Clear CUDA cache only when the backend option explicitly requests it."""
    if enabled and torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def saved_tensors_context(
    *,
    enabled: bool,
    train_device: str,
    pin_memory: bool = True,
    min_numel: int = 0,
) -> Iterator[None]:
    """Offload autograd saved tensors to CPU for memory-constrained training."""
    if enabled and train_device.startswith("cuda") and torch.cuda.is_available():
        if min_numel > 0:

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

        with torch.autograd.graph.save_on_cpu(pin_memory=pin_memory):
            yield
    else:
        yield
