"""Small PyTorch runtime support shared by local training and inference."""

from __future__ import annotations

import time

import torch


def parse_device_spec(device_spec: str) -> str:
    """Convert retrain's device spec into a PyTorch device string."""
    device_spec = device_spec.strip()
    if device_spec.startswith("gpu:"):
        return device_spec.replace("gpu:", "cuda:")
    if device_spec == "cpu":
        return "cpu"
    return "cuda:0"


def pad_to_width(tensor: torch.Tensor, width: int, value: float | int) -> torch.Tensor:
    """Right-pad a batch-major tensor to ``width`` columns."""
    if tensor.shape[1] >= width:
        return tensor
    pad = tensor.new_full((tensor.shape[0], width - tensor.shape[1]), value)
    return torch.cat([tensor, pad], dim=1)


def is_cuda_device(device: object) -> bool:
    return (
        isinstance(device, str)
        and device.startswith("cuda")
        and torch.cuda.is_available()
    )


def timer_start(device: str):
    if is_cuda_device(device):
        with torch.cuda.device(torch.device(device)):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        return ("cuda", device, event)
    return ("cpu", device, time.perf_counter())


def timer_stop(start) -> float:
    kind, device, marker = start
    if kind == "cuda":
        with torch.cuda.device(torch.device(device)):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
        return marker.elapsed_time(end) / 1000.0
    return time.perf_counter() - marker


def reset_cuda_peak(device: str) -> None:
    if is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def cuda_peak_metrics(prefix: str, device: str) -> dict[str, float]:
    if not is_cuda_device(device):
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
