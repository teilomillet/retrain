"""Device planning for the local backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from retrain.backends.torch import parse_device_spec


EXTERNAL_ENGINE_TYPES = frozenset(
    {
        "max",
        "vllm",
        "sglang",
        "trtllm",
        "mlx",
        "openai",
    }
)

SERVER_ENGINE_TYPES = frozenset(
    {
        "vllm",
        "sglang",
        "trtllm",
        "mlx",
        "openai",
    }
)


@dataclass(frozen=True)
class DevicePlan:
    infer_device: str
    train_device: str
    split_mode: bool
    external_engine: bool
    server_engine: bool
    use_amp: bool
    dtype: torch.dtype


def resolve(devices: str, engine_type: str) -> DevicePlan:
    raw_devices = [device.strip() for device in devices.split(",") if device.strip()]
    parsed_devices = [parse_device_spec(device) for device in raw_devices]
    cuda_available = torch.cuda.is_available()
    cuda_devices = [device for device in parsed_devices if device.startswith("cuda")]
    external_engine = engine_type in EXTERNAL_ENGINE_TYPES
    server_engine = engine_type in SERVER_ENGINE_TYPES

    split_mode = len(cuda_devices) > 1 and cuda_available
    if external_engine:
        device = parsed_devices[-1] if parsed_devices else "cuda:0"
        device = _fallback_cuda_to_cpu(device, cuda_available)
        split_mode = False
        infer_device = device
        train_device = device
    elif split_mode:
        infer_device = cuda_devices[0]
        train_device = cuda_devices[-1]
    else:
        device = parsed_devices[0] if parsed_devices else "cuda:0"
        device = _fallback_cuda_to_cpu(device, cuda_available)
        infer_device = device
        train_device = device

    use_amp = train_device != "cpu"
    dtype = torch.bfloat16 if use_amp else torch.float32
    return DevicePlan(
        infer_device=infer_device,
        train_device=train_device,
        split_mode=split_mode,
        external_engine=external_engine,
        server_engine=server_engine,
        use_amp=use_amp,
        dtype=dtype,
    )


def _fallback_cuda_to_cpu(device: str, cuda_available: bool) -> str:
    if device.startswith("cuda") and not cuda_available:
        print("CUDA not available, falling back to CPU")
        return "cpu"
    return device
