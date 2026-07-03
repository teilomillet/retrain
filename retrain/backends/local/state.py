"""Adapter state persistence for the local backend."""

from __future__ import annotations

import os
from typing import cast

import torch


ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def lora_state_dict(model: object, *, clone: bool) -> dict[str, torch.Tensor]:
    """Collect LoRA tensors by parameter name.

    Initial inference sync can share live tensor storage. Cross-thread snapshots
    must clone, so the caller chooses the ownership policy explicitly.
    """
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return {}
    state: dict[str, torch.Tensor] = {}
    for name, param in named_parameters():
        if "lora_" not in str(name):
            continue
        data = cast(torch.Tensor, getattr(param, "data"))
        state[str(name)] = data.clone() if clone else data
    return state


AdapterName = str | os.PathLike[str]


def resolve_adapter_dir(adapter_path: str, name: AdapterName) -> str:
    candidate_dir = os.fspath(name)
    if _contains_adapter_weights(candidate_dir):
        return candidate_dir
    return os.path.join(adapter_path, candidate_dir)


def load_adapter_weights(save_dir: str, train_device: str) -> dict[str, torch.Tensor]:
    safetensors_path = os.path.join(save_dir, "adapter_model.safetensors")
    bin_path = os.path.join(save_dir, "adapter_model.bin")

    if os.path.isfile(safetensors_path):
        from safetensors.torch import load_file

        return load_file(safetensors_path, device=str(train_device))
    if os.path.isfile(bin_path):
        return cast(
            dict[str, torch.Tensor],
            torch.load(bin_path, map_location=train_device, weights_only=True),
        )
    raise FileNotFoundError(
        f"No adapter weights in {save_dir}. "
        f"Expected adapter_model.safetensors or adapter_model.bin."
    )


def load_into_model(
    model: object,
    *,
    adapter_path: str,
    name: AdapterName,
    train_device: str,
) -> str:
    save_dir = resolve_adapter_dir(adapter_path, name)
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(
            f"Adapter checkpoint not found: {save_dir}. "
            f"Cannot resume from checkpoint '{name}'."
        )

    from peft import set_peft_model_state_dict

    adapter_state = load_adapter_weights(save_dir, train_device)
    set_peft_model_state_dict(model, adapter_state)
    return save_dir


def save_model(model: object, *, path: str, name: str) -> str:
    save_dir = os.path.join(path, name)
    os.makedirs(save_dir, exist_ok=True)
    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TypeError("Local train model does not expose save_pretrained().")
    save_pretrained(save_dir)
    return save_dir


def _contains_adapter_weights(path: str) -> bool:
    return any(
        os.path.isfile(os.path.join(path, filename))
        for filename in ADAPTER_WEIGHT_FILES
    )
