"""LoRA setup for the local backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from peft import LoraConfig, TaskType

from retrain.kernels.lora import (
    infer_transformer_layer_count,
    parse_lora_layers_to_transform,
    patch_lora_fast_linear_modules,
)
from retrain.models.gemma4 import (
    parse_lora_target_module_suffixes,
    resolve_lora_target_modules,
)


@dataclass(frozen=True)
class PeftBuild:
    config: LoraConfig
    selected_layers: list[int] | None


def build_config(
    base_model,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    layers_spec: str,
    layers_pattern: str,
    target_module_suffixes: tuple[str, ...],
) -> PeftBuild:
    effective_alpha = alpha if alpha > 0 else rank * 2
    layer_count = infer_transformer_layer_count(base_model)
    selected_layers = parse_lora_layers_to_transform(layers_spec, layer_count)
    selected_layers_pattern = layers_pattern if selected_layers is not None else None
    return PeftBuild(
        config=LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=effective_alpha,
            lora_dropout=dropout,
            target_modules=resolve_lora_target_modules(
                base_model,
                target_module_suffixes,
            ),
            layers_to_transform=selected_layers,
            layers_pattern=selected_layers_pattern,
        ),
        selected_layers=selected_layers,
    )


def freeze_a(model, *, enabled: bool) -> int:
    if not enabled:
        return 0
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return 0
    frozen = 0
    for name, param in named_parameters():
        if ".lora_A." not in f".{name}.":
            continue
        param.requires_grad_(False)
        frozen += 1
    return frozen


def _detach_first_tensor_input(_module, inputs):
    if not inputs:
        return inputs
    first = inputs[0]
    if torch.is_tensor(first):
        return (first.detach(), *inputs[1:])
    return inputs


def detach_input(model, *, enabled: bool):
    if not enabled:
        return []
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return []
    handles = []
    for name, module in named_modules():
        if ".lora_A." not in f".{name}.":
            continue
        if not torch.is_tensor(getattr(module, "weight", None)):
            continue
        register = getattr(module, "register_forward_pre_hook", None)
        if not callable(register):
            continue
        handles.append(register(_detach_first_tensor_input))
    return handles


def patch_fast(model, *, enabled: bool, detach: bool, freeze: bool) -> int:
    if not enabled:
        return 0
    return patch_lora_fast_linear_modules(
        model,
        detach_input=detach,
        freeze_a=freeze,
    )


def metrics(
    model,
    *,
    selected_layers: list[int] | None,
    layers_pattern: str,
    target_module_suffixes: tuple[str, ...],
    freeze_a_enabled: bool,
    frozen_a_tensors: int,
    detach_input_enabled: bool,
    detach_input_hooks: int,
    fast_enabled: bool,
    fast_patches: int,
) -> dict[str, float | int | str]:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return {}
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
    return {
        "local_lora_layer_selection_enabled": int(selected_layers is not None),
        "local_lora_selected_layer_count": (
            0 if selected_layers is None else len(selected_layers)
        ),
        "local_lora_selected_layers": (
            "" if selected_layers is None else ",".join(map(str, selected_layers))
        ),
        "local_lora_layers_pattern": layers_pattern,
        "local_lora_target_modules": ",".join(target_module_suffixes),
        "local_lora_target_module_count": len(target_module_suffixes),
        "local_lora_parameter_count": lora_param_count,
        "local_lora_parameter_tensor_count": lora_tensor_count,
        "local_lora_trainable_parameter_count": trainable_param_count,
        "local_lora_freeze_a_enabled": int(freeze_a_enabled),
        "local_lora_frozen_a_tensor_count": frozen_a_tensors,
        "local_lora_detach_input_enabled": int(detach_input_enabled),
        "local_lora_detach_input_hook_count": detach_input_hooks,
        "local_lora_fast_linear_enabled": int(fast_enabled),
        "local_lora_fast_linear_patch_count": fast_patches,
        "local_lora_fast_linear_detach_input_enabled": int(
            fast_enabled and detach_input_enabled
        ),
    }


DEFAULT_TARGET_SUFFIXES = parse_lora_target_module_suffixes("")
