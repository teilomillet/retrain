"""Device, model, engine, and optimizer bootstrap for the local backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch

from retrain.backends.determinism import (
    add_model_attention_proof,
    establish_strict_determinism,
)
from retrain.backends.local import device as local_device
from retrain.backends.local.checkpointing import configure_gradient_checkpointing
from retrain.backends.local.lora import (
    detach_input as detach_lora_input,
    freeze_a as freeze_lora_a,
    metrics as lora_metrics,
    patch_fast as patch_fast_lora,
)
from retrain.backends.local.memory import configure_cuda_allocator
from retrain.kernels.accelerators import (
    accelerator_status,
    apply_liger_kernel_if_available,
    install_cudnn_causal_conv1d_shim,
)
from retrain.models.qwen35 import patch_qwen35_gated_delta_kernel

if TYPE_CHECKING:
    from retrain.backends.local.train import LocalTrainHelper


def initialize(
    helper: "LocalTrainHelper",
    *,
    model_name: str,
    devices,
    engine_type: str,
    inference_url: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    optim_beta1: float,
    optim_beta2: float,
    optim_eps: float,
    engine_factory,
) -> None:
    """Construct runtime resources after backend options have been normalized."""

    device_plan = local_device.resolve(devices, engine_type)
    helper.infer_device = device_plan.infer_device
    helper.train_device = device_plan.train_device
    helper.split_mode = device_plan.split_mode
    helper._server_engine = device_plan.server_engine
    helper._external_engine = device_plan.external_engine
    helper.use_amp = device_plan.use_amp
    dtype = device_plan.dtype
    helper.amp_dtype = dtype
    if helper.sample_kv_quantization == "oscar" and not helper.split_mode:
        raise ValueError(
            "sample_kv_quantization='oscar' is sampling-only and requires "
            "local split mode so the training model remains the standard "
            "HF/PEFT model. Use [backend] devices = 'cuda:0,cuda:0' for "
            "an experimental same-GPU split, or provide separate CUDA "
            "devices for train and inference."
        )

    helper._cuda_allocator_metrics = configure_cuda_allocator(
        mode=helper.cuda_expandable_segments,
        train_device=helper.train_device,
        gradient_checkpointing_skip_last_n=(helper.gradient_checkpointing_skip_last_n),
    )
    helper._accelerator_metrics = install_cudnn_causal_conv1d_shim(
        enabled=helper.cudnn_causal_conv1d_shim,
    )
    helper._accelerator_metrics.update(accelerator_status())
    helper._accelerator_metrics.update(
        apply_liger_kernel_if_available(model_name, enabled=helper.liger_kernel)
    )

    print(f"Loading train model: {model_name} on {helper.train_device}...")
    helper.train_model, peft_config = helper._load_train_model(
        model_name,
        dtype,
        lora_rank,
        lora_alpha,
        lora_dropout,
    )
    helper._determinism_metrics.update(
        establish_strict_determinism(enabled=helper.strict_deterministic)
    )
    helper._determinism_metrics = add_model_attention_proof(
        helper._determinism_metrics,
        model=helper.train_model,
        requested_attention_kernel=helper.attention_kernel,
    )
    helper._accelerator_metrics.update(
        patch_qwen35_gated_delta_kernel(
            helper.train_model,
            mode=helper.qwen35_gated_delta_kernel,
            device=helper.train_device,
        )
    )
    helper._lora_frozen_a_tensor_count = freeze_lora_a(
        helper.train_model,
        enabled=helper.lora_freeze_a,
    )
    helper._lora_detach_input_hook_handles = detach_lora_input(
        helper.train_model,
        enabled=helper.lora_detach_input,
    )
    helper._lora_detach_input_hook_count = len(helper._lora_detach_input_hook_handles)
    helper._lora_fast_linear_patch_count = patch_fast_lora(
        helper.train_model,
        enabled=helper.lora_fast_linear,
        detach=helper.lora_detach_input,
        freeze=helper.lora_freeze_a,
    )
    helper._move_train_model_to_device()
    helper._lora_model_metrics = lora_metrics(
        helper.train_model,
        selected_layers=helper._lora_layers_to_transform,
        layers_pattern=helper.lora_layers_pattern,
        target_module_suffixes=helper.lora_target_module_suffixes,
        freeze_a_enabled=helper.lora_freeze_a,
        frozen_a_tensors=helper._lora_frozen_a_tensor_count,
        detach_input_enabled=helper.lora_detach_input,
        detach_input_hooks=helper._lora_detach_input_hook_count,
        fast_enabled=helper.lora_fast_linear,
        fast_patches=helper._lora_fast_linear_patch_count,
    )
    if hasattr(helper.train_model, "print_trainable_parameters"):
        helper.train_model.print_trainable_parameters()

    helper._gradient_checkpointing_layer_metrics = configure_gradient_checkpointing(
        helper.train_model,
        enabled=helper.gradient_checkpointing,
        use_reentrant=helper.gradient_checkpointing_use_reentrant,
        skip_last_n=helper.gradient_checkpointing_skip_last_n,
    )
    helper._train_future = None
    helper._pending_loss = 0.0
    helper._weight_snapshot = None
    helper._weights_dirty = False

    engine_options = {
        "model_name": model_name,
        "device": helper.infer_device,
        "dtype": dtype,
        "sample_use_cache": helper.sample_use_cache,
        "prefix_caching": helper.prefix_caching,
        "attention_kernel": helper.attention_kernel,
        "liger_kernel": helper.liger_kernel,
        "sample_kv_quantization": helper.sample_kv_quantization,
        "sample_oscar_options": helper.sample_oscar_options,
        "model_revision": helper.model_revision,
        "model_local_files_only": helper.model_local_files_only,
    }
    if helper._external_engine:
        helper.engine = engine_factory(
            engine_type=engine_type,
            peft_config=peft_config,
            inference_url=inference_url,
            **engine_options,
        )
    elif helper.split_mode:
        helper._train_executor = ThreadPoolExecutor(max_workers=1)
        helper.engine = engine_factory(
            engine_type="pytorch",
            peft_config=peft_config,
            **engine_options,
        )
        helper._do_initial_sync()
    else:
        helper.engine = engine_factory(
            engine_type="pytorch",
            peft_config=None,
            existing_model=helper.train_model,
            **engine_options,
        )

    helper.optimizer = torch.optim.AdamW(
        helper.train_model.parameters(),
        lr=4e-5,
        betas=(optim_beta1, optim_beta2),
        eps=optim_eps,
        weight_decay=0.0,
    )
    helper.scaler = torch.amp.GradScaler(
        enabled=helper.use_amp and helper.amp_dtype == torch.float16
    )
    print(
        "LocalTrainHelper ready "
        f"(engine={engine_type}, split_mode={helper.split_mode})."
    )
