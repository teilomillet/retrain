"""Local PyTorch backend: PEFT training + pluggable inference.

This backend provides:
- PyTorch model with PEFT LoRA for training (gradient computation)
- Pluggable InferenceEngine for sampling (PyTorch fallback or server-based)
- Adapter save/load for weight synchronization

GPU split mode (multi-GPU): separate devices for inference and training.
- engine on first device (sampling)
- train_model on last device (gradient updates)
- checkpoint() syncs LoRA weights train -> engine
"""

import gc
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model

from retrain.backends.determinism import (
    add_model_attention_proof,
    establish_strict_determinism,
    seed_strict_determinism,
)
from retrain.kernels.accelerators import (
    accelerator_status,
    apply_liger_kernel_if_available,
    from_pretrained_attention_kwargs,
    install_cudnn_causal_conv1d_shim,
    module_available,
)
from retrain.inference_engine import create_engine
from retrain.backends.local.lora import (
    DEFAULT_TARGET_SUFFIXES as _DEFAULT_LORA_TARGET_SUFFIXES,
    build_config as _build_lora_config,
    detach_input as _detach_lora_input,
    freeze_a as _freeze_lora_a,
    metrics as _lora_metrics,
    patch_fast as _patch_fast_lora,
)
from retrain.backends.local import batch as local_batch
from retrain.backends.local import device as local_device
from retrain.backends.local import logprobs as local_logprobs
from retrain.backends.local import loss as local_loss
from retrain.backends.local import metrics as local_metrics
from retrain.backends.local import sampling as local_sampling
from retrain.backends.local import sft as local_sft
from retrain.backends.local import state as local_state
from retrain.backends.local import sync as local_sync
from retrain.backends.local.steps import hybrid as local_hybrid_step
from retrain.backends.local.steps import rl as local_rl_step
from retrain.backends.local.steps import sft as local_sft_step
from retrain.backends.local.checkpointing import (
    configure_gradient_checkpointing,
)
from retrain.backends.local.memory import (
    configure_cuda_allocator,
    normalize_expandable_segments_mode,
)
from retrain.models.gemma4 import (
    forward_hidden_states_and_lm_head,
    forward_logits,
    parse_lora_target_module_suffixes,
)
from retrain.models.oscar_qwen3 import (
    OscarQwen3Options,
    normalize_sample_kv_quantization,
    oscar_options_from_mapping,
)
from retrain.models.qwen35 import patch_qwen35_gated_delta_kernel
from retrain.training.batch_digest import (
    local_rl_effective_rows_sha256,
    local_sft_effective_rows_sha256,
)


class LocalTrainHelper:
    """Local GPU backend: pluggable inference engine + PyTorch/PEFT training."""

    def __init__(self, model_name, adapter_path, devices, lora_rank=32,
                 engine_type="pytorch", inference_url="",
                 lora_alpha=0, lora_dropout=0.0,
                 optim_beta1=0.9, optim_beta2=0.95, optim_eps=1e-8,
                 clip_eps=0.0, clip_eps_high=0.0,
                 policy_loss_mode="standard",
                 kl_cov_percent=0.2,
                 kl_cov_coef=1.0,
                 clip_cov_ratio=0.0002,
                 clip_cov_min=1.0,
                 clip_cov_max=5.0,
                 train_microbatch_size=0,
                 train_sft_microbatch_token_budget=0,
                 train_logprob_chunk_size=0,
                 liger_kernel=True,
                 liger_fused_linear_ce=True,
                 cuda_empty_cache=False,
                 cuda_expandable_segments="auto",
                 strict_deterministic=False,
                 strict_deterministic_seed=-1,
                 sample_use_cache=True,
                 gradient_checkpointing=True,
                 gradient_checkpointing_use_reentrant="auto",
                 gradient_checkpointing_skip_last_n=0,
                 cudnn_causal_conv1d_shim=False,
                 attention_kernel="default",
                 prefix_caching=True,
                 train_selective_suffix_logits=False,
                 train_save_on_cpu=False,
                 train_save_on_cpu_pin_memory=True,
                 train_save_on_cpu_min_numel=0,
                 train_supervised_context_tokens=0,
                 train_unsloth_fused_ce="off",
                 train_unsloth_fused_ce_target_gb=0.0,
                 train_unsloth_fused_ce_torch_compile=True,
                 train_compile_selective_ce="off",
                 train_compile_selective_ce_min_tokens=128,
                 lora_target_modules="",
                 lora_layers_to_transform="",
                 lora_layers_pattern="layers",
                 lora_detach_input=False,
                 lora_fast_linear=False,
                 lora_freeze_a=False,
                 qwen35_gated_delta_kernel="auto",
                 sample_kv_quantization="off",
                 sample_oscar_repo="",
                 sample_oscar_bits=2,
                 sample_oscar_quant_mode="k-channel",
                 sample_oscar_group_size=0,
                 sample_oscar_kv_rotation="hadamard",
                 sample_oscar_kv_norm="1",
                 sample_oscar_residual_block_size=128,
                 sample_oscar_attn_implementation="sdpa",
                 trust_remote_code=False):
        self.strict_deterministic = bool(strict_deterministic)
        self._determinism_metrics = establish_strict_determinism(
            enabled=self.strict_deterministic
        )
        if self.strict_deterministic:
            self._determinism_metrics.update(
                seed_strict_determinism(int(strict_deterministic_seed))
            )
        self.adapter_path = adapter_path
        self.model_name = model_name
        self.engine_type = engine_type
        self.trust_remote_code = bool(trust_remote_code)
        self.clip_eps = clip_eps
        self.clip_eps_high = clip_eps_high
        self._clip_fraction = 0.0
        self.policy_loss_mode = policy_loss_mode
        self.kl_cov_percent = float(kl_cov_percent)
        self.kl_cov_coef = float(kl_cov_coef)
        self.clip_cov_ratio = float(clip_cov_ratio)
        self.clip_cov_min = float(clip_cov_min)
        self.clip_cov_max = float(clip_cov_max)
        self._policy_cov_fraction = 0.0
        self._policy_abs_kl = 0.0
        self.train_microbatch_size = max(0, int(train_microbatch_size))
        self.train_sft_microbatch_token_budget = max(
            0,
            int(train_sft_microbatch_token_budget),
        )
        self.train_logprob_chunk_size = max(0, int(train_logprob_chunk_size))
        self.liger_kernel = bool(liger_kernel)
        self.liger_fused_linear_ce = bool(liger_fused_linear_ce)
        self.attention_kernel = str(attention_kernel or "default")
        self.cuda_empty_cache = bool(cuda_empty_cache)
        self.cuda_expandable_segments = normalize_expandable_segments_mode(
            cuda_expandable_segments
        )
        self._cuda_allocator_metrics = {
            "enabled": 0,
            "env_preset": 0,
            "set_failed": 0,
        }
        self.sample_use_cache = bool(sample_use_cache)
        self.sample_kv_quantization = normalize_sample_kv_quantization(
            sample_kv_quantization
        )
        self.sample_oscar_options = OscarQwen3Options(
            repo=str(sample_oscar_repo or ""),
            bits=int(sample_oscar_bits),
            quant_mode=str(sample_oscar_quant_mode or "k-channel"),
            group_size=int(sample_oscar_group_size),
            kv_rotation=str(sample_oscar_kv_rotation or "hadamard"),
            kv_norm=str(sample_oscar_kv_norm or "1"),
            residual_block_size=int(sample_oscar_residual_block_size),
            attn_implementation=str(sample_oscar_attn_implementation or "sdpa"),
        )
        # Reuse the central validator for range checks.
        self.sample_oscar_options = oscar_options_from_mapping(
            {
                "sample_oscar_repo": self.sample_oscar_options.repo,
                "sample_oscar_bits": self.sample_oscar_options.bits,
                "sample_oscar_quant_mode": self.sample_oscar_options.quant_mode,
                "sample_oscar_group_size": self.sample_oscar_options.group_size,
                "sample_oscar_kv_rotation": self.sample_oscar_options.kv_rotation,
                "sample_oscar_kv_norm": self.sample_oscar_options.kv_norm,
                "sample_oscar_residual_block_size": (
                    self.sample_oscar_options.residual_block_size
                ),
                "sample_oscar_attn_implementation": (
                    self.sample_oscar_options.attn_implementation
                ),
            }
        )
        if self.sample_kv_quantization == "oscar":
            if engine_type != "pytorch":
                raise ValueError(
                    "sample_kv_quantization='oscar' is only supported with "
                    "inference.engine='pytorch'."
                )
            if not self.sample_use_cache:
                raise ValueError(
                    "sample_kv_quantization='oscar' requires sample_use_cache=true."
                )
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.gradient_checkpointing_use_reentrant = str(
            gradient_checkpointing_use_reentrant or "auto"
        ).lower()
        self.gradient_checkpointing_skip_last_n = max(
            0,
            int(gradient_checkpointing_skip_last_n),
        )
        self._gradient_checkpointing_layer_metrics = {
            "total": 0,
            "enabled": 0,
            "skipped_last_n": 0,
        }
        self.cudnn_causal_conv1d_shim = bool(cudnn_causal_conv1d_shim)
        self.prefix_caching = bool(prefix_caching)
        self.train_selective_suffix_logits = bool(train_selective_suffix_logits)
        self.train_save_on_cpu = bool(train_save_on_cpu)
        self.train_save_on_cpu_pin_memory = bool(train_save_on_cpu_pin_memory)
        self.train_save_on_cpu_min_numel = max(0, int(train_save_on_cpu_min_numel))
        self.train_supervised_context_tokens = max(
            0,
            int(train_supervised_context_tokens),
        )
        self.train_unsloth_fused_ce = self._normalize_unsloth_fused_ce_mode(
            train_unsloth_fused_ce
        )
        self.train_unsloth_fused_ce_target_gb = max(
            0.0,
            float(train_unsloth_fused_ce_target_gb),
        )
        self.train_unsloth_fused_ce_torch_compile = bool(
            train_unsloth_fused_ce_torch_compile
        )
        self.train_compile_selective_ce = self._normalize_compile_mode(
            train_compile_selective_ce,
            option_name="train_compile_selective_ce",
        )
        self.train_compile_selective_ce_min_tokens = max(
            0,
            int(train_compile_selective_ce_min_tokens),
        )
        self.lora_target_module_suffixes = parse_lora_target_module_suffixes(
            lora_target_modules
        )
        self.lora_layers_to_transform_spec = str(lora_layers_to_transform or "")
        self.lora_layers_pattern = str(lora_layers_pattern or "layers")
        self.lora_detach_input = bool(lora_detach_input)
        self.lora_fast_linear = bool(lora_fast_linear)
        self.lora_freeze_a = bool(lora_freeze_a)
        self.qwen35_gated_delta_kernel = str(qwen35_gated_delta_kernel or "auto")
        self._lora_detach_input_hook_handles = []
        self._lora_detach_input_hook_count = 0
        self._lora_fast_linear_patch_count = 0
        self._lora_frozen_a_tensor_count = 0
        self._lora_layers_to_transform: list[int] | None = None
        self._lora_model_metrics: dict[str, float | int | str] = {}
        self._last_sample_metrics: dict[str, float | int] = {}
        self._last_train_metrics: dict[str, float | int | str] = {}
        self._last_sync_metrics: dict[str, float | int] = {}
        self._last_context_crop_metrics: dict[str, float | int] = {}
        self._last_effective_optimizer_rows_sha256 = ""
        self._train_logits_to_keep_supported: bool | None = None
        self._selective_logprob_path_counts: dict[str, int] = {}
        self._loss_path_counts: dict[str, int] = {}
        self._unsloth_fused_ce_available: bool | None = None
        self._unsloth_fused_ce_unavailable_reason = ""
        self._unsloth_fused_ce_fallback_reason = ""
        self._last_unsloth_fused_ce_effective_target_gb = 0.0
        self._compiled_selective_ce_fallback_reason = ""
        self._compiled_selective_ce_available: bool | None = None

        device_plan = local_device.resolve(devices, engine_type)
        self.infer_device = device_plan.infer_device
        self.train_device = device_plan.train_device
        self.split_mode = device_plan.split_mode
        self._server_engine = device_plan.server_engine
        self._external_engine = device_plan.external_engine
        self.use_amp = device_plan.use_amp
        dtype = device_plan.dtype
        self.amp_dtype = dtype
        if self.sample_kv_quantization == "oscar" and not self.split_mode:
            raise ValueError(
                "sample_kv_quantization='oscar' is sampling-only and requires "
                "local split mode so the training model remains the standard "
                "HF/PEFT model. Use [backend] devices = 'cuda:0,cuda:0' for "
                "an experimental same-GPU split, or provide separate CUDA "
                "devices for train and inference."
            )

        # Configure the allocator before the first training-model allocation so
        # every training segment can use the expandable policy.
        self._cuda_allocator_metrics = configure_cuda_allocator(
            mode=self.cuda_expandable_segments,
            train_device=self.train_device,
            gradient_checkpointing_skip_last_n=self.gradient_checkpointing_skip_last_n,
        )

        self._accelerator_metrics = install_cudnn_causal_conv1d_shim(
            enabled=self.cudnn_causal_conv1d_shim,
        )
        self._accelerator_metrics.update(accelerator_status())
        self._accelerator_metrics.update(
            apply_liger_kernel_if_available(
                model_name,
                enabled=self.liger_kernel,
            )
        )

        # Create train model
        print(f"Loading train model: {model_name} on {self.train_device}...")
        self.train_model, peft_config = self._load_train_model(
            model_name,
            dtype,
            lora_rank,
            lora_alpha,
            lora_dropout,
        )
        # Re-read the controls after model/CUDA construction. Strict mode only
        # survives this boundary if the same verified process preflight ran
        # before CUDA initialization.
        self._determinism_metrics.update(
            establish_strict_determinism(enabled=self.strict_deterministic)
        )
        self._determinism_metrics = add_model_attention_proof(
            self._determinism_metrics,
            model=self.train_model,
            requested_attention_kernel=self.attention_kernel,
        )
        self._accelerator_metrics.update(
            patch_qwen35_gated_delta_kernel(
                self.train_model,
                mode=self.qwen35_gated_delta_kernel,
                device=self.train_device,
            )
        )
        self._lora_frozen_a_tensor_count = _freeze_lora_a(
            self.train_model,
            enabled=self.lora_freeze_a,
        )
        self._lora_detach_input_hook_handles = _detach_lora_input(
            self.train_model,
            enabled=self.lora_detach_input,
        )
        self._lora_detach_input_hook_count = len(
            self._lora_detach_input_hook_handles
        )
        self._lora_fast_linear_patch_count = _patch_fast_lora(
            self.train_model,
            enabled=self.lora_fast_linear,
            detach=self.lora_detach_input,
            freeze=self.lora_freeze_a,
        )
        self._move_train_model_to_device()
        self._lora_model_metrics = _lora_metrics(
            self.train_model,
            selected_layers=self._lora_layers_to_transform,
            layers_pattern=self.lora_layers_pattern,
            target_module_suffixes=self.lora_target_module_suffixes,
            freeze_a_enabled=self.lora_freeze_a,
            frozen_a_tensors=self._lora_frozen_a_tensor_count,
            detach_input_enabled=self.lora_detach_input,
            detach_input_hooks=self._lora_detach_input_hook_count,
            fast_enabled=self.lora_fast_linear,
            fast_patches=self._lora_fast_linear_patch_count,
        )
        if hasattr(self.train_model, "print_trainable_parameters"):
            self.train_model.print_trainable_parameters()

        self._gradient_checkpointing_layer_metrics = configure_gradient_checkpointing(
            self.train_model,
            enabled=self.gradient_checkpointing,
            use_reentrant=self.gradient_checkpointing_use_reentrant,
            skip_last_n=self.gradient_checkpointing_skip_last_n,
        )

        # Async pipeline state — initialized before first _sync_lora_weights call
        self._train_future = None       # Future from last submitted training
        self._pending_loss = 0.0        # Loss from last completed training
        self._weight_snapshot = None    # LoRA weight snapshot for safe cross-thread sync
        self._weights_dirty = False     # Track if weights changed since last server sync

        # Create inference engine
        if self._external_engine:
            # External engine (MAX in-process or server-based) — manages its own model
            self.engine = create_engine(
                engine_type=engine_type,
                model_name=model_name,
                device=self.infer_device,
                peft_config=peft_config,
                dtype=dtype,
                inference_url=inference_url,
                sample_use_cache=self.sample_use_cache,
                prefix_caching=self.prefix_caching,
                attention_kernel=self.attention_kernel,
                liger_kernel=self.liger_kernel,
                sample_kv_quantization=self.sample_kv_quantization,
                sample_oscar_options=self.sample_oscar_options,
            )
        elif self.split_mode:
            self._train_executor = ThreadPoolExecutor(max_workers=1)

            # PyTorch engine on separate inference device
            self.engine = create_engine(
                engine_type="pytorch",
                model_name=model_name,
                device=self.infer_device,
                peft_config=peft_config,
                dtype=dtype,
                sample_use_cache=self.sample_use_cache,
                prefix_caching=self.prefix_caching,
                attention_kernel=self.attention_kernel,
                liger_kernel=self.liger_kernel,
                sample_kv_quantization=self.sample_kv_quantization,
                sample_oscar_options=self.sample_oscar_options,
            )
            # Initial sync: copy LoRA weights from train to infer
            self._do_initial_sync()
        else:
            # Single-model mode: engine wraps the train model directly
            # We use a thin wrapper that points to train_model
            self.engine = create_engine(
                engine_type="pytorch",
                model_name=model_name,
                device=self.infer_device,
                peft_config=None,
                dtype=dtype,
                existing_model=self.train_model,
                sample_use_cache=self.sample_use_cache,
                prefix_caching=self.prefix_caching,
                attention_kernel=self.attention_kernel,
                liger_kernel=self.liger_kernel,
                sample_kv_quantization=self.sample_kv_quantization,
                sample_oscar_options=self.sample_oscar_options,
            )

        # Optimizer (only for train_model)
        self.optimizer = torch.optim.AdamW(
            self.train_model.parameters(),
            lr=4e-5,
            betas=(optim_beta1, optim_beta2),
            eps=optim_eps,
            weight_decay=0.0,
        )

        # BF16 autocast does not need loss scaling. Enabling GradScaler for
        # BF16 can silently skip every optimizer step after scaled-grad overflow.
        self.scaler = torch.amp.GradScaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        print(f"LocalTrainHelper ready (engine={engine_type}, split_mode={self.split_mode}).")

    def _autocast_context(self):
        device_type = self.train_device.split(":")[0]
        return torch.amp.autocast(
            device_type=device_type,
            enabled=self.use_amp,
            dtype=self.amp_dtype if self.use_amp else None,
        )

    def _load_train_model(self, model_name, dtype, lora_rank, lora_alpha, lora_dropout):
        model_kwargs = from_pretrained_attention_kwargs(self.attention_kernel)
        base_train = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
            **model_kwargs,
        )
        target_module_suffixes = getattr(
            self,
            "lora_target_module_suffixes",
            _DEFAULT_LORA_TARGET_SUFFIXES,
        )
        peft_build = _build_lora_config(
            base_train,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            layers_spec=self.lora_layers_to_transform_spec,
            layers_pattern=self.lora_layers_pattern,
            target_module_suffixes=target_module_suffixes,
        )
        self._lora_layers_to_transform = peft_build.selected_layers
        return get_peft_model(base_train, peft_build.config), peft_build.config

    def _move_train_model_to_device(self):
        self.train_model.to(self.train_device)

    def _liger_fused_linear_ce_loss(self):
        if not (
            getattr(self, "liger_fused_linear_ce", False)
            and module_available("liger_kernel")
        ):
            return None
        loss = getattr(self, "_liger_fused_ce_loss", None)
        if loss is not None:
            return loss
        try:
            from liger_kernel.transformers import (  # type: ignore[unresolved-import]
                LigerFusedLinearCrossEntropyLoss,
            )

            loss = LigerFusedLinearCrossEntropyLoss(reduction="none")
        except Exception:  # Optional accelerator path.
            return None
        self._liger_fused_ce_loss = loss
        return loss

    def _do_initial_sync(self):
        local_sync.initial_sync(self)

    def _sync_lora_weights(self):
        local_sync.sync_lora_weights(self)

    def checkpoint(self, name):
        """Prepare for sampling by syncing LoRA weights from snapshot -> engine.

        Non-blocking: if _train_future is done, collects loss into _pending_loss.
        Syncs from latest weight snapshot. Does NOT block-wait for in-flight
        training — this enables overlap between sample(N) and train(N-1).
        """
        if self._train_future is not None and self._train_future.done():
            self._pending_loss = self._train_future.result()
            self._train_future = None
        self._sync_lora_weights()
        self._clear_inference_prefix_cache()

    def _shared_model_sampling_cache_context(self):
        return local_sampling.shared_model_cache_context(self)

    def _sample_groups(self, prompt_ids_list, num_samples, max_tokens,
                       temperature, top_p, *, compute_entropy):
        return local_sampling.sample_groups(
            self,
            prompt_ids_list=prompt_ids_list,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            compute_entropy=compute_entropy,
        )

    def sample(self, prompt_ids_list, num_samples, max_tokens, temperature, top_p):
        """Generate completions with per-token logprobs.

        Delegates to the configured InferenceEngine, then converts
        SampleResult objects to the plain (token_ids, logprobs) tuples
        the trainer consumes.

        Args:
            prompt_ids_list: List of lists of token IDs (one per prompt).
            num_samples: Number of completions per prompt.
            max_tokens: Maximum new tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            List of lists of (token_ids, logprobs) tuples.
        """
        engine_results = self._sample_groups(
            prompt_ids_list, num_samples, max_tokens, temperature, top_p,
            compute_entropy=False,
        )
        return [
            [(sr.token_ids, sr.logprobs) for sr in group]
            for group in engine_results
        ]

    def sample_with_entropy(self, prompt_ids_list, num_samples, max_tokens,
                            temperature, top_p):
        """Generate completions with per-token logprobs and Shannon entropy.

        Like sample(), but requests per-token entropy from the engine and
        returns 3-tuples (token_ids, logprobs, token_entropies).
        """
        engine_results = self._sample_groups(
            prompt_ids_list, num_samples, max_tokens, temperature, top_p,
            compute_entropy=True,
        )
        return [
            [(sr.token_ids, sr.logprobs, sr.token_entropies) for sr in group]
            for group in engine_results
        ]

    def runtime_metrics(self):
        """Expose optional engine-level runtime counters to the trainer."""
        return local_metrics.runtime_metrics(self)

    def _snapshot_lora_weights_if_needed(self) -> float:
        return local_sync.snapshot_lora_weights_if_needed(self)

    def _clear_inference_prefix_cache(self):
        local_sync.clear_inference_prefix_cache(self)

    def shutdown(self) -> None:
        """Release model, optimizer, executor, and CUDA allocator state."""
        future = getattr(self, "_train_future", None)
        if future is not None:
            try:
                future.result()
            except Exception:  # Cleanup should not mask failures.
                pass

        executor = getattr(self, "_train_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=True)

        engine = getattr(self, "engine", None)
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:  # Best-effort cleanup.
                pass

        for name in (
            "engine",
            "train_model",
            "optimizer",
            "scaler",
            "_weight_snapshot",
            "_train_future",
        ):
            if hasattr(self, name):
                try:
                    delattr(self, name)
                except Exception:  # Best-effort cleanup.
                    pass

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:  # CUDA may be recovering from OOM.
                pass
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:  # Cleanup must not raise.
                pass

    @staticmethod
    def _normalize_compile_mode(raw, *, option_name: str) -> str:
        return local_loss.normalize_compile_mode(raw, option_name=option_name)

    @staticmethod
    def _normalize_unsloth_fused_ce_mode(raw) -> str:
        return local_loss.normalize_unsloth_fused_ce_mode(raw)

    def _reject_compiled_selective_ce(self, reason: str):
        return local_loss.reject_compiled_selective_ce(self, reason)

    def _compiled_selective_ce_logprobs(self, selected_hidden, lm_head, target_ids):
        return local_loss.compiled_selective_ce_logprobs(
            self,
            selected_hidden,
            lm_head,
            target_ids,
        )

    def _unsloth_fused_ce_loss(self):
        return local_loss.unsloth_fused_ce_loss(self)

    def _effective_unsloth_fused_ce_target_gb(self) -> float:
        return local_loss.effective_unsloth_fused_ce_target_gb(self)

    @staticmethod
    def _is_cuda_oom_exception(exc: BaseException) -> bool:
        return local_loss.is_cuda_oom_exception(exc)

    def _disable_unsloth_fused_ce_after_runtime_failure(
        self,
        exc: BaseException,
    ) -> str:
        return local_loss.disable_unsloth_fused_ce_after_runtime_failure(self, exc)

    @staticmethod
    def _constant_positive_weight(weights, target_mask):
        return local_loss.constant_positive_weight(weights, target_mask)

    def _maybe_compute_unsloth_fused_sft_loss(
        self,
        input_ids,
        attention_mask,
        weights,
        target_mask,
        token_count,
    ):
        return local_loss.maybe_compute_unsloth_fused_sft_loss(
            self,
            input_ids,
            attention_mask,
            weights,
            target_mask,
            token_count,
            hidden_and_head_fn=forward_hidden_states_and_lm_head,
        )

    def _supports_train_logits_to_keep(self, input_ids, attention_mask) -> bool:
        return local_logprobs.supports_train_logits_to_keep(
            self,
            input_ids,
            attention_mask,
        )

    def _selective_suffix_token_logprobs(self, input_ids, attention_mask, target_mask):
        return local_logprobs.selective_suffix_token_logprobs(
            self,
            input_ids,
            attention_mask,
            target_mask,
        )

    def _selective_hidden_token_logprobs(self, input_ids, attention_mask, target_mask):
        return local_logprobs.selective_hidden_token_logprobs(
            self,
            input_ids,
            attention_mask,
            target_mask,
            hidden_and_head_fn=forward_hidden_states_and_lm_head,
        )

    def _shifted_token_logprobs(self, input_ids, attention_mask, target_mask=None):
        return local_logprobs.shifted_token_logprobs(
            self,
            input_ids,
            attention_mask,
            target_mask=target_mask,
            hidden_and_head_fn=forward_hidden_states_and_lm_head,
            forward_logits_fn=forward_logits,
        )

    def _compute_train_loss(self, input_ids, old_logprobs, advantages, attention_mask):
        return local_rl_step.compute_loss(
            self,
            input_ids,
            old_logprobs,
            advantages,
            attention_mask,
        )

    def _compute_sft_loss(self, input_ids, advantages, attention_mask):
        return local_sft_step.compute_loss(self, input_ids, advantages, attention_mask)

    def _backward_sft_microbatch(
        self,
        input_ids,
        advantages,
        attention_mask,
        total_tokens,
    ):
        return local_sft_step.backward_microbatch(
            self,
            input_ids,
            advantages,
            attention_mask,
            total_tokens,
        )

    def _do_train_impl(self, input_ids, old_logprobs, advantages, attention_mask):
        return local_rl_step.run(
            self,
            input_ids,
            old_logprobs,
            advantages,
            attention_mask,
        )

    def _do_train_sequence_impl(
        self,
        all_tokens,
        all_logprobs,
        all_advantages,
    ):
        return local_rl_step.run_sequence(
            self,
            all_tokens,
            all_logprobs,
            all_advantages,
        )

    def _do_sft_impl(self, input_ids, advantages, attention_mask):
        return local_sft_step.run_padded(self, input_ids, advantages, attention_mask)

    def _do_sft_sequence_impl(self, all_tokens, all_advantages):
        return local_sft_step.run_sequence(self, all_tokens, all_advantages)

    def _do_hybrid_mask_impl(
        self,
        input_ids,
        old_logprobs,
        advantages,
        attention_mask,
        echo_advantages,
        echo_full_observation_counts,
        echo_loss_fn,
        echo_rollout_denominator=0,
    ):
        return local_hybrid_step.run(
            self,
            input_ids,
            old_logprobs,
            advantages,
            attention_mask,
            echo_advantages,
            echo_full_observation_counts,
            echo_loss_fn,
            echo_rollout_denominator,
        )

    def _do_hybrid_mask_sequence_impl(
        self,
        all_tokens,
        all_logprobs,
        all_advantages,
        echo_advantages,
        echo_full_observation_counts,
        echo_loss_fn,
        echo_rollout_denominator=0,
    ):
        return local_hybrid_step.run_sequence(
            self,
            all_tokens,
            all_logprobs,
            all_advantages,
            echo_advantages,
            echo_full_observation_counts,
            echo_loss_fn,
            echo_rollout_denominator,
        )

    def _maybe_crop_supervised_context(
        self,
        all_tokens,
        all_logprobs=None,
        all_advantages=None,
        echo_advantages=None,
    ):
        context_tokens = int(getattr(self, "train_supervised_context_tokens", 0))
        enabled = (
            context_tokens > 0
            and getattr(self, "train_selective_suffix_logits", False)
        )
        result = local_sft.crop_supervised_context(
            all_tokens,
            all_logprobs=all_logprobs,
            all_advantages=all_advantages,
            echo_advantages=echo_advantages,
            context_tokens=context_tokens,
            enabled=enabled,
        )
        self._last_context_crop_metrics = result.metrics
        return result.tokens, result.logprobs, result.advantages, result.echo_advantages

    def _clear_effective_optimizer_rows(self) -> None:
        """Prevent a failed or empty step from emitting a previous row digest."""

        self._last_effective_optimizer_rows_sha256 = ""

    def _record_effective_rl_rows(
        self,
        all_tokens,
        all_logprobs,
        all_advantages,
        *,
        echo_advantages=None,
        echo_full_observation_counts=None,
        echo_rollout_denominator=None,
    ) -> None:
        """Record post-crop RL rows without claiming optimizer equivalence."""

        self._last_effective_optimizer_rows_sha256 = (
            local_rl_effective_rows_sha256(
                all_tokens,
                all_logprobs,
                all_advantages,
                echo_observation_masks=echo_advantages,
                echo_full_observation_counts=echo_full_observation_counts,
                echo_rollout_denominator=echo_rollout_denominator,
            )
        )

    def _record_effective_sft_rows(self, all_tokens, all_advantages) -> None:
        """Record post-crop cross-entropy SFT rows and target weights."""

        self._last_effective_optimizer_rows_sha256 = (
            local_sft_effective_rows_sha256(all_tokens, all_advantages)
        )

    def train_step(self, all_tokens, all_logprobs, all_advantages, lr, weight_decay):
        """Run one training step with importance sampling loss.

        Split mode (async): waits for any previous training future (back pressure),
        prepares tensors on train_device, submits _do_train_impl to the executor,
        and returns _pending_loss (loss from the previously completed step).
        Step 0 returns 0.0 since no training has completed yet.

        Single mode: unchanged synchronous path returning current step's loss.

        Args:
            all_tokens: List of full token sequences (prompt + completion).
            all_logprobs: List of per-token logprobs (0-padded for prompt).
            all_advantages: List of per-token advantages (0-padded for prompt).
            lr: Learning rate.
            weight_decay: Weight decay coefficient.

        Returns:
            Mean loss (from previous step in split mode, current step otherwise).
        """
        self._clear_effective_optimizer_rows()
        n = len(all_tokens)
        if n == 0:
            return 0.0

        all_tokens, all_logprobs, all_advantages, _ = (
            self._maybe_crop_supervised_context(
                all_tokens,
                all_logprobs,
                all_advantages,
            )
        )
        self._record_effective_rl_rows(
            all_tokens,
            all_logprobs,
            all_advantages,
        )

        self._clear_inference_prefix_cache()

        # Update optimizer hyperparameters
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = weight_decay

        microbatch_size = self.train_microbatch_size or n
        if microbatch_size < n:
            if self.split_mode:
                if self._train_future is not None:
                    self._pending_loss = self._train_future.result()
                    self._train_future = None
                # The previous tensor path snapshotted caller-owned rows during
                # padding. Preserve that isolation for the asynchronous path.
                tokens_snapshot = tuple(tuple(row) for row in all_tokens)
                logprobs_snapshot = tuple(tuple(row) for row in all_logprobs)
                advantages_snapshot = tuple(tuple(row) for row in all_advantages)
                self._train_future = self._train_executor.submit(
                    self._do_train_sequence_impl,
                    tokens_snapshot,
                    logprobs_snapshot,
                    advantages_snapshot,
                )
                return self._pending_loss
            return self._do_train_sequence_impl(
                all_tokens,
                all_logprobs,
                all_advantages,
            )

        batch = local_batch.policy(
            all_tokens,
            all_logprobs,
            all_advantages,
            device=self.train_device,
        )

        if self.split_mode:
            # Async path: wait for previous training to finish (back pressure)
            if self._train_future is not None:
                self._pending_loss = self._train_future.result()
                self._train_future = None

            # Submit training to background thread
            self._train_future = self._train_executor.submit(
                self._do_train_impl,
                batch.input_ids,
                batch.old_logprobs,
                batch.advantages,
                batch.attention_mask,
            )

            # Return loss from the previously completed training step
            return self._pending_loss
        else:
            # Synchronous path: run training inline, return current loss
            return self._do_train_impl(
                batch.input_ids,
                batch.old_logprobs,
                batch.advantages,
                batch.attention_mask,
            )

    def sft_train_step(self, all_tokens, all_advantages, lr, weight_decay):
        """Run one SFT/ECHO update.

        ``importance_sampling`` preserves the historical warmup behavior.
        ``cross_entropy`` gives the direct next-token world-modeling objective
        ECHO needs for prompt-side environment/tool tokens.
        """
        self._clear_effective_optimizer_rows()
        loss_fn = getattr(self, "sft_loss_fn", "importance_sampling")
        if loss_fn == "importance_sampling":
            all_logprobs = [[0.0] * len(tokens) for tokens in all_tokens]
            return self.train_step(
                all_tokens,
                all_logprobs,
                all_advantages,
                lr,
                weight_decay,
            )
        if loss_fn != "cross_entropy":
            raise ValueError(
                "sft_loss_fn must be 'importance_sampling' or 'cross_entropy'."
            )

        n = len(all_tokens)
        if n == 0:
            return 0.0

        all_tokens, _, all_advantages, _ = self._maybe_crop_supervised_context(
            all_tokens,
            all_advantages=all_advantages,
        )
        self._record_effective_sft_rows(all_tokens, all_advantages)

        self._clear_inference_prefix_cache()

        if self._train_future is not None:
            self._pending_loss = self._train_future.result()
            self._train_future = None

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = weight_decay

        microbatch_size = self.train_microbatch_size or n
        token_budget = getattr(self, "train_sft_microbatch_token_budget", 0)
        if microbatch_size < n or token_budget > 0:
            return self._do_sft_sequence_impl(all_tokens, all_advantages)

        batch = local_batch.sft(
            all_tokens,
            all_advantages,
            device=self.train_device,
        )
        return self._do_sft_impl(
            batch.input_ids,
            batch.advantages,
            batch.attention_mask,
        )

    def train_step_with_echo_masks(
        self,
        all_tokens,
        all_logprobs,
        all_advantages,
        echo_advantages,
        echo_full_observation_counts,
        echo_loss_fn,
        lr,
        weight_decay,
        echo_rollout_denominator=0,
    ):
        """Run RL + ECHO masks over the same rollout rows."""
        self._clear_effective_optimizer_rows()
        if not echo_advantages:
            return self.train_step(
                all_tokens,
                all_logprobs,
                all_advantages,
                lr,
                weight_decay,
            ), 0.0

        if echo_loss_fn != "cross_entropy":
            raise ValueError(
                "echo_loss_fn must be 'cross_entropy' for paper-faithful ECHO."
            )
        if len(echo_advantages) != len(all_tokens):
            raise ValueError("echo_advantages must have one row per training datum.")
        if len(echo_full_observation_counts) != len(all_tokens):
            raise ValueError(
                "echo_full_observation_counts must have one value per training datum."
            )

        all_tokens, all_logprobs, all_advantages, echo_advantages = (
            self._maybe_crop_supervised_context(
                all_tokens,
                all_logprobs,
                all_advantages,
                echo_advantages,
            )
        )
        effective_echo_denominator = echo_rollout_denominator or len(all_tokens)
        self._record_effective_rl_rows(
            all_tokens,
            all_logprobs,
            all_advantages,
            echo_advantages=echo_advantages,
            echo_full_observation_counts=echo_full_observation_counts,
            echo_rollout_denominator=effective_echo_denominator,
        )

        self._clear_inference_prefix_cache()

        if self._train_future is not None:
            self._pending_loss = self._train_future.result()
            self._train_future = None

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = weight_decay

        microbatch_size = self.train_microbatch_size or len(all_tokens)
        if microbatch_size < len(all_tokens):
            return self._do_hybrid_mask_sequence_impl(
                all_tokens,
                all_logprobs,
                all_advantages,
                echo_advantages,
                echo_full_observation_counts,
                echo_loss_fn,
                effective_echo_denominator,
            )

        batch = local_batch.echo(
            all_tokens,
            all_logprobs,
            all_advantages,
            echo_advantages,
            echo_full_observation_counts,
            device=self.train_device,
        )

        return self._do_hybrid_mask_impl(
            batch.input_ids,
            batch.old_logprobs,
            batch.advantages,
            batch.attention_mask,
            batch.echo_advantages,
            batch.echo_counts,
            echo_loss_fn,
            effective_echo_denominator,
        )

    def load_state(self, name):
        """Load adapter weights from a saved checkpoint or adapter directory.

        Restores LoRA adapter weights into the train model and re-syncs
        to the inference engine. Optimizer state (Adam momentum/variance)
        is NOT restored — training will re-warm the optimizer.

        Args:
            name: Checkpoint name under adapter_path, or a direct path to a
                PEFT adapter directory containing adapter_model.*.
        """
        save_dir = local_state.load_into_model(
            self.train_model,
            adapter_path=self.adapter_path,
            name=name,
            train_device=self.train_device,
        )

        # Re-sync weights to inference engine
        if self.split_mode or self._external_engine:
            self._weight_snapshot = local_state.lora_state_dict(
                self.train_model,
                clone=True,
            )
            self._weights_dirty = True

        print(f"Loaded adapter checkpoint: {save_dir} (optimizer state not restored)")

    def save_adapter(self, path, name) -> str:
        """Save LoRA adapter to disk.

        Flushes any pending async training before saving to ensure
        the saved weights include the latest completed training step.

        Args:
            path: Base directory for adapter storage.
            name: Checkpoint name (creates subdirectory).

        Returns:
            Path to the saved adapter directory.
        """
        # Flush pending training before saving
        if self._train_future is not None:
            self._pending_loss = self._train_future.result()
            self._train_future = None

        save_dir = local_state.save_model(self.train_model, path=path, name=name)
        print(f"Adapter saved to {save_dir}")
        return save_dir
