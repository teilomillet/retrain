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

from concurrent.futures import Future, ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel, get_peft_model

from retrain.backends.determinism import establish_strict_determinism
from retrain.backends.determinism import seed_strict_determinism
from retrain.kernels.accelerators import (
    from_pretrained_attention_kwargs,
    module_available,
)
from retrain.backends.local import bootstrap as local_bootstrap
from retrain.backends.local.lora import (
    DEFAULT_TARGET_SUFFIXES as _DEFAULT_LORA_TARGET_SUFFIXES,
    build_config as _build_lora_config,
)
from retrain.backends.local import logprobs as local_logprobs
from retrain.backends.local import lifecycle as local_lifecycle
from retrain.backends.local import loss as local_loss
from retrain.backends.local import metrics as local_metrics
from retrain.backends.local import sampling as local_sampling
from retrain.backends.local import sync as local_sync
from retrain.backends.local.steps import hybrid as local_hybrid_step
from retrain.backends.local.steps import rl as local_rl_step
from retrain.backends.local.steps import sft as local_sft_step
from retrain.backends.local.steps import dispatch as local_step_dispatch
from retrain.backends.local.memory import (
    normalize_expandable_segments_mode,
)
from retrain.inference_engine import InferenceEngine, create_engine
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


class LocalTrainHelper:
    """Local GPU backend: pluggable inference engine + PyTorch/PEFT training."""

    # Runtime resources are constructed by local_bootstrap.initialize().  Keep
    # their ownership explicit here so static analysis sees the same contract
    # that Python establishes during __init__.
    infer_device: str
    train_device: str
    split_mode: bool
    _server_engine: bool
    _external_engine: bool
    use_amp: bool
    amp_dtype: torch.dtype
    _accelerator_metrics: dict[str, object]
    train_model: PeftModel
    _train_future: Future[float] | None
    _pending_loss: float
    _weight_snapshot: dict[str, torch.Tensor] | None
    _weights_dirty: bool
    engine: InferenceEngine
    _train_executor: ThreadPoolExecutor
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler

    def __init__(
        self,
        model_name,
        adapter_path,
        devices,
        lora_rank=32,
        engine_type="pytorch",
        inference_url="",
        model_revision="",
        model_local_files_only=False,
        lora_alpha=0,
        lora_dropout=0.0,
        optim_beta1=0.9,
        optim_beta2=0.95,
        optim_eps=1e-8,
        clip_eps=0.0,
        clip_eps_high=0.0,
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
        trust_remote_code=False,
    ):
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
        self.model_revision = str(model_revision or "")
        self.model_local_files_only = bool(model_local_files_only)
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

        local_bootstrap.initialize(
            self,
            model_name=model_name,
            devices=devices,
            engine_type=engine_type,
            inference_url=inference_url,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            optim_beta1=optim_beta1,
            optim_beta2=optim_beta2,
            optim_eps=optim_eps,
            engine_factory=create_engine,
        )

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
            revision=self.model_revision or None,
            local_files_only=self.model_local_files_only,
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
        """Prepare the latest completed weights for sampling."""
        local_lifecycle.checkpoint(self, name)

    def _shared_model_sampling_cache_context(self):
        return local_sampling.shared_model_cache_context(self)

    def _sample_groups(
        self,
        prompt_ids_list,
        num_samples,
        max_tokens,
        temperature,
        top_p,
        *,
        compute_entropy,
    ):
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
            prompt_ids_list,
            num_samples,
            max_tokens,
            temperature,
            top_p,
            compute_entropy=False,
        )
        return [
            [(sr.token_ids, sr.logprobs) for sr in group] for group in engine_results
        ]

    def sample_with_finish_reason(
        self, prompt_ids_list, num_samples, max_tokens, temperature, top_p
    ):
        """Generate completions while retaining per-sample stop metadata."""
        engine_results = self._sample_groups(
            prompt_ids_list,
            num_samples,
            max_tokens,
            temperature,
            top_p,
            compute_entropy=False,
        )
        return [
            [
                (sr.token_ids, sr.logprobs, getattr(sr, "finish_reason", None))
                for sr in group
            ]
            for group in engine_results
        ]

    def sample_with_entropy(
        self, prompt_ids_list, num_samples, max_tokens, temperature, top_p
    ):
        """Generate completions with per-token logprobs and Shannon entropy.

        Like sample(), but requests per-token entropy from the engine and
        returns 3-tuples (token_ids, logprobs, token_entropies).
        """
        engine_results = self._sample_groups(
            prompt_ids_list,
            num_samples,
            max_tokens,
            temperature,
            top_p,
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
        local_lifecycle.shutdown(self)

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
        return local_step_dispatch.crop_supervised_context(
            self,
            all_tokens,
            all_logprobs,
            all_advantages,
            echo_advantages,
        )

    def _clear_effective_optimizer_rows(self) -> None:
        """Prevent a failed or empty step from emitting a previous row digest."""

        local_step_dispatch.clear_effective_optimizer_rows(self)

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

        local_step_dispatch.record_effective_rl_rows(
            self,
            all_tokens,
            all_logprobs,
            all_advantages,
            echo_advantages=echo_advantages,
            echo_full_observation_counts=echo_full_observation_counts,
            echo_rollout_denominator=echo_rollout_denominator,
        )

    def _record_effective_sft_rows(self, all_tokens, all_advantages) -> None:
        """Record post-crop cross-entropy SFT rows and target weights."""

        local_step_dispatch.record_effective_sft_rows(
            self,
            all_tokens,
            all_advantages,
        )

    def train_step(self, all_tokens, all_logprobs, all_advantages, lr, weight_decay):
        """Run one local importance-sampling training step."""
        return local_step_dispatch.train_step(
            self,
            all_tokens,
            all_logprobs,
            all_advantages,
            lr,
            weight_decay,
        )

    def sft_train_step(self, all_tokens, all_advantages, lr, weight_decay):
        """Run one local SFT/ECHO update."""
        return local_step_dispatch.sft_train_step(
            self,
            all_tokens,
            all_advantages,
            lr,
            weight_decay,
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
        return local_step_dispatch.train_step_with_echo_masks(
            self,
            all_tokens,
            all_logprobs,
            all_advantages,
            echo_advantages,
            echo_full_observation_counts,
            echo_loss_fn,
            lr,
            weight_decay,
            echo_rollout_denominator,
        )

    def load_state(self, name):
        """Load adapter weights and re-synchronize the inference engine."""
        local_lifecycle.load_state(self, name)

    def save_adapter(self, path, name) -> str:
        """Flush pending training and save the latest LoRA adapter."""
        return local_lifecycle.save_adapter(self, path, name)
