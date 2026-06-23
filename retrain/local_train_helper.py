"""Python helper for LocalBackend: PyTorch/PEFT training + pluggable inference.

This helper provides:
- PyTorch model with PEFT LoRA for training (gradient computation)
- Pluggable InferenceEngine for sampling (PyTorch fallback or server-based)
- Adapter save/load for weight synchronization

GPU split mode (multi-GPU): separate devices for inference and training.
- engine on first device (sampling)
- train_model on last device (gradient updates)
- checkpoint() syncs LoRA weights train -> engine

The LocalBackend in Mojo calls into this module via Python interop.
"""

import gc
import os
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from retrain.accelerators import (
    accelerator_status,
    apply_liger_kernel_if_available,
    from_pretrained_attention_kwargs,
    module_available,
)
from retrain.gemma4_text import (
    DEFAULT_LORA_TARGET_MODULES,
    forward_hidden_states_and_lm_head,
    forward_logits,
    resolve_lora_target_modules,
)
from retrain.inference_engine import create_engine


def _masked_mean(values, mask):
    denom = mask.float().sum().clamp(min=1)
    return (values * mask.float()).sum() / denom


def _compute_policy_loss(
    old_logprobs,
    new_logprobs,
    adv,
    mask,
    clip_eps,
    clip_eps_high,
    policy_loss_mode="standard",
    kl_cov_percent=0.2,
    kl_cov_coef=1.0,
    clip_cov_ratio=0.0002,
    clip_cov_min=1.0,
    clip_cov_max=5.0,
):
    """Compute policy loss, optionally with covariance-aware entropy control.

    Args:
        old_logprobs: Log probability under rollout policy.
        new_logprobs: Log probability under current policy.
        adv: Per-token advantages.
        mask: Attention mask (1 for real tokens, 0 for padding).
        clip_eps: Lower clipping epsilon. 0 = no clipping.
        clip_eps_high: Upper clipping epsilon. 0 = use clip_eps (symmetric).
        policy_loss_mode: ``standard``, ``kl_cov``, or ``clip_cov``.

    Returns:
        ``(masked_loss, clip_fraction, cov_fraction, abs_kl)`` tuple.
    """
    logprob_delta = new_logprobs - old_logprobs
    ratio = torch.exp(logprob_delta)
    abs_kl = _masked_mean(logprob_delta.abs(), mask)

    def _standard_loss():
        if clip_eps > 0:
            eps_high = clip_eps_high if clip_eps_high > 0 else clip_eps
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + eps_high)
            surr1 = ratio * adv
            surr2 = clipped_ratio * adv
            per_token_loss = -torch.min(surr1, surr2)
            with torch.no_grad():
                clipped = (
                    (ratio < 1.0 - clip_eps) | (ratio > 1.0 + eps_high)
                ).float()
                frac = (clipped * mask).sum().item() / mask.sum().clamp(min=1).item()
        else:
            per_token_loss = -(ratio * adv)
            frac = 0.0
        return per_token_loss, frac

    if policy_loss_mode == "kl_cov":
        per_token_loss = -(ratio * adv)
        valid = mask > 0
        valid_count = int(valid.sum().item())
        cov_selected = torch.zeros_like(mask, dtype=torch.bool)
        if valid_count > 0 and kl_cov_percent > 0:
            with torch.no_grad():
                valid_adv = adv[valid]
                valid_logp = new_logprobs[valid]
                cov_values = (
                    (valid_adv - valid_adv.mean())
                    * (valid_logp - valid_logp.mean())
                )
                k_tokens = max(1, int(valid_count * min(kl_cov_percent, 100.0) / 100.0))
                selected_flat = torch.topk(cov_values, k_tokens, largest=True).indices
                valid_indices = torch.nonzero(valid.reshape(-1), as_tuple=True)[0]
                selected_indices = valid_indices[selected_flat]
                cov_selected = cov_selected.reshape(-1)
                cov_selected[selected_indices] = True
                cov_selected = cov_selected.reshape_as(mask)
        if cov_selected.any():
            per_token_loss = torch.where(
                cov_selected,
                per_token_loss + kl_cov_coef * logprob_delta.abs(),
                per_token_loss,
            )
        cov_fraction = (cov_selected.float() * mask.float()).sum() / mask.float().sum().clamp(min=1)
        masked_loss = (per_token_loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
        return masked_loss, 0.0, float(cov_fraction.detach().item()), float(abs_kl.detach().item())

    if policy_loss_mode == "clip_cov":
        cov_selected = torch.zeros_like(mask, dtype=torch.bool)
        eps_low = clip_eps if clip_eps > 0 else 1.0
        eps_high = clip_eps_high if clip_eps_high > 0 else eps_low
        pg_loss_unclipped = -(ratio * adv)
        pg_loss_clipped = -(
            torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
        )
        clip_by_origin = (pg_loss_clipped > pg_loss_unclipped) & (mask > 0)
        per_token_loss = torch.maximum(pg_loss_unclipped, pg_loss_clipped)
        with torch.no_grad():
            cov_all = (
                (adv - _masked_mean(adv, mask))
                * (new_logprobs - _masked_mean(new_logprobs, mask))
            )
            eligible = (
                (mask > 0)
                & ~clip_by_origin
                & (cov_all > clip_cov_min)
                & (cov_all < clip_cov_max)
            )
            eligible_idx = torch.nonzero(eligible)
            clip_num = max(int(float(clip_cov_ratio) * mask.sum().item()), 1)
            if len(eligible_idx) > 0:
                perm = torch.randperm(len(eligible_idx), device=eligible_idx.device)
                selected = eligible_idx[perm[: min(clip_num, len(eligible_idx))]]
                cov_selected[selected[:, 0], selected[:, 1]] = True
        per_token_loss = torch.where(
            cov_selected,
            torch.zeros_like(per_token_loss),
            per_token_loss,
        )
        cov_fraction = (cov_selected.float() * mask.float()).sum() / mask.float().sum().clamp(min=1)
        masked_loss = (per_token_loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
        return masked_loss, float(cov_fraction.detach().item()), float(cov_fraction.detach().item()), float(abs_kl.detach().item())

    if policy_loss_mode != "standard":
        raise ValueError(
            "policy_loss_mode must be 'standard', 'kl_cov', or 'clip_cov'."
        )

    per_token_loss, frac = _standard_loss()
    masked_loss = (per_token_loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
    return masked_loss, frac, 0.0, float(abs_kl.detach().item())


def _parse_device(device_str):
    """Convert a device spec like 'gpu:0' to a torch device string like 'cuda:0'."""
    device_str = device_str.strip()
    if device_str.startswith("gpu:"):
        return device_str.replace("gpu:", "cuda:")
    elif device_str == "cpu":
        return "cpu"
    else:
        return "cuda:0"


def _pad_to_width(tensor, width, value):
    """Right-pad a batch-major tensor to ``width`` columns."""
    if tensor.shape[1] >= width:
        return tensor
    pad = tensor.new_full((tensor.shape[0], width - tensor.shape[1]), value)
    return torch.cat([tensor, pad], dim=1)


def _is_cuda_device(device) -> bool:
    return isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()


def _timer_start(device):
    if _is_cuda_device(device):
        with torch.cuda.device(torch.device(device)):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        return ("cuda", device, event)
    return ("cpu", device, time.perf_counter())


def _timer_stop(start) -> float:
    kind, device, marker = start
    if kind == "cuda":
        with torch.cuda.device(torch.device(device)):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
        return marker.elapsed_time(end) / 1000.0
    return time.perf_counter() - marker


def _reset_cuda_peak(device) -> None:
    if _is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def _cuda_peak_metrics(prefix: str, device) -> dict[str, float]:
    if not _is_cuda_device(device):
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


class LocalTrainHelper:
    """Local GPU helper: pluggable inference engine + PyTorch/PEFT training."""

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
                 train_logprob_chunk_size=0,
                 liger_kernel=True,
                 liger_fused_linear_ce=True,
                 cuda_empty_cache=False,
                 sample_use_cache=True,
                 gradient_checkpointing=True,
                 attention_kernel="default",
                 prefix_caching=True,
                 train_selective_suffix_logits=False,
                 train_save_on_cpu=False,
                 train_save_on_cpu_pin_memory=True,
                 train_save_on_cpu_min_numel=0,
                 train_supervised_context_tokens=0):
        self.adapter_path = adapter_path
        self.model_name = model_name
        self.engine_type = engine_type
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
        self.train_logprob_chunk_size = max(0, int(train_logprob_chunk_size))
        self.liger_kernel = bool(liger_kernel)
        self.liger_fused_linear_ce = bool(liger_fused_linear_ce)
        self.attention_kernel = str(attention_kernel or "default")
        self.cuda_empty_cache = bool(cuda_empty_cache)
        self.sample_use_cache = bool(sample_use_cache)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.prefix_caching = bool(prefix_caching)
        self.train_selective_suffix_logits = bool(train_selective_suffix_logits)
        self.train_save_on_cpu = bool(train_save_on_cpu)
        self.train_save_on_cpu_pin_memory = bool(train_save_on_cpu_pin_memory)
        self.train_save_on_cpu_min_numel = max(0, int(train_save_on_cpu_min_numel))
        self.train_supervised_context_tokens = max(
            0,
            int(train_supervised_context_tokens),
        )
        self._last_sample_metrics: dict[str, float | int] = {}
        self._last_train_metrics: dict[str, float | int] = {}
        self._last_sync_metrics: dict[str, float | int] = {}
        self._last_context_crop_metrics: dict[str, float | int] = {}

        # Parse all devices from comma-separated spec
        raw_devices = [d.strip() for d in devices.split(",") if d.strip()]
        parsed_devices = [_parse_device(d) for d in raw_devices]

        # Determine split mode: need >1 device and CUDA available
        cuda_devices = [d for d in parsed_devices if d.startswith("cuda")]
        self.split_mode = len(cuda_devices) > 1 and torch.cuda.is_available()

        # Non-PyTorch engines manage their own inference independently.
        # Server engines use a remote process; MAX engine uses its own in-process model.
        self._server_engine = engine_type in (
            "vllm",
            "sglang",
            "trtllm",
            "mlx",
            "openai",
        )
        self._external_engine = engine_type in (
            "max",
            "vllm",
            "sglang",
            "trtllm",
            "mlx",
            "openai",
        )

        if self._external_engine:
            # Server handles inference — all local GPUs for training
            device = parsed_devices[-1] if parsed_devices else "cuda:0"
            if device.startswith("cuda") and not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                device = "cpu"
            self.infer_device = device  # not used for sampling, but kept for compat
            self.train_device = device
            self.split_mode = False
        elif self.split_mode:
            self.infer_device = cuda_devices[0]   # first GPU for inference
            self.train_device = cuda_devices[-1]   # last GPU for training
        else:
            # Single-model mode: use first device (with CUDA fallback)
            device = parsed_devices[0] if parsed_devices else "cuda:0"
            if device.startswith("cuda") and not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                device = "cpu"
            self.infer_device = device
            self.train_device = device

        self.use_amp = self.train_device != "cpu"
        dtype = torch.bfloat16 if self.train_device != "cpu" else torch.float32

        self._accelerator_metrics: dict[str, object] = accelerator_status()
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
        self._move_train_model_to_device()
        if hasattr(self.train_model, "print_trainable_parameters"):
            self.train_model.print_trainable_parameters()

        self._configure_gradient_checkpointing()

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
            )

        # Optimizer (only for train_model)
        self.optimizer = torch.optim.AdamW(
            self.train_model.parameters(),
            lr=4e-5,
            betas=(optim_beta1, optim_beta2),
            eps=optim_eps,
            weight_decay=0.0,
        )

        # AMP scaler for mixed precision
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        print(f"LocalTrainHelper ready (engine={engine_type}, split_mode={self.split_mode}).")

    def _build_peft_config(self, base_model, lora_rank, lora_alpha, lora_dropout):
        effective_alpha = lora_alpha if lora_alpha > 0 else lora_rank * 2
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=effective_alpha,
            lora_dropout=lora_dropout,
            target_modules=resolve_lora_target_modules(
                base_model,
                DEFAULT_LORA_TARGET_MODULES,
            ),
        )

    def _load_train_model(self, model_name, dtype, lora_rank, lora_alpha, lora_dropout):
        model_kwargs = from_pretrained_attention_kwargs(self.attention_kernel)
        base_train = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            **model_kwargs,
        )
        peft_config = self._build_peft_config(
            base_train,
            lora_rank,
            lora_alpha,
            lora_dropout,
        )
        return get_peft_model(base_train, peft_config), peft_config

    def _move_train_model_to_device(self):
        self.train_model.to(self.train_device)

    def _configure_gradient_checkpointing(self):
        # Gradient checkpointing trades compute for VRAM; benchmarks must be
        # able to toggle it because it changes both memory fit and throughput.
        if self.gradient_checkpointing and hasattr(
            self.train_model,
            "gradient_checkpointing_enable",
        ):
            self.train_model.gradient_checkpointing_enable()
        elif hasattr(self.train_model, "gradient_checkpointing_disable"):
            self.train_model.gradient_checkpointing_disable()

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
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

            loss = LigerFusedLinearCrossEntropyLoss(reduction="none")
        except Exception:  # noqa: BLE001 - optional accelerator path.
            return None
        self._liger_fused_ce_loss = loss
        return loss

    def _do_initial_sync(self):
        """Copy LoRA weights from train_model to engine at init time.

        Used once during __init__ before any training has occurred (no snapshot
        exists yet). After this, all syncs go through _sync_lora_weights which
        reads from the weight snapshot.
        """
        train_state = dict(self.train_model.named_parameters())
        lora_dict = {}
        for name, param in train_state.items():
            if "lora_" in name:
                lora_dict[name] = param.data
        self.engine.sync_from_state_dict(lora_dict)

    def _sync_lora_weights(self):
        """Copy LoRA weights from snapshot to engine.

        In split mode: copies from _weight_snapshot (created after each completed
        training step) rather than live train_model weights. This is safe to call
        while training is running on GPU 1 — reads snapshot, not live weights.

        For external engines (MAX, server-based): saves adapter to disk and
        tells engine to reload.

        No-op when split_mode is False (engine shares train_model) or when
        no training has completed yet (_weight_snapshot is None).
        """
        sync_start = time.perf_counter()
        save_s = 0.0
        reload_s = 0.0
        copied = 0
        try:
            if self._external_engine:
                if self._weights_dirty and self._weight_snapshot is not None:
                    # Save adapter to disk, then tell engine to reload
                    save_dir = os.path.join(self.adapter_path, "_live_adapter")
                    os.makedirs(save_dir, exist_ok=True)
                    save_start = time.perf_counter()
                    self.train_model.save_pretrained(save_dir)
                    save_s = time.perf_counter() - save_start
                    reload_start = time.perf_counter()
                    self.engine.reload_weights(save_dir)
                    reload_s = time.perf_counter() - reload_start
                    self._weights_dirty = False
                    copied = 1
                return

            if not self.split_mode:
                return
            if self._weight_snapshot is None:
                return

            self.engine.sync_from_state_dict(self._weight_snapshot)
            copied = 1
        finally:
            self._last_sync_metrics = {
                "local_adapter_sync_s": time.perf_counter() - sync_start,
                "local_adapter_save_s": save_s,
                "local_adapter_reload_s": reload_s,
                "local_adapter_sync_copied": copied,
            }

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
        if hasattr(self.engine, "clear_prefix_cache"):
            self.engine.clear_prefix_cache()

    @contextmanager
    def _shared_model_sampling_cache_context(self):
        toggled = False
        config = None
        previous_use_cache = None
        if (
            getattr(self, "gradient_checkpointing", False)
            and getattr(self, "sample_use_cache", True)
            and not getattr(self, "_external_engine", False)
            and not getattr(self, "split_mode", False)
        ):
            model = self.train_model
            config = getattr(model, "config", None)
            previous_use_cache = getattr(config, "use_cache", None)
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
                toggled = True
            if config is not None and previous_use_cache is not None:
                config.use_cache = True
        self._last_sample_gc_disabled_for_cache = int(toggled)
        try:
            yield
        finally:
            if toggled:
                model = self.train_model
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                if config is not None and previous_use_cache is not None:
                    config.use_cache = previous_use_cache

    def sample(self, prompt_ids_list, num_samples, max_tokens, temperature, top_p):
        """Generate completions with per-token logprobs.

        Delegates to the configured InferenceEngine, then converts
        SampleResult objects to (token_ids, logprobs) tuples for
        backward compatibility with the Mojo caller.

        Args:
            prompt_ids_list: List of lists of token IDs (one per prompt).
            num_samples: Number of completions per prompt.
            max_tokens: Maximum new tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            List of lists of (token_ids, logprobs) tuples.
        """
        sample_start = time.perf_counter()
        _reset_cuda_peak(getattr(self, "infer_device", getattr(self, "train_device", "cpu")))
        try:
            with self._shared_model_sampling_cache_context():
                engine_results = self.engine.generate(
                    prompt_ids_list, num_samples, max_tokens, temperature, top_p
                )
        finally:
            self._empty_cuda_cache_if_requested()
        self._record_sample_metrics(sample_start, prompt_ids_list, num_samples, engine_results)

        # Convert SampleResult -> tuples for Mojo interop
        results = []
        for group in engine_results:
            converted = []
            for sr in group:
                converted.append((sr.token_ids, sr.logprobs))
            results.append(converted)
        return results

    def sample_with_entropy(self, prompt_ids_list, num_samples, max_tokens,
                            temperature, top_p):
        """Generate completions with per-token logprobs and Shannon entropy.

        Like sample(), but requests per-token entropy from the engine.
        Returns 3-tuples (token_ids, logprobs, token_entropies).

        Args:
            prompt_ids_list: List of lists of token IDs (one per prompt).
            num_samples: Number of completions per prompt.
            max_tokens: Maximum new tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            List of lists of (token_ids, logprobs, token_entropies) tuples.
        """
        sample_start = time.perf_counter()
        _reset_cuda_peak(getattr(self, "infer_device", getattr(self, "train_device", "cpu")))
        try:
            with self._shared_model_sampling_cache_context():
                engine_results = self.engine.generate(
                    prompt_ids_list, num_samples, max_tokens, temperature, top_p,
                    compute_entropy=True,
                )
        finally:
            self._empty_cuda_cache_if_requested()
        self._record_sample_metrics(sample_start, prompt_ids_list, num_samples, engine_results)

        results = []
        for group in engine_results:
            converted = []
            for sr in group:
                converted.append((sr.token_ids, sr.logprobs, sr.token_entropies))
            results.append(converted)
        return results

    def runtime_metrics(self):
        """Expose optional engine-level runtime counters to the trainer."""
        metrics = {
            "local_gradient_checkpointing_enabled": int(
                getattr(self, "gradient_checkpointing", False)
            ),
            "local_sample_use_cache": int(getattr(self, "sample_use_cache", True)),
            "local_train_microbatch_size": int(
                getattr(self, "train_microbatch_size", 0)
            ),
            "local_train_selective_suffix_logits": int(
                getattr(self, "train_selective_suffix_logits", False)
            ),
            "local_train_save_on_cpu": int(
                getattr(self, "train_save_on_cpu", False)
            ),
            "local_train_save_on_cpu_pin_memory": int(
                getattr(self, "train_save_on_cpu_pin_memory", True)
            ),
            "local_train_save_on_cpu_min_numel": int(
                getattr(self, "train_save_on_cpu_min_numel", 0)
            ),
            "local_train_supervised_context_tokens": int(
                getattr(self, "train_supervised_context_tokens", 0)
            ),
            "local_train_logprob_chunk_size": int(
                getattr(self, "train_logprob_chunk_size", 0)
            ),
            "local_liger_kernel_enabled": int(
                getattr(self, "liger_kernel", False)
            ),
            "local_liger_fused_linear_ce_enabled": int(
                getattr(self, "liger_fused_linear_ce", False)
            ),
            "local_prefix_caching": int(getattr(self, "prefix_caching", True)),
        }
        metrics.update(getattr(self, "_accelerator_metrics", {}))
        engine = getattr(self, "engine", None)
        if hasattr(engine, "performance_counters"):
            counters = engine.performance_counters()
            if isinstance(counters, dict):
                metrics.update(counters)
        metrics.update(getattr(self, "_last_context_crop_metrics", {}))
        metrics.update(getattr(self, "_last_sample_metrics", {}))
        metrics.update(getattr(self, "_last_train_metrics", {}))
        metrics.update(getattr(self, "_last_sync_metrics", {}))
        return metrics

    def _record_sample_metrics(self, start_s, prompt_ids_list, num_samples, engine_results):
        wall_s = time.perf_counter() - start_s
        prompt_tokens = sum(len(prompt) for prompt in prompt_ids_list) * int(num_samples)
        generated_tokens = sum(
            len(result.token_ids)
            for group in engine_results
            for result in group
        )
        metrics: dict[str, float | int] = {
            "local_sample_wall_s": wall_s,
            "local_sample_prompt_tokens": prompt_tokens,
            "local_sample_generated_tokens": generated_tokens,
            "local_sample_generation_tokens_per_s": (
                generated_tokens / wall_s if wall_s > 0 else 0.0
            ),
            "local_sample_gc_disabled_for_cache": int(
                getattr(self, "_last_sample_gc_disabled_for_cache", 0)
            ),
        }
        metrics.update(
            _cuda_peak_metrics(
                "local_sample_gpu",
                getattr(self, "infer_device", getattr(self, "train_device", "cpu")),
            )
        )
        self._last_sample_metrics = metrics

    def _snapshot_lora_weights_if_needed(self) -> float:
        if not (self.split_mode or self._external_engine):
            return 0.0
        timer = _timer_start(self.train_device)
        snapshot = {}
        for name, param in self.train_model.named_parameters():
            if "lora_" in name:
                snapshot[name] = param.data.clone()
        self._weight_snapshot = snapshot
        self._weights_dirty = True
        return _timer_stop(timer)

    def _record_train_metrics(
        self,
        *,
        kind: str,
        wall_start_s: float,
        forward_s: float,
        backward_s: float,
        optimizer_s: float,
        snapshot_s: float,
        microbatches: int,
        total_tokens: float,
        batch_size: int,
    ) -> None:
        wall_s = time.perf_counter() - wall_start_s
        metrics: dict[str, float | int] = {
            "local_train_kind": kind,
            "local_train_wall_s": wall_s,
            "local_train_forward_s": forward_s,
            "local_train_backward_s": backward_s,
            "local_train_optimizer_s": optimizer_s,
            "local_train_snapshot_s": snapshot_s,
            "local_train_microbatches": microbatches,
            "local_train_tokens": total_tokens,
            "local_train_batch_size": batch_size,
            "local_train_tokens_per_s": total_tokens / wall_s if wall_s > 0 else 0.0,
        }
        metrics.update(_cuda_peak_metrics("local_train_gpu", self.train_device))
        self._last_train_metrics = metrics

    def _empty_cuda_cache_if_requested(self):
        if self.cuda_empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _clear_inference_prefix_cache(self):
        engine = getattr(self, "engine", None)
        if engine is not None and hasattr(engine, "clear_prefix_cache"):
            engine.clear_prefix_cache()

    def shutdown(self) -> None:
        """Release model, optimizer, executor, and CUDA allocator state."""
        future = getattr(self, "_train_future", None)
        if future is not None:
            try:
                future.result()
            except Exception:  # noqa: BLE001 - cleanup should not mask failures.
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
            except Exception:  # noqa: BLE001 - best-effort cleanup.
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
                except Exception:  # noqa: BLE001 - best-effort cleanup.
                    pass

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001 - CUDA may be recovering from OOM.
                pass
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:  # noqa: BLE001 - cleanup must not raise.
                pass

    def _selective_suffix_token_logprobs(self, input_ids, attention_mask, target_mask):
        if not getattr(self, "train_selective_suffix_logits", False):
            return None
        if target_mask is None or not bool(target_mask.any().item()):
            return None

        selected = torch.nonzero(target_mask, as_tuple=False)
        if selected.numel() == 0:
            return None

        seq_len = int(input_ids.shape[1])
        min_target_pos = int(selected[:, 1].min().item())
        logits_to_keep = seq_len - min_target_pos
        if logits_to_keep <= 1 or logits_to_keep >= seq_len:
            return None

        try:
            outputs = self.train_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=logits_to_keep,
            )
        except TypeError:
            return None

        logits = getattr(outputs, "logits", None)
        if logits is None:
            return None

        logits_start = seq_len - logits_to_keep
        target_suffix = input_ids[:, logits_start + 1 :]
        usable_logits = logits[:, : target_suffix.shape[1], :]
        suffix_logprobs = F.log_softmax(usable_logits.float(), dim=-1)
        suffix_logprobs = suffix_logprobs.gather(
            2,
            target_suffix.unsqueeze(2),
        ).squeeze(2)
        full = suffix_logprobs.new_zeros((input_ids.shape[0], seq_len - 1))
        full[:, logits_start : logits_start + suffix_logprobs.shape[1]] = (
            suffix_logprobs
        )
        return full

    def _shifted_token_logprobs(self, input_ids, attention_mask, target_mask=None):
        """Compute next-token logprobs, optionally chunking the LM head."""
        selective = self._selective_suffix_token_logprobs(
            input_ids,
            attention_mask,
            target_mask,
        )
        if selective is not None:
            return selective

        liger_loss = self._liger_fused_linear_ce_loss()
        if liger_loss is not None:
            hidden_and_head = forward_hidden_states_and_lm_head(
                self.train_model,
                input_ids,
                attention_mask,
            )
            if hidden_and_head is not None:
                hidden_states, lm_head = hidden_and_head
                shifted_hidden = hidden_states[:, :-1, :]
                target_ids = input_ids[:, 1:]
                flat_hidden = shifted_hidden.reshape(-1, shifted_hidden.shape[-1])
                flat_target_ids = target_ids.reshape(-1)
                weight = getattr(lm_head, "weight")
                try:
                    nll = liger_loss(weight, flat_hidden, flat_target_ids)
                except TypeError:
                    nll = liger_loss(flat_hidden, weight, flat_target_ids)
                return -nll.reshape_as(target_ids)

        chunk_size = int(getattr(self, "train_logprob_chunk_size", 0))
        if chunk_size <= 0:
            logits = forward_logits(self.train_model, input_ids, attention_mask)[:, :-1]
            new_logprobs = F.log_softmax(logits.float(), dim=-1)
            target_ids = input_ids[:, 1:]
            return new_logprobs.gather(2, target_ids.unsqueeze(2)).squeeze(2)

        hidden_and_head = forward_hidden_states_and_lm_head(
            self.train_model,
            input_ids,
            attention_mask,
        )
        if hidden_and_head is None:
            logits = forward_logits(self.train_model, input_ids, attention_mask)[:, :-1]
            new_logprobs = F.log_softmax(logits.float(), dim=-1)
            target_ids = input_ids[:, 1:]
            return new_logprobs.gather(2, target_ids.unsqueeze(2)).squeeze(2)

        hidden_states, lm_head = hidden_and_head
        shifted_hidden = hidden_states[:, :-1, :]
        target_ids = input_ids[:, 1:]
        chunks = []
        for start in range(0, shifted_hidden.shape[1], chunk_size):
            stop = min(start + chunk_size, shifted_hidden.shape[1])
            logits = lm_head(shifted_hidden[:, start:stop, :])
            logprobs = F.log_softmax(logits.float(), dim=-1)
            chunks.append(
                logprobs.gather(
                    2,
                    target_ids[:, start:stop].unsqueeze(2),
                ).squeeze(2)
            )
        if not chunks:
            return shifted_hidden.new_empty((shifted_hidden.shape[0], 0))
        return torch.cat(chunks, dim=1)

    @contextmanager
    def _saved_tensors_context(self):
        if (
            getattr(self, "train_save_on_cpu", False)
            and self.train_device.startswith("cuda")
            and torch.cuda.is_available()
        ):
            min_numel = int(getattr(self, "train_save_on_cpu_min_numel", 0))
            if min_numel > 0:
                pin_memory = bool(getattr(self, "train_save_on_cpu_pin_memory", True))

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

            with torch.autograd.graph.save_on_cpu(
                pin_memory=getattr(self, "train_save_on_cpu_pin_memory", True)
            ):
                yield
        else:
            yield

    def _compute_train_loss(self, input_ids, old_logprobs, advantages, attention_mask):
        """Compute masked policy loss for one already-padded microbatch."""
        with torch.amp.autocast(device_type=self.train_device.split(":")[0], enabled=self.use_amp):
            old_lp = old_logprobs[:, 1:]  # [N, max_len-1]
            adv = advantages[:, 1:]       # [N, max_len-1]
            mask = attention_mask[:, 1:]   # [N, max_len-1] — exclude padding
            loss_mask = mask
            target_mask = None
            if getattr(self, "train_selective_suffix_logits", False):
                target_mask = (mask > 0) & (adv != 0)
                loss_mask = target_mask.to(mask.dtype)

            new_logprobs = self._shifted_token_logprobs(
                input_ids,
                attention_mask,
                target_mask=target_mask,
            )

            masked_loss, clip_frac, cov_frac, abs_kl = _compute_policy_loss(
                old_lp,
                new_logprobs,
                adv,
                loss_mask,
                self.clip_eps,
                self.clip_eps_high,
                getattr(self, "policy_loss_mode", "standard"),
                getattr(self, "kl_cov_percent", 0.2),
                getattr(self, "kl_cov_coef", 1.0),
                getattr(self, "clip_cov_ratio", 0.0002),
                getattr(self, "clip_cov_min", 1.0),
                getattr(self, "clip_cov_max", 5.0),
            )
            token_count = loss_mask.sum().clamp(min=1)

        return masked_loss, token_count, clip_frac, cov_frac, abs_kl

    def _compute_sft_loss(self, input_ids, advantages, attention_mask):
        """Compute weighted next-token cross-entropy for SFT/ECHO datums."""
        with torch.amp.autocast(device_type=self.train_device.split(":")[0], enabled=self.use_amp):
            weights = torch.clamp(advantages[:, 1:], min=0.0)
            weights = weights * attention_mask[:, 1:].float()
            target_mask = None
            if getattr(self, "train_selective_suffix_logits", False):
                target_mask = weights > 0

            new_logprobs = self._shifted_token_logprobs(
                input_ids,
                attention_mask,
                target_mask=target_mask,
            )

            token_mask = (weights > 0).float()
            token_count = token_mask.sum().clamp(min=1)
            loss = (-new_logprobs * weights).sum() / token_count

        return loss, token_count

    def _do_train_impl(self, input_ids, old_logprobs, advantages, attention_mask):
        """Execute training forward/backward/step on pre-prepared tensors.

        After the optimizer step, clones LoRA params into _weight_snapshot
        for safe cross-thread syncing. Runs on background thread in split mode.

        Returns:
            Scalar loss value.
        """
        self.train_model.train()
        self.optimizer.zero_grad()
        wall_start = time.perf_counter()
        _reset_cuda_peak(self.train_device)
        forward_s = 0.0
        backward_s = 0.0
        optimizer_s = 0.0
        snapshot_s = 0.0
        microbatches = 0

        try:
            batch_size = int(input_ids.shape[0])
            microbatch_size = self.train_microbatch_size or batch_size
            if getattr(self, "train_selective_suffix_logits", False):
                total_tokens = (
                    ((attention_mask[:, 1:] > 0) & (advantages[:, 1:] != 0))
                    .sum()
                    .clamp(min=1)
                )
            else:
                total_tokens = attention_mask[:, 1:].sum().clamp(min=1)
            total_tokens_value = float(total_tokens.item())
            loss_sum = 0.0
            clip_count = 0.0
            cov_count = 0.0
            abs_kl_sum = 0.0

            for start in range(0, batch_size, microbatch_size):
                stop = min(start + microbatch_size, batch_size)
                microbatches += 1
                timer = _timer_start(self.train_device)
                with self._saved_tensors_context():
                    masked_loss, token_count, clip_frac, cov_frac, abs_kl = (
                        self._compute_train_loss(
                            input_ids[start:stop],
                            old_logprobs[start:stop],
                            advantages[start:stop],
                            attention_mask[start:stop],
                        )
                    )
                    forward_s += _timer_stop(timer)
                    token_count_value = float(token_count.item())
                    scaled_loss = masked_loss * (token_count / total_tokens)
                    timer = _timer_start(self.train_device)
                    self.scaler.scale(scaled_loss).backward()
                    backward_s += _timer_stop(timer)
                loss_sum += float(masked_loss.detach().item()) * token_count_value
                clip_count += clip_frac * token_count_value
                cov_count += cov_frac * token_count_value
                abs_kl_sum += abs_kl * token_count_value

            timer = _timer_start(self.train_device)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_s += _timer_stop(timer)

            self._clip_fraction = clip_count / total_tokens_value
            self._policy_cov_fraction = cov_count / total_tokens_value
            self._policy_abs_kl = abs_kl_sum / total_tokens_value
            loss_val = loss_sum / total_tokens_value

            # Snapshot LoRA weights for safe cross-thread sync
            snapshot_s = self._snapshot_lora_weights_if_needed()
            self._record_train_metrics(
                kind="rl",
                wall_start_s=wall_start,
                forward_s=forward_s,
                backward_s=backward_s,
                optimizer_s=optimizer_s,
                snapshot_s=snapshot_s,
                microbatches=microbatches,
                total_tokens=total_tokens_value,
                batch_size=batch_size,
            )

            return loss_val
        finally:
            self.optimizer.zero_grad()
            self._empty_cuda_cache_if_requested()

    def _do_sft_impl(self, input_ids, advantages, attention_mask):
        """Execute weighted cross-entropy SFT/ECHO update synchronously."""
        self.train_model.train()
        self.optimizer.zero_grad()
        wall_start = time.perf_counter()
        _reset_cuda_peak(self.train_device)
        forward_s = 0.0
        backward_s = 0.0
        optimizer_s = 0.0
        snapshot_s = 0.0
        microbatches = 0

        try:
            batch_size = int(input_ids.shape[0])
            microbatch_size = self.train_microbatch_size or batch_size
            weights = torch.clamp(advantages[:, 1:], min=0.0)
            total_tokens = (weights * attention_mask[:, 1:].float() > 0).sum().clamp(min=1)
            total_tokens_value = float(total_tokens.item())
            loss_sum = 0.0

            for start in range(0, batch_size, microbatch_size):
                stop = min(start + microbatch_size, batch_size)
                microbatches += 1
                timer = _timer_start(self.train_device)
                with self._saved_tensors_context():
                    masked_loss, token_count = self._compute_sft_loss(
                        input_ids[start:stop],
                        advantages[start:stop],
                        attention_mask[start:stop],
                    )
                    forward_s += _timer_stop(timer)
                    token_count_value = float(token_count.item())
                    scaled_loss = masked_loss * (token_count / total_tokens)
                    timer = _timer_start(self.train_device)
                    self.scaler.scale(scaled_loss).backward()
                    backward_s += _timer_stop(timer)
                loss_sum += float(masked_loss.detach().item()) * token_count_value

            timer = _timer_start(self.train_device)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_s += _timer_stop(timer)
            loss_val = loss_sum / total_tokens_value

            snapshot_s = self._snapshot_lora_weights_if_needed()
            self._record_train_metrics(
                kind="sft",
                wall_start_s=wall_start,
                forward_s=forward_s,
                backward_s=backward_s,
                optimizer_s=optimizer_s,
                snapshot_s=snapshot_s,
                microbatches=microbatches,
                total_tokens=total_tokens_value,
                batch_size=batch_size,
            )

            return loss_val
        finally:
            self.optimizer.zero_grad()
            self._empty_cuda_cache_if_requested()

    def _do_hybrid_mask_impl(
        self,
        input_ids,
        old_logprobs,
        advantages,
        attention_mask,
        echo_advantages,
        echo_full_observation_counts,
        echo_loss_fn,
    ):
        """Execute RL + ECHO on the same rollout rows in one optimizer step."""
        self.train_model.train()
        self.optimizer.zero_grad()
        wall_start = time.perf_counter()
        _reset_cuda_peak(self.train_device)
        forward_s = 0.0
        backward_s = 0.0
        optimizer_s = 0.0
        snapshot_s = 0.0
        microbatches = 0

        try:
            batch_size = int(input_ids.shape[0])
            microbatch_size = self.train_microbatch_size or batch_size
            if getattr(self, "train_selective_suffix_logits", False):
                total_tokens = (
                    ((attention_mask[:, 1:] > 0) & (advantages[:, 1:] != 0))
                    .sum()
                    .clamp(min=1)
                )
            else:
                total_tokens = attention_mask[:, 1:].sum().clamp(min=1)
            total_tokens_value = float(total_tokens.item())
            loss_sum = 0.0
            clip_count = 0.0
            cov_count = 0.0
            abs_kl_sum = 0.0
            echo_loss_sum = 0.0

            for start in range(0, batch_size, microbatch_size):
                stop = min(start + microbatch_size, batch_size)
                microbatches += 1
                mb_input_ids = input_ids[start:stop]
                mb_old_logprobs = old_logprobs[start:stop]
                mb_advantages = advantages[start:stop]
                mb_attention_mask = attention_mask[start:stop]
                mb_echo_advantages = echo_advantages[start:stop]
                mb_echo_counts = echo_full_observation_counts[start:stop]

                timer = _timer_start(self.train_device)
                with self._saved_tensors_context():
                    with torch.amp.autocast(
                        device_type=self.train_device.split(":")[0],
                        enabled=self.use_amp,
                    ):
                        old_lp = mb_old_logprobs[:, 1:]
                        adv = mb_advantages[:, 1:]
                        mask = mb_attention_mask[:, 1:]
                        echo_weights = torch.clamp(
                            mb_echo_advantages[:, 1:],
                            min=0.0,
                        )
                        echo_weights = echo_weights * mask.float()
                        loss_mask = mask
                        target_mask = None
                        if getattr(self, "train_selective_suffix_logits", False):
                            target_mask = (mask > 0) & (
                                (adv != 0) | (echo_weights > 0.0)
                            )
                            loss_mask = ((mask > 0) & (adv != 0)).to(mask.dtype)
                        token_logprobs = self._shifted_token_logprobs(
                            mb_input_ids,
                            mb_attention_mask,
                            target_mask=target_mask,
                        )
                        masked_loss, clip_frac, cov_frac, abs_kl = (
                            _compute_policy_loss(
                                old_lp,
                                token_logprobs,
                                adv,
                                loss_mask,
                                self.clip_eps,
                                self.clip_eps_high,
                                getattr(self, "policy_loss_mode", "standard"),
                                getattr(self, "kl_cov_percent", 0.2),
                                getattr(self, "kl_cov_coef", 1.0),
                                getattr(self, "clip_cov_ratio", 0.0002),
                                getattr(self, "clip_cov_min", 1.0),
                                getattr(self, "clip_cov_max", 5.0),
                            )
                        )
                        token_count = loss_mask.sum().clamp(min=1)
                        token_count_value = float(token_count.item())
                        scaled_loss = masked_loss * (token_count / total_tokens)

                        if echo_loss_fn != "cross_entropy":
                            raise ValueError(
                                "echo_loss_fn must be 'cross_entropy' for "
                                "paper-faithful ECHO."
                            )
                        echo_selected = (echo_weights > 0.0).float()
                        if echo_selected.sum() > 0:
                            denom = mb_echo_counts.float().clamp(min=1e-3).unsqueeze(1)
                            echo_loss = (
                                (-token_logprobs * echo_weights) / denom
                            ).sum()
                        else:
                            echo_loss = token_logprobs.sum() * 0.0

                        scaled_loss = scaled_loss + echo_loss / max(batch_size, 1)
                    forward_s += _timer_stop(timer)

                    timer = _timer_start(self.train_device)
                    self.scaler.scale(scaled_loss).backward()
                    backward_s += _timer_stop(timer)
                loss_sum += float(masked_loss.detach().item()) * token_count_value
                clip_count += clip_frac * token_count_value
                cov_count += cov_frac * token_count_value
                abs_kl_sum += abs_kl * token_count_value
                echo_loss_sum += float(echo_loss.detach().item())

            timer = _timer_start(self.train_device)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_s += _timer_stop(timer)

            self._clip_fraction = clip_count / total_tokens_value
            self._policy_cov_fraction = cov_count / total_tokens_value
            self._policy_abs_kl = abs_kl_sum / total_tokens_value
            loss_val = loss_sum / total_tokens_value
            echo_loss_val = echo_loss_sum / max(batch_size, 1)

            snapshot_s = self._snapshot_lora_weights_if_needed()
            self._record_train_metrics(
                kind="hybrid_echo",
                wall_start_s=wall_start,
                forward_s=forward_s,
                backward_s=backward_s,
                optimizer_s=optimizer_s,
                snapshot_s=snapshot_s,
                microbatches=microbatches,
                total_tokens=total_tokens_value,
                batch_size=batch_size,
            )

            return loss_val, echo_loss_val
        finally:
            self.optimizer.zero_grad()
            self._empty_cuda_cache_if_requested()

    @staticmethod
    def _first_supervised_token_index(*rows) -> int | None:
        earliest: int | None = None
        for row in rows:
            if row is None:
                continue
            for idx, value in enumerate(row[1:], start=1):
                if float(value) != 0.0:
                    earliest = idx if earliest is None else min(earliest, idx)
                    break
        return earliest

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
        original_lengths = [len(row) for row in all_tokens]
        if not enabled or not original_lengths:
            self._last_context_crop_metrics = {
                "local_train_context_rows_cropped": 0,
                "local_train_context_tokens_removed": 0,
                "local_train_context_original_max_tokens": max(original_lengths, default=0),
                "local_train_context_cropped_max_tokens": max(original_lengths, default=0),
            }
            return all_tokens, all_logprobs, all_advantages, echo_advantages

        cropped_tokens = []
        cropped_logprobs = [] if all_logprobs is not None else None
        cropped_advantages = [] if all_advantages is not None else None
        cropped_echo = [] if echo_advantages is not None else None
        rows_cropped = 0
        tokens_removed = 0

        for idx, tokens in enumerate(all_tokens):
            logprobs = all_logprobs[idx] if all_logprobs is not None else None
            advantages = all_advantages[idx] if all_advantages is not None else None
            echo = echo_advantages[idx] if echo_advantages is not None else None
            earliest = self._first_supervised_token_index(advantages, echo)
            start = 0
            if earliest is not None:
                start = max(0, int(earliest) - context_tokens)
                if start >= earliest:
                    start = max(0, int(earliest) - 1)
            if start > 0:
                rows_cropped += 1
                tokens_removed += start
            cropped_tokens.append(tokens[start:])
            if cropped_logprobs is not None:
                cropped_logprobs.append(logprobs[start:])
            if cropped_advantages is not None:
                cropped_advantages.append(advantages[start:])
            if cropped_echo is not None:
                cropped_echo.append(echo[start:])

        cropped_lengths = [len(row) for row in cropped_tokens]
        self._last_context_crop_metrics = {
            "local_train_context_rows_cropped": rows_cropped,
            "local_train_context_tokens_removed": tokens_removed,
            "local_train_context_original_max_tokens": max(original_lengths, default=0),
            "local_train_context_cropped_max_tokens": max(cropped_lengths, default=0),
        }
        return cropped_tokens, cropped_logprobs, cropped_advantages, cropped_echo

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

        self._clear_inference_prefix_cache()

        # Update optimizer hyperparameters
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = weight_decay

        # Pad all sequences into a single batch
        token_tensors = [torch.tensor(t, dtype=torch.long) for t in all_tokens]
        lp_tensors = [torch.tensor(lp, dtype=torch.float32) for lp in all_logprobs]
        adv_tensors = [torch.tensor(a, dtype=torch.float32) for a in all_advantages]

        # pad_sequence pads to max length in batch (batch_first=True -> [N, max_len])
        input_ids = pad_sequence(token_tensors, batch_first=True, padding_value=0).to(self.train_device)
        old_logprobs = pad_sequence(lp_tensors, batch_first=True, padding_value=0.0).to(self.train_device)
        advantages = pad_sequence(adv_tensors, batch_first=True, padding_value=0.0).to(self.train_device)

        # Build attention mask: 1 where real tokens, 0 where padding
        lengths = torch.tensor([len(t) for t in all_tokens], device=self.train_device)
        max_len = input_ids.shape[1]
        attention_mask = torch.arange(max_len, device=self.train_device).unsqueeze(0) < lengths.unsqueeze(1)

        if self.split_mode:
            # Async path: wait for previous training to finish (back pressure)
            if self._train_future is not None:
                self._pending_loss = self._train_future.result()
                self._train_future = None

            # Submit training to background thread
            self._train_future = self._train_executor.submit(
                self._do_train_impl, input_ids, old_logprobs, advantages, attention_mask
            )

            # Return loss from the previously completed training step
            return self._pending_loss
        else:
            # Synchronous path: run training inline, return current loss
            return self._do_train_impl(input_ids, old_logprobs, advantages, attention_mask)

    def sft_train_step(self, all_tokens, all_advantages, lr, weight_decay):
        """Run one SFT/ECHO update.

        ``importance_sampling`` preserves the historical warmup behavior.
        ``cross_entropy`` gives the direct next-token world-modeling objective
        ECHO needs for prompt-side environment/tool tokens.
        """
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

        self._clear_inference_prefix_cache()

        if self._train_future is not None:
            self._pending_loss = self._train_future.result()
            self._train_future = None

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = weight_decay

        token_tensors = [torch.tensor(t, dtype=torch.long) for t in all_tokens]
        adv_tensors = [torch.tensor(a, dtype=torch.float32) for a in all_advantages]

        input_ids = pad_sequence(token_tensors, batch_first=True, padding_value=0).to(self.train_device)
        advantages = pad_sequence(adv_tensors, batch_first=True, padding_value=0.0).to(self.train_device)

        lengths = torch.tensor([len(t) for t in all_tokens], device=self.train_device)
        max_len = input_ids.shape[1]
        attention_mask = torch.arange(max_len, device=self.train_device).unsqueeze(0) < lengths.unsqueeze(1)

        return self._do_sft_impl(input_ids, advantages, attention_mask)

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
    ):
        """Run RL + ECHO masks over the same rollout rows."""
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

        self._clear_inference_prefix_cache()

        if self._train_future is not None:
            self._pending_loss = self._train_future.result()
            self._train_future = None

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = weight_decay

        token_tensors = [torch.tensor(t, dtype=torch.long) for t in all_tokens]
        lp_tensors = [torch.tensor(lp, dtype=torch.float32) for lp in all_logprobs]
        adv_tensors = [torch.tensor(a, dtype=torch.float32) for a in all_advantages]
        echo_adv_tensors = [
            torch.tensor(a, dtype=torch.float32) for a in echo_advantages
        ]

        input_ids = pad_sequence(token_tensors, batch_first=True, padding_value=0).to(self.train_device)
        old_logprobs = pad_sequence(lp_tensors, batch_first=True, padding_value=0.0).to(self.train_device)
        advantages = pad_sequence(adv_tensors, batch_first=True, padding_value=0.0).to(self.train_device)
        echo_advantages_tensor = pad_sequence(echo_adv_tensors, batch_first=True, padding_value=0.0).to(self.train_device)
        echo_counts = torch.tensor(
            [float(c) for c in echo_full_observation_counts],
            dtype=torch.float32,
            device=self.train_device,
        )

        lengths = torch.tensor([len(t) for t in all_tokens], device=self.train_device)
        max_len = input_ids.shape[1]
        attention_mask = torch.arange(max_len, device=self.train_device).unsqueeze(0) < lengths.unsqueeze(1)

        return self._do_hybrid_mask_impl(
            input_ids,
            old_logprobs,
            advantages,
            attention_mask,
            echo_advantages_tensor,
            echo_counts,
            echo_loss_fn,
        )

    def load_state(self, name):
        """Load adapter weights from a saved checkpoint for resume.

        Restores LoRA adapter weights into the train model and re-syncs
        to the inference engine. Optimizer state (Adam momentum/variance)
        is NOT restored — training will re-warm the optimizer.

        Args:
            name: Checkpoint name (subdirectory under adapter_path).
        """
        save_dir = os.path.join(self.adapter_path, name)
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(
                f"Adapter checkpoint not found: {save_dir}. "
                f"Cannot resume from checkpoint '{name}'."
            )

        from peft import set_peft_model_state_dict

        safetensors_path = os.path.join(save_dir, "adapter_model.safetensors")
        bin_path = os.path.join(save_dir, "adapter_model.bin")

        if os.path.isfile(safetensors_path):
            from safetensors.torch import load_file
            adapter_state = load_file(safetensors_path, device=str(self.train_device))
        elif os.path.isfile(bin_path):
            adapter_state = torch.load(bin_path, map_location=self.train_device, weights_only=True)
        else:
            raise FileNotFoundError(
                f"No adapter weights in {save_dir}. "
                f"Expected adapter_model.safetensors or adapter_model.bin."
            )

        set_peft_model_state_dict(self.train_model, adapter_state)

        # Re-sync weights to inference engine
        if self.split_mode or self._external_engine:
            snapshot = {}
            for pname, param in self.train_model.named_parameters():
                if "lora_" in pname:
                    snapshot[pname] = param.data.clone()
            self._weight_snapshot = snapshot
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

        save_dir = os.path.join(path, name)
        os.makedirs(save_dir, exist_ok=True)
        self.train_model.save_pretrained(save_dir)
        print(f"Adapter saved to {save_dir}")
        return save_dir
