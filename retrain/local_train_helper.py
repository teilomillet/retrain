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

import importlib
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from peft import get_peft_model, LoraConfig, TaskType

from retrain.inference_engine import create_engine


def _compute_policy_loss(ratio, adv, mask, clip_eps, clip_eps_high):
    """Compute PPO-style clipped policy loss.

    Args:
        ratio: Importance sampling ratio (new_prob / old_prob).
        adv: Per-token advantages.
        mask: Attention mask (1 for real tokens, 0 for padding).
        clip_eps: Lower clipping epsilon. 0 = no clipping.
        clip_eps_high: Upper clipping epsilon. 0 = use clip_eps (symmetric).

    Returns:
        (masked_loss, clip_fraction) tuple.
    """
    if clip_eps > 0:
        eps_high = clip_eps_high if clip_eps_high > 0 else clip_eps
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + eps_high)
        surr1 = ratio * adv
        surr2 = clipped_ratio * adv
        per_token_loss = -torch.min(surr1, surr2)
        with torch.no_grad():
            clipped = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + eps_high)).float()
            frac = (clipped * mask).sum().item() / mask.sum().clamp(min=1).item()
    else:
        per_token_loss = -(ratio * adv)
        frac = 0.0
    masked_loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1)
    return masked_loss, frac


def _parse_device(device_str):
    """Convert a local backend device spec to a torch device string."""
    device_str = device_str.strip().lower()
    if device_str in ("gpu", "cuda"):
        return "cuda:0"
    if device_str.startswith("gpu:"):
        device_index = device_str.split(":", 1)[1]
        if device_index.isdigit():
            return f"cuda:{device_index}"
    if device_str.startswith("cuda:"):
        device_index = device_str.split(":", 1)[1]
        if device_index.isdigit():
            return device_str
    if device_str == "cpu":
        return "cpu"
    if device_str in ("mps", "mps:0"):
        return "mps"
    raise ValueError(
        f"Unsupported local backend device {device_str!r}. "
        "Expected gpu:N, cuda:N, mps, mps:0, or cpu."
    )


def _device_type(device):
    if str(device).startswith("cuda"):
        return "cuda"
    if str(device).startswith("mps"):
        return "mps"
    if str(device) == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported torch device {device!r}.")


def _mps_is_available():
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def _resolve_available_device(device):
    device_kind = _device_type(device)
    if device_kind == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return "cpu"
    if device_kind == "mps" and not _mps_is_available():
        raise RuntimeError(
            "MPS was requested for the local backend, but PyTorch reports "
            "torch.backends.mps.is_available() == False."
        )
    return device


def _model_dtype_for_device(device):
    device_kind = _device_type(device)
    if device_kind == "cuda":
        return torch.bfloat16
    if device_kind == "mps":
        return torch.float16
    return torch.float32


def _use_amp_for_device(device):
    return _device_type(device) in ("cuda", "mps")


def _use_grad_scaler_for_device(device):
    return _device_type(device) == "cuda"


def _empty_accelerator_cache(device):
    device_kind = _device_type(device)
    if device_kind == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_kind == "mps" and _mps_is_available() and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _import_error(module_name):
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return exc
    return None


def _preflight_model_type(model_name, trust_remote_code):
    if trust_remote_code:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return str(getattr(config, "model_type", ""))

    config_dict, _ = PretrainedConfig.get_config_dict(model_name)
    return str(config_dict.get("model_type", ""))


def _preflight_local_model_prerequisites(
    model_name,
    trust_remote_code,
    require_causal_conv1d,
    devices,
):
    model_type = _preflight_model_type(model_name, trust_remote_code)
    if model_type != "nemotron_h":
        return

    if not trust_remote_code:
        raise RuntimeError(
            "Nemotron-H models require Hugging Face custom model code. "
            "Set [backend.options] trust_remote_code = true for local training."
        )

    mamba_error = _import_error("mamba_ssm")
    if mamba_error is not None:
        raise RuntimeError(
            "Nemotron-H local Transformers loading requires mamba-ssm "
            "(import mamba_ssm failed). Install mamba-ssm in the retrain "
            f"environment before training {model_name!r}."
        ) from mamba_error

    uses_cuda = torch.cuda.is_available() and any(
        str(device).startswith("cuda") for device in devices
    )
    if not uses_cuda:
        return

    causal_conv_error = _import_error("causal_conv1d")
    if causal_conv_error is None:
        return

    message = (
        "Nemotron-H on CUDA is missing causal-conv1d; the fast Mamba path "
        "will be unavailable. Install causal-conv1d for the BF16 CUDA path."
    )
    if require_causal_conv1d:
        raise RuntimeError(message) from causal_conv_error
    warnings.warn(message, RuntimeWarning, stacklevel=2)


class LocalTrainHelper:
    """Local GPU helper: pluggable inference engine + PyTorch/PEFT training."""

    def __init__(self, model_name, adapter_path, devices, lora_rank=32,
                 engine_type="pytorch", inference_url="",
                 lora_alpha=0, lora_dropout=0.0,
                 optim_beta1=0.9, optim_beta2=0.95, optim_eps=1e-8,
                 clip_eps=0.0, clip_eps_high=0.0,
                 trust_remote_code=False, require_causal_conv1d=False,
                 train_microbatch_size=0, cuda_empty_cache=False,
                 sample_use_cache=True):
        self.adapter_path = adapter_path
        self.model_name = model_name
        self.engine_type = engine_type
        self.clip_eps = clip_eps
        self.clip_eps_high = clip_eps_high
        self._clip_fraction = 0.0
        self.train_microbatch_size = int(train_microbatch_size or 0)
        self.cuda_empty_cache = bool(cuda_empty_cache)
        self.sample_use_cache = bool(sample_use_cache)

        # Parse all devices from comma-separated spec
        raw_devices = [d.strip() for d in devices.split(",") if d.strip()]
        parsed_devices = [_parse_device(d) for d in raw_devices]

        # Determine split mode: need >1 device and CUDA available
        cuda_devices = [d for d in parsed_devices if d.startswith("cuda")]
        self.split_mode = len(cuda_devices) > 1 and torch.cuda.is_available()

        # Non-PyTorch engines manage their own inference independently.
        # Server engines use a remote process; MAX engine uses its own in-process model.
        self._server_engine = engine_type in ("vllm", "sglang", "mlx", "openai")
        self._external_engine = engine_type in ("max", "vllm", "sglang", "mlx", "openai")

        if self._external_engine:
            # Server handles inference — all local GPUs for training
            device = parsed_devices[-1] if parsed_devices else "cuda:0"
            device = _resolve_available_device(device)
            self.infer_device = device  # not used for sampling, but kept for compat
            self.train_device = device
            self.split_mode = False
        elif self.split_mode:
            self.infer_device = cuda_devices[0]   # first GPU for inference
            self.train_device = cuda_devices[-1]   # last GPU for training
        else:
            # Single-model mode: use first device (with CUDA fallback)
            device = parsed_devices[0] if parsed_devices else "cuda:0"
            device = _resolve_available_device(device)
            self.infer_device = device
            self.train_device = device

        self.train_device_type = _device_type(self.train_device)
        self.use_amp = _use_amp_for_device(self.train_device)
        self.autocast_dtype = _model_dtype_for_device(self.train_device)
        dtype = self.autocast_dtype

        _preflight_local_model_prerequisites(
            model_name,
            trust_remote_code,
            require_causal_conv1d,
            devices=(self.train_device, self.infer_device),
        )

        effective_alpha = lora_alpha if lora_alpha > 0 else lora_rank * 2
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=effective_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

        # Create train model
        print(f"Loading train model: {model_name} on {self.train_device}...")
        base_train = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
        self.train_model = get_peft_model(base_train, peft_config)
        self.train_model.to(self.train_device)
        self.train_model.print_trainable_parameters()

        # Gradient checkpointing — trade compute for VRAM
        self.train_model.gradient_checkpointing_enable()

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
                trust_remote_code=trust_remote_code,
                use_cache=self.sample_use_cache,
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
                trust_remote_code=trust_remote_code,
                use_cache=self.sample_use_cache,
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
                peft_config=peft_config,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                use_cache=self.sample_use_cache,
            )
            # Point the engine's model to the train model (same object)
            self.engine.model = self.train_model

        # Optimizer (only for train_model)
        self.optimizer = torch.optim.AdamW(
            self.train_model.parameters(),
            lr=4e-5,
            betas=(optim_beta1, optim_beta2),
            eps=optim_eps,
            weight_decay=0.0,
        )

        # AMP scaler for mixed precision
        self.scaler = torch.amp.GradScaler(
            enabled=self.use_amp and _use_grad_scaler_for_device(self.train_device)
        )

        print(f"LocalTrainHelper ready (engine={engine_type}, split_mode={self.split_mode}).")

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
        if self._external_engine:
            if self._weights_dirty and self._weight_snapshot is not None:
                # Save adapter to disk, then tell engine to reload
                save_dir = os.path.join(self.adapter_path, "_live_adapter")
                os.makedirs(save_dir, exist_ok=True)
                self.train_model.save_pretrained(save_dir)
                self.engine.reload_weights(save_dir)
                self._weights_dirty = False
            return

        if not self.split_mode:
            return
        if self._weight_snapshot is None:
            return

        self.engine.sync_from_state_dict(self._weight_snapshot)

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
        engine_results = self.engine.generate(
            prompt_ids_list, num_samples, max_tokens, temperature, top_p
        )
        self._maybe_empty_cuda_cache()

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
        engine_results = self.engine.generate(
            prompt_ids_list, num_samples, max_tokens, temperature, top_p,
            compute_entropy=True,
        )
        self._maybe_empty_cuda_cache()

        results = []
        for group in engine_results:
            converted = []
            for sr in group:
                converted.append((sr.token_ids, sr.logprobs, sr.token_entropies))
            results.append(converted)
        return results

    def runtime_metrics(self):
        """Expose optional engine-level runtime counters to the trainer."""
        if hasattr(self.engine, "performance_counters"):
            counters = self.engine.performance_counters()
            if isinstance(counters, dict):
                return dict(counters)
        return {}

    def _maybe_empty_cuda_cache(self):
        if self.cuda_empty_cache:
            _empty_accelerator_cache(self.train_device)

    def _snapshot_lora_weights_after_step(self):
        if self.split_mode or self._external_engine:
            snapshot = {}
            for name, param in self.train_model.named_parameters():
                if "lora_" in name:
                    snapshot[name] = param.data.clone()
            self._weight_snapshot = snapshot
            self._weights_dirty = True
        self._maybe_empty_cuda_cache()

    def _loss_for_batch(self, input_ids, old_logprobs, advantages, attention_mask):
        with torch.amp.autocast(
            device_type=self.train_device_type,
            dtype=self.autocast_dtype,
            enabled=self.use_amp,
        ):
            outputs = self.train_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1]  # [N, max_len-1, vocab] — shift
            new_logprobs = F.log_softmax(logits.float(), dim=-1)

            # Gather logprobs for target tokens (shifted by 1)
            target_ids = input_ids[:, 1:]  # [N, max_len-1]
            new_logprobs = new_logprobs.gather(2, target_ids.unsqueeze(2)).squeeze(2)  # [N, max_len-1]

            # Align with advantages/old_logprobs (also shifted by 1)
            old_lp = old_logprobs[:, 1:]  # [N, max_len-1]
            adv = advantages[:, 1:]       # [N, max_len-1]
            mask = attention_mask[:, 1:]   # [N, max_len-1] — exclude padding

            # Importance sampling loss with optional PPO-style ratio clipping
            ratio = torch.exp(new_logprobs - old_lp)
            masked_loss, clip_frac = _compute_policy_loss(
                ratio, adv, mask, self.clip_eps, self.clip_eps_high
            )
        token_count = mask.sum().clamp(min=1).item()
        return masked_loss, clip_frac, token_count

    def _do_train_impl(self, input_ids, old_logprobs, advantages, attention_mask):
        """Execute training forward/backward/step on pre-prepared tensors.

        After the optimizer step, clones LoRA params into _weight_snapshot
        for safe cross-thread syncing. Runs on background thread in split mode.

        Returns:
            Scalar loss value.
        """
        self.train_model.train()
        self.optimizer.zero_grad()

        masked_loss, clip_frac, _ = self._loss_for_batch(
            input_ids, old_logprobs, advantages, attention_mask
        )
        self._clip_fraction = clip_frac

        self.scaler.scale(masked_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        loss_val = masked_loss.item()
        self._snapshot_lora_weights_after_step()

        return loss_val

    def _do_train_microbatched_impl(self, input_ids, old_logprobs, advantages, attention_mask):
        """Run one optimizer step using smaller forward/backward microbatches."""
        microbatch_size = self.train_microbatch_size
        if microbatch_size <= 0 or input_ids.shape[0] <= microbatch_size:
            return self._do_train_impl(input_ids, old_logprobs, advantages, attention_mask)

        self.train_model.train()
        self.optimizer.zero_grad()

        total_tokens = attention_mask[:, 1:].sum().clamp(min=1).item()
        weighted_loss = 0.0
        weighted_clip = 0.0
        for start in range(0, input_ids.shape[0], microbatch_size):
            end = start + microbatch_size
            chunk_loss, chunk_clip, chunk_tokens = self._loss_for_batch(
                input_ids[start:end],
                old_logprobs[start:end],
                advantages[start:end],
                attention_mask[start:end],
            )
            weight = float(chunk_tokens) / float(total_tokens)
            self.scaler.scale(chunk_loss * weight).backward()
            weighted_loss += float(chunk_loss.item()) * weight
            weighted_clip += float(chunk_clip) * weight

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self._clip_fraction = weighted_clip
        self._snapshot_lora_weights_after_step()

        return weighted_loss

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
                self._do_train_microbatched_impl,
                input_ids,
                old_logprobs,
                advantages,
                attention_mask,
            )

            # Return loss from the previously completed training step
            return self._pending_loss
        else:
            # Synchronous path: run training inline, return current loss
            return self._do_train_microbatched_impl(
                input_ids, old_logprobs, advantages, attention_mask
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
