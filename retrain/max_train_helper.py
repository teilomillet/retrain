"""Python helper for MAXBackend: PyTorch/PEFT training + inference fallback.

MAX is inference-only (no autograd). This helper provides:
- PyTorch model with PEFT LoRA for training (gradient computation)
- PyTorch generate with logprobs for inference (fallback, upgradeable to MAX)
- Adapter save/load for weight synchronization

GPU split mode (multi-GPU): separate devices for inference and training.
- infer_model on first device (sampling)
- train_model on last device (gradient updates)
- checkpoint() syncs LoRA weights train -> infer

The MAXBackend in Mojo calls into this module via Python interop.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


def _parse_device(device_str):
    """Convert a device spec like 'gpu:0' to a torch device string like 'cuda:0'."""
    device_str = device_str.strip()
    if device_str.startswith("gpu:"):
        return device_str.replace("gpu:", "cuda:")
    elif device_str == "cpu":
        return "cpu"
    else:
        return "cuda:0"


class MAXTrainHelper:
    """Hybrid helper: PyTorch/PEFT training + inference."""

    def __init__(self, model_name, adapter_path, devices, lora_rank=32):
        self.adapter_path = adapter_path
        self.model_name = model_name

        # Parse all devices from comma-separated spec
        raw_devices = [d.strip() for d in devices.split(",") if d.strip()]
        parsed_devices = [_parse_device(d) for d in raw_devices]

        # Determine split mode: need >1 device and CUDA available
        cuda_devices = [d for d in parsed_devices if d.startswith("cuda")]
        self.split_mode = len(cuda_devices) > 1 and torch.cuda.is_available()

        if self.split_mode:
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

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

        # Create train model
        print(f"Loading train model: {model_name} on {self.train_device}...")
        base_train = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
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

        if self.split_mode:
            self._train_executor = ThreadPoolExecutor(max_workers=1)

            # Create separate infer model on a different device
            print(f"Loading infer model: {model_name} on {self.infer_device}...")
            base_infer = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype
            )
            self.infer_model = get_peft_model(base_infer, peft_config)
            self.infer_model.to(self.infer_device)
            # Initial sync: copy directly from train_model (no snapshot yet)
            self._do_initial_sync()
        else:
            # Single-model mode: same object for both roles
            self.infer_model = self.train_model

        # Optimizer (only for train_model)
        self.optimizer = torch.optim.AdamW(
            self.train_model.parameters(),
            lr=4e-5,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

        # AMP scaler for mixed precision
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        print(f"MAXTrainHelper ready (split_mode={self.split_mode}).")

    def _do_initial_sync(self):
        """Copy LoRA weights from train_model to infer_model at init time.

        Used once during __init__ before any training has occurred (no snapshot
        exists yet). After this, all syncs go through _sync_lora_weights which
        reads from the weight snapshot.
        """
        train_state = dict(self.train_model.named_parameters())
        for name, infer_param in self.infer_model.named_parameters():
            if "lora_" in name and name in train_state:
                infer_param.data.copy_(train_state[name].data.to(infer_param.device))

    def _sync_lora_weights(self):
        """Copy LoRA weights from snapshot to infer_model.

        In split mode: copies from _weight_snapshot (created after each completed
        training step) rather than live train_model weights. This is safe to call
        while training is running on GPU 1 — reads snapshot, not live weights.

        No-op when split_mode is False (infer_model is train_model) or when
        no training has completed yet (_weight_snapshot is None).
        """
        if not self.split_mode:
            return
        if self._weight_snapshot is None:
            return

        for name, infer_param in self.infer_model.named_parameters():
            if "lora_" in name and name in self._weight_snapshot:
                infer_param.data.copy_(self._weight_snapshot[name].to(infer_param.device))

    def checkpoint(self, name):
        """Prepare for sampling by syncing LoRA weights from snapshot -> infer.

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

        Args:
            prompt_ids_list: List of lists of token IDs (one per prompt).
            num_samples: Number of completions per prompt.
            max_tokens: Maximum new tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            List of lists of (token_ids, logprobs) tuples.
        """
        results = []
        self.infer_model.eval()

        with torch.no_grad():
            for prompt_ids in prompt_ids_list:
                # Expand prompt for num_samples
                input_ids = torch.tensor(
                    [prompt_ids] * num_samples, device=self.infer_device
                )
                prompt_len = len(prompt_ids)

                outputs = self.infer_model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 1e-7),
                    top_p=top_p,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                # Vectorized logprob extraction: stack all scores at once
                # outputs.scores is a tuple of (num_steps,) tensors each [num_samples, vocab]
                if len(outputs.scores) > 0:
                    all_scores = torch.stack(outputs.scores, dim=1)  # [num_samples, steps, vocab]
                    all_log_probs = F.log_softmax(all_scores.float(), dim=-1)

                group = []
                for i in range(num_samples):
                    gen_ids = outputs.sequences[i][prompt_len:].tolist()
                    gen_len = len(gen_ids)

                    if gen_len > 0 and len(outputs.scores) > 0:
                        # Gather logprobs for chosen tokens in one shot
                        gen_tensor = outputs.sequences[i][prompt_len:prompt_len + gen_len]
                        logprobs = all_log_probs[i, :gen_len].gather(
                            1, gen_tensor.unsqueeze(1)
                        ).squeeze(1).tolist()
                    else:
                        logprobs = []

                    group.append((gen_ids, logprobs))
                results.append(group)

        return results

    def _do_train_impl(self, input_ids, old_logprobs, advantages, attention_mask):
        """Execute training forward/backward/step on pre-prepared tensors.

        After the optimizer step, clones LoRA params into _weight_snapshot
        for safe cross-thread syncing. Runs on background thread in split mode.

        Returns:
            Scalar loss value.
        """
        self.train_model.train()
        self.optimizer.zero_grad()

        # Single batched forward pass with AMP
        with torch.amp.autocast(device_type=self.train_device.split(":")[0], enabled=self.use_amp):
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

            # Importance sampling loss: -mean(ratio * advantage) over real tokens
            ratio = torch.exp(new_logprobs - old_lp)
            per_token_loss = -(ratio * adv)
            # Mask out padding, average over real tokens
            masked_loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1)

        self.scaler.scale(masked_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        loss_val = masked_loss.item()

        # Snapshot LoRA weights for safe cross-thread sync
        if self.split_mode:
            snapshot = {}
            for name, param in self.train_model.named_parameters():
                if "lora_" in name:
                    snapshot[name] = param.data.clone()
            self._weight_snapshot = snapshot

        return loss_val

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
                self._do_train_impl, input_ids, old_logprobs, advantages, attention_mask
            )

            # Return loss from the previously completed training step
            return self._pending_loss
        else:
            # Synchronous path: run training inline, return current loss
            return self._do_train_impl(input_ids, old_logprobs, advantages, attention_mask)

    def save_adapter(self, path, name):
        """Save LoRA adapter to disk.

        Flushes any pending async training before saving to ensure
        the saved weights include the latest completed training step.

        Args:
            path: Base directory for adapter storage.
            name: Checkpoint name (creates subdirectory).
        """
        # Flush pending training before saving
        if self._train_future is not None:
            self._pending_loss = self._train_future.result()
            self._train_future = None

        save_dir = os.path.join(path, name)
        os.makedirs(save_dir, exist_ok=True)
        self.train_model.save_pretrained(save_dir)
        print(f"Adapter saved to {save_dir}")
