"""Python helper for MAXBackend: PyTorch/PEFT training + inference fallback.

MAX is inference-only (no autograd). This helper provides:
- PyTorch model with PEFT LoRA for training (gradient computation)
- PyTorch generate with logprobs for inference (fallback, upgradeable to MAX)
- Adapter save/load for weight synchronization

The MAXBackend in Mojo calls into this module via Python interop.
"""

import os

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class MAXTrainHelper:
    """Hybrid helper: PyTorch/PEFT training + inference."""

    def __init__(self, model_name, adapter_path, devices, lora_rank=32):
        self.adapter_path = adapter_path
        self.model_name = model_name

        # Parse device from devices spec (e.g. "gpu:0" -> "cuda:0", "cpu" -> "cpu")
        device_str = devices.split(",")[0].strip()
        if device_str.startswith("gpu:"):
            self.device = device_str.replace("gpu:", "cuda:")
        elif device_str == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda:0"

        # Check if CUDA is actually available
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"

        self.use_amp = self.device != "cpu"

        print(f"Loading base model: {model_name} on {self.device}...")
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        )

        # Apply LoRA
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
        self.model = get_peft_model(self.model, peft_config)
        self.model.to(self.device)
        self.model.print_trainable_parameters()

        # Gradient checkpointing — trade compute for VRAM
        self.model.gradient_checkpointing_enable()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=4e-5,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

        # AMP scaler for mixed precision
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        print("MAXTrainHelper ready.")

    def checkpoint(self, name):
        """Prepare for sampling. In PyTorch fallback, model is always current."""
        pass

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
        self.model.eval()

        with torch.no_grad():
            for prompt_ids in prompt_ids_list:
                # Expand prompt for num_samples
                input_ids = torch.tensor(
                    [prompt_ids] * num_samples, device=self.device
                )
                prompt_len = len(prompt_ids)

                outputs = self.model.generate(
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

    def train_step(self, all_tokens, all_logprobs, all_advantages, lr, weight_decay):
        """Run one training step with importance sampling loss.

        Pads sequences into a single batch for one forward pass instead
        of looping per-datum. Uses torch.cuda.amp for mixed precision.

        Args:
            all_tokens: List of full token sequences (prompt + completion).
            all_logprobs: List of per-token logprobs (0-padded for prompt).
            all_advantages: List of per-token advantages (0-padded for prompt).
            lr: Learning rate.
            weight_decay: Weight decay coefficient.

        Returns:
            Mean loss over the batch.
        """
        n = len(all_tokens)
        if n == 0:
            return 0.0

        # Update optimizer hyperparameters
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = weight_decay

        self.model.train()
        self.optimizer.zero_grad()

        # Pad all sequences into a single batch
        token_tensors = [torch.tensor(t, dtype=torch.long) for t in all_tokens]
        lp_tensors = [torch.tensor(lp, dtype=torch.float32) for lp in all_logprobs]
        adv_tensors = [torch.tensor(a, dtype=torch.float32) for a in all_advantages]

        # pad_sequence pads to max length in batch (batch_first=True -> [N, max_len])
        input_ids = pad_sequence(token_tensors, batch_first=True, padding_value=0).to(self.device)
        old_logprobs = pad_sequence(lp_tensors, batch_first=True, padding_value=0.0).to(self.device)
        advantages = pad_sequence(adv_tensors, batch_first=True, padding_value=0.0).to(self.device)

        # Build attention mask: 1 where real tokens, 0 where padding
        lengths = torch.tensor([len(t) for t in all_tokens], device=self.device)
        max_len = input_ids.shape[1]
        attention_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)

        # Single batched forward pass with AMP
        with torch.amp.autocast(device_type=self.device.split(":")[0], enabled=self.use_amp):
            outputs = self.model(input_ids, attention_mask=attention_mask)
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

        return masked_loss.item()

    def save_adapter(self, path, name):
        """Save LoRA adapter to disk.

        Args:
            path: Base directory for adapter storage.
            name: Checkpoint name (creates subdirectory).
        """
        save_dir = os.path.join(path, name)
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        print(f"Adapter saved to {save_dir}")
