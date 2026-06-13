"""PyTorch inference engine — local GPU sampling with PEFT LoRA.

Extracted from LocalTrainHelper.sample(). Loads its own PEFT model on
the inference device, generates completions with per-token logprobs
via model.generate + log_softmax.

Supports two weight-sync modes:
- sync_from_state_dict(lora_dict): fast in-memory copy (split mode)
- reload_weights(path): load adapter from disk (not typically used)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import get_peft_model

from retrain.gemma4_text import eos_token_ids, is_gemma4_text_model, unwrap_peft_model
from retrain.inference_engine.base import InferenceEngine, SampleResult


def _sample_next_token(logits, temperature, top_p):
    scaled = logits / max(float(temperature), 1e-7)
    probs = F.softmax(scaled.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs > top_p
        filtered = sorted_probs.masked_fill(remove, 0.0)
        filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled_sorted = torch.multinomial(filtered, num_samples=1)
        next_token = sorted_indices.gather(1, sampled_sorted)
        next_prob = filtered.gather(1, sampled_sorted).squeeze(1)
    else:
        next_token = torch.multinomial(probs, num_samples=1)
        next_prob = probs.gather(1, next_token).squeeze(1)

    return next_token, next_prob.clamp_min(1e-12).log(), entropy


class PyTorchEngine(InferenceEngine):
    """Local PyTorch/PEFT inference engine."""

    def __init__(
        self,
        model_name,
        device,
        peft_config,
        dtype,
        existing_model=None,
        sample_use_cache=True,
    ):
        """Load a PEFT-wrapped model for inference.

        Args:
            model_name: HuggingFace model ID.
            device: Torch device string (e.g. "cuda:0", "cpu").
            peft_config: LoraConfig for PEFT wrapping. Must be None when
                existing_model is provided, because that model is expected to
                already be PEFT-wrapped and configured.
            dtype: Model dtype (bfloat16 or float32).
        """
        self.device = device
        self.model_name = model_name
        self.sample_use_cache = bool(sample_use_cache)

        if existing_model is not None:
            if peft_config is not None:
                raise ValueError(
                    "peft_config must be None when existing_model is provided; "
                    "existing_model is expected to be already PEFT-wrapped."
                )
            self.model = existing_model.to(device)
        else:
            print(f"Loading infer model: {model_name} on {device}...")
            base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
            self.model = get_peft_model(base, peft_config)
            self.model.to(device)

    def generate(self, prompt_ids_list, num_samples, max_tokens, temperature, top_p,
                 compute_entropy=False):
        """Generate completions with per-token logprobs via PyTorch."""
        results = []
        self.model.eval()

        if is_gemma4_text_model(self.model):
            return self._generate_gemma4_text(
                prompt_ids_list,
                num_samples,
                max_tokens,
                temperature,
                top_p,
                compute_entropy,
            )

        with torch.no_grad():
            for prompt_ids in prompt_ids_list:
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

                # Vectorized logprob extraction
                all_entropies = None
                if len(outputs.scores) > 0:
                    all_scores = torch.stack(outputs.scores, dim=1)
                    all_log_probs = F.log_softmax(all_scores.float(), dim=-1)

                    if compute_entropy:
                        all_probs = F.softmax(all_scores.float(), dim=-1)
                        # H(t) = -sum(p * log(p)) per position
                        all_entropies = -(all_probs * all_log_probs).sum(dim=-1)

                group = []
                for i in range(num_samples):
                    gen_ids = outputs.sequences[i][prompt_len:].tolist()
                    gen_len = len(gen_ids)

                    if gen_len > 0 and len(outputs.scores) > 0:
                        gen_tensor = outputs.sequences[i][prompt_len:prompt_len + gen_len]
                        logprobs = all_log_probs[i, :gen_len].gather(
                            1, gen_tensor.unsqueeze(1)
                        ).squeeze(1).tolist()
                    else:
                        logprobs = []

                    token_entropies = None
                    if all_entropies is not None and gen_len > 0:
                        token_entropies = all_entropies[i, :gen_len].tolist()

                    group.append(SampleResult(
                        token_ids=gen_ids,
                        logprobs=logprobs,
                        token_entropies=token_entropies,
                    ))
                results.append(group)

        return results

    def _generate_gemma4_text(self, prompt_ids_list, num_samples, max_tokens, temperature, top_p,
                              compute_entropy=False):
        """Text-only Gemma4 sampling path avoiding the multimodal wrapper."""
        results = []
        unwrapped = unwrap_peft_model(self.model)
        text_model = unwrapped.model.language_model
        lm_head = unwrapped.lm_head
        eos_ids = eos_token_ids(self.model)

        with torch.no_grad():
            for prompt_ids in prompt_ids_list:
                input_ids = torch.tensor([prompt_ids] * num_samples, device=self.device)
                generated = input_ids
                attention_mask = torch.ones_like(generated, device=self.device)
                past_key_values = None
                shared_kv_states = None
                generated_tokens = [[] for _ in range(num_samples)]
                generated_logprobs = [[] for _ in range(num_samples)]
                generated_entropies = [[] for _ in range(num_samples)] if compute_entropy else None
                finished = torch.zeros(num_samples, dtype=torch.bool, device=self.device)

                for _ in range(max_tokens):
                    use_cache = self.sample_use_cache
                    step_input_ids = (
                        generated
                        if past_key_values is None or not use_cache
                        else generated[:, -1:]
                    )
                    kwargs = {}
                    if use_cache:
                        kwargs["return_shared_kv_states"] = True
                        if shared_kv_states is not None:
                            kwargs["shared_kv_states"] = shared_kv_states

                    model_kwargs = {
                        "input_ids": step_input_ids,
                        "attention_mask": attention_mask,
                        "use_cache": use_cache,
                        **kwargs,
                    }
                    if use_cache:
                        model_kwargs["past_key_values"] = past_key_values
                    outputs = text_model(**model_kwargs)
                    logits = lm_head(outputs.last_hidden_state[:, -1:, :])[:, -1, :]
                    next_token, logprob, entropy = _sample_next_token(logits, temperature, top_p)
                    generated = torch.cat([generated, next_token], dim=-1)
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
                    past_key_values = (
                        getattr(outputs, "past_key_values", None) if use_cache else None
                    )
                    shared_kv_states = (
                        getattr(outputs, "shared_kv_states", None) if use_cache else None
                    )

                    for i in range(num_samples):
                        if finished[i]:
                            continue
                        token = int(next_token[i].item())
                        generated_tokens[i].append(token)
                        generated_logprobs[i].append(float(logprob[i].item()))
                        if generated_entropies is not None:
                            generated_entropies[i].append(float(entropy[i].item()))
                        if token in eos_ids:
                            finished[i] = True

                    if bool(finished.all()):
                        break

                group = []
                for i in range(num_samples):
                    group.append(SampleResult(
                        token_ids=generated_tokens[i],
                        logprobs=generated_logprobs[i],
                        token_entropies=generated_entropies[i] if generated_entropies is not None else None,
                    ))
                results.append(group)

        return results

    def reload_weights(self, adapter_path):
        """Load adapter weights from disk."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        import os

        safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path, device=str(self.device))
            set_peft_model_state_dict(self.model, state_dict)
        else:
            # Fall back to bin format
            bin_path = os.path.join(adapter_path, "adapter_model.bin")
            if os.path.exists(bin_path):
                state_dict = torch.load(bin_path, map_location=self.device)
                set_peft_model_state_dict(self.model, state_dict)

    def sync_from_state_dict(self, lora_dict):
        """Fast in-memory LoRA weight sync (split mode).

        Args:
            lora_dict: Dict mapping parameter names to tensors
                       (from train_model.named_parameters snapshot).
        """
        for name, param in self.model.named_parameters():
            if "lora_" in name and name in lora_dict:
                param.data.copy_(lora_dict[name].to(param.device))

    def shutdown(self):
        """Release model from GPU memory."""
        del self.model
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
