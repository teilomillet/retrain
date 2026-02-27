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
from peft import get_peft_model, LoraConfig, TaskType

from retrain.inference_engine.base import InferenceEngine, SampleResult


class PyTorchEngine(InferenceEngine):
    """Local PyTorch/PEFT inference engine."""

    def __init__(self, model_name, device, peft_config, dtype):
        """Load a PEFT-wrapped model for inference.

        Args:
            model_name: HuggingFace model ID.
            device: Torch device string (e.g. "cuda:0", "cpu").
            peft_config: LoraConfig for PEFT wrapping.
            dtype: Model dtype (bfloat16 or float32).
        """
        self.device = device
        self.model_name = model_name

        print(f"Loading infer model: {model_name} on {device}...")
        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model = get_peft_model(base, peft_config)
        self.model.to(device)

    def generate(self, prompt_ids_list, num_samples, max_tokens, temperature, top_p,
                 compute_entropy=False):
        """Generate completions with per-token logprobs via PyTorch."""
        results = []
        self.model.eval()

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
