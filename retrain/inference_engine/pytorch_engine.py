"""PyTorch inference engine — local GPU sampling with PEFT LoRA.

Extracted from LocalTrainHelper.sample(). Loads its own PEFT model on
the inference device, generates completions with per-token logprobs
via model.generate + log_softmax.

Supports two weight-sync modes:
- sync_from_state_dict(lora_dict): fast in-memory copy (split mode)
- reload_weights(path): load adapter from disk (not typically used)
"""

import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import get_peft_model

from retrain.accelerators import (
    accelerator_status,
    apply_liger_kernel_if_available,
    from_pretrained_attention_kwargs,
)
from retrain.gemma4_text import is_gemma4_text_model, unwrap_peft_model
from retrain.inference_engine.base import InferenceEngine, SampleResult
from retrain.token_ids import model_eos_token_ids
from retrain.backends.torch import (
    is_cuda_device,
    timer_start as _timer_start,
    timer_stop as _timer_stop,
)


def _shannon_entropy_from_probs_logprobs(probs, log_probs):
    safe_log_probs = log_probs.masked_fill(probs == 0, 0.0)
    return -(probs * safe_log_probs).sum(dim=-1)


def _sample_next_token(logits, temperature, top_p, compute_entropy=False):
    scaled = logits / max(float(temperature), 1e-7)
    probs = F.softmax(scaled.float(), dim=-1)
    entropy = None
    if compute_entropy:
        entropy = _shannon_entropy_from_probs_logprobs(
            probs,
            probs.clamp_min(1e-12).log(),
        )

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


class _TimingAccumulator:
    """Accumulate CUDA timings without synchronizing every generated token."""

    def __init__(self, device):
        self.device = device
        self._cuda = is_cuda_device(device)
        self._events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._totals = {"prefill": 0.0, "decode": 0.0}

    def start(self):
        if self._cuda:
            with torch.cuda.device(torch.device(self.device)):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
            return event
        return time.perf_counter()

    def stop(self, start, bucket: str) -> None:
        if self._cuda:
            with torch.cuda.device(torch.device(self.device)):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
            self._events.append((bucket, start, end))
            return
        self._totals[bucket] = self._totals.get(bucket, 0.0) + (
            time.perf_counter() - start
        )

    def totals(self) -> dict[str, float]:
        if self._cuda and self._events:
            with torch.cuda.device(torch.device(self.device)):
                torch.cuda.synchronize()
            for bucket, start, end in self._events:
                self._totals[bucket] = self._totals.get(bucket, 0.0) + (
                    start.elapsed_time(end) / 1000.0
                )
            self._events.clear()
        return dict(self._totals)


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
        prefix_caching=True,
        attention_kernel="default",
        liger_kernel=True,
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
        self.prefix_caching = bool(prefix_caching)
        self.attention_kernel = str(attention_kernel or "default")
        self.liger_kernel = bool(liger_kernel)
        self._accelerator_metrics: dict[str, object] = accelerator_status()
        self._last_generation_metrics: dict[str, float | int] = {}
        self._prefix_cache: dict[
            tuple[int, ...],
            tuple[object, int],
        ] = {}
        self._prefix_cache_order: list[tuple[int, ...]] = []
        self._prefix_cache_max_entries = 64
        self._prefix_cache_hits = 0
        self._prefix_cache_misses = 0
        self._prefix_cache_fallbacks = 0
        self._prefix_cache_reused_tokens = 0
        self._prefix_cache_append_s = 0.0

        if existing_model is not None:
            if peft_config is not None:
                raise ValueError(
                    "peft_config must be None when existing_model is provided; "
                    "existing_model is expected to be already PEFT-wrapped."
                )
            self.model = existing_model
        else:
            self._accelerator_metrics.update(
                apply_liger_kernel_if_available(
                    model_name,
                    enabled=self.liger_kernel,
                )
            )
            print(f"Loading infer model: {model_name} on {device}...")
            model_kwargs = from_pretrained_attention_kwargs(self.attention_kernel)
            base = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                **model_kwargs,
            )
            self.model = get_peft_model(base, peft_config)
            self.model.to(device)

    def generate(self, prompt_ids_list, num_samples, max_tokens, temperature, top_p,
                 compute_entropy=False):
        """Generate completions with per-token logprobs via PyTorch."""
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

        results = []
        generation_start = time.perf_counter()
        timings = _TimingAccumulator(self.device)
        eos_ids = model_eos_token_ids(self.model, unwrap_model=unwrap_peft_model)

        with torch.no_grad():
            for prompt_ids in prompt_ids_list:
                generated = torch.tensor([prompt_ids] * num_samples, device=self.device)
                attention_mask = torch.ones_like(generated, device=self.device)
                past_key_values = None
                token_steps = []
                logprob_steps = []
                entropy_steps = [] if compute_entropy else None
                generated_lengths = torch.zeros(
                    num_samples,
                    dtype=torch.long,
                    device=self.device,
                )
                finished = torch.zeros(num_samples, dtype=torch.bool, device=self.device)

                for step_idx in range(max_tokens):
                    use_cache = self.sample_use_cache
                    timer = timings.start()
                    if step_idx == 0:
                        outputs = self._prefill_prompt(
                            prompt_ids,
                            generated,
                            attention_mask,
                            num_samples,
                            use_cache,
                        )
                    else:
                        model_kwargs = {
                            "input_ids": (
                                generated
                                if past_key_values is None or not use_cache
                                else generated[:, -1:]
                            ),
                            "attention_mask": attention_mask,
                            "use_cache": use_cache,
                            "logits_to_keep": 1,
                        }
                        if use_cache and past_key_values is not None:
                            model_kwargs["past_key_values"] = past_key_values
                        outputs = self._forward_model(
                            model_kwargs,
                            fallback_input_ids=generated,
                            fallback_attention_mask=attention_mask,
                        )
                    timings.stop(timer, "prefill" if step_idx == 0 else "decode")

                    logits = outputs.logits[:, -1, :]
                    next_token, logprob, entropy = _sample_next_token(
                        logits,
                        temperature,
                        top_p,
                        compute_entropy=compute_entropy,
                    )
                    generated = torch.cat([generated, next_token], dim=-1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(next_token)],
                        dim=-1,
                    )
                    past_key_values = (
                        getattr(outputs, "past_key_values", None) if use_cache else None
                    )

                    active = ~finished
                    token_values = next_token.squeeze(1)
                    token_steps.append(token_values.detach())
                    logprob_steps.append(logprob.detach())
                    if entropy_steps is not None and entropy is not None:
                        entropy_steps.append(entropy.detach())
                    generated_lengths += active.to(torch.long)

                    if eos_ids:
                        eos_hit = torch.zeros_like(finished)
                        for eos_id in eos_ids:
                            eos_hit |= token_values.eq(int(eos_id))
                        finished |= active & eos_hit

                    if bool(finished.all()):
                        break

                generated_tokens, generated_logprobs, generated_entropies = (
                    self._materialize_generated_steps(
                        token_steps,
                        logprob_steps,
                        entropy_steps,
                        generated_lengths,
                        num_samples,
                    )
                )

                if (
                    self.prefix_caching
                    and self.sample_use_cache
                    and num_samples == 1
                    and past_key_values is not None
                    and generated_tokens
                ):
                    full_prefix_ids = list(prompt_ids) + list(generated_tokens[0])
                    past_key_values = self._advance_past_for_prefix_cache(
                        past_key_values,
                        full_prefix_ids,
                        generated,
                    )
                    self._store_prefix_cache(
                        full_prefix_ids,
                        past_key_values,
                        num_samples,
                    )

                group = [
                    SampleResult(
                        token_ids=generated_tokens[i],
                        logprobs=generated_logprobs[i],
                        token_entropies=(
                            generated_entropies[i]
                            if generated_entropies is not None
                            else None
                        ),
                    )
                    for i in range(num_samples)
                ]
                results.append(group)

        timing_totals = timings.totals()
        self._record_generation_metrics(
            generation_start,
            prompt_ids_list,
            num_samples,
            results,
            timing_totals.get("prefill", 0.0),
            timing_totals.get("decode", 0.0),
        )
        return results

    def _materialize_generated_steps(
        self,
        token_steps,
        logprob_steps,
        entropy_steps,
        generated_lengths,
        num_samples,
    ):
        lengths = [int(length) for length in generated_lengths.detach().cpu().tolist()]
        if token_steps:
            token_matrix = torch.stack(token_steps, dim=0).detach().cpu()
            logprob_matrix = torch.stack(logprob_steps, dim=0).detach().cpu()
            entropy_matrix = (
                torch.stack(entropy_steps, dim=0).detach().cpu()
                if entropy_steps is not None and entropy_steps
                else None
            )
        else:
            token_matrix = torch.empty((0, num_samples), dtype=torch.long)
            logprob_matrix = torch.empty((0, num_samples), dtype=torch.float32)
            entropy_matrix = None

        generated_tokens = []
        generated_logprobs = []
        generated_entropies = [] if entropy_steps is not None else None
        for sample_idx, length in enumerate(lengths):
            generated_tokens.append(
                [int(token) for token in token_matrix[:length, sample_idx].tolist()]
            )
            generated_logprobs.append(
                [
                    float(logprob)
                    for logprob in logprob_matrix[:length, sample_idx].tolist()
                ]
            )
            if generated_entropies is not None:
                if entropy_matrix is None:
                    generated_entropies.append([])
                else:
                    generated_entropies.append(
                        [
                            float(entropy)
                            for entropy in entropy_matrix[:length, sample_idx].tolist()
                        ]
                    )
        return generated_tokens, generated_logprobs, generated_entropies

    def _prefill_prompt(
        self,
        prompt_ids,
        generated,
        attention_mask,
        num_samples,
        use_cache,
    ):
        cached = self._find_prefix_cache(prompt_ids, num_samples)
        if use_cache and cached is not None:
            prefix_ids, past_key_values = cached
            suffix_ids = list(prompt_ids[len(prefix_ids):])
            if suffix_ids:
                suffix_input_ids = torch.tensor(
                    [suffix_ids] * num_samples,
                    device=self.device,
                )
                try:
                    outputs = self._forward_model(
                        {
                            "input_ids": suffix_input_ids,
                            "attention_mask": attention_mask,
                            "use_cache": True,
                            "past_key_values": past_key_values,
                            "logits_to_keep": 1,
                        },
                        fallback_input_ids=generated,
                        fallback_attention_mask=attention_mask,
                    )
                    self._prefix_cache_hits += 1
                    self._prefix_cache_reused_tokens += len(prefix_ids) * num_samples
                    self._store_prefix_cache(
                        prompt_ids,
                        getattr(outputs, "past_key_values", None),
                        num_samples,
                    )
                    return outputs
                except Exception:  # Exact-prefix reuse must fall back to normal sampling.
                    self._prefix_cache_fallbacks += 1

        self._prefix_cache_misses += 1
        outputs = self._forward_model(
            {
                "input_ids": generated,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
                "logits_to_keep": 1,
            },
            fallback_input_ids=generated,
            fallback_attention_mask=attention_mask,
        )
        self._store_prefix_cache(
            prompt_ids,
            getattr(outputs, "past_key_values", None),
            num_samples,
        )
        return outputs

    def _find_prefix_cache(self, prompt_ids, num_samples):
        if not (self.prefix_caching and self.sample_use_cache):
            return None
        prompt_key = tuple(int(token_id) for token_id in prompt_ids)
        for key in sorted(self._prefix_cache.keys(), key=lambda item: len(item), reverse=True):
            if len(key) >= len(prompt_key):
                continue
            if prompt_key[:len(key)] != key:
                continue
            past_key_values, batch_size = self._prefix_cache[key]
            if batch_size != num_samples:
                continue
            return key, past_key_values
        return None

    def _store_prefix_cache(self, prompt_ids, past_key_values, num_samples):
        if not (self.prefix_caching and self.sample_use_cache):
            return
        if past_key_values is None:
            return
        key = tuple(int(token_id) for token_id in prompt_ids)
        if not key:
            return
        if key not in self._prefix_cache:
            self._prefix_cache_order.append(key)
        self._prefix_cache[key] = (self._detach_past_key_values(past_key_values), num_samples)
        while len(self._prefix_cache_order) > self._prefix_cache_max_entries:
            evicted = self._prefix_cache_order.pop(0)
            self._prefix_cache.pop(evicted, None)

    def _advance_past_for_prefix_cache(self, past_key_values, full_prefix_ids, generated):
        if not full_prefix_ids:
            return past_key_values
        last_token = torch.tensor([[full_prefix_ids[-1]]], device=self.device)
        attention_mask = torch.ones(
            (1, len(full_prefix_ids)),
            dtype=torch.long,
            device=self.device,
        )
        timer = _timer_start(self.device)
        try:
            outputs = self._forward_model(
                {
                    "input_ids": last_token,
                    "attention_mask": attention_mask,
                    "use_cache": True,
                    "past_key_values": past_key_values,
                    "logits_to_keep": 1,
                },
                fallback_input_ids=generated,
                fallback_attention_mask=torch.ones_like(generated),
            )
        except Exception:  # Cache advance must not break sampling.
            self._prefix_cache_fallbacks += 1
            return past_key_values
        finally:
            self._prefix_cache_append_s += _timer_stop(timer)
        return getattr(outputs, "past_key_values", past_key_values)

    def _detach_past_key_values(self, value):
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, tuple):
            return tuple(self._detach_past_key_values(item) for item in value)
        if isinstance(value, list):
            return [self._detach_past_key_values(item) for item in value]
        return value

    def clear_prefix_cache(self):
        self._prefix_cache.clear()
        self._prefix_cache_order.clear()

    def _forward_model(
        self,
        model_kwargs,
        *,
        fallback_input_ids=None,
        fallback_attention_mask=None,
    ):
        try:
            return self.model(**model_kwargs)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword" not in message and "got an unexpected" not in message:
                raise
            if "logits_to_keep" in model_kwargs:
                retry_kwargs = dict(model_kwargs)
                retry_kwargs.pop("logits_to_keep", None)
                try:
                    return self.model(**retry_kwargs)
                except TypeError as retry_exc:
                    retry_message = str(retry_exc)
                    if (
                        "unexpected keyword" not in retry_message
                        and "got an unexpected" not in retry_message
                    ):
                        raise
            fallback = {
                "input_ids": (
                    fallback_input_ids
                    if fallback_input_ids is not None
                    else model_kwargs["input_ids"]
                )
            }
            if fallback_attention_mask is not None:
                fallback["attention_mask"] = fallback_attention_mask
            try:
                return self.model(**fallback)
            except TypeError as fallback_exc:
                fallback_message = str(fallback_exc)
                if (
                    "unexpected keyword" not in fallback_message
                    and "got an unexpected" not in fallback_message
                ):
                    raise
                fallback.pop("attention_mask", None)
                return self.model(**fallback)

    def _record_generation_metrics(
        self,
        start_s,
        prompt_ids_list,
        num_samples,
        results,
        prefill_s,
        decode_s,
    ):
        wall_s = time.perf_counter() - start_s
        prompt_tokens = sum(len(prompt) for prompt in prompt_ids_list) * int(num_samples)
        generated_tokens = sum(
            len(result.token_ids)
            for group in results
            for result in group
        )
        self._last_generation_metrics = {
            "engine_generation_wall_s": wall_s,
            "engine_prompt_prefill_s": prefill_s,
            "engine_decode_s": decode_s,
            "engine_prompt_tokens": prompt_tokens,
            "engine_generated_tokens": generated_tokens,
            "engine_generation_tokens_per_s": (
                generated_tokens / wall_s if wall_s > 0 else 0.0
            ),
            "engine_sample_use_cache": int(self.sample_use_cache),
            **self._prefix_cache_metrics(),
        }

    def _prefix_cache_metrics(self):
        return {
            "engine_prefix_caching_enabled": int(self.prefix_caching),
            "engine_prefix_cache_entries": len(self._prefix_cache),
            "engine_prefix_cache_hits": self._prefix_cache_hits,
            "engine_prefix_cache_misses": self._prefix_cache_misses,
            "engine_prefix_cache_fallbacks": self._prefix_cache_fallbacks,
            "engine_prefix_cache_reused_tokens": self._prefix_cache_reused_tokens,
            "engine_prefix_cache_append_s": self._prefix_cache_append_s,
        }

    def _generate_gemma4_text(self, prompt_ids_list, num_samples, max_tokens, temperature, top_p,
                              compute_entropy=False):
        """Text-only Gemma4 sampling path avoiding the multimodal wrapper."""
        results = []
        unwrapped = unwrap_peft_model(self.model)
        text_model = unwrapped.model.language_model
        lm_head = unwrapped.lm_head
        eos_ids = model_eos_token_ids(self.model, unwrap_model=unwrap_peft_model)

        with torch.no_grad():
            generation_start = time.perf_counter()
            prefill_s = 0.0
            decode_s = 0.0
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

                for step_idx in range(max_tokens):
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
                    timer = _timer_start(self.device)
                    outputs = text_model(**model_kwargs)
                    elapsed = _timer_stop(timer)
                    if step_idx == 0:
                        prefill_s += elapsed
                    else:
                        decode_s += elapsed
                    logits = lm_head(outputs.last_hidden_state[:, -1:, :])[:, -1, :]
                    next_token, logprob, entropy = _sample_next_token(
                        logits,
                        temperature,
                        top_p,
                        compute_entropy=compute_entropy,
                    )
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

        self._record_generation_metrics(
            generation_start,
            prompt_ids_list,
            num_samples,
            results,
            prefill_s,
            decode_s,
        )
        return results

    def performance_counters(self):
        """Return last-generation timing counters."""
        counters = dict(self._last_generation_metrics)
        counters.update(self._prefix_cache_metrics())
        counters.update(getattr(self, "_accelerator_metrics", {}))
        return counters

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
