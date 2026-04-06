"""TinkerTrainHelper — wraps the Tinker remote GPU API.

Provides the same interface as LocalTrainHelper so the trainer can
dispatch between local and tinker backends transparently:
  - checkpoint(name)
  - sample(prompt_ids_list, num_samples, max_tokens, temperature, top_p)
  - train_step(all_tokens, all_logprobs, all_advantages, lr, weight_decay)
  - save_adapter(path, name)

Ports src/tinker_backend.mojo into pure Python.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from retrain.tinker_throttle import NoOpThrottle, TinkerThrottle


class TinkerTrainHelper:
    """Training helper using the Tinker remote GPU service."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        lora_rank: int = 32,
        optim_beta1: float = 0.9,
        optim_beta2: float = 0.95,
        optim_eps: float = 1e-8,
        throttle_dir: str = "",
        max_concurrent: int = 4,
        clip_eps: float = 0.0,
        clip_eps_high: float = 0.0,
        grad_clip_norm: float = 0.0,
        clip_ratio_c: float = 0.0,
        sample_log_dir: str = "",
    ) -> None:
        """Create Tinker service client and LoRA training client."""
        import tinker

        if throttle_dir:
            self._throttle: TinkerThrottle | NoOpThrottle = TinkerThrottle(
                throttle_dir, max_concurrent
            )
        else:
            self._throttle = NoOpThrottle()

        print("Connecting to Tinker...")
        if base_url:
            service_client = tinker.ServiceClient(base_url=base_url, max_retries=100)
        else:
            service_client = tinker.ServiceClient(max_retries=100)

        print(
            f"Creating LoRA training client (model={model_name}, rank={lora_rank})..."
        )
        self.training_client = service_client.create_lora_training_client(
            base_model=model_name,
            rank=lora_rank,
        )
        self.sampling_client = None
        self._optim_beta1 = optim_beta1
        self._optim_beta2 = optim_beta2
        self._optim_eps = optim_eps
        self._clip_eps = clip_eps
        self._clip_eps_high = clip_eps_high
        self._grad_clip_norm = grad_clip_norm
        self._clip_ratio_c = clip_ratio_c
        self._sample_diag_path: Path | None = None
        self._sample_diag_lock = threading.Lock()
        if sample_log_dir:
            diag_dir = Path(sample_log_dir).resolve()
            diag_dir.mkdir(parents=True, exist_ok=True)
            self._sample_diag_path = diag_dir / "tinker_sample_diagnostics.jsonl"
            print(f"Tinker sample diagnostics: {self._sample_diag_path}")
        self._use_custom_ppo = clip_eps > 0
        if clip_eps > 0:
            eps_hi = clip_eps_high or clip_eps
            dc = f", dual-clip c={clip_ratio_c}" if clip_ratio_c > 0 else ""
            print(f"PPO dual-clip enabled via manual forward+backward pipeline "
                  f"(eps_lo={clip_eps}, eps_hi={eps_hi}{dc})")
        if grad_clip_norm > 0:
            print(f"Gradient clipping enabled (max_norm={grad_clip_norm})")
        print("Training client ready.")
        self._write_sample_diag(
            {
                "event": "helper_initialized",
                "sample_log_dir": str(self._sample_diag_path.parent)
                if self._sample_diag_path is not None
                else "",
            }
        )

    def _write_sample_diag(self, payload: dict[str, object]) -> None:
        if self._sample_diag_path is None:
            return
        record = dict(payload)
        record.setdefault("ts", time.time())
        line = json.dumps(record, sort_keys=True) + "\n"
        with self._sample_diag_lock:
            self._sample_diag_path.parent.mkdir(parents=True, exist_ok=True)
            with self._sample_diag_path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def checkpoint(self, name: str) -> None:
        """Save weights and get a sampling client for the current checkpoint."""
        with self._throttle:
            self.sampling_client = (
                self.training_client.save_weights_and_get_sampling_client(name=name)
            )

    def sample(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[tuple[list[int], list[float]]]]:
        """Generate completions via the Tinker sampling API.

        Returns list of groups, each group is a list of (token_ids, logprobs) tuples
        — same shape as LocalTrainHelper.sample().
        """
        import tinker.types as types

        if self.sampling_client is None:
            raise RuntimeError(
                "No sampling client — call checkpoint() before sample()."
            )

        with self._throttle:
            # Fire all sampling futures
            futures = []
            for prompt_idx, prompt_ids in enumerate(prompt_ids_list):
                model_input = types.ModelInput.from_ints(prompt_ids)
                sampling_params = types.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                dispatch_started_at = time.perf_counter()
                futures.append(
                    (
                        prompt_idx,
                        len(prompt_ids),
                        dispatch_started_at,
                        self.sampling_client.sample(
                            prompt=model_input,
                            num_samples=num_samples,
                            sampling_params=sampling_params,
                        ),
                    )
                )
                self._write_sample_diag(
                    {
                        "event": "dispatch",
                        "prompt_idx": prompt_idx,
                        "prompt_tokens": len(prompt_ids),
                        "num_samples": num_samples,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                )

            # Collect results (with timeout to prevent hanging on context overflow)
            results: list[list[tuple[list[int], list[float]]]] = []
            for prompt_idx, prompt_tokens, dispatch_started_at, future in futures:
                wait_started_at = time.perf_counter()
                try:
                    sample_result = future.result(timeout=300)  # 5 min timeout per sample
                    group: list[tuple[list[int], list[float]]] = []
                    completion_lengths: list[int] = []
                    for seq in sample_result.sequences:
                        token_ids = list(seq.tokens)
                        logprobs = [float(lp) for lp in seq.logprobs]
                        completion_lengths.append(len(token_ids))
                        group.append((token_ids, logprobs))
                    results.append(group)
                    self._write_sample_diag(
                        {
                            "event": "result",
                            "prompt_idx": prompt_idx,
                            "prompt_tokens": prompt_tokens,
                            "completion_count": len(group),
                            "completion_tokens": completion_lengths,
                            "dispatch_latency_s": round(
                                wait_started_at - dispatch_started_at, 6
                            ),
                            "result_latency_s": round(
                                time.perf_counter() - wait_started_at, 6
                            ),
                            "status": "ok",
                        }
                    )
                except Exception as exc:
                    # Context overflow or timeout — return empty sequence
                    # instead of hanging forever.
                    import sys
                    print(
                        f"WARNING: Tinker sample {prompt_idx} failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._write_sample_diag(
                        {
                            "event": "result",
                            "prompt_idx": prompt_idx,
                            "prompt_tokens": prompt_tokens,
                            "completion_count": 0,
                            "completion_tokens": [],
                            "dispatch_latency_s": round(
                                wait_started_at - dispatch_started_at, 6
                            ),
                            "result_latency_s": round(
                                time.perf_counter() - wait_started_at, 6
                            ),
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    results.append([])
            return results

    def sample_with_entropy(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[tuple[list[int], list[float], list[float] | None]]]:
        """Generate completions with entropy (stub — Tinker lacks entropy).

        Delegates to sample() and appends None for token_entropies.
        """
        groups = self.sample(prompt_ids_list, num_samples, max_tokens,
                             temperature, top_p)
        return [
            [(ids, lps, None) for ids, lps in group]
            for group in groups
        ]

    def sft_train_step(
        self,
        all_tokens: list[list[int]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float:
        """SFT with configurable loss function (importance_sampling or cross_entropy).

        Set ``self.sft_loss_fn`` to control:
        - ``"importance_sampling"`` (default): Softer push via exp(logprob).
          Works well for models that already produce plausible outputs (Qwen).
        - ``"cross_entropy"``: Direct log-prob maximization.  Stronger signal
          for models that initially assign near-zero probability (Nemotron).

        advantages=0 for prompt tokens (no gradient), advantages=1 for response.
        """
        import torch
        import tinker.types as types
        from tinker.types.tensor_data import TensorData

        loss_fn = getattr(self, "sft_loss_fn", "importance_sampling")

        with self._throttle:
            datums = []
            for i, tokens in enumerate(all_tokens):
                advs = all_advantages[i] if i < len(all_advantages) else [1.0] * len(tokens)
                model_input = types.ModelInput.from_ints(tokens)

                if loss_fn == "cross_entropy":
                    loss_mask = [1.0 if a > 0 else 0.0 for a in advs]
                    loss_fn_inputs = {
                        "target_tokens": TensorData.from_torch(
                            torch.tensor(tokens, dtype=torch.long)
                        ),
                        "weights": TensorData.from_torch(
                            torch.tensor(loss_mask, dtype=torch.float32)
                        ),
                    }
                else:
                    # importance_sampling (default)
                    loss_fn_inputs = {
                        "target_tokens": TensorData.from_torch(
                            torch.tensor(tokens, dtype=torch.long)
                        ),
                        "logprobs": TensorData.from_torch(
                            torch.zeros(len(tokens), dtype=torch.float32)
                        ),
                        "advantages": TensorData.from_torch(
                            torch.tensor(advs, dtype=torch.float32)
                        ),
                    }

                datums.append(types.Datum(
                    model_input=model_input,
                    loss_fn_inputs=loss_fn_inputs,
                ))

            fwd_bwd_future = self.training_client.forward_backward(
                datums, loss_fn=loss_fn
            )
            adam_params = types.AdamParams(
                learning_rate=lr,
                beta1=self._optim_beta1,
                beta2=self._optim_beta2,
                eps=self._optim_eps,
                weight_decay=weight_decay,
            )
            optim_future = self.training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            if hasattr(fwd_bwd_result, "metrics") and fwd_bwd_result.metrics:
                n = max(len(datums), 1)
                loss_sum = fwd_bwd_result.metrics.get("loss:sum", 0.0)
                return float(loss_sum) / n
            return 0.0

    def _ppo_dual_clip_loss(
        self,
        data: list,
        new_logprobs_list: list,
    ) -> tuple:
        """PPO dual-clip loss matching the MaxRL paper (local PyTorch math).

        Args:
            data: Datum list with old logprobs + advantages in loss_fn_inputs.
            new_logprobs_list: New-policy logprobs from forward pass (with grad).

        Returns:
            (loss_tensor, metrics_dict)
        """
        import torch

        eps_lo = self._clip_eps
        eps_hi = self._clip_eps_high or self._clip_eps
        clip_c = self._clip_ratio_c

        total_loss = torch.tensor(0.0)
        total_tokens = 0
        total_clipped = 0
        ratios_all = []

        for datum, new_lp in zip(data, new_logprobs_list):
            old_lp = torch.tensor(datum.loss_fn_inputs["logprobs"].data, dtype=torch.float32)
            adv = torch.tensor(datum.loss_fn_inputs["advantages"].data, dtype=torch.float32)

            seq_len = min(len(new_lp), len(old_lp), len(adv))
            ratio = torch.exp(new_lp[:seq_len] - old_lp[:seq_len])
            adv_t = adv[:seq_len]

            # Standard PPO clip (asymmetric eps)
            surr1 = -adv_t * ratio
            surr2 = -adv_t * torch.clamp(ratio, 1.0 - eps_lo, 1.0 + eps_hi)
            clipped = torch.max(surr1, surr2)

            # Dual-clip for negative advantages (if clip_ratio_c > 0)
            if clip_c > 0:
                surr3 = -adv_t * clip_c
                dual = torch.min(surr3, clipped)
                per_token_loss = torch.where(adv_t < 0, dual, clipped)
            else:
                per_token_loss = clipped

            mask = (adv_t != 0).float()
            n_tokens = mask.sum().clamp(min=1)
            total_loss = total_loss + (per_token_loss * mask).sum() / n_tokens
            total_tokens += int(n_tokens.item())

            with torch.no_grad():
                clip_frac = ((ratio < 1.0 - eps_lo) | (ratio > 1.0 + eps_hi)).float()
                total_clipped += (clip_frac * mask).sum().item()
                ratios_all.append(ratio.detach())

        loss = total_loss / max(len(data), 1)

        all_ratios = torch.cat(ratios_all) if ratios_all else torch.tensor([1.0])
        metrics = {
            "ppo_custom_loss": loss.item(),
            "ppo_clip_fraction": total_clipped / max(total_tokens, 1),
            "ppo_mean_ratio": all_ratios.mean().item(),
            "ppo_max_ratio": all_ratios.max().item(),
            "ppo_kl_approx": (all_ratios - 1 - all_ratios.log()).mean().item(),
        }
        return loss, metrics

    def train_step(
        self,
        all_tokens: list[list[int]],
        all_logprobs: list[list[float]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float:
        """Build datums, run forward-backward + optimizer step, return mean loss."""
        import torch
        import tinker.types as types
        from tinker.types.tensor_data import TensorData

        with self._throttle:
            # Build datums
            datums = []
            for i in range(len(all_tokens)):
                model_input = types.ModelInput.from_ints(all_tokens[i])
                loss_fn_inputs = {
                    "target_tokens": TensorData.from_torch(
                        torch.tensor(all_tokens[i], dtype=torch.long)
                    ),
                    "logprobs": TensorData.from_torch(
                        torch.tensor(all_logprobs[i], dtype=torch.float32)
                    ),
                    "advantages": TensorData.from_torch(
                        torch.tensor(all_advantages[i], dtype=torch.float32)
                    ),
                }
                datums.append(
                    types.Datum(
                        model_input=model_input,
                        loss_fn_inputs=loss_fn_inputs,
                    )
                )

            # Forward-backward: custom PPO or standard loss
            if self._use_custom_ppo:
                # Custom dual-clip PPO via manual forward + backward pipeline.
                # SDK's forward_backward_custom uses cross_entropy internally,
                # which is broken on the Tinker server. Instead we:
                #   1. forward(IS) -> get new-policy logprobs
                #   2. Compute PPO dual-clip loss locally (PyTorch)
                #   3. Backward pass: forward_backward(IS) with
                #      old_lp=new_lp (ratio=1), advantages=grad
                #      -> chain rule gives correct parameter gradients

                # Phase 1: Forward pass to get new-policy logprobs
                fwd_future = self.training_client.forward(
                    datums, loss_fn="importance_sampling"
                )
                fwd_result = fwd_future.result()

                # Extract new logprobs as differentiable tensors
                new_lp_list = []
                for out in fwd_result.loss_fn_outputs:
                    lp = torch.tensor(
                        out["logprobs"].data, dtype=torch.float32
                    ).requires_grad_(True)
                    new_lp_list.append(lp)

                # Phase 2: Compute custom PPO dual-clip loss
                loss, ppo_metrics = self._ppo_dual_clip_loss(datums, new_lp_list)
                loss.backward()

                # Phase 3: Backward pass via importance_sampling with ratio=1
                # Set old_lp = new_lp so ratio = exp(new-new) = 1,
                # and advantages = grad so effective gradient = grad * d(lp)/d(params)
                grad_datums = []
                for i, lp_tensor in enumerate(new_lp_list):
                    grad = lp_tensor.grad
                    if grad is None:
                        raise RuntimeError(
                            f"No gradient for logprob tensor {i}"
                        )
                    new_lp_data = [float(x) for x in lp_tensor.detach()]
                    grad_data = [float(x) for x in grad]
                    grad_datums.append(
                        types.Datum(
                            model_input=datums[i].model_input,
                            loss_fn_inputs={
                                "target_tokens": datums[i].loss_fn_inputs[
                                    "target_tokens"
                                ],
                                "logprobs": TensorData.from_torch(
                                    torch.tensor(new_lp_data, dtype=torch.float32)
                                ),
                                "advantages": TensorData.from_torch(
                                    torch.tensor(grad_data, dtype=torch.float32)
                                ),
                            },
                        )
                    )
                fwd_bwd_future = self.training_client.forward_backward(
                    grad_datums, loss_fn="importance_sampling"
                )
            else:
                loss_fn = "ppo" if self._clip_eps > 0 else "importance_sampling"
                fwd_bwd_future = self.training_client.forward_backward(
                    datums, loss_fn=loss_fn
                )
                ppo_metrics = None

            # Optimizer step
            adam_params = types.AdamParams(
                learning_rate=lr,
                beta1=self._optim_beta1,
                beta2=self._optim_beta2,
                eps=self._optim_eps,
                weight_decay=weight_decay,
                grad_clip_norm=self._grad_clip_norm,
            )
            optim_future = self.training_client.optim_step(adam_params)

            # Collect results
            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            # Extract mean loss
            if ppo_metrics is not None:
                return float(ppo_metrics.get("ppo_custom_loss", 0.0))
            if hasattr(fwd_bwd_result, "metrics") and fwd_bwd_result.metrics:
                n = max(len(datums), 1)
                loss_sum = fwd_bwd_result.metrics.get("loss:sum", 0.0)
                return float(loss_sum) / n
            return 0.0

    def load_state(self, name: str) -> None:
        """Load training state from a Tinker checkpoint.

        Requires the Tinker SDK to support load_state on the training client.
        Raises AttributeError with a clear message if not available.
        """
        if not hasattr(self.training_client, "load_state"):
            raise AttributeError(
                f"Tinker training client does not support load_state(). "
                f"Cannot resume from checkpoint '{name}'. "
                f"Check your Tinker SDK version supports checkpoint loading."
            )
        try:
            self.training_client.load_state(name=name)
        except TypeError:
            # Some SDK builds expose load_state(path_or_name) positionally only.
            self.training_client.load_state(name)
        print(f"Tinker checkpoint loaded: {name}")

    def save_adapter(self, path: str, name: str) -> str:
        """Save training state + sampler weights via Tinker.

        save_state() creates a training checkpoint (for resume), but
        the Tinker archive download API only works with sampler weights.
        So we also create sampler weights (downloadable for squeeze).

        Returns the tinker:// path for the sampler weights checkpoint.
        """
        with self._throttle:
            # Training state (for resume)
            result = self.training_client.save_state(name=name)
            response = result.result()
            training_path = response.path

            # Sampler weights (downloadable for squeeze analysis)
            squeeze_name = f"{name}_squeeze"
            self.training_client.save_weights_and_get_sampling_client(name=squeeze_name)

            # Construct tinker:// path for sampler weights from training path
            # training_path: tinker://run-id/weights/{name}
            # sampler_path:  tinker://run-id/weights/{squeeze_name}
            base = training_path.rsplit("/", 1)[0]
            sampler_path = f"{base}/{squeeze_name}"

            print(f"Tinker checkpoint saved: {name} ({sampler_path})")
            return sampler_path
