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
        print("Training client ready.")

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
            for prompt_ids in prompt_ids_list:
                model_input = types.ModelInput.from_ints(prompt_ids)
                sampling_params = types.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                futures.append(
                    self.sampling_client.sample(
                        prompt=model_input,
                        num_samples=num_samples,
                        sampling_params=sampling_params,
                    )
                )

            # Collect results
            results: list[list[tuple[list[int], list[float]]]] = []
            for future in futures:
                sample_result = future.result()
                group: list[tuple[list[int], list[float]]] = []
                for seq in sample_result.sequences:
                    token_ids = list(seq.tokens)
                    logprobs = [float(lp) for lp in seq.logprobs]
                    group.append((token_ids, logprobs))
                results.append(group)
            return results

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

            # Forward-backward
            fwd_bwd_future = self.training_client.forward_backward(
                datums, loss_fn="importance_sampling"
            )

            # Optimizer step
            adam_params = types.AdamParams(
                learning_rate=lr,
                beta1=self._optim_beta1,
                beta2=self._optim_beta2,
                eps=self._optim_eps,
                weight_decay=weight_decay,
            )
            optim_future = self.training_client.optim_step(adam_params)

            # Collect results
            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            # Extract mean loss
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
        self.training_client.load_state(name=name)
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
