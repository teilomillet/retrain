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


class TinkerTrainHelper:
    """Training helper using the Tinker remote GPU service."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        lora_rank: int = 32,
    ) -> None:
        """Create Tinker service client and LoRA training client."""
        import tinker

        print("Connecting to Tinker...")
        if base_url:
            service_client = tinker.ServiceClient(base_url=base_url)
        else:
            service_client = tinker.ServiceClient()

        print(
            f"Creating LoRA training client (model={model_name}, rank={lora_rank})..."
        )
        self.training_client = service_client.create_lora_training_client(
            base_model=model_name,
            rank=lora_rank,
        )
        self.sampling_client = None
        print("Training client ready.")

    def checkpoint(self, name: str) -> None:
        """Save weights and get a sampling client for the current checkpoint."""
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
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
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

    def save_adapter(self, path: str, name: str) -> None:
        """Save training state checkpoint via Tinker."""
        self.training_client.save_state(name=name)
        print(f"Tinker checkpoint saved: {name}")
