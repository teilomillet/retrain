"""Base classes for pluggable inference engines.

InferenceEngine defines the contract: generate completions with logprobs,
reload LoRA weights, and shutdown. Engines run on their own device(s)
independently from the training model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SampleResult:
    """Single completion with per-token logprobs.

    Attributes:
        token_ids: Generated token IDs (completion only, no prompt).
        logprobs: Log-probability of each generated token under the model.
        token_entropies: Per-token Shannon entropy (optional, GPU-computed).
        finish_reason: Why generation stopped, when the engine reports it.
    """

    token_ids: list[int]
    logprobs: list[float]
    token_entropies: list[float] | None = None
    finish_reason: str | None = None


class InferenceEngine(ABC):
    """Abstract base for inference engines used during GRPO sampling."""

    @abstractmethod
    def generate(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        compute_entropy: bool = False,
    ) -> list[list[SampleResult]]:
        """Generate completions for a batch of prompts.

        Args:
            prompt_ids_list: List of token-ID sequences (one per prompt).
            num_samples: Number of completions per prompt.
            max_tokens: Maximum new tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            compute_entropy: If True, compute per-token Shannon entropy
                on GPU and populate SampleResult.token_entropies.

        Returns:
            Nested list [num_prompts][num_samples] of SampleResult.
        """
        ...

    @abstractmethod
    def reload_weights(self, adapter_path: str) -> None:
        """Reload LoRA adapter weights (e.g. after a training checkpoint).

        Args:
            adapter_path: Path to saved adapter directory.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources (models, connections)."""
        ...
