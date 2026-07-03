"""Formal interfaces for training backends.

``TrainHelper`` is the minimum lifecycle required by the trainer.  Optional
capabilities live in separate protocols so remote backends can expose only the
features their wire format supports.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from retrain.type_defs import EnrichedSampleBatch, SampleBatch


@runtime_checkable
class TrainHelper(Protocol):
    """Backend interface for model training and sampling."""

    def checkpoint(self, name: str) -> None: ...

    def sample(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> SampleBatch: ...

    def train_step(
        self,
        all_tokens: list[list[int]],
        all_logprobs: list[list[float]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float: ...

    def save_adapter(self, path: str, name: str) -> str: ...

    def load_state(self, name: str) -> None: ...


@runtime_checkable
class EntropySamplingHelper(Protocol):
    """Optional backend capability for token-entropy sampling."""

    def sample_with_entropy(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> EnrichedSampleBatch: ...
