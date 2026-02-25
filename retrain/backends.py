"""TrainHelper Protocol — formal interface for training backends.

Both ``LocalTrainHelper`` and ``TinkerTrainHelper`` already satisfy this
protocol by convention.  Making it explicit enables type-checking and
documentation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


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
    ) -> list: ...

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
