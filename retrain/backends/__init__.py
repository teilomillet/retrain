"""Formal interfaces for training backends.

``TrainHelper`` is the minimum lifecycle required by the trainer.  Optional
capabilities live in separate protocols so remote backends can expose only the
features their wire format supports.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from retrain.types import EnrichedSampleBatch, SampleBatch

type RuntimeMetric = int | float | str


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


@runtime_checkable
class RuntimeMetricsHelper(Protocol):
    """Optional backend capability for runtime telemetry."""

    def runtime_metrics(self) -> Mapping[str, object]: ...


@runtime_checkable
class SftTrainHelper(Protocol):
    """Optional backend capability for native supervised fine-tuning loss."""

    def sft_train_step(
        self,
        all_tokens: list[list[int]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float: ...


def collect_runtime_metrics(helper: object) -> dict[str, RuntimeMetric]:
    """Return backend runtime telemetry when the helper exposes it safely."""
    runtime_metrics = getattr(helper, "runtime_metrics", None)
    if not callable(runtime_metrics):
        return {}
    metrics = runtime_metrics()
    if not isinstance(metrics, Mapping):
        return {}
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(key, str) and isinstance(value, (int, float, str))
    }


def run_sft_train_step(
    helper: TrainHelper,
    all_tokens: list[list[int]],
    all_advantages: list[list[float]],
    lr: float,
    weight_decay: float,
) -> float:
    """Use a backend's native SFT path, falling back to TrainHelper semantics."""
    sft_train_step = getattr(helper, "sft_train_step", None)
    if callable(sft_train_step):
        return float(sft_train_step(all_tokens, all_advantages, lr, weight_decay))

    all_logprobs = [[0.0] * len(tokens) for tokens in all_tokens]
    return float(
        helper.train_step(
            all_tokens,
            all_logprobs,
            all_advantages,
            lr,
            weight_decay,
        )
    )
