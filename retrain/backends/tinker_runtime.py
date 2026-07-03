"""Typed boundary for the optional Tinker SDK."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Protocol, TypeVar, cast

_T_co = TypeVar("_T_co", covariant=True)


class FutureLike(Protocol[_T_co]):
    def result(self, timeout: float | None = None) -> _T_co: ...


class CheckpointArchiveUrl(Protocol):
    url: str


class CheckpointState(Protocol):
    path: str


class LossResult(Protocol):
    metrics: Mapping[str, float] | None


class TensorDataValue(Protocol):
    data: Sequence[float]


class ForwardResult(Protocol):
    loss_fn_outputs: Sequence[Mapping[str, TensorDataValue]]


class SampleSequence(Protocol):
    tokens: Sequence[int]
    logprobs: Sequence[float]


class SampleResult(Protocol):
    sequences: Sequence[SampleSequence]


class SamplingClient(Protocol):
    def sample(
        self,
        *,
        prompt: object,
        num_samples: int,
        sampling_params: object,
    ) -> FutureLike[SampleResult]: ...


class RestClient(Protocol):
    def get_checkpoint_archive_url_from_tinker_path(
        self,
        tinker_path: str,
    ) -> FutureLike[CheckpointArchiveUrl]: ...


class TrainingClient(Protocol):
    def save_weights_and_get_sampling_client(self, *, name: str) -> SamplingClient: ...
    def forward(
        self,
        datums: Sequence[object],
        *,
        loss_fn: str,
    ) -> FutureLike[ForwardResult]: ...
    def forward_backward(
        self,
        datums: Sequence[object],
        *,
        loss_fn: str,
    ) -> FutureLike[LossResult]: ...
    def optim_step(self, adam_params: object) -> FutureLike[object]: ...
    def save_state(self, *, name: str) -> FutureLike[CheckpointState]: ...


class ServiceClient(Protocol):
    def create_rest_client(self) -> RestClient: ...
    def create_lora_training_client(
        self,
        *,
        base_model: str,
        rank: int,
    ) -> TrainingClient: ...


class ServiceClientFactory(Protocol):
    def __call__(self, **kwargs: object) -> ServiceClient: ...


class TinkerModule(Protocol):
    ServiceClient: ServiceClientFactory


class ModelInputFactory(Protocol):
    def from_ints(self, tokens: Sequence[int]) -> object: ...


class SamplingParamsFactory(Protocol):
    def __call__(
        self,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> object: ...


class DatumFactory(Protocol):
    def __call__(
        self,
        *,
        model_input: object,
        loss_fn_inputs: Mapping[str, object],
    ) -> object: ...


class AdamParamsFactory(Protocol):
    def __call__(
        self,
        *,
        learning_rate: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        grad_clip_norm: float = 0.0,
    ) -> object: ...


class TinkerTypes(Protocol):
    ModelInput: ModelInputFactory
    SamplingParams: SamplingParamsFactory
    Datum: DatumFactory
    AdamParams: AdamParamsFactory


class TensorDataFactory(Protocol):
    def from_torch(self, tensor: object) -> object: ...


def load_tinker() -> TinkerModule:
    return cast(TinkerModule, importlib.import_module("tinker"))


def load_tinker_types() -> TinkerTypes:
    return cast(TinkerTypes, importlib.import_module("tinker.types"))


def load_tensor_data() -> TensorDataFactory:
    module = importlib.import_module("tinker.types.tensor_data")
    return cast(TensorDataFactory, getattr(module, "TensorData"))
