"""Typed boundary for Modular MAX's optional Python runtime."""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Protocol, cast


class Tokenizer(Protocol):
    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str: ...


class PipelineConfigFactory(Protocol):
    def __call__(self, *, model_path: str) -> object: ...


class Pipeline(Protocol):
    def generate_async(self, request: object) -> AsyncIterator[object]: ...


class LLM(Protocol):
    _pipeline: Pipeline


class LLMFactory(Protocol):
    def __call__(self, config: object) -> LLM: ...


class SamplingParamsFactory(Protocol):
    def __call__(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> object: ...


class TextGenerationRequestFactory(Protocol):
    def __call__(
        self,
        *,
        model_name: str,
        prompt: str,
        messages: list[object],
        images: list[object],
        tools: object | None,
        response_format: object | None,
        logprobs: int,
        echo: bool,
        sampling_params: object,
    ) -> object: ...


class TokenLogProbabilities(Protocol):
    token_log_probabilities: Sequence[float] | None


class TextGenerationOutput(Protocol):
    tokens: Sequence[int] | None
    log_probabilities: Sequence[TokenLogProbabilities | None] | None


class TextGenerationOutputFactory(Protocol):
    def __call__(
        self,
        *,
        request_id: str,
        tokens: list[int],
        final_status: object | None,
        log_probabilities: object | None,
    ) -> TextGenerationOutput: ...

    def merge(self, chunks: Sequence[object]) -> TextGenerationOutput: ...


def local_factories() -> tuple[LLMFactory, PipelineConfigFactory]:
    """Load MAX local classes lazily so server mode stays zero-MAX-dependency."""
    llm_module = importlib.import_module("max.entrypoints.llm")
    pipelines_module = importlib.import_module("max.pipelines")
    return (
        cast(LLMFactory, getattr(llm_module, "LLM")),
        cast(PipelineConfigFactory, getattr(pipelines_module, "PipelineConfig")),
    )


def generation_factories() -> tuple[
    TextGenerationRequestFactory,
    SamplingParamsFactory,
]:
    text_generation = importlib.import_module(
        "max.interfaces.pipeline_variants.text_generation"
    )
    sampling_params = importlib.import_module("max.interfaces.sampling_params")
    return (
        cast(
            TextGenerationRequestFactory,
            getattr(text_generation, "TextGenerationRequest"),
        ),
        cast(SamplingParamsFactory, getattr(sampling_params, "SamplingParams")),
    )


def output_factory() -> TextGenerationOutputFactory:
    text_generation = importlib.import_module(
        "max.interfaces.pipeline_variants.text_generation"
    )
    return cast(
        TextGenerationOutputFactory,
        getattr(text_generation, "TextGenerationOutput"),
    )


def load_tokenizer(model_name: str) -> Tokenizer:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer is None or not callable(getattr(tokenizer, "decode", None)):
        raise TypeError(f"Tokenizer for {model_name!r} does not expose decode().")
    return cast(Tokenizer, tokenizer)
