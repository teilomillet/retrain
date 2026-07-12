"""Tests for the optional MAX inference adapter."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from types import ModuleType

import pytest

from retrain.inference_engine.max_engine import MAXLocalEngine, MAXServeEngine


class _FakeTokenizer:
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class _FakeLogProbChunk:
    def __init__(self, values: list[float]) -> None:
        self.token_log_probabilities = values


class _FakeTextGenerationOutput:
    def __init__(
        self,
        *,
        request_id: str,
        tokens: list[int],
        final_status: object | None,
        log_probabilities: list[_FakeLogProbChunk] | None,
    ) -> None:
        self.request_id = request_id
        self.tokens = tokens
        self.final_status = final_status
        self.log_probabilities = log_probabilities

    @staticmethod
    def merge(chunks: list[object]) -> "_FakeTextGenerationOutput":
        tokens: list[int] = []
        log_probabilities: list[_FakeLogProbChunk] = []
        for chunk in chunks:
            output = chunk
            if not isinstance(output, _FakeTextGenerationOutput):
                raise TypeError("unexpected fake MAX chunk")
            tokens.extend(output.tokens)
            if output.log_probabilities is not None:
                log_probabilities.extend(output.log_probabilities)
        return _FakeTextGenerationOutput(
            request_id="merged",
            tokens=tokens,
            final_status=None,
            log_probabilities=log_probabilities,
        )


class _FakeTextGenerationRequest:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeSamplingParams:
    def __init__(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p


class _FakePipelineConfig:
    def __init__(self, *, model_path: str) -> None:
        self.model_path = model_path


class _FakePipeline:
    async def generate_async(
        self,
        request: object,
    ) -> AsyncIterator[_FakeTextGenerationOutput]:
        del request
        yield _FakeTextGenerationOutput(
            request_id="first",
            tokens=[101, 102],
            final_status=None,
            log_probabilities=[_FakeLogProbChunk([-0.1, -0.2])],
        )
        yield _FakeTextGenerationOutput(
            request_id="second",
            tokens=[103],
            final_status=None,
            log_probabilities=[_FakeLogProbChunk([-0.3])],
        )


class _FakeLLM:
    def __init__(self, config: object) -> None:
        self.config = config
        self._pipeline = _FakePipeline()


def _module(name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_fake_max(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        "max": _module("max"),
        "max.entrypoints": _module("max.entrypoints"),
        "max.entrypoints.llm": _module("max.entrypoints.llm", LLM=_FakeLLM),
        "max.pipelines": _module(
            "max.pipelines",
            PipelineConfig=_FakePipelineConfig,
        ),
        "max.interfaces": _module("max.interfaces"),
        "max.interfaces.pipeline_variants": _module("max.interfaces.pipeline_variants"),
        "max.interfaces.pipeline_variants.text_generation": _module(
            "max.interfaces.pipeline_variants.text_generation",
            TextGenerationRequest=_FakeTextGenerationRequest,
            TextGenerationOutput=_FakeTextGenerationOutput,
        ),
        "max.interfaces.sampling_params": _module(
            "max.interfaces.sampling_params",
            SamplingParams=_FakeSamplingParams,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_max_server_engine_does_not_import_local_max_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import AutoTokenizer

    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda model_name: _FakeTokenizer(),
    )

    engine = MAXServeEngine("model", "http://localhost:8000/v1")

    assert engine.model_name == "model"
    assert "max.entrypoints.llm" not in sys.modules


def test_max_local_engine_generates_logprobs_with_fake_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_max(monkeypatch)

    from transformers import AutoTokenizer

    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda model_name: _FakeTokenizer(),
    )

    engine = MAXLocalEngine("fake-model")

    groups = engine.generate(
        [[1, 2, 3]],
        num_samples=2,
        max_tokens=4,
        temperature=0.0,
        top_p=0.9,
    )

    assert len(groups) == 1
    assert [sample.token_ids for sample in groups[0]] == [[101, 102, 103]] * 2
    assert [sample.logprobs for sample in groups[0]] == [[-0.1, -0.2, -0.3]] * 2
    assert engine.performance_counters() == {"engine_prompt_decode_calls": 1}
