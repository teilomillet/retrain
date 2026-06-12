"""Tests for Scaleway backend response parsing and training-server glue."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from retrain.scaleway_backend import ScalewayTrainHelper


class _Response:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or {}

    def json(self) -> dict:
        return self.payload

    def raise_for_status(self) -> None:
        return None


class _Client:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.requests: list[dict] = []

    def post(self, url: str, json: dict, timeout: float):
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        return _Response(self.payload)


class _Tokenizer:
    unk_token_id = -1

    def __init__(self) -> None:
        self.token_map = {"a": 101, "b": 102}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        return "".join(str(token_id) for token_id in token_ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        if text == "ab":
            return [101, 102]
        if text == "a":
            return [101]
        if text == "too many":
            return [1, 2]
        return [ord(char) for char in text]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_map.get(token, self.unk_token_id)


class _FakeFastAPI:
    def __init__(self, *args, **kwargs) -> None:
        _ = args, kwargs

    def get(self, *_args, **_kwargs):
        return lambda func: func

    def post(self, *_args, **_kwargs):
        return lambda func: func


class _FakeResponse:
    def __init__(self, content: bytes = b"", media_type: str = "") -> None:
        self.content = content
        self.media_type = media_type


class _FakeBaseModel:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _helper_with_response(payload: dict) -> ScalewayTrainHelper:
    helper = ScalewayTrainHelper.__new__(ScalewayTrainHelper)
    helper._model = "test-model"
    helper._inference_url = "http://inference"
    helper._client = _Client(payload)
    helper._tokenizer = _Tokenizer()
    return helper


def test_scaleway_sample_one_requires_integer_token_ids_or_single_token_fallback():
    helper = _helper_with_response({
        "choices": [{
            "message": {"content": "ab"},
            "logprobs": {"content": [
                {"token": "a", "token_id": 101, "logprob": -0.1},
                {"token": "b", "top_logprobs": [{"logprob": -0.2}]},
            ]},
        }],
    })

    token_ids, logprobs = helper._sample_one("prompt", 8, 0.7, 0.9)

    assert token_ids == [101, 102]
    assert logprobs == [-0.1, -0.2]


def test_scaleway_sample_one_rejects_ambiguous_token_text():
    helper = _helper_with_response({
        "choices": [{
            "message": {"content": "too many"},
            "logprobs": {"content": [
                {"token": "too many", "logprob": -0.1},
            ]},
        }],
    })

    with pytest.raises(RuntimeError, match="does not map to exactly one token ID"):
        helper._sample_one("prompt", 8, 0.7, 0.9)


def test_training_server_checkpoint_reloads_saved_adapter_path(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "fastapi", SimpleNamespace(
        FastAPI=_FakeFastAPI,
        HTTPException=Exception,
        status=SimpleNamespace(HTTP_503_SERVICE_UNAVAILABLE=503),
    ))
    monkeypatch.setitem(sys.modules, "fastapi.responses", SimpleNamespace(Response=_FakeResponse))
    monkeypatch.setitem(sys.modules, "pydantic", SimpleNamespace(BaseModel=_FakeBaseModel))
    sys.modules.pop("retrain.scaleway.training_server", None)
    from retrain.scaleway import training_server
    training_server = importlib.reload(training_server)

    calls: list[tuple[str, object]] = []

    class _Helper:
        adapter_path = str(tmp_path / "adapters")

        def checkpoint(self, name: str) -> None:
            calls.append(("checkpoint", name))

        def save_adapter(self, path: str, name: str) -> str:
            calls.append(("save_adapter", (path, name)))
            saved = tmp_path / "adapters" / name
            saved.mkdir(parents=True)
            return str(saved)

    def fake_post(url: str, json: dict, timeout: float):
        calls.append(("post", {"url": url, "json": json, "timeout": timeout}))
        return _Response()

    monkeypatch.setattr(training_server, "_helper", _Helper())
    monkeypatch.setattr(training_server, "_inference_url", "http://inference")
    monkeypatch.setattr(training_server, "_inference_engine", "vllm")
    monkeypatch.setattr(training_server.httpx, "post", fake_post)

    out = training_server.checkpoint(training_server.CheckpointRequest(name="step_1"))

    saved_path = str((tmp_path / "adapters" / "step_1").resolve())
    assert out == {"adapter_path": str(tmp_path / "adapters" / "step_1")}
    assert calls == [
        ("checkpoint", "step_1"),
        ("save_adapter", (str(tmp_path / "adapters"), "step_1")),
        ("post", {
            "url": "http://inference/v1/load_lora_adapter",
            "json": {"lora_name": "step_1", "lora_path": saved_path},
            "timeout": 30,
        }),
    ]
