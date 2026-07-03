"""Tests for OpenAI-compatible engine behaviors."""

from __future__ import annotations

import pytest

from retrain.inference_engine.openai_engine import OpenAIEngine


class _FakeTokenizer:
    def __init__(self) -> None:
        self.decode_calls = 0

    def decode(self, token_ids, skip_special_tokens=False):
        self.decode_calls += 1
        return "prompt"

    def encode(self, text, add_special_tokens=False):
        return [1, 2]

    def convert_tokens_to_ids(self, tokens):
        return [1 for _ in tokens]


class _FallbackTokenizer(_FakeTokenizer):
    def encode(self, text, add_special_tokens=False):
        return [101, 102]

    def convert_tokens_to_ids(self, tokens):
        return [None for _ in tokens]


def test_mlx_reload_uses_adapters_payload(monkeypatch):
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: _FakeTokenizer(),
    )

    engine = OpenAIEngine(
        base_url="http://localhost:8080",
        model_name="mlx-community/FakeModel",
        engine_type="mlx",
    )

    calls = []

    def _fake_post(endpoint, payload, max_retries=3):
        calls.append((endpoint, dict(payload)))
        return {
            "choices": [
                {
                    "text": "x",
                    "logprobs": {"tokens": ["x"], "token_logprobs": [-0.1]},
                }
            ]
        }

    monkeypatch.setattr(engine, "_post", _fake_post)

    engine.reload_weights("/tmp/retrain_adapter/_live_adapter")
    engine.generate(
        prompt_ids_list=[[1, 2, 3]],
        num_samples=1,
        max_tokens=8,
        temperature=0.7,
        top_p=0.95,
    )

    assert calls
    endpoint, payload = calls[0]
    assert endpoint == "/v1/completions"
    assert payload["adapters"] == "/tmp/retrain_adapter/_live_adapter"


def test_vllm_reload_calls_lora_endpoint(monkeypatch):
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: _FakeTokenizer(),
    )

    engine = OpenAIEngine(
        base_url="http://localhost:8000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="vllm",
    )

    calls = []

    def _fake_post(endpoint, payload, max_retries=3, **kwargs):
        calls.append((endpoint, dict(payload)))
        return {}

    monkeypatch.setattr(engine, "_post", _fake_post)

    engine.reload_weights("/tmp/retrain_adapter/_live_adapter")

    assert calls == [
        (
            "/v1/load_lora_adapter",
            {
                "lora_name": "default",
                "lora_path": "/tmp/retrain_adapter/_live_adapter",
                "load_inplace": True,
            },
        )
    ]


def test_sglang_reload_calls_lora_endpoint(monkeypatch):
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: _FakeTokenizer(),
    )

    engine = OpenAIEngine(
        base_url="http://localhost:30000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="sglang",
    )

    calls = []

    def _fake_post(endpoint, payload, max_retries=3, **kwargs):
        calls.append((endpoint, dict(payload)))
        return {}

    monkeypatch.setattr(engine, "_post", _fake_post)

    engine.reload_weights("/tmp/retrain_adapter/_live_adapter")

    assert calls == [
        (
            "/load_lora_adapter",
            {
                "lora_name": "default",
                "lora_path": "/tmp/retrain_adapter/_live_adapter",
            },
        )
    ]


def test_sglang_reload_unloads_existing_adapter_before_reloading(monkeypatch):
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: _FakeTokenizer(),
    )

    engine = OpenAIEngine(
        base_url="http://localhost:30000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="sglang",
    )

    calls = []

    def _fake_post(endpoint, payload, max_retries=3, **kwargs):
        calls.append((endpoint, dict(payload)))
        return {}

    monkeypatch.setattr(engine, "_post", _fake_post)

    engine.reload_weights("/tmp/retrain_adapter/_live_adapter")
    engine.reload_weights("/tmp/retrain_adapter/_live_adapter")

    assert calls == [
        (
            "/load_lora_adapter",
            {
                "lora_name": "default",
                "lora_path": "/tmp/retrain_adapter/_live_adapter",
            },
        ),
        ("/unload_lora_adapter", {"lora_name": "default"}),
        (
            "/load_lora_adapter",
            {
                "lora_name": "default",
                "lora_path": "/tmp/retrain_adapter/_live_adapter",
            },
        ),
    ]


def test_trtllm_reload_sets_per_request_lora_without_http_reload(monkeypatch):
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: _FakeTokenizer(),
    )

    engine = OpenAIEngine(
        base_url="http://localhost:31000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="trtllm",
    )

    calls = []

    def _fake_post(endpoint, payload, max_retries=3, **kwargs):
        calls.append((endpoint, dict(payload)))
        return {"choices": []}

    monkeypatch.setattr(engine, "_post", _fake_post)

    engine.reload_weights("/tmp/retrain_adapter/_live_adapter")
    engine.reload_weights("/tmp/retrain_adapter/_live_adapter")
    engine.generate([[1, 2, 3]], num_samples=1, max_tokens=8, temperature=0.7, top_p=0.95)

    assert calls == [
        (
            "/v1/completions",
            {
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                "prompt": "prompt",
                "max_tokens": 8,
                "temperature": 0.7,
                "top_p": 0.95,
                "n": 1,
                "logprobs": 1,
                "lora_request": {
                    "lora_name": "default",
                    "lora_int_id": 0,
                    "lora_path": "/tmp/retrain_adapter/_live_adapter",
                },
            },
        )
    ]
    counters = engine.performance_counters()
    assert counters["engine_adapter_reload_calls"] == 2
    assert counters["engine_adapter_reload_failures"] == 0
    assert counters["engine_token_native_prompt_enabled"] == 0


def test_vllm_generate_uses_token_prompt_without_decode(monkeypatch):
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: tokenizer,
    )

    engine = OpenAIEngine(
        base_url="http://localhost:8000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="vllm",
    )

    calls = []

    def _fake_post(endpoint, payload, max_retries=3):
        calls.append((endpoint, dict(payload)))
        return {"choices": []}

    monkeypatch.setattr(engine, "_post", _fake_post)

    engine.generate([[1, 2, 3]], num_samples=1, max_tokens=8, temperature=0.7, top_p=0.95)
    engine.generate([[1, 2, 3]], num_samples=1, max_tokens=8, temperature=0.7, top_p=0.95)

    assert [payload["prompt"] for _endpoint, payload in calls] == [[1, 2, 3], [1, 2, 3]]
    assert tokenizer.decode_calls == 0
    assert engine.performance_counters() == {
        "engine_prompt_decode_calls": 0,
        "engine_prompt_cache_hits": 0,
        "engine_prompt_cache_size": 0,
        "engine_token_prompt_calls": 2,
        "engine_token_prompt_fallbacks": 0,
        "engine_token_native_prompt_enabled": 1,
        "engine_adapter_reload_calls": 0,
        "engine_adapter_reload_failures": 0,
        "engine_adapter_reload_skips": 0,
    }


def test_openai_generate_reuses_cached_prompt_decode(monkeypatch):
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: tokenizer,
    )

    engine = OpenAIEngine(
        base_url="http://localhost:8000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="openai",
    )

    monkeypatch.setattr(engine, "_post", lambda endpoint, payload, max_retries=3: {"choices": []})

    engine.generate([[1, 2, 3]], num_samples=1, max_tokens=8, temperature=0.7, top_p=0.95)
    engine.generate([[1, 2, 3]], num_samples=1, max_tokens=8, temperature=0.7, top_p=0.95)

    assert tokenizer.decode_calls == 1
    counters = engine.performance_counters()
    assert counters["engine_prompt_decode_calls"] == 1
    assert counters["engine_prompt_cache_hits"] == 1
    assert counters["engine_prompt_cache_size"] == 1
    assert counters["engine_token_prompt_calls"] == 0
    assert counters["engine_token_native_prompt_enabled"] == 0


def test_token_prompt_rejection_falls_back_to_text(monkeypatch):
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: tokenizer,
    )

    engine = OpenAIEngine(
        base_url="http://localhost:8000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="sglang",
    )

    calls = []

    def _fake_post(endpoint, payload, max_retries=3):
        calls.append((endpoint, dict(payload)))
        if len(calls) == 1:
            raise RuntimeError("HTTP error from server: 422 Client Error")
        return {"choices": []}

    monkeypatch.setattr(engine, "_post", _fake_post)

    engine.generate([[1, 2, 3]], num_samples=1, max_tokens=8, temperature=0.7, top_p=0.95)

    assert calls[0][1]["prompt"] == [1, 2, 3]
    assert calls[1][1]["prompt"] == "prompt"
    assert tokenizer.decode_calls == 1
    counters = engine.performance_counters()
    assert counters["engine_token_prompt_calls"] == 1
    assert counters["engine_token_prompt_fallbacks"] == 1


def test_vllm_reload_same_live_adapter_path_still_posts(monkeypatch):
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: _FakeTokenizer(),
    )

    engine = OpenAIEngine(
        base_url="http://localhost:8000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="vllm",
    )

    calls = []

    def _fake_post(endpoint, payload, max_retries=3, **kwargs):
        calls.append((endpoint, dict(payload)))
        return {}

    monkeypatch.setattr(engine, "_post", _fake_post)

    engine.reload_weights("/tmp/retrain_adapter/_live_adapter")
    engine.reload_weights("/tmp/retrain_adapter/_live_adapter")

    assert [endpoint for endpoint, _payload in calls] == [
        "/v1/load_lora_adapter",
        "/v1/load_lora_adapter",
    ]
    counters = engine.performance_counters()
    assert counters["engine_adapter_reload_calls"] == 2
    assert counters["engine_adapter_reload_failures"] == 0


def test_vllm_reload_accepts_plain_text_success(monkeypatch):
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: _FakeTokenizer(),
    )

    engine = OpenAIEngine(
        base_url="http://localhost:8000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="vllm",
    )

    class _Response:
        status_code = 200
        text = "Success: LoRA adapter loaded."

        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(engine.session, "post", lambda *args, **kwargs: _Response())

    engine.reload_weights("/tmp/retrain_adapter/_live_adapter")

    counters = engine.performance_counters()
    assert counters["engine_adapter_reload_calls"] == 1
    assert counters["engine_adapter_reload_failures"] == 0


def test_vllm_generate_uses_loaded_lora_model_name_after_reload(monkeypatch):
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: _FakeTokenizer(),
    )

    engine = OpenAIEngine(
        base_url="http://localhost:8000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="vllm",
    )

    calls = []

    def _fake_post(endpoint, payload, max_retries=3, **kwargs):
        calls.append((endpoint, dict(payload)))
        if endpoint == "/v1/load_lora_adapter":
            return {}
        return {"choices": []}

    monkeypatch.setattr(engine, "_post", _fake_post)

    engine.reload_weights("/tmp/retrain_adapter/_live_adapter")
    engine.generate([[1, 2, 3]], num_samples=1, max_tokens=8, temperature=0.7, top_p=0.95)

    assert calls[-1][0] == "/v1/completions"
    assert calls[-1][1]["model"] == "default"


def test_sglang_generate_uses_model_adapter_suffix_after_reload(monkeypatch):
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: _FakeTokenizer(),
    )

    engine = OpenAIEngine(
        base_url="http://localhost:30000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="sglang",
    )

    calls = []

    def _fake_post(endpoint, payload, max_retries=3, **kwargs):
        calls.append((endpoint, dict(payload)))
        if endpoint == "/load_lora_adapter":
            return {}
        return {"choices": []}

    monkeypatch.setattr(engine, "_post", _fake_post)

    engine.reload_weights("/tmp/retrain_adapter/_live_adapter")
    engine.generate([[1, 2, 3]], num_samples=1, max_tokens=8, temperature=0.7, top_p=0.95)

    assert calls[-1][0] == "/v1/completions"
    assert calls[-1][1]["model"] == "Qwen/Qwen3-4B-Instruct-2507:default"


def test_vllm_reload_failure_is_counted(monkeypatch):
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: _FakeTokenizer(),
    )

    engine = OpenAIEngine(
        base_url="http://localhost:8000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="vllm",
    )

    def _fake_post(endpoint, payload, max_retries=3, **kwargs):
        raise RuntimeError("HTTP error from server: 404 Client Error")

    monkeypatch.setattr(engine, "_post", _fake_post)

    engine.reload_weights("/tmp/retrain_adapter/_live_adapter")

    counters = engine.performance_counters()
    assert counters["engine_adapter_reload_calls"] == 1
    assert counters["engine_adapter_reload_failures"] == 1


def test_direct_token_ids_payload_is_used(monkeypatch):
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: _FakeTokenizer(),
    )

    engine = OpenAIEngine(
        base_url="http://localhost:8000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="vllm",
    )

    response = {
        "choices": [
            {
                "text": "world",
                "token_ids": [101, 102],
                "logprobs": {
                    "token_logprobs": [-0.5, -0.3],
                    "tokens": ["wor", "ld"],
                },
            }
        ]
    }
    monkeypatch.setattr(engine, "_post", lambda endpoint, payload, max_retries=3: response)

    results = engine.generate(
        [[1, 2, 3]],
        num_samples=1,
        max_tokens=10,
        temperature=0.7,
        top_p=0.95,
    )

    assert results[0][0].token_ids == [101, 102]
    assert results[0][0].logprobs == [-0.5, -0.3]


def test_null_choice_token_ids_with_unconvertible_tokens_reencodes_text(monkeypatch):
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: _FallbackTokenizer(),
    )

    engine = OpenAIEngine(
        base_url="http://localhost:8000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="vllm",
    )

    response = {
        "choices": [
            {
                "text": ". I'm",
                "token_ids": None,
                "logprobs": {
                    "token_logprobs": [-0.5, -0.3],
                    "tokens": [".", " I'm"],
                },
            }
        ]
    }
    monkeypatch.setattr(engine, "_post", lambda endpoint, payload, max_retries=3: response)

    results = engine.generate(
        [[1, 2, 3]],
        num_samples=1,
        max_tokens=10,
        temperature=0.7,
        top_p=0.95,
    )

    assert results[0][0].token_ids == [101, 102]
    assert results[0][0].logprobs == [-0.5, -0.3]


def test_content_logprobs_with_token_ids_use_content_path(monkeypatch):
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: _FallbackTokenizer(),
    )

    engine = OpenAIEngine(
        base_url="http://localhost:8000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="openai",
    )

    response = {
        "choices": [
            {
                "text": "world",
                "logprobs": {
                    "content": [
                        {"token_id": 701, "logprob": "-0.7"},
                        {"token_id": "702", "logprob": -0.2},
                    ]
                },
            }
        ]
    }
    monkeypatch.setattr(engine, "_post", lambda endpoint, payload, max_retries=3: response)

    results = engine.generate(
        [[1, 2, 3]],
        num_samples=1,
        max_tokens=10,
        temperature=0.7,
        top_p=0.95,
    )

    assert results[0][0].token_ids == [701, 702]
    assert results[0][0].logprobs == [-0.7, -0.2]


def test_malformed_choices_response_raises(monkeypatch):
    monkeypatch.setattr(
        "retrain.inference_engine.openai_engine.AutoTokenizer.from_pretrained",
        lambda _model_name: _FakeTokenizer(),
    )

    engine = OpenAIEngine(
        base_url="http://localhost:8000",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        engine_type="openai",
    )
    monkeypatch.setattr(
        engine,
        "_post",
        lambda endpoint, payload, max_retries=3: {"choices": {"text": "bad"}},
    )

    with pytest.raises(RuntimeError, match="'choices' must be a list"):
        engine.generate(
            [[1, 2, 3]],
            num_samples=1,
            max_tokens=10,
            temperature=0.7,
            top_p=0.95,
        )
