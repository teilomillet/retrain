"""Tests for OpenAI-compatible engine behaviors."""

from __future__ import annotations

from retrain.inference_engine.openai_engine import OpenAIEngine


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return "prompt"

    def encode(self, text, add_special_tokens=False):
        return [1, 2]

    def convert_tokens_to_ids(self, tokens):
        return [1 for _ in tokens]


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

    def _fake_post(endpoint, payload, max_retries=3):  # noqa: ARG001
        calls.append((endpoint, payload))
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

    def _fake_post(endpoint, payload, max_retries=3):  # noqa: ARG001
        calls.append((endpoint, payload))
        return {}

    monkeypatch.setattr(engine, "_post", _fake_post)

    engine.reload_weights("/tmp/retrain_adapter/_live_adapter")

    assert calls == [
        (
            "/v1/load_lora_adapter",
            {
                "lora_name": "default",
                "lora_path": "/tmp/retrain_adapter/_live_adapter",
            },
        )
    ]
