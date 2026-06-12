"""Focused tests for Gemma4 local-backend integration helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from retrain.gemma4_text import (
    DEFAULT_LORA_TARGET_MODULES,
    eos_token_ids,
    forward_logits,
    is_gemma4_text_model,
    resolve_lora_target_modules,
)
from retrain.inference_engine.pytorch_engine import PyTorchEngine, _sample_next_token


class _ExistingModel:
    def __init__(self) -> None:
        self.to_calls: list[str] = []

    def to(self, device: str):
        self.to_calls.append(device)
        return self


class _FakeLanguageModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        self.calls.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "kwargs": kwargs,
        })
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeGemma4TextModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(model_type="gemma4", eos_token_id=[7, 8])
        self.generation_config = SimpleNamespace(eos_token_id=9)
        self.language_model = _FakeLanguageModel()
        self.model = SimpleNamespace(language_model=self.language_model)
        self.lm_head = lambda hidden: hidden + 1.0

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("Gemma4 multimodal wrapper should not be called")

    def named_modules(self):
        names = [
            "",
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.self_attn.k_proj",
            "model.language_model.layers.0.mlp.down_proj",
            "model.vision_tower.layers.0.self_attn.q_proj",
        ]
        return [(name, object()) for name in names]


class _WrappedGemma4TextModel:
    def __init__(self, model: _FakeGemma4TextModel) -> None:
        self.base_model = SimpleNamespace(model=model)

    def named_modules(self):
        return [("base_model.model.model.language_model.layers.0.self_attn.q_proj", object())]


def test_gemma4_lora_targets_are_language_tower_exact_names():
    model = _FakeGemma4TextModel()

    targets = resolve_lora_target_modules(model, DEFAULT_LORA_TARGET_MODULES)

    assert targets == [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.mlp.down_proj",
    ]


def test_gemma4_lora_targets_scan_unwrapped_peft_model():
    model = _FakeGemma4TextModel()
    wrapped = _WrappedGemma4TextModel(model)

    targets = resolve_lora_target_modules(wrapped, DEFAULT_LORA_TARGET_MODULES)

    assert targets == [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.mlp.down_proj",
    ]


def test_gemma4_forward_logits_bypasses_multimodal_wrapper():
    model = _FakeGemma4TextModel()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[True, True, True]])

    logits = forward_logits(model, input_ids, attention_mask)

    assert is_gemma4_text_model(model)
    assert logits.shape == (1, 3, 3)
    assert model.language_model.calls[0]["attention_mask"] is attention_mask
    assert model.language_model.calls[0]["use_cache"] is False
    assert torch.equal(logits[0, :, 0], torch.tensor([2.0, 3.0, 4.0]))


def test_gemma4_eos_ids_prefer_generation_config():
    model = _FakeGemma4TextModel()

    assert eos_token_ids(model) == {9}


def test_pytorch_engine_moves_existing_model_to_requested_device():
    model = _ExistingModel()

    engine = PyTorchEngine(
        model_name="unused",
        device="cuda:7",
        peft_config=None,
        dtype=None,
        existing_model=model,
    )

    assert engine.model is model
    assert model.to_calls == ["cuda:7"]


def test_pytorch_engine_rejects_peft_config_with_existing_model():
    with pytest.raises(ValueError, match="peft_config must be None"):
        PyTorchEngine(
            model_name="unused",
            device="cpu",
            peft_config=object(),
            dtype=None,
            existing_model=_ExistingModel(),
        )


def test_top_p_sampling_entropy_uses_full_distribution():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    probs = F.softmax(logits, dim=-1)
    expected_entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

    token, logprob, entropy = _sample_next_token(logits, temperature=1.0, top_p=0.4)

    assert token.tolist() == [[0]]
    assert logprob.tolist() == pytest.approx([0.0])
    assert entropy.tolist() == pytest.approx(expected_entropy.tolist())
