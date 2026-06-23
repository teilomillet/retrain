from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from retrain import local_train_helper as local_mod
from retrain.local_train_helper import LocalTrainHelper


class _TinyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.lm_head = torch.nn.Linear(8, 16)

    def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
        _ = attention_mask
        return SimpleNamespace(logits=self.lm_head(self.embed(input_ids)))


class _ScalarLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))


class _ClearCacheEngine:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear_prefix_cache(self) -> None:
        self.clear_calls += 1


def _helper(model: torch.nn.Module) -> LocalTrainHelper:
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.optimizer = torch.optim.SGD(helper.train_model.parameters(), lr=0.05)
    helper.scaler = torch.amp.GradScaler(enabled=False)
    helper.train_microbatch_size = 0
    helper.train_device = "cpu"
    helper.use_amp = False
    helper.clip_eps = 0.0
    helper.clip_eps_high = 0.0
    helper.split_mode = False
    helper._external_engine = False
    helper._train_future = None
    helper.cuda_empty_cache = False
    helper._clip_fraction = 0.0
    helper.policy_loss_mode = "standard"
    return helper


def test_local_echo_hybrid_masks_use_one_forward_and_full_observation_denominator(monkeypatch) -> None:
    helper = _helper(_ScalarLM())
    forward_calls = 0

    def counted_forward_logits(model, input_ids, attention_mask):  # noqa: ANN001
        _ = attention_mask
        nonlocal forward_calls
        forward_calls += 1
        vocab_size = 16
        return model.bias * 0.0 + torch.zeros(
            (*input_ids.shape, vocab_size),
            dtype=torch.float32,
        )

    monkeypatch.setattr(local_mod, "forward_logits", counted_forward_logits)

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    old_logprobs = torch.zeros_like(input_ids, dtype=torch.float32)
    advantages = torch.tensor([[0.0, 0.25, -0.10, 0.0, 0.15]], dtype=torch.float32)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    echo_advantages = torch.tensor(
        [[0.0, 0.0, 0.2, 0.2, 0.0]],
        dtype=torch.float32,
    )
    echo_counts = torch.tensor([4.0], dtype=torch.float32)
    before = [p.detach().clone() for p in helper.train_model.parameters()]

    rl_loss, echo_loss = helper._do_hybrid_mask_impl(
        input_ids,
        old_logprobs,
        advantages,
        attention_mask,
        echo_advantages,
        echo_counts,
        "cross_entropy",
    )

    changed = any(
        not torch.equal(old, new)
        for old, new in zip(before, helper.train_model.parameters())
    )
    assert forward_calls == 1
    assert math.isfinite(float(rl_loss))
    assert float(echo_loss) == pytest.approx(0.1 * math.log(16), rel=1e-5)
    # The zero-logit fixture keeps the exact loss easy to inspect; gradients are
    # zero by construction, so the parameter need not change in this test.
    assert not changed


def test_local_echo_mask_train_step_updates_model_with_real_logits(monkeypatch) -> None:
    helper = _helper(_TinyLM())
    forward_calls = 0
    real_forward_logits = local_mod.forward_logits

    def counted_forward_logits(model, input_ids, attention_mask):  # noqa: ANN001
        nonlocal forward_calls
        forward_calls += 1
        return real_forward_logits(model, input_ids, attention_mask)

    monkeypatch.setattr(local_mod, "forward_logits", counted_forward_logits)
    before = [p.detach().clone() for p in helper.train_model.parameters()]

    rl_loss, echo_loss = helper.train_step_with_echo_masks(
        all_tokens=[[1, 2, 3, 4]],
        all_logprobs=[[0.0, 0.0, -0.2, 0.0]],
        all_advantages=[[0.0, 0.0, 0.3, 0.0]],
        echo_advantages=[[0.0, 0.0, 0.2, 0.0]],
        echo_full_observation_counts=[1],
        echo_loss_fn="cross_entropy",
        lr=0.05,
        weight_decay=0.0,
    )

    changed = any(
        not torch.equal(old, new)
        for old, new in zip(before, helper.train_model.parameters())
    )
    assert forward_calls == 1
    assert math.isfinite(float(rl_loss))
    assert math.isfinite(float(echo_loss))
    assert float(echo_loss) > 0.0
    assert changed


def test_local_echo_train_step_clears_inference_prefix_cache() -> None:
    helper = _helper(_TinyLM())
    engine = _ClearCacheEngine()
    helper.engine = engine

    helper.train_step_with_echo_masks(
        all_tokens=[[1, 2, 3, 4]],
        all_logprobs=[[0.0, 0.0, -0.2, 0.0]],
        all_advantages=[[0.0, 0.0, 0.3, 0.0]],
        echo_advantages=[[0.0, 0.0, 0.2, 0.0]],
        echo_full_observation_counts=[1],
        echo_loss_fn="cross_entropy",
        lr=0.05,
        weight_decay=0.0,
    )

    assert engine.clear_calls == 1
