from __future__ import annotations

import math
from types import SimpleNamespace

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


def test_local_echo_hybrid_uses_one_forward_for_rl_and_echo(monkeypatch) -> None:
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = _TinyLM()
    helper.optimizer = torch.optim.SGD(helper.train_model.parameters(), lr=0.05)
    helper.scaler = torch.amp.GradScaler(enabled=False)
    helper.train_microbatch_size = 0
    helper.train_device = "cpu"
    helper.use_amp = False
    helper.clip_eps = 0.0
    helper.clip_eps_high = 0.0
    helper.split_mode = False
    helper._external_engine = False
    helper.cuda_empty_cache = False
    helper._clip_fraction = 0.0

    forward_calls = 0
    real_forward_logits = local_mod.forward_logits

    def counted_forward_logits(model, input_ids, attention_mask):  # noqa: ANN001
        nonlocal forward_calls
        forward_calls += 1
        return real_forward_logits(model, input_ids, attention_mask)

    monkeypatch.setattr(local_mod, "forward_logits", counted_forward_logits)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    old_logprobs = torch.zeros_like(input_ids, dtype=torch.float32)
    advantages = torch.tensor([[0.0, 0.25, -0.10, 0.15]], dtype=torch.float32)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    echo_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    echo_advantages = torch.tensor(
        [[0.0, 0.0, 0.2, 0.2, 0.0]],
        dtype=torch.float32,
    )
    echo_attention_mask = torch.ones_like(echo_input_ids, dtype=torch.bool)
    before = [p.detach().clone() for p in helper.train_model.parameters()]

    rl_loss, echo_loss = helper._do_hybrid_impl(
        input_ids,
        old_logprobs,
        advantages,
        attention_mask,
        echo_input_ids,
        echo_advantages,
        echo_attention_mask,
        "cross_entropy",
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
