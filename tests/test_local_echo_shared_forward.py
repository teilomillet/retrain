from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

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


class _TinyBackbone(torch.nn.Module):
    def __init__(self, embed: torch.nn.Embedding) -> None:
        super().__init__()
        self.embed = embed

    def forward(self, input_ids, attention_mask=None, use_cache=False):  # noqa: ANN001
        _ = attention_mask, use_cache
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class _TinyHiddenLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.model = _TinyBackbone(self.embed)
        self.lm_head = torch.nn.Linear(8, 16)

    def forward(self, input_ids, attention_mask=None, **kwargs):  # noqa: ANN001
        _ = kwargs
        hidden = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden))


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


def test_selective_hidden_logprobs_match_dense_logits_on_selected_tokens() -> None:
    torch.manual_seed(123)
    model = _TinyHiddenLM()
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.train_selective_suffix_logits = True
    helper.train_logprob_chunk_size = 1
    helper._train_logits_to_keep_supported = False
    helper._selective_logprob_path_counts = {}

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    target_mask = torch.tensor([[False, False, True, True]])

    actual = helper._shifted_token_logprobs(
        input_ids,
        attention_mask,
        target_mask=target_mask,
    )

    hidden = model.model(input_ids, attention_mask=attention_mask).last_hidden_state
    dense_logits = model.lm_head(hidden[:, :-1, :])
    target_ids = input_ids[:, 1:]
    expected = F.log_softmax(dense_logits.float(), dim=-1).gather(
        2,
        target_ids.unsqueeze(2),
    ).squeeze(2)

    assert torch.allclose(actual[target_mask], expected[target_mask])
    assert torch.equal(actual[~target_mask], torch.zeros_like(actual[~target_mask]))
    assert helper._selective_logprob_path_counts == {"hidden": 1}


def test_positive_logprob_chunk_size_skips_logits_to_keep_suffix_path() -> None:
    torch.manual_seed(321)
    model = _TinyHiddenLM()
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.train_selective_suffix_logits = True
    helper.train_logprob_chunk_size = 1
    helper._train_logits_to_keep_supported = True
    helper._selective_logprob_path_counts = {}

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    target_mask = torch.tensor([[False, False, True, True]])

    actual = helper._shifted_token_logprobs(
        input_ids,
        attention_mask,
        target_mask=target_mask,
    )

    assert actual.shape == target_mask.shape
    assert helper._train_logits_to_keep_supported is True
    assert helper._selective_logprob_path_counts == {"hidden": 1}


def test_sparse_long_target_mask_skips_suffix_logits_for_hidden_path() -> None:
    torch.manual_seed(654)
    model = _TinyHiddenLM()
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.train_selective_suffix_logits = True
    helper.train_logprob_chunk_size = 0
    helper._train_logits_to_keep_supported = True
    helper._selective_logprob_path_counts = {}

    input_ids = (torch.arange(3008, dtype=torch.long).unsqueeze(0) % 16)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    target_mask = torch.zeros((1, input_ids.shape[1] - 1), dtype=torch.bool)
    target_mask[:, 10] = True
    target_mask[:, 20] = True

    actual = helper._shifted_token_logprobs(
        input_ids,
        attention_mask,
        target_mask=target_mask,
    )

    hidden = model.model(input_ids, attention_mask=attention_mask).last_hidden_state
    dense_logits = model.lm_head(hidden[:, :-1, :])
    target_ids = input_ids[:, 1:]
    expected = F.log_softmax(dense_logits.float(), dim=-1).gather(
        2,
        target_ids.unsqueeze(2),
    ).squeeze(2)

    assert actual.shape == target_mask.shape
    assert torch.allclose(actual[target_mask], expected[target_mask])
    assert torch.equal(actual[~target_mask], torch.zeros_like(actual[~target_mask]))
    assert helper._train_logits_to_keep_supported is True
    assert helper._selective_logprob_path_counts == {
        "sparse_suffix_skip": 1,
        "hidden": 1,
    }


def test_unsloth_fused_sft_loss_matches_dense_ce_on_constant_weights(monkeypatch) -> None:
    torch.manual_seed(456)
    model = _TinyHiddenLM()
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.train_device = "cpu"
    helper.use_amp = False
    helper.train_selective_suffix_logits = True
    helper.train_unsloth_fused_ce = "require"
    helper.train_unsloth_fused_ce_target_gb = 0.0
    helper.train_unsloth_fused_ce_torch_compile = False
    helper._loss_path_counts = {}
    captured: dict[str, torch.Tensor | object] = {}

    def fake_fused_ce(
        trainer,  # noqa: ANN001
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: torch.Tensor | None,
        labels: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        n_items: torch.Tensor | None = None,
        shift_labels: bool = True,
        **kwargs,  # noqa: ANN003
    ) -> torch.Tensor:
        _ = trainer, kwargs
        assert shift_labels is True
        captured["labels"] = labels.detach().clone()
        captured["mask"] = None if mask is None else mask.detach().clone()
        logits = F.linear(hidden_states[:, :-1, :], lm_head_weight, lm_head_bias)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        return loss / n_items.to(loss.device)

    monkeypatch.setattr(helper, "_unsloth_fused_ce_loss", lambda: fake_fused_ce)

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    advantages = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

    actual, token_count = helper._compute_sft_loss(
        input_ids,
        advantages,
        attention_mask,
    )

    hidden = model.model(input_ids, attention_mask=attention_mask).last_hidden_state
    dense_logits = model.lm_head(hidden[:, :-1, :])
    target_ids = input_ids[:, 1:]
    expected_logprobs = F.log_softmax(dense_logits.float(), dim=-1).gather(
        2,
        target_ids.unsqueeze(2),
    ).squeeze(2)
    weights = advantages[:, 1:]
    expected = (-expected_logprobs * weights).sum() / (weights > 0).sum()

    assert torch.allclose(actual, expected)
    assert float(token_count.item()) == 2.0
    assert helper._loss_path_counts == {"unsloth_fused_ce": 1}
    labels = captured["labels"]
    mask = captured["mask"]
    assert isinstance(labels, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert labels.tolist() == [[-100, -100, -100, 4, 5]]
    assert mask.tolist() == [[False, False, False, True, True]]


def test_unsloth_fused_sft_loss_falls_back_for_nonconstant_weights() -> None:
    helper = object.__new__(LocalTrainHelper)
    helper.train_device = "cpu"
    helper.use_amp = False
    helper.train_selective_suffix_logits = False
    helper.train_unsloth_fused_ce = "auto"
    helper._loss_path_counts = {}

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    advantages = torch.tensor([[0.0, 0.0, 0.5, 1.0]], dtype=torch.float32)

    def shifted_logprobs(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
        return torch.tensor([[-0.1, -0.2, -0.3]], dtype=torch.float32)

    helper._shifted_token_logprobs = shifted_logprobs
    loss, token_count = helper._compute_sft_loss(
        input_ids,
        advantages,
        attention_mask,
    )

    expected = (0.2 * 0.5 + 0.3 * 1.0) / 2.0
    assert float(loss.item()) == pytest.approx(expected)
    assert float(token_count.item()) == 2.0
    assert helper._unsloth_fused_ce_fallback_reason == "non_constant_token_weights"


def test_unsloth_fused_sft_loss_falls_back_for_long_sparse_targets() -> None:
    helper = object.__new__(LocalTrainHelper)
    helper.train_device = "cpu"
    helper.use_amp = False
    helper.train_selective_suffix_logits = False
    helper.train_unsloth_fused_ce = "auto"
    helper._loss_path_counts = {}

    input_ids = torch.arange(2052, dtype=torch.long).unsqueeze(0) % 16
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    advantages = torch.zeros_like(input_ids, dtype=torch.float32)
    advantages[:, -2:] = 1.0

    def shifted_logprobs(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
        return torch.full((1, input_ids.shape[1] - 1), -0.25, dtype=torch.float32)

    helper._shifted_token_logprobs = shifted_logprobs
    loss, token_count = helper._compute_sft_loss(
        input_ids,
        advantages,
        attention_mask,
    )

    assert float(loss.item()) == pytest.approx(0.25)
    assert float(token_count.item()) == 2.0
    assert helper._unsloth_fused_ce_fallback_reason == "sparse_supervised_tokens"


def test_unsloth_fused_sft_loss_rejects_saved_tensor_offload() -> None:
    helper = object.__new__(LocalTrainHelper)
    helper.train_device = "cpu"
    helper.use_amp = False
    helper.train_selective_suffix_logits = True
    helper.train_save_on_cpu = True
    helper.train_unsloth_fused_ce = "require"
    helper._loss_path_counts = {}

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    advantages = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="saved_tensor_hooks_incompatible"):
        helper._compute_sft_loss(
            input_ids,
            advantages,
            attention_mask,
        )


def test_unsloth_fused_ce_auto_target_uses_small_gpu_heuristic(monkeypatch) -> None:
    helper = object.__new__(LocalTrainHelper)
    helper.train_device = "cuda:0"
    helper.train_unsloth_fused_ce_target_gb = 0.0

    monkeypatch.setattr(local_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        local_mod.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=12 * 1024**3),
    )

    assert helper._effective_unsloth_fused_ce_target_gb() == pytest.approx(0.25)

    helper.train_unsloth_fused_ce_target_gb = 0.75
    assert helper._effective_unsloth_fused_ce_target_gb() == pytest.approx(0.75)


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
