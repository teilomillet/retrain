from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from retrain.backends.local import metrics as local_metrics
from retrain.backends.local import train as local_mod
from retrain.backends.local import LocalTrainHelper
from retrain.kernels.logprobs import (
    packed_quantized_linear_target_logprobs,
    unpack_quantized_weight,
)


class _TinyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.lm_head = torch.nn.Linear(8, 16)

    def forward(self, input_ids, attention_mask=None):
        _ = attention_mask
        return SimpleNamespace(logits=self.lm_head(self.embed(input_ids)))


class _TinyBackbone(torch.nn.Module):
    def __init__(self, embed: torch.nn.Embedding) -> None:
        super().__init__()
        self.embed = embed

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        _ = attention_mask, use_cache
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class _TinyHiddenLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.model = _TinyBackbone(self.embed)
        self.lm_head = torch.nn.Linear(8, 16)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        _ = kwargs
        hidden = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden))


def _pack_int2(values: torch.Tensor) -> torch.Tensor:
    unsigned = (values + 2).to(torch.uint8)
    pad = (-unsigned.shape[-1]) % 4
    if pad:
        unsigned = F.pad(unsigned, (0, pad))
    return (
        unsigned[:, 0::4]
        | (unsigned[:, 1::4] << 2)
        | (unsigned[:, 2::4] << 4)
        | (unsigned[:, 3::4] << 6)
    )


class _FakePackedQuantizedHead(torch.nn.Module):
    def __init__(self, in_features: int = 8, out_features: int = 16) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = 2
        int_weight = (
            torch.arange(out_features * in_features, dtype=torch.int8)
            .reshape(out_features, in_features)
            .remainder(4)
            - 2
        )
        self.weight = torch.nn.Parameter(_pack_int2(int_weight), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(
            torch.linspace(0.25, 1.0, out_features).unsqueeze(1),
            requires_grad=False,
        )
        self.bias = torch.nn.Parameter(torch.linspace(-0.5, 0.5, out_features))
        self.input_activation_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.output_activation_scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = unpack_quantized_weight(
            self.weight,
            self.num_bits,
            self.in_features,
        ).to(hidden.dtype)
        weight = weight * self.weight_scale.to(hidden.dtype)
        return F.linear(hidden, weight, self.bias.to(hidden.dtype))


class _TinyPackedHiddenLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.model = _TinyBackbone(self.embed)
        self.lm_head = _FakePackedQuantizedHead(8, 16)

    def forward(self, input_ids, attention_mask=None, **kwargs):
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

    def counted_forward_logits(model, input_ids, attention_mask):
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

    def counted_forward_logits(model, input_ids, attention_mask):
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


def test_packed_quantized_lm_head_target_logprobs_match_dense() -> None:
    torch.manual_seed(123)
    lm_head = _FakePackedQuantizedHead(in_features=8, out_features=16)
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    target_ids = torch.tensor([[0, 5, 15], [3, 8, 12]], dtype=torch.long)

    actual = packed_quantized_linear_target_logprobs(
        hidden,
        lm_head,
        target_ids,
        vocab_chunk_size=5,
    )

    dense_logits = lm_head(hidden)
    expected = F.log_softmax(dense_logits.float(), dim=-1).gather(
        2,
        target_ids.unsqueeze(2),
    ).squeeze(2)

    assert actual is not None
    assert torch.allclose(actual, expected, atol=1e-6)
    (-actual.sum()).backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_selective_hidden_uses_packed_quantized_lm_head_path() -> None:
    torch.manual_seed(321)
    model = _TinyPackedHiddenLM()
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.train_selective_suffix_logits = True
    helper.train_logprob_chunk_size = 1
    helper.train_compile_selective_ce = "off"
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

    assert torch.allclose(actual[target_mask], expected[target_mask], atol=1e-6)
    assert torch.equal(actual[~target_mask], torch.zeros_like(actual[~target_mask]))
    assert helper._selective_logprob_path_counts == {
        "hidden": 1,
        "packed_quantized_lm_head": 1,
    }
    metrics = helper.runtime_metrics()
    assert metrics["local_train_selective_packed_quantized_lm_head_batches"] == 1
    assert metrics["local_train_packed_quantized_lm_head_batches"] == 0


def test_selective_compile_auto_falls_back_on_cpu() -> None:
    torch.manual_seed(123)
    model = _TinyHiddenLM()
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.train_selective_suffix_logits = True
    helper.train_logprob_chunk_size = 1
    helper.train_compile_selective_ce = "auto"
    helper.train_compile_selective_ce_min_tokens = 1
    helper._train_logits_to_keep_supported = False
    helper._selective_logprob_path_counts = {}
    helper._compiled_selective_ce_fallback_reason = ""
    helper._compiled_selective_ce_available = None

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    target_mask = torch.tensor([[False, False, True, True]])

    actual = helper._shifted_token_logprobs(
        input_ids,
        attention_mask,
        target_mask=target_mask,
    )

    assert actual.shape == target_mask.shape
    assert helper._selective_logprob_path_counts == {"hidden": 1}
    assert helper._compiled_selective_ce_fallback_reason == "non_cuda"


def test_selective_compile_require_raises_on_cpu() -> None:
    model = _TinyHiddenLM()
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.train_selective_suffix_logits = True
    helper.train_logprob_chunk_size = 1
    helper.train_compile_selective_ce = "require"
    helper.train_compile_selective_ce_min_tokens = 1
    helper._train_logits_to_keep_supported = False
    helper._selective_logprob_path_counts = {}
    helper._compiled_selective_ce_fallback_reason = ""
    helper._compiled_selective_ce_available = None

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    target_mask = torch.tensor([[False, False, True, True]])

    with pytest.raises(RuntimeError, match="non_cuda"):
        helper._shifted_token_logprobs(
            input_ids,
            attention_mask,
            target_mask=target_mask,
        )


def test_selective_compiled_path_scatter_and_metrics(monkeypatch) -> None:
    model = _TinyHiddenLM()
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.train_selective_suffix_logits = True
    helper.train_logprob_chunk_size = 1
    helper.train_compile_selective_ce = "auto"
    helper.train_compile_selective_ce_min_tokens = 1
    helper._train_logits_to_keep_supported = False
    helper._selective_logprob_path_counts = {}
    helper._compiled_selective_ce_fallback_reason = ""
    helper._compiled_selective_ce_available = None

    def fake_compiled(self, selected_hidden, lm_head, target_ids):
        _ = selected_hidden, lm_head, target_ids
        return torch.tensor([-1.0, -2.0], dtype=torch.float32)

    monkeypatch.setattr(
        LocalTrainHelper,
        "_compiled_selective_ce_logprobs",
        fake_compiled,
    )

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    target_mask = torch.tensor([[False, False, True, True]])

    actual = helper._shifted_token_logprobs(
        input_ids,
        attention_mask,
        target_mask=target_mask,
    )

    assert torch.equal(
        actual,
        torch.tensor([[0.0, 0.0, -1.0, -2.0]], dtype=torch.float32),
    )
    assert helper._selective_logprob_path_counts == {
        "hidden": 1,
        "compiled_ce": 1,
    }


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
        trainer,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: torch.Tensor | None,
        labels: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        n_items: torch.Tensor | None = None,
        shift_labels: bool = True,
        **kwargs,
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
        assert n_items is not None
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


def test_unsloth_fused_sft_loss_auto_falls_back_on_runtime_failure(monkeypatch) -> None:
    model = _TinyHiddenLM()
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.train_device = "cpu"
    helper.use_amp = False
    helper.train_selective_suffix_logits = False
    helper.train_unsloth_fused_ce = "auto"
    helper.train_unsloth_fused_ce_target_gb = 0.0
    helper.train_unsloth_fused_ce_torch_compile = False
    helper._loss_path_counts = {}

    def failing_fused_ce(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("Hard failure due to fullgraph=True")

    monkeypatch.setattr(helper, "_unsloth_fused_ce_loss", lambda: failing_fused_ce)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    advantages = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

    def shifted_logprobs(*args, **kwargs):
        _ = args, kwargs
        return torch.tensor([[-0.1, -0.2, -0.3]], dtype=torch.float32)

    helper._shifted_token_logprobs = shifted_logprobs
    loss, token_count = helper._compute_sft_loss(
        input_ids,
        advantages,
        attention_mask,
    )

    assert float(loss.item()) == pytest.approx(0.25)
    assert float(token_count.item()) == 2.0
    assert helper._unsloth_fused_ce_available is False
    assert helper._unsloth_fused_ce_unavailable_reason == "runtime_RuntimeError"
    assert helper._unsloth_fused_ce_fallback_reason == "runtime_RuntimeError"
    assert helper._unsloth_fused_ce_runtime_disabled is True


def test_unsloth_fused_sft_loss_auto_falls_back_on_hidden_runtime_failure(monkeypatch) -> None:
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = _TinyHiddenLM()
    helper.train_device = "cpu"
    helper.use_amp = False
    helper.train_selective_suffix_logits = False
    helper.train_unsloth_fused_ce = "auto"
    helper.train_unsloth_fused_ce_target_gb = 0.0
    helper.train_unsloth_fused_ce_torch_compile = False
    helper._loss_path_counts = {}

    def failing_hidden(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("Hard failure due to fullgraph=True")

    monkeypatch.setattr(local_mod, "forward_hidden_states_and_lm_head", failing_hidden)
    monkeypatch.setattr(helper, "_unsloth_fused_ce_loss", lambda: object())

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    advantages = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

    def shifted_logprobs(*args, **kwargs):
        _ = args, kwargs
        return torch.tensor([[-0.1, -0.2, -0.3]], dtype=torch.float32)

    helper._shifted_token_logprobs = shifted_logprobs
    loss, token_count = helper._compute_sft_loss(
        input_ids,
        advantages,
        attention_mask,
    )

    assert float(loss.item()) == pytest.approx(0.25)
    assert float(token_count.item()) == 2.0
    assert helper._unsloth_fused_ce_unavailable_reason == "runtime_RuntimeError"
    assert helper._unsloth_fused_ce_fallback_reason == "runtime_RuntimeError"
    assert helper._unsloth_fused_ce_runtime_disabled is True


def test_unsloth_fused_sft_loss_require_raises_on_runtime_failure(monkeypatch) -> None:
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = _TinyHiddenLM()
    helper.train_device = "cpu"
    helper.use_amp = False
    helper.train_selective_suffix_logits = False
    helper.train_unsloth_fused_ce = "require"
    helper.train_unsloth_fused_ce_target_gb = 0.0
    helper.train_unsloth_fused_ce_torch_compile = False
    helper._loss_path_counts = {}

    def failing_fused_ce(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("Hard failure due to fullgraph=True")

    monkeypatch.setattr(helper, "_unsloth_fused_ce_loss", lambda: failing_fused_ce)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    advantages = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="runtime_RuntimeError"):
        helper._compute_sft_loss(
            input_ids,
            advantages,
            attention_mask,
        )


def test_unsloth_fused_sft_loss_auto_reraises_cuda_oom(monkeypatch) -> None:
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = _TinyHiddenLM()
    helper.train_device = "cpu"
    helper.use_amp = False
    helper.train_selective_suffix_logits = False
    helper.train_unsloth_fused_ce = "auto"
    helper.train_unsloth_fused_ce_target_gb = 0.0
    helper.train_unsloth_fused_ce_torch_compile = False
    helper._loss_path_counts = {}

    def oom_fused_ce(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")

    monkeypatch.setattr(helper, "_unsloth_fused_ce_loss", lambda: oom_fused_ce)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    advantages = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        helper._compute_sft_loss(
            input_ids,
            advantages,
            attention_mask,
        )


def test_sft_train_step_auto_retries_after_fused_ce_runtime_failure(monkeypatch) -> None:
    helper = _helper(_ScalarLM())
    helper.train_unsloth_fused_ce = "auto"
    helper._unsloth_fused_ce_fallback_reason = ""
    helper._unsloth_fused_ce_unavailable_reason = ""
    helper._unsloth_fused_ce_available = None
    helper._unsloth_fused_ce_runtime_disabled = False
    calls = 0

    def fake_compute_sft_loss(input_ids, advantages, attention_mask):
        _ = input_ids, advantages, attention_mask
        nonlocal calls
        calls += 1
        if calls == 1:
            helper._unsloth_fused_ce_attempts = 1
            local_metrics.record_loss_path(helper, "unsloth_fused_ce")
            raise RuntimeError("Hard failure due to fullgraph=True")
        token_count = (advantages[:, 1:] > 0).sum().clamp(min=1)
        return helper.train_model.bias * 0.0 + torch.tensor(0.5), token_count

    monkeypatch.setattr(helper, "_compute_sft_loss", fake_compute_sft_loss)

    loss = helper._do_sft_impl(
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32),
        torch.ones((1, 3), dtype=torch.bool),
    )

    assert calls == 2
    assert loss == pytest.approx(0.5)
    assert helper._unsloth_fused_ce_available is False
    assert helper._unsloth_fused_ce_unavailable_reason == "runtime_RuntimeError"
    assert helper._unsloth_fused_ce_fallback_reason == "runtime_RuntimeError"
    assert helper._unsloth_fused_ce_runtime_disabled is True
    metrics = helper.runtime_metrics()
    assert metrics["local_train_unsloth_fused_ce_attempts"] == 1
    assert metrics["local_train_unsloth_fused_ce_batches"] == 0


def test_sft_sequence_train_step_auto_retries_after_fused_ce_runtime_failure(
    monkeypatch,
) -> None:
    helper = _helper(_ScalarLM())
    helper.train_unsloth_fused_ce = "auto"
    helper.train_microbatch_size = 1
    helper.train_sft_microbatch_token_budget = 0
    helper._unsloth_fused_ce_fallback_reason = ""
    helper._unsloth_fused_ce_unavailable_reason = ""
    helper._unsloth_fused_ce_available = None
    helper._unsloth_fused_ce_runtime_disabled = False
    calls = 0

    def fake_compute_sft_loss(input_ids, advantages, attention_mask):
        _ = input_ids, advantages, attention_mask
        nonlocal calls
        calls += 1
        if calls == 1:
            helper._unsloth_fused_ce_attempts = 1
            local_metrics.record_loss_path(helper, "unsloth_fused_ce")
            raise RuntimeError("Hard failure due to fullgraph=True")
        token_count = (advantages[:, 1:] > 0).sum().clamp(min=1)
        return helper.train_model.bias * 0.0 + torch.tensor(0.5), token_count

    monkeypatch.setattr(helper, "_compute_sft_loss", fake_compute_sft_loss)

    loss = helper._do_sft_sequence_impl(
        all_tokens=[[1, 2, 3], [4, 5]],
        all_advantages=[[0.0, 1.0, 1.0], [0.0, 1.0]],
    )

    assert calls == 3
    assert loss == pytest.approx(0.5)
    assert helper._unsloth_fused_ce_available is False
    assert helper._unsloth_fused_ce_unavailable_reason == "runtime_RuntimeError"
    assert helper._unsloth_fused_ce_fallback_reason == "runtime_RuntimeError"
    assert helper._unsloth_fused_ce_runtime_disabled is True
    metrics = helper.runtime_metrics()
    assert metrics["local_train_microbatches"] == 2
    assert metrics["local_train_unsloth_fused_ce_attempts"] == 1
    assert metrics["local_train_unsloth_fused_ce_batches"] == 0


def test_sft_train_step_auto_does_not_retry_unrelated_runtime_failure(monkeypatch) -> None:
    helper = _helper(_ScalarLM())
    helper.train_unsloth_fused_ce = "auto"
    helper._unsloth_fused_ce_runtime_disabled = False
    calls = 0

    def fake_compute_sft_loss(input_ids, advantages, attention_mask):
        _ = input_ids, advantages, attention_mask
        nonlocal calls
        calls += 1
        raise RuntimeError("unrelated train-step failure")

    monkeypatch.setattr(helper, "_compute_sft_loss", fake_compute_sft_loss)

    with pytest.raises(RuntimeError, match="unrelated train-step failure"):
        helper._do_sft_impl(
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32),
            torch.ones((1, 3), dtype=torch.bool),
        )

    assert calls == 1
    assert helper._unsloth_fused_ce_runtime_disabled is False


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

    def shifted_logprobs(*args, **kwargs):
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

    def shifted_logprobs(*args, **kwargs):
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
