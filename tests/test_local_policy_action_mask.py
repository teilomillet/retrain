from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from retrain.backends.local.steps import rl as local_rl_step
from retrain.backends.local.steps import shared
from retrain.backends.local.steps import hybrid as local_hybrid_step


class _PolicyLossHelper:
    def __init__(
        self,
        new_logprobs: torch.Tensor,
        *,
        selective: bool,
        policy_loss_mode: str,
    ) -> None:
        self.new_logprobs = new_logprobs
        self.train_selective_suffix_logits = selective
        self.policy_loss_mode = policy_loss_mode
        self.clip_eps = 0.2
        self.clip_eps_high = 0.2
        self.kl_cov_percent = 100.0
        self.kl_cov_coef = 0.5
        self.clip_cov_ratio = 1.0
        self.clip_cov_min = -100.0
        self.clip_cov_max = 100.0
        self.target_masks: list[torch.Tensor | None] = []

    def _autocast_context(self):
        return nullcontext()

    def _shifted_token_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        del input_ids, attention_mask
        self.target_masks.append(target_mask)
        return self.new_logprobs


class _ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))


class _HybridLossHelper(_PolicyLossHelper):
    def __init__(self, *, selective: bool, policy_loss_mode: str) -> None:
        self.train_model = _ScalarModel()
        super().__init__(
            torch.tensor([[0.31, -0.05, -0.27, 0.07, 0.22]]),
            selective=selective,
            policy_loss_mode=policy_loss_mode,
        )
        self.optimizer = torch.optim.SGD(self.train_model.parameters(), lr=0.0)
        self.scaler = torch.amp.GradScaler(enabled=False)
        self.train_device = "cpu"
        self.cuda_empty_cache = False

    def _shifted_token_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        del input_ids, attention_mask
        self.target_masks.append(target_mask)
        return self.new_logprobs + self.train_model.bias


@pytest.mark.parametrize("policy_loss_mode", ["standard", "kl_cov", "clip_cov"])
def test_dense_policy_loss_matches_selective_action_masking(
    policy_loss_mode: str,
) -> None:
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    old_logprobs = torch.zeros_like(input_ids, dtype=torch.float32)
    advantages = torch.tensor(
        [[0.0, 0.0, 2.0, 0.0, -1.0, 0.0]],
        dtype=torch.float32,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    new_values = torch.tensor(
        [[0.31, -0.05, -0.27, 0.07, 0.22]],
        dtype=torch.float32,
    )

    dense = _PolicyLossHelper(
        new_values.clone(),
        selective=False,
        policy_loss_mode=policy_loss_mode,
    )
    selective = _PolicyLossHelper(
        new_values.clone(),
        selective=True,
        policy_loss_mode=policy_loss_mode,
    )
    torch.manual_seed(7)
    dense_result = local_rl_step.compute_loss(
        dense,
        input_ids,
        old_logprobs,
        advantages,
        attention_mask,
    )
    torch.manual_seed(7)
    selective_result = local_rl_step.compute_loss(
        selective,
        input_ids,
        old_logprobs,
        advantages,
        attention_mask,
    )

    assert float(dense_result[0].item()) == pytest.approx(
        float(selective_result[0].item())
    )
    assert float(dense_result[1].item()) == 2.0
    assert float(selective_result[1].item()) == 2.0
    assert dense_result[2:] == pytest.approx(selective_result[2:])
    assert dense.target_masks == [None]
    assert selective.target_masks[0] is not None
    assert selective.target_masks[0].tolist() == [[False, True, False, True, False]]


@pytest.mark.parametrize("policy_loss_mode", ["standard", "kl_cov", "clip_cov"])
def test_dense_policy_loss_has_no_gradient_on_zero_advantage_context(
    policy_loss_mode: str,
) -> None:
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    old_logprobs = torch.zeros_like(input_ids, dtype=torch.float32)
    advantages = torch.tensor(
        [[0.0, 0.0, 2.0, 0.0, -1.0, 0.0]],
        dtype=torch.float32,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    new_values = torch.tensor(
        [[0.31, -0.05, -0.27, 0.07, 0.22]],
        dtype=torch.float32,
        requires_grad=True,
    )
    helper = _PolicyLossHelper(
        new_values,
        selective=False,
        policy_loss_mode=policy_loss_mode,
    )

    torch.manual_seed(7)
    loss, token_count, *_ = local_rl_step.compute_loss(
        helper,
        input_ids,
        old_logprobs,
        advantages,
        attention_mask,
    )
    loss.backward()

    assert float(token_count.item()) == 2.0
    assert new_values.grad is not None
    assert new_values.grad[0, [0, 2, 4]].tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("selective", [False, True])
def test_policy_token_normalization_always_counts_actions(selective: bool) -> None:
    helper = type(
        "_Helper",
        (),
        {"train_selective_suffix_logits": selective},
    )()
    attention_mask = torch.ones((1, 6), dtype=torch.bool)
    advantages = torch.tensor(
        [[0.0, 0.0, 2.0, 0.0, -1.0, 0.0]],
        dtype=torch.float32,
    )

    tensor_count = shared.policy_total_tokens(helper, advantages, attention_mask)
    row_count = shared.policy_total_tokens_from_rows(
        helper,
        [[1, 2, 3, 4, 5, 6]],
        advantages.tolist(),
    )

    assert float(tensor_count.item()) == 2.0
    assert row_count == 2.0


@pytest.mark.parametrize("policy_loss_mode", ["standard", "kl_cov", "clip_cov"])
def test_dense_hybrid_policy_term_matches_selective_action_masking(
    policy_loss_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_hybrid_step.shared,
        "snapshot_and_record",
        lambda *args, **kwargs: None,
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    old_logprobs = torch.zeros_like(input_ids, dtype=torch.float32)
    advantages = torch.tensor(
        [[0.0, 0.0, 2.0, 0.0, -1.0, 0.0]],
        dtype=torch.float32,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    echo_advantages = torch.tensor(
        [[0.0, 0.0, 0.0, 0.2, 0.0, 0.0]],
        dtype=torch.float32,
    )
    echo_counts = torch.tensor([1.0], dtype=torch.float32)

    def run_once(selective: bool) -> tuple[float, float, _HybridLossHelper]:
        helper = _HybridLossHelper(
            selective=selective,
            policy_loss_mode=policy_loss_mode,
        )
        torch.manual_seed(7)
        losses = local_hybrid_step._run_batches(
            helper,
            [
                (
                    input_ids,
                    old_logprobs,
                    advantages,
                    attention_mask,
                    echo_advantages,
                    echo_counts,
                )
            ],
            "cross_entropy",
            batch_size=1,
            total_tokens_value=2.0,
            echo_rollout_denominator=1,
        )
        return *losses, helper

    dense_policy, dense_echo, dense = run_once(False)
    selective_policy, selective_echo, selective = run_once(True)

    assert dense_policy == pytest.approx(selective_policy)
    assert dense_echo == pytest.approx(selective_echo)
    assert dense.target_masks == [None]
    assert selective.target_masks[0] is not None
    assert selective.target_masks[0].tolist() == [[False, True, True, True, False]]
