from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from retrain.backends.local.steps import planning
from retrain.training.batch_digest import (
    local_rl_effective_rows_sha256,
    local_sft_effective_rows_sha256,
)


def _config(**overrides) -> planning.StepConfig:
    values = {
        "train_microbatch_size": 0,
        "train_sft_microbatch_token_budget": 0,
        "train_supervised_context_tokens": 0,
        "train_selective_suffix_logits": False,
        "split_mode": False,
    }
    values.update(overrides)
    return planning.config_from_values(**values)


def test_step_config_is_normalized_and_immutable() -> None:
    config = planning.config_from_values(
        train_microbatch_size=-2,
        train_sft_microbatch_token_budget=-3,
        train_supervised_context_tokens=-4,
        train_selective_suffix_logits=1,
        split_mode=1,
    )

    assert config == planning.StepConfig(
        train_microbatch_size=0,
        train_sft_microbatch_token_budget=0,
        train_supervised_context_tokens=0,
        train_selective_suffix_logits=True,
        split_mode=True,
    )
    with pytest.raises(FrozenInstanceError):
        config.split_mode = False  # type: ignore[misc]


def test_empty_rl_plan_has_no_effective_rows_or_execution_work() -> None:
    plan = planning.plan_rl_step(
        _config(split_mode=True),
        [],
        [],
        [],
        learning_rate=3e-5,
        weight_decay=0.1,
    )

    assert plan.execution == "empty"
    assert plan.rows == planning.RlRows((), (), ())
    assert plan.effective_rows_sha256 == ""
    assert plan.context_crop_metrics == ()
    assert plan.snapshot_for_async is False
    assert plan.optimizer == planning.OptimizerOptions(3e-5, 0.1)


def test_rl_plan_crops_rows_hashes_effective_values_and_selects_async_sequence() -> (
    None
):
    tokens = [[10, 11, 12, 13], [20, 21, 22]]
    logprobs = [[0.0, 0.0, -0.2, -0.3], [0.0, 0.0, -0.4]]
    advantages = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, -1.0]]

    plan = planning.plan_rl_step(
        _config(
            train_microbatch_size=1,
            train_supervised_context_tokens=1,
            train_selective_suffix_logits=True,
            split_mode=True,
        ),
        tokens,
        logprobs,
        advantages,
        learning_rate=2e-5,
        weight_decay=0.2,
    )

    assert plan.execution == "sequence"
    assert plan.snapshot_for_async is True
    assert plan.rows == planning.RlRows(
        tokens=((11, 12, 13), (21, 22)),
        logprobs=((0.0, -0.2, -0.3), (0.0, -0.4)),
        advantages=((0.0, 1.0, 1.0), (0.0, -1.0)),
    )
    assert plan.effective_rows_sha256 == local_rl_effective_rows_sha256(
        plan.rows.tokens,
        plan.rows.logprobs,
        plan.rows.advantages,
    )
    assert dict(plan.context_crop_metrics)["local_train_context_rows_cropped"] == 2
    assert tokens == [[10, 11, 12, 13], [20, 21, 22]]


def test_sft_plan_exposes_token_budget_routing_and_digest() -> None:
    plan = planning.plan_sft_step(
        _config(train_sft_microbatch_token_budget=8),
        [[1, 2, 3], [4, 5]],
        [[0.0, 1.0, 1.0], [0.0, 0.5]],
        learning_rate=1e-5,
        weight_decay=0.0,
    )

    assert plan.execution == "sequence"
    assert plan.rows == planning.SftRows(
        tokens=((1, 2, 3), (4, 5)),
        advantages=((0.0, 1.0, 1.0), (0.0, 0.5)),
    )
    assert plan.effective_rows_sha256 == local_sft_effective_rows_sha256(
        plan.rows.tokens,
        plan.rows.advantages,
    )


def test_echo_plan_validates_contract_and_makes_denominator_explicit() -> None:
    with pytest.raises(ValueError, match="paper-faithful ECHO"):
        planning.plan_echo_step(
            _config(),
            [[1, 2]],
            [[0.0, -0.1]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [1],
            echo_loss_fn="importance_sampling",
            echo_rollout_denominator=0,
            learning_rate=1e-5,
            weight_decay=0.0,
        )

    plan = planning.plan_echo_step(
        _config(),
        [[1, 2], [3, 4]],
        [[0.0, -0.1], [0.0, -0.2]],
        [[0.0, 1.0], [0.0, -1.0]],
        [[0.0, 0.5], [0.0, 0.25]],
        [2, 1],
        echo_loss_fn="cross_entropy",
        echo_rollout_denominator=0,
        learning_rate=1e-5,
        weight_decay=0.0,
    )

    assert plan.execution == "padded"
    assert plan.rows.rollout_denominator == 2
    assert plan.effective_rows_sha256 == local_rl_effective_rows_sha256(
        plan.rows.tokens,
        plan.rows.logprobs,
        plan.rows.advantages,
        echo_observation_masks=plan.rows.echo_advantages,
        echo_full_observation_counts=plan.rows.full_observation_counts,
        echo_rollout_denominator=2,
    )


def test_sft_mode_and_zero_logprobs_are_pure_value_transforms() -> None:
    assert planning.validate_sft_loss_fn("cross_entropy") == "cross_entropy"
    assert planning.zero_logprob_rows([[1, 2], [3]]) == ((0.0, 0.0), (0.0,))
    with pytest.raises(ValueError, match="sft_loss_fn"):
        planning.validate_sft_loss_fn("unknown")
