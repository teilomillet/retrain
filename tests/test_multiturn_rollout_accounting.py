"""Focused accounting checks for multi-turn token-limit telemetry."""

from __future__ import annotations

from typing import cast

from retrain.backends import TrainHelper
from retrain.config import TrainConfig
from retrain.environments.rollout import VerifiersRolloutTiming, VerifiersTurnSample
from retrain.io.log import JsonlLogger
from retrain.training.prompts import PromptBatch
from retrain.training.rollout.multi import run_multiturn
from retrain.training.rollout.state import RolloutAccumulator
from retrain.training.rollouts import TokenTextLookup


def _turn(token_ids: list[int], *, truncated: bool = False) -> VerifiersTurnSample:
    return VerifiersTurnSample(
        prompt_ids=[10],
        completion_ids=token_ids,
        completion_logprobs=[-0.1] * len(token_ids),
        completion_text="action",
        finish_reason="length" if truncated else "stop",
        is_truncated=truncated,
        truncation_reason="action_token_limit" if truncated else None,
    )


def test_multiturn_cap_rate_counts_actions_not_episode_token_sum(tmp_path) -> None:
    def group_runner(*args, **kwargs):
        del args, kwargs
        turns = [
            [_turn([1]), _turn([2]), _turn([3])],
            [_turn([4, 5], truncated=True)],
        ]
        return (
            [1.0, 0.0],
            turns,
            ["good", "truncated"],
            [[], []],
            [[], []],
            [[], []],
            [],
            VerifiersRolloutTiming(model_tokens=5, turns=4),
        )

    config = TrainConfig(model="dummy", max_tokens=2)
    prompts = PromptBatch(
        objs=[[{"role": "user", "content": "task"}]],
        previews=["task"],
        ids=[[10]],
        answers=["answer"],
        tasks=["task"],
        infos=[{}],
    )
    acc = RolloutAccumulator()
    generations_logger = JsonlLogger(
        str(tmp_path / "generations.jsonl"),
        enabled=False,
    )

    run_multiturn(
        config,
        cast(TrainHelper, object()),
        object(),
        object(),
        prompts,
        acc,
        step=0,
        group_size=2,
        sepa_lambda=0.0,
        algorithm_params={},
        transform_params={},
        needs_planning=False,
        detector=None,
        token_lookup=TokenTextLookup(object()),
        generations_logger=generations_logger,
        group_runner=group_runner,
    )

    assert acc.total_completions == 4
    assert acc.max_token_hits == 1
    assert acc.sampled_completion_token_count == 5
    assert acc.eligible_completion_token_count == 3
    assert all(token not in {4, 5} for row in acc.datum_tokens for token in row)


def test_truncated_action_is_excluded_while_prior_echo_observation_survives(
    tmp_path,
) -> None:
    valid_turn = VerifiersTurnSample(
        prompt_ids=[10],
        completion_ids=[1],
        completion_logprobs=[-0.1],
        completion_text="valid action",
        finish_reason="stop",
        echo_observation_capture_supported=True,
        post_observation_ids=[10, 1, 20],
        post_observation_mask=[0, 0, 1],
        post_observation_seen=True,
    )

    def group_runner(*args, **kwargs):
        del args, kwargs
        return (
            [0.0],
            [[valid_turn, _turn([4, 5], truncated=True)]],
            ["truncated"],
            [[]],
            [[]],
            [[]],
            [],
            VerifiersRolloutTiming(model_tokens=3, turns=2),
        )

    config = TrainConfig(
        model="dummy",
        max_tokens=2,
        echo_enabled=True,
        echo_weight=0.05,
        echo_target_retention="all",
        echo_entropy_floor=0.0,
    )
    prompts = PromptBatch(
        objs=[[{"role": "user", "content": "task"}]],
        previews=["task"],
        ids=[[10]],
        answers=["answer"],
        tasks=["task"],
        infos=[{}],
    )
    acc = RolloutAccumulator()
    generations_logger = JsonlLogger(
        str(tmp_path / "generations.jsonl"),
        enabled=False,
    )

    run_multiturn(
        config,
        cast(TrainHelper, object()),
        object(),
        object(),
        prompts,
        acc,
        step=0,
        group_size=1,
        sepa_lambda=0.0,
        algorithm_params={},
        transform_params={},
        needs_planning=False,
        detector=None,
        token_lookup=TokenTextLookup(object()),
        generations_logger=generations_logger,
        group_runner=group_runner,
    )

    assert acc.max_token_hits == 1
    assert acc.sampled_completion_token_count == 3
    assert acc.echo_build.candidate_tokens == 1
    assert any(20 in row for row in acc.datum_tokens)
    assert all(token not in {4, 5} for row in acc.datum_tokens for token in row)


def test_truncated_rollout_cannot_manufacture_grpo_reward_variance(tmp_path) -> None:
    completed_turn = VerifiersTurnSample(
        prompt_ids=[10],
        completion_ids=[1],
        completion_logprobs=[-0.1],
        completion_text="submit",
        finish_reason="stop",
        echo_observation_capture_supported=True,
        post_observation_ids=[10, 1, 20],
        post_observation_mask=[0, 0, 1],
        post_observation_seen=True,
    )
    prior_truncated_turn = VerifiersTurnSample(
        prompt_ids=[10],
        completion_ids=[2],
        completion_logprobs=[-0.2],
        completion_text="inspect",
        finish_reason="stop",
        echo_observation_capture_supported=True,
        post_observation_ids=[10, 2, 30],
        post_observation_mask=[0, 0, 1],
        post_observation_seen=True,
    )

    def group_runner(*args, **kwargs):
        del args, kwargs
        return (
            [1.0, 0.0],
            [
                [completed_turn],
                [prior_truncated_turn, _turn([4, 5], truncated=True)],
            ],
            ["submitted", "truncated"],
            [[], []],
            [[], []],
            [[], []],
            [],
            VerifiersRolloutTiming(model_tokens=4, turns=3),
        )

    config = TrainConfig(
        model="dummy",
        max_tokens=2,
        echo_enabled=True,
        echo_weight=0.05,
        echo_target_retention="all",
        echo_entropy_floor=0.0,
    )
    prompts = PromptBatch(
        objs=[[{"role": "user", "content": "task"}]],
        previews=["task"],
        ids=[[10]],
        answers=["answer"],
        tasks=["task"],
        infos=[{}],
    )
    acc = RolloutAccumulator()
    generations_logger = JsonlLogger(
        str(tmp_path / "generations.jsonl"),
        enabled=False,
    )

    run_multiturn(
        config,
        cast(TrainHelper, object()),
        object(),
        object(),
        prompts,
        acc,
        step=0,
        group_size=2,
        sepa_lambda=0.0,
        algorithm_params={},
        transform_params={},
        needs_planning=False,
        detector=None,
        token_lookup=TokenTextLookup(object()),
        generations_logger=generations_logger,
        group_runner=group_runner,
    )

    # Only the completed rollout enters reward centering. Its singleton GRPO
    # advantage is zero, so the cap event cannot create a positive update.
    assert acc.rewards == [1.0, 0.0]
    assert acc.eligible_completion_token_count == 1
    assert acc.rl_completion_token_count == 1
    assert acc.pre_optimizer_nonzero_advantage_token_count == 0
    assert all(advantage == 0.0 for row in acc.datum_advantages for advantage in row)

    # Both completed action-to-observation transitions remain ECHO targets.
    assert acc.echo_build.executed_transition_datums == 2
    assert acc.echo_build.candidate_tokens == 2
    assert any(20 in row for row in acc.datum_tokens)
    assert any(30 in row for row in acc.datum_tokens)
    assert all(token not in {4, 5} for row in acc.datum_tokens for token in row)
