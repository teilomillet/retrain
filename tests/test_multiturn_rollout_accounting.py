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
