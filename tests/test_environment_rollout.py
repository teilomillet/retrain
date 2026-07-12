"""Tests for multi-turn rollout support."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from retrain.backends import TrainHelper
from retrain.environments.rollout import (
    RolloutScheduler,
    VerifiersRolloutTiming,
    rollout_temperatures,
    sample_active_rollouts,
)
from retrain.environments.verifiers import VerifiersRolloutTiming as BridgeTiming


class _RecordingHelper:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def sample(self, prompt_ids_batch, num_samples, max_tokens, temperature, top_p):
        self.calls.append(
            {
                "prompt_ids_batch": [list(row) for row in prompt_ids_batch],
                "num_samples": num_samples,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return [
            [
                (
                    [100 + (sample_idx if num_samples > 1 else prompt_idx)],
                    [-0.1],
                )
                for sample_idx in range(num_samples)
            ]
            for prompt_idx, _row in enumerate(prompt_ids_batch)
        ]


def _helper() -> tuple[TrainHelper, _RecordingHelper]:
    helper = _RecordingHelper()
    return cast(TrainHelper, helper), helper


def test_bridge_reexports_rollout_timing() -> None:
    assert BridgeTiming is VerifiersRolloutTiming


def test_rollout_temperatures_apply_stable_spread() -> None:
    assert rollout_temperatures(
        temperature=1.0,
        temperature_spread=0.3,
        num_rollouts=3,
    ) == pytest.approx([0.7, 1.0, 1.3])
    assert rollout_temperatures(
        temperature=0.2,
        temperature_spread=0.5,
        num_rollouts=3,
    ) == pytest.approx([0.1, 0.2, 0.7])
    assert rollout_temperatures(
        temperature=0.8,
        temperature_spread=0.0,
        num_rollouts=3,
    ) == [0.8, 0.8, 0.8]


def test_sample_active_rollouts_batches_equal_temperatures() -> None:
    helper, recorder = _helper()

    groups = sample_active_rollouts(
        helper=helper,
        active=[
            (0, "a", [1], None),
            (1, "b", [2], None),
            (2, "c", [3], None),
        ],
        rollout_temps=[0.5, 0.5, 0.8],
        max_tokens=4,
        top_p=0.9,
    )

    assert [[sample.token_ids for sample in group] for group in groups] == [
        [[100]],
        [[101]],
        [[100]],
    ]
    assert [call["temperature"] for call in recorder.calls] == [0.5, 0.8]
    assert [call["prompt_ids_batch"] for call in recorder.calls] == [[[1], [2]], [[3]]]


def test_sample_active_rollouts_collapses_identical_prompts_to_num_samples() -> None:
    helper, recorder = _helper()

    groups = sample_active_rollouts(
        helper=helper,
        active=[
            (0, "a", [1, 2], None),
            (1, "a", [1, 2], None),
            (2, "a", [1, 2], None),
        ],
        rollout_temps=[0.7, 0.7, 0.7],
        max_tokens=4,
        top_p=0.9,
    )

    assert [[sample.token_ids for sample in group] for group in groups] == [
        [[100]],
        [[101]],
        [[102]],
    ]
    assert recorder.calls == [
        {
            "prompt_ids_batch": [[1, 2]],
            "num_samples": 3,
            "max_tokens": 4,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    ]


def test_sample_active_rollouts_uses_exact_finish_reason_before_length_fallback() -> (
    None
):
    class FinishReasonHelper:
        def sample_with_finish_reason(
            self,
            prompt_ids_batch,
            num_samples,
            max_tokens,
            temperature,
            top_p,
        ):
            del prompt_ids_batch, num_samples, temperature, top_p
            return [
                [([1] * max_tokens, [-0.1] * max_tokens, "stop")],
                [([2] * max_tokens, [-0.2] * max_tokens, "length")],
            ]

        def sample(self, *args, **kwargs):
            raise AssertionError("metadata sampling must be preferred")

    groups = sample_active_rollouts(
        helper=cast(TrainHelper, FinishReasonHelper()),
        active=[
            (0, "a", [1], None),
            (1, "b", [2], None),
        ],
        rollout_temps=[0.5, 0.5],
        max_tokens=2,
        top_p=0.9,
    )

    assert groups[0][0].finish_reason == "stop"
    assert groups[0][0].hit_token_limit is False
    assert groups[1][0].finish_reason == "length"
    assert groups[1][0].hit_token_limit is True


def test_sample_active_rollouts_fails_closed_on_non_stop_finish_reason() -> None:
    class FilteredHelper:
        def sample_with_finish_reason(
            self,
            prompt_ids_batch,
            num_samples,
            max_tokens,
            temperature,
            top_p,
        ):
            del prompt_ids_batch, num_samples, max_tokens, temperature, top_p
            return [[([1], [-0.1], "content_filter")]]

        def sample(self, *args, **kwargs):
            raise AssertionError("metadata sampling must be preferred")

    groups = sample_active_rollouts(
        helper=cast(TrainHelper, FilteredHelper()),
        active=[(0, "a", [1], None)],
        rollout_temps=[0.5],
        max_tokens=2,
        top_p=0.9,
    )

    assert groups[0][0].finish_reason == "content_filter"
    assert groups[0][0].hit_token_limit is True


@pytest.mark.parametrize("finish_reason", [None, 7])
def test_metadata_sampler_fails_closed_without_valid_finish_reason(
    finish_reason: object,
) -> None:
    class MissingFinishReasonHelper:
        def sample_with_finish_reason(
            self,
            prompt_ids_batch,
            num_samples,
            max_tokens,
            temperature,
            top_p,
        ):
            del prompt_ids_batch, num_samples, max_tokens, temperature, top_p
            if finish_reason is None:
                return [[([1], [-0.1])]]
            return [[([1], [-0.1], finish_reason)]]

        def sample(self, *args, **kwargs):
            raise AssertionError("metadata sampling must be preferred")

    groups = sample_active_rollouts(
        helper=cast(TrainHelper, MissingFinishReasonHelper()),
        active=[(0, "a", [1], None)],
        rollout_temps=[0.5],
        max_tokens=2,
        top_p=0.9,
    )

    assert groups[0][0].finish_reason is None
    assert groups[0][0].hit_token_limit is True


def test_legacy_sampler_retains_token_count_fallback() -> None:
    helper, _recorder = _helper()

    below_cap = sample_active_rollouts(
        helper=helper,
        active=[(0, "a", [1], None)],
        rollout_temps=[0.5],
        max_tokens=2,
        top_p=0.9,
    )
    at_cap = sample_active_rollouts(
        helper=helper,
        active=[(0, "a", [1], None)],
        rollout_temps=[0.5],
        max_tokens=1,
        top_p=0.9,
    )

    assert below_cap[0][0].hit_token_limit is False
    assert at_cap[0][0].hit_token_limit is True


def test_sample_active_rollouts_rejects_stop_past_cap() -> None:
    class OversizedHelper:
        def sample_with_finish_reason(
            self,
            prompt_ids_batch,
            num_samples,
            max_tokens,
            temperature,
            top_p,
        ):
            del prompt_ids_batch, num_samples, temperature, top_p
            return [[([1] * (max_tokens + 1), [-0.1] * (max_tokens + 1), "stop")]]

        def sample(self, *args, **kwargs):
            raise AssertionError("metadata sampling must be preferred")

    groups = sample_active_rollouts(
        helper=cast(TrainHelper, OversizedHelper()),
        active=[(0, "a", [1], None)],
        rollout_temps=[0.5],
        max_tokens=2,
        top_p=0.9,
    )

    assert groups[0][0].finish_reason == "stop"
    assert groups[0][0].hit_token_limit is True


def test_sample_active_rollouts_rejects_token_logprob_length_mismatch() -> None:
    class MismatchedHelper:
        def sample_with_finish_reason(
            self,
            prompt_ids_batch,
            num_samples,
            max_tokens,
            temperature,
            top_p,
        ):
            del prompt_ids_batch, num_samples, max_tokens, temperature, top_p
            return [[([1, 2], [-0.1], "stop")]]

        def sample(self, *args, **kwargs):
            raise AssertionError("metadata sampling must be preferred")

    with pytest.raises(ValueError, match="identical lengths"):
        sample_active_rollouts(
            helper=cast(TrainHelper, MismatchedHelper()),
            active=[(0, "a", [1], None)],
            rollout_temps=[0.5],
            max_tokens=2,
            top_p=0.9,
        )


@pytest.mark.parametrize("logprob", [float("nan"), float("inf"), float("-inf")])
def test_sample_active_rollouts_rejects_non_finite_logprobs(logprob: float) -> None:
    class NonFiniteHelper:
        def sample_with_finish_reason(
            self,
            prompt_ids_batch,
            num_samples,
            max_tokens,
            temperature,
            top_p,
        ):
            del prompt_ids_batch, num_samples, max_tokens, temperature, top_p
            return [[([1], [logprob], "stop")]]

        def sample(self, *args, **kwargs):
            raise AssertionError("metadata sampling must be preferred")

    with pytest.raises(ValueError, match="logprobs must be finite"):
        sample_active_rollouts(
            helper=cast(TrainHelper, NonFiniteHelper()),
            active=[(0, "a", [1], None)],
            rollout_temps=[0.5],
            max_tokens=2,
            top_p=0.9,
        )


def test_rollout_scheduler_preserves_order_with_bounded_workers() -> None:
    timing = VerifiersRolloutTiming()
    scheduler = RolloutScheduler(max_env_workers=2, max_buffered_rollouts=2)

    async def worker(raw: object) -> object:
        value = int(cast(int, raw))
        await asyncio.sleep(0)
        return value * 10

    result = asyncio.run(scheduler.map_ordered([3, 1, 2], worker, timing))

    assert result == [30, 10, 20]
    assert scheduler.max_env_workers == 2
    assert scheduler.max_buffered_rollouts == 2
    assert timing.scheduler_worker_s >= 0.0
