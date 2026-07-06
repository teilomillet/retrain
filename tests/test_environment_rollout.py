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

    assert groups == [[([100], [-0.1])], [([101], [-0.1])], [([100], [-0.1])]]
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

    assert groups == [
        [([100], [-0.1])],
        [([101], [-0.1])],
        [([102], [-0.1])],
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
