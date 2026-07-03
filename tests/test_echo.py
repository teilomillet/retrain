from __future__ import annotations

from types import SimpleNamespace

import pytest

from retrain.training.echo import (
    EchoBuildStats,
    build_rollout_echo_datum,
    common_prefix_len,
    limit_echo_masks,
    merge_echo_build_stats,
    run_rl_echo_train_step,
)


def test_common_prefix_len_stops_at_first_mismatch() -> None:
    assert common_prefix_len([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix_len([1, 2], [1, 2, 3]) == 2


def test_build_rollout_echo_datum_interleaves_actions_and_observations() -> None:
    turns = [
        SimpleNamespace(
            prompt_ids=[1, 2],
            completion_ids=[3],
            completion_logprobs=[-0.1],
        ),
        SimpleNamespace(
            prompt_ids=[1, 2, 3, 50, 51, 99],
            completion_ids=[4, 5],
            completion_logprobs=[-0.2, -0.3],
            observation_mask=[0, 0, 0, 1, 1, 0],
        ),
    ]

    datum, stats = build_rollout_echo_datum(
        turns,
        completion_advantages=[[0.7], [-0.2, 0.4]],
        weight=0.05,
        min_prompt_overlap=1.0,
    )

    assert datum is not None
    assert datum.tokens == [1, 2, 3, 50, 51, 99, 4, 5]
    assert datum.logprobs == [0.0, 0.0, -0.1, 0.0, 0.0, 0.0, -0.2, -0.3]
    assert datum.advantages == [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, -0.2, 0.4]
    assert datum.echo_advantages == [0.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0]
    assert datum.full_observation_count == 2
    assert datum.positive_tokens == 2
    assert stats.candidate_datums == 1
    assert stats.observation_mask_datums == 1
    assert stats.skipped_first_turns == 1


def test_build_rollout_echo_datum_rejects_unstable_prompt_stitching() -> None:
    turns = [
        SimpleNamespace(
            prompt_ids=[1, 2],
            completion_ids=[3],
            completion_logprobs=[-0.1],
        ),
        SimpleNamespace(
            prompt_ids=[9, 8, 7],
            completion_ids=[4],
            completion_logprobs=[-0.2],
        ),
    ]

    datum, stats = build_rollout_echo_datum(
        turns,
        completion_advantages=[[0.1], [0.2]],
        weight=0.05,
        min_prompt_overlap=0.5,
    )

    assert datum is None
    assert stats.skipped_low_overlap == 1


def test_build_rollout_echo_datum_falls_back_to_prompt_suffix() -> None:
    turns = [
        SimpleNamespace(
            prompt_ids=[1, 2],
            completion_ids=[3],
            completion_logprobs=[-0.1],
        ),
        SimpleNamespace(
            prompt_ids=[1, 2, 3, 50, 51],
            completion_ids=[4],
            completion_logprobs=[-0.2],
        ),
    ]

    datum, stats = build_rollout_echo_datum(
        turns,
        completion_advantages=[[0.7], [0.4]],
        weight=0.05,
        min_prompt_overlap=1.0,
    )

    assert datum is not None
    assert datum.tokens == [1, 2, 3, 50, 51, 4]
    assert datum.echo_advantages == [0.0, 0.0, 0.0, 0.05, 0.05, 0.0]
    assert datum.full_observation_count == 2
    assert datum.positive_tokens == 2
    assert stats.candidate_datums == 1
    assert stats.observation_mask_datums == 0


def test_build_rollout_echo_datum_skips_bad_observation_mask() -> None:
    turns = [
        SimpleNamespace(
            prompt_ids=[1, 2],
            completion_ids=[3],
            completion_logprobs=[-0.1],
        ),
        SimpleNamespace(
            prompt_ids=[1, 2, 3, 50],
            completion_ids=[4],
            completion_logprobs=[-0.2],
            observation_mask=[0, 1],
        ),
    ]

    datum, stats = build_rollout_echo_datum(
        turns,
        completion_advantages=[[0.7], [0.4]],
        weight=0.05,
        min_prompt_overlap=1.0,
    )

    assert datum is not None
    assert datum.echo_advantages == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert datum.full_observation_count == 0
    assert datum.positive_tokens == 0
    assert stats.candidate_datums == 0
    assert stats.skipped_bad_observation_mask == 1


def test_limit_echo_masks_caps_without_truncating_rollout_rows() -> None:
    rows = [
        [0.0, 0.2, 0.2, 0.0],
        [0.1, 0.0, 0.1],
    ]

    limited, stats = limit_echo_masks(rows, max_positive_tokens=3)

    assert limited == [
        [0.0, 0.2, 0.2, 0.0],
        [0.1, 0.0, 0.0],
    ]
    assert [len(row) for row in limited] == [4, 3]
    assert stats.kept_datums == 2
    assert stats.kept_tokens == 3
    assert stats.truncated_tokens == 1


def test_merge_echo_build_stats_adds_all_counters() -> None:
    left = EchoBuildStats(
        candidate_datums=1,
        candidate_tokens=2,
        observation_mask_datums=3,
        skipped_first_turns=4,
        skipped_no_suffix=5,
        skipped_low_overlap=6,
        skipped_bad_observation_mask=7,
    )
    right = EchoBuildStats(
        candidate_datums=8,
        candidate_tokens=9,
        observation_mask_datums=10,
        skipped_first_turns=11,
        skipped_no_suffix=12,
        skipped_low_overlap=13,
        skipped_bad_observation_mask=14,
    )

    assert merge_echo_build_stats(left, right) == EchoBuildStats(
        candidate_datums=9,
        candidate_tokens=11,
        observation_mask_datums=13,
        skipped_first_turns=15,
        skipped_no_suffix=17,
        skipped_low_overlap=19,
        skipped_bad_observation_mask=21,
    )


def test_echo_treats_rl_advantages_as_opaque_algorithm_output() -> None:
    class Helper:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def train_step_with_echo_masks(
            self,
            all_tokens,
            all_logprobs,
            all_advantages,
            echo_advantages,
            echo_full_observation_counts,
            echo_loss_fn,
            lr,
            weight_decay,
        ):
            self.calls.append(
                {
                    "tokens": all_tokens,
                    "logprobs": all_logprobs,
                    "advantages": all_advantages,
                    "echo_advantages": echo_advantages,
                    "echo_full_observation_counts": echo_full_observation_counts,
                    "echo_loss_fn": echo_loss_fn,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )
            return 0.25, 0.125

    helper = Helper()
    algorithm_advantages = [[0.7, -0.2, 0.4], [-0.5, -0.5]]

    rl_loss, echo_loss, joint = run_rl_echo_train_step(
        helper,
        all_tokens=[[10, 11, 12], [20, 21]],
        all_logprobs=[[-0.1, -0.2, -0.3], [-0.4, -0.5]],
        all_advantages=algorithm_advantages,
        echo_advantages=[
            [0.0, 0.0, 0.3],
            [0.0, 0.0],
        ],
        echo_full_observation_counts=[1, 0],
        echo_loss_fn="cross_entropy",
        lr=1e-4,
        weight_decay=0.01,
    )

    assert (rl_loss, echo_loss, joint) == (0.25, 0.125, True)
    assert helper.calls[0]["advantages"] is algorithm_advantages
    assert helper.calls[0]["echo_advantages"] == [
        [0.0, 0.0, 0.3],
        [0.0, 0.0],
    ]
    assert helper.calls[0]["echo_full_observation_counts"] == [1, 0]


def test_echo_rejects_helper_without_shared_forward_step() -> None:
    class Helper:
        def train_step(self, *args, **kwargs):
            raise AssertionError("separate RL step must not run")

    with pytest.raises(RuntimeError, match="same rollout rows"):
        run_rl_echo_train_step(
            Helper(),
            all_tokens=[[10, 11]],
            all_logprobs=[[-0.1, -0.2]],
            all_advantages=[[0.5, -0.5]],
            echo_advantages=[[0.0, 0.0, 0.3]],
            echo_full_observation_counts=[1],
            echo_loss_fn="cross_entropy",
            lr=1e-4,
            weight_decay=0.0,
        )
