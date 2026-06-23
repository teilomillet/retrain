from __future__ import annotations

from types import SimpleNamespace

from retrain.echo import (
    build_rollout_echo_datum,
    common_prefix_len,
    limit_echo_masks,
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
