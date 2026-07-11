from __future__ import annotations

import pytest

from retrain.training.batch_digest import (
    local_rl_effective_rows_sha256,
    local_sft_effective_rows_sha256,
    logical_optimizer_batch_sha256,
)


def _digest(
    *,
    echo: bool = False,
    token: int = 3,
    logprob: float = -0.25,
    advantage: float = 1.0,
) -> str:
    if echo:
        return logical_optimizer_batch_sha256(
            [[2, token]],
            [[0.0, logprob]],
            [[0.0, advantage]],
            echo_observation_masks=[[0.0, 0.5]],
            echo_full_observation_counts=[1],
            echo_rollout_denominator=1,
        )
    return logical_optimizer_batch_sha256(
        [[2, token]],
        [[0.0, logprob]],
        [[0.0, advantage]],
    )


def test_optimizer_batch_digest_is_stable_and_canonical() -> None:
    digest = _digest()

    assert digest == _digest()
    assert digest == "50025d1b4c9727f3b532ef4927128c14a9c7fda340dcb4446909934969d092dd"
    assert len(digest) == 64


@pytest.mark.parametrize(
    "changed",
    [
        _digest(token=4),
        _digest(logprob=-0.25000000000000006),
        _digest(advantage=-1.0),
        _digest(echo=True),
    ],
)
def test_optimizer_batch_digest_changes_with_optimizer_inputs(changed: str) -> None:
    assert changed != _digest()


def test_optimizer_batch_digest_distinguishes_row_order() -> None:
    first = logical_optimizer_batch_sha256(
        [[1], [2]],
        [[-0.1], [-0.2]],
        [[1.0], [-1.0]],
    )
    second = logical_optimizer_batch_sha256(
        [[2], [1]],
        [[-0.2], [-0.1]],
        [[-1.0], [1.0]],
    )

    assert first != second


def test_optimizer_batch_digest_rejects_incomplete_echo_inputs() -> None:
    with pytest.raises(ValueError, match="must be supplied together"):
        logical_optimizer_batch_sha256(
            [[1]],
            [[0.0]],
            [[1.0]],
            echo_observation_masks=[[0.5]],
        )


def test_optimizer_batch_digest_rejects_misaligned_rows() -> None:
    with pytest.raises(ValueError, match=r"old_logprobs\[0\]"):
        logical_optimizer_batch_sha256([[1, 2]], [[0.0]], [[0.0, 1.0]])


def test_local_effective_rows_digest_uses_float32_values() -> None:
    base = local_rl_effective_rows_sha256(
        [[1, 2]],
        [[0.0, -0.25]],
        [[0.0, 1.0]],
    )
    below_float32_resolution = local_rl_effective_rows_sha256(
        [[1, 2]],
        [[0.0, -0.25000000000000006]],
        [[0.0, 1.0]],
    )

    assert base == below_float32_resolution
    assert base != _digest()


def test_local_effective_rows_digest_includes_echo_scaling_inputs() -> None:
    base = local_rl_effective_rows_sha256(
        [[1, 2]],
        [[0.0, -0.25]],
        [[0.0, 1.0]],
        echo_observation_masks=[[0.0, 0.5]],
        echo_full_observation_counts=[1],
        echo_rollout_denominator=1,
    )
    changed_denominator = local_rl_effective_rows_sha256(
        [[1, 2]],
        [[0.0, -0.25]],
        [[0.0, 1.0]],
        echo_observation_masks=[[0.0, 0.5]],
        echo_full_observation_counts=[1],
        echo_rollout_denominator=2,
    )

    assert base != changed_denominator


def test_local_sft_effective_rows_use_distinct_float32_framing() -> None:
    base = local_sft_effective_rows_sha256(
        [[1, 2]],
        [[0.0, 1.0]],
    )
    below_float32_resolution = local_sft_effective_rows_sha256(
        [[1, 2]],
        [[0.0, 1.0000000000000002]],
    )
    rl_rows = local_rl_effective_rows_sha256(
        [[1, 2]],
        [[0.0, 0.0]],
        [[0.0, 1.0]],
    )

    assert base == below_float32_resolution
    assert base != rl_rows
