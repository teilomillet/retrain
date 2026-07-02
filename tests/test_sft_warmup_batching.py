from __future__ import annotations

import pytest

from retrain.sft import build_sft_example_order as build_tokenized_sft_order
from retrain.trainer import _build_sft_example_order, _select_sft_batch_indices


def test_build_sft_example_order_is_deterministic_permutation() -> None:
    order_a = _build_sft_example_order(12, 42)
    order_b = _build_sft_example_order(12, 42)

    assert order_a == order_b
    assert sorted(order_a) == list(range(12))


def test_select_sft_batch_indices_covers_shuffled_set_before_wrap() -> None:
    order = _build_sft_example_order(20, 7)

    seen: set[int] = set()
    for step in range(4):
        batch = _select_sft_batch_indices(order, batch_size=5, step=step)
        assert len(batch) == 5
        assert len(set(batch)) == 5
        seen.update(batch)

    assert seen == set(range(20))


def test_select_sft_batch_indices_wraps_after_full_cycle() -> None:
    order = _build_sft_example_order(6, 3)

    first = _select_sft_batch_indices(order, batch_size=4, step=0)
    second = _select_sft_batch_indices(order, batch_size=4, step=1)

    assert len(first) == 4
    assert len(second) == 4
    assert second[:2] == order[4:]
    assert second[2:] == order[:2]


def test_length_order_uses_true_token_lengths_after_shuffle() -> None:
    lengths = [30, 10, 20, 40]
    order = build_tokenized_sft_order(
        len(lengths),
        11,
        lengths=lengths,
        batch_order="length",
    )

    assert order == [1, 2, 0, 3]


def test_length_bucket_sorts_only_inside_shuffled_windows() -> None:
    lengths = [100, 10, 90, 20, 80, 30, 70, 40]
    shuffled = build_tokenized_sft_order(len(lengths), 5)
    bucketed = build_tokenized_sft_order(
        len(lengths),
        5,
        lengths=lengths,
        batch_order="length_bucket",
        length_bucket_size=4,
    )

    expected = []
    for start in range(0, len(shuffled), 4):
        expected.extend(sorted(shuffled[start : start + 4], key=lambda idx: (lengths[idx], idx)))

    assert bucketed == expected
    assert sorted(bucketed) == list(range(len(lengths)))


def test_length_order_requires_lengths() -> None:
    with pytest.raises(ValueError, match="requires one token length"):
        build_tokenized_sft_order(3, 1, batch_order="length_bucket")
