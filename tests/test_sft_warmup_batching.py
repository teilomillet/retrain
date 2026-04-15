from __future__ import annotations

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
