from __future__ import annotations

import pytest

from retrain.config import TrainConfig
from retrain.training.sft import build_sft_example_order as build_tokenized_sft_order
from retrain.training.sft import (
    SftDataProvenance,
    build_sft_epoch_order,
    build_sft_example_order,
    build_sft_resume_schedule_contract,
    build_sft_schedule_metrics,
    describe_sft_batch_position,
    select_sft_batch_indices,
    sft_indices_sha256,
    verify_sft_resume_schedule_contract,
)


def test_build_sft_example_order_is_deterministic_permutation() -> None:
    order_a = build_sft_example_order(12, 42)
    order_b = build_sft_example_order(12, 42)

    assert order_a == order_b
    assert sorted(order_a) == list(range(12))


def test_select_sft_batch_indices_covers_shuffled_set_before_wrap() -> None:
    order = build_sft_example_order(20, 7)

    seen: set[int] = set()
    for step in range(4):
        batch = select_sft_batch_indices(order, batch_size=5, step=step)
        assert len(batch) == 5
        assert len(set(batch)) == 5
        seen.update(batch)

    assert seen == set(range(20))


def test_select_sft_batch_indices_wraps_after_full_cycle() -> None:
    order = build_sft_example_order(6, 3)

    first = select_sft_batch_indices(order, batch_size=4, step=0)
    second = select_sft_batch_indices(order, batch_size=4, step=1)

    assert len(first) == 4
    assert len(second) == 4
    assert second[:2] == order[4:]
    assert second[2:] == order[:2]


def test_epoch_reshuffle_uses_absolute_sample_schedule_across_boundary() -> None:
    seed = 17
    order = build_sft_example_order(5, seed)
    next_order = build_sft_example_order(5, seed + 1)

    first = select_sft_batch_indices(
        order,
        batch_size=3,
        step=0,
        seed=seed,
        reshuffle_each_epoch=True,
    )
    crossing = select_sft_batch_indices(
        order,
        batch_size=3,
        step=1,
        seed=seed,
        reshuffle_each_epoch=True,
    )
    after_resume = select_sft_batch_indices(
        order,
        batch_size=3,
        step=2,
        seed=seed,
        reshuffle_each_epoch=True,
    )

    assert first == order[:3]
    assert crossing == order[3:] + next_order[:1]
    assert after_resume == next_order[1:4]

    flattened = [
        index
        for step in range(5)
        for index in select_sft_batch_indices(
            order,
            batch_size=3,
            step=step,
            seed=seed,
            reshuffle_each_epoch=True,
        )
    ]
    assert flattened == (
        order
        + next_order
        + build_sft_example_order(5, seed + 2)
    )


def test_epoch_reshuffle_is_deterministic_when_resuming_at_later_step() -> None:
    seed = 29
    order = build_sft_example_order(7, seed)
    epoch_order_cache = {0: order}
    resumed = select_sft_batch_indices(
        order,
        batch_size=4,
        step=5,
        seed=seed,
        reshuffle_each_epoch=True,
        epoch_order_cache=epoch_order_cache,
    )

    assert resumed == select_sft_batch_indices(
        order,
        batch_size=4,
        step=5,
        seed=seed,
        reshuffle_each_epoch=True,
    )
    assert set(epoch_order_cache) == {0, 2, 3}


def test_schedule_hashes_cover_exact_batch_and_crossed_epoch_orders() -> None:
    seed = 17
    order = build_sft_example_order(5, seed)
    cache = {0: order}
    digest_cache: dict[int, str] = {}
    selected = select_sft_batch_indices(
        order,
        batch_size=3,
        step=1,
        seed=seed,
        reshuffle_each_epoch=True,
        epoch_order_cache=cache,
    )

    metrics = build_sft_schedule_metrics(
        order,
        selected,
        batch_size=3,
        step=1,
        seed=seed,
        reshuffle_each_epoch=True,
        epoch_order_cache=cache,
        epoch_order_sha256_cache=digest_cache,
    )

    assert metrics["sft_batch_indices_sha256"] == sft_indices_sha256(selected)
    assert metrics["sft_epoch_start_order_sha256"] == sft_indices_sha256(order)
    assert metrics["sft_epoch_end_order_sha256"] == sft_indices_sha256(
        build_sft_example_order(5, seed + 1)
    )
    assert set(cache) == {0, 1}
    assert set(digest_cache) == {0, 1}


def test_sft_index_hash_has_stable_uint64_big_endian_encoding() -> None:
    assert sft_indices_sha256([1, 2, 3]) == (
        "ca73761ddabfffcbe51170be0b07f67bafcdbed202545c60707573d36dc935b4"
    )


def test_sft_resume_schedule_contract_binds_traversal_and_data() -> None:
    config = TrainConfig(
        trainer="sft",
        sft_data_path="dataset.jsonl",
        sft_batch_size=4,
        sft_batch_order="length_bucket",
        sft_length_bucket_size=8,
        sft_reshuffle_each_epoch=True,
        sft_max_tokens=512,
        seed=42,
        model="test-model",
    )
    provenance = SftDataProvenance(
        data_path="/tmp/dataset.jsonl",
        data_sha256="a" * 64,
        data_rows=12,
        data_bytes=100,
        data_path_status="scratch",
    )
    order = build_sft_example_order(
        12,
        42,
        lengths=list(range(12)),
        batch_order="length_bucket",
        length_bucket_size=8,
    )

    contract = build_sft_resume_schedule_contract(
        config,
        provenance,
        batch_size=4,
        max_tokens=512,
        example_order=order,
    )

    assert contract["seed"] == 42
    assert contract["batch_size"] == 4
    assert contract["batch_order"] == "length_bucket"
    assert contract["length_bucket_size"] == 8
    assert contract["effective_length_bucket_size"] == 8
    assert contract["reshuffle_each_epoch"] is True
    assert contract["data_sha256"] == "a" * 64
    assert contract["data_rows"] == 12
    assert contract["audit_sha256"] == ""
    assert contract["audit_schema"] == ""
    assert contract["sft_warmup_steps"] == 0
    assert contract["epoch_zero_order_sha256"] == sft_indices_sha256(order)
    verify_sft_resume_schedule_contract(contract, contract)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("seed", 43),
        ("batch_size", 2),
        ("batch_order", "shuffle"),
        ("length_bucket_size", 16),
        ("reshuffle_each_epoch", False),
        ("data_sha256", "b" * 64),
        ("data_rows", 13),
        ("audit_sha256", "e" * 64),
        ("sft_warmup_steps", 3),
        ("max_tokens", 256),
        ("epoch_zero_order_sha256", "c" * 64),
    ],
)
def test_sft_resume_schedule_contract_rejects_changed_semantics(
    field: str,
    changed: object,
) -> None:
    current: dict[str, object] = {
        "version": 2,
        "seed": 42,
        "batch_size": 4,
        "batch_order": "length_bucket",
        "length_bucket_size": 8,
        "reshuffle_each_epoch": True,
        "data_sha256": "a" * 64,
        "data_rows": 12,
        "audit_sha256": "",
        "audit_schema": "",
        "sft_warmup_steps": 0,
        "max_tokens": 512,
        "epoch_zero_order_sha256": "d" * 64,
    }
    saved = dict(current)
    saved[field] = changed

    with pytest.raises(ValueError, match=field):
        verify_sft_resume_schedule_contract(saved, current)


def test_sft_resume_schedule_contract_fails_closed_when_missing() -> None:
    with pytest.raises(ValueError, match="missing from trainer_state.json"):
        verify_sft_resume_schedule_contract(None, {"version": 2})


def test_fullhist_609_by_7_epochs_are_distinct_full_permutations() -> None:
    example_count = 609
    batch_size = 7
    steps_per_epoch = example_count // batch_size
    seed = 42
    order = build_sft_example_order(example_count, seed)
    cache = {0: order}

    epochs = []
    for epoch in range(2):
        selected = [
            index
            for step in range(
                epoch * steps_per_epoch,
                (epoch + 1) * steps_per_epoch,
            )
            for index in select_sft_batch_indices(
                order,
                batch_size=batch_size,
                step=step,
                seed=seed,
                reshuffle_each_epoch=True,
                epoch_order_cache=cache,
            )
        ]
        epochs.append(selected)

    assert sorted(epochs[0]) == list(range(example_count))
    assert sorted(epochs[1]) == list(range(example_count))
    assert epochs[0] != epochs[1]
    assert epochs[0] == build_sft_epoch_order(
        order,
        epoch=0,
        seed=seed,
        reshuffle_each_epoch=True,
    )
    assert epochs[1] == build_sft_epoch_order(
        order,
        epoch=1,
        seed=seed,
        reshuffle_each_epoch=True,
    )


def test_length_bucket_order_is_rebuilt_for_each_epoch() -> None:
    lengths = [12, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6]
    seed = 5
    order = build_sft_example_order(
        len(lengths),
        seed,
        lengths=lengths,
        batch_order="length_bucket",
        length_bucket_size=4,
    )
    cache = {0: order}
    selected = [
        index
        for step in range(8)
        for index in select_sft_batch_indices(
            order,
            batch_size=3,
            step=step,
            seed=seed,
            lengths=lengths,
            batch_order="length_bucket",
            length_bucket_size=4,
            reshuffle_each_epoch=True,
            epoch_order_cache=cache,
        )
    ]
    first_epoch = selected[: len(lengths)]
    second_epoch = selected[len(lengths) :]

    assert first_epoch == order
    assert second_epoch == build_sft_example_order(
        len(lengths),
        seed + 1,
        lengths=lengths,
        batch_order="length_bucket",
        length_bucket_size=4,
    )
    assert first_epoch != second_epoch


def test_legacy_fixed_order_can_be_selected_explicitly() -> None:
    order = build_sft_example_order(5, 17)

    fixed = select_sft_batch_indices(
        order,
        batch_size=3,
        step=1,
        seed=17,
        reshuffle_each_epoch=False,
    )

    assert fixed == order[3:] + order[:1]


def test_describe_sft_batch_position_reports_cross_epoch_batch() -> None:
    assert describe_sft_batch_position(5, batch_size=3, step=1) == {
        "sft_epoch": 0,
        "sft_epoch_end": 1,
        "sft_epoch_sample_offset": 3,
        "sft_absolute_sample": 3,
    }


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
        expected.extend(
            sorted(shuffled[start : start + 4], key=lambda idx: (lengths[idx], idx))
        )

    assert bucketed == expected
    assert sorted(bucketed) == list(range(len(lengths)))


def test_length_order_requires_lengths() -> None:
    with pytest.raises(ValueError, match="requires one token length"):
        build_tokenized_sft_order(3, 1, batch_order="length_bucket")
