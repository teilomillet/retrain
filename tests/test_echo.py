from __future__ import annotations

from types import SimpleNamespace

from retrain.echo import (
    EchoDatum,
    build_prompt_suffix_echo_datums,
    common_prefix_len,
    limit_echo_datums,
)


def test_common_prefix_len_stops_at_first_mismatch() -> None:
    assert common_prefix_len([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix_len([1, 2], [1, 2, 3]) == 2


def test_build_prompt_suffix_echo_datums_uses_new_prompt_suffix() -> None:
    turns = [
        [
            SimpleNamespace(prompt_ids=[1, 2], completion_ids=[3]),
            SimpleNamespace(prompt_ids=[1, 2, 3, 50, 51], completion_ids=[4]),
        ]
    ]

    datums, stats = build_prompt_suffix_echo_datums(
        turns,
        weight=0.2,
        min_prompt_overlap=0.5,
    )

    assert stats.candidate_datums == 1
    assert stats.candidate_tokens == 2
    assert stats.skipped_first_turns == 1
    assert datums[0].tokens == [1, 2, 3, 50, 51]
    assert datums[0].advantages == [0.0, 0.0, 0.0, 0.2, 0.2]


def test_build_prompt_suffix_echo_datums_skips_low_overlap() -> None:
    turns = [
        [
            SimpleNamespace(prompt_ids=[1, 2], completion_ids=[3]),
            SimpleNamespace(prompt_ids=[9, 8, 7], completion_ids=[4]),
        ]
    ]

    datums, stats = build_prompt_suffix_echo_datums(
        turns,
        weight=0.2,
        min_prompt_overlap=0.5,
    )

    assert datums == []
    assert stats.skipped_low_overlap == 1


def test_limit_echo_datums_caps_positive_tokens_only() -> None:
    datums = [
        EchoDatum(
            tokens=[1, 2, 3, 4],
            advantages=[0.0, 0.0, 0.1, 0.1],
            positive_tokens=2,
        ),
        EchoDatum(
            tokens=[5, 6, 7],
            advantages=[0.0, 0.1, 0.1],
            positive_tokens=2,
        ),
    ]

    kept, stats = limit_echo_datums(datums, max_positive_tokens=3)

    assert stats.kept_datums == 2
    assert stats.kept_tokens == 3
    assert stats.truncated_tokens == 1
    assert kept[0].tokens == [1, 2, 3, 4]
    assert kept[1].tokens == [5, 6]
    assert kept[1].advantages == [0.0, 0.1]
