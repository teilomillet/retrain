"""Exact overlap operations for token-id sequences."""

from __future__ import annotations

from collections.abc import Sequence


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the length of the exact token prefix shared by two sequences."""

    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def common_suffix_length(
    left: Sequence[int],
    right: Sequence[int],
    *,
    prefix_len: int,
) -> int:
    """Return the shared suffix length without overlapping a known prefix."""

    limit = min(len(left), len(right)) - prefix_len
    count = 0
    while count < limit and left[-1 - count] == right[-1 - count]:
        count += 1
    return count
