"""Reward function protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RewardFunction(Protocol):
    """Any object that can score a completion against a reference answer."""

    def score(self, response: str, reference: str) -> float: ...
