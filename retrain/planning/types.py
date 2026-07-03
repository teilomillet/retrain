"""Planning detector protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PlanningDetector(Protocol):
    """Detects which tokens belong to planning/metacognitive spans."""

    def detect(self, token_strs: list[str]) -> list[int]: ...
