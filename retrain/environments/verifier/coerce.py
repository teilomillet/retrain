"""Value coercion for verifiers state payloads."""

from __future__ import annotations

from typing import cast


def float_list(raw: object) -> list[float]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    result: list[float] = []
    for item in raw:
        try:
            result.append(float(cast(int | float | str, item)))
        except (TypeError, ValueError):
            result.append(0.0)
    return result


def reward(raw: object) -> float:
    if raw is None:
        return 0.0
    try:
        return float(cast(int | float | str, raw))
    except (TypeError, ValueError):
        return 0.0
