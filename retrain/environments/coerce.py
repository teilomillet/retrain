"""Value coercion shared by environment payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast


def integer(raw: object) -> int:
    try:
        return int(cast(int | str | float, raw))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected int-like value, got {raw!r}.") from exc


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


def field(container: object, name: str) -> object:
    """Read one field from either a mapping or an attribute-bearing object."""

    if isinstance(container, Mapping):
        return cast(Mapping[str, object], container).get(name)
    return getattr(container, name, None)
