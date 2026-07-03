"""Shared backend option value coercion for constructors."""

from __future__ import annotations

from collections.abc import Mapping


def backend_option_int(
    options: Mapping[str, object],
    key: str,
    default: int = 0,
) -> int:
    raw = options.get(key, default)
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int | float | str):
        return int(raw)
    return default


def backend_option_float(
    options: Mapping[str, object],
    key: str,
    default: float = 0.0,
) -> float:
    raw = options.get(key, default)
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, int | float | str):
        return float(raw)
    return default
