"""TOML table typing helpers."""

from __future__ import annotations

import typing


def as_object_table(value: object) -> dict[str, object] | None:
    """Return TOML-style tables as object mappings."""
    if not isinstance(value, dict):
        return None
    return typing.cast(dict[str, object], value)
