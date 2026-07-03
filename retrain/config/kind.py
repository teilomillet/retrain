"""Classify a TOML config by which workflow it drives."""

from __future__ import annotations

import tomllib


def config_kind(path: str) -> str:
    """Return ``"campaign"``, ``"squeeze"``, or ``"single"`` for a TOML file.

    A [campaign] section wins over [squeeze]: campaign conditions may embed
    squeeze-like keys, but the reverse never holds.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if "campaign" in data:
        return "campaign"
    if "squeeze" in data:
        return "squeeze"
    return "single"
