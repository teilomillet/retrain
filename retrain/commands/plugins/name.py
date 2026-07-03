"""Plugin module name normalization."""

from __future__ import annotations

import re


def sanitize(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        return "my_plugin"
    if cleaned[0].isdigit():
        cleaned = f"plugin_{cleaned}"
    return cleaned
