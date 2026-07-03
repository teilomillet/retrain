"""Environment-derived config defaults."""

from __future__ import annotations

import os


def _first_non_empty_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""
