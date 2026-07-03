"""CLI display-name resolution."""

from __future__ import annotations

import sys
from pathlib import Path


def resolve(argv0: str | None = None) -> str:
    """Best-effort CLI binary name for help text."""
    name = Path(sys.argv[0] if argv0 is None else argv0).name.strip()
    if (
        not name
        or name in {"python", "python3", "pytest", "py.test"}
        or name.endswith(".py")
        or "pytest" in name
    ):
        return "retrain"
    return name
