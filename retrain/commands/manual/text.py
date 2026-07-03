"""Manual text loading."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

ManualPath = Callable[[], Path]


def load_text(cli_name: str, manual_path: ManualPath) -> str:
    """Load manual text and substitute runtime command name."""
    path = manual_path()
    if not path.is_file():
        # Backward-compat fallback for older installs.
        legacy = path.with_name("vauban.man")
        if legacy.is_file():
            path = legacy
    if path.is_file():
        text = path.read_text()
    else:
        text = (
            "RETRAIN(1)\n\nNAME\n"
            "    retrain - manual file missing (reinstall package)\n"
        )
    return text.replace("{{CLI}}", cli_name).rstrip() + "\n"
