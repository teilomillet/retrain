"""JSONL logger -- append JSON to files via json.dumps."""

from __future__ import annotations

import json
from pathlib import Path


class JsonlLogger:
    """Append-only JSONL file logger."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def log(self, entry: dict) -> None:
        """Write a dict as a single JSON line to the log file."""
        with open(self.file_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
