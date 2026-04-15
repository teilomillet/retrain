"""JSONL logger -- buffered append of JSON records."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import TextIO


class JsonlLogger:
    """Append-only JSONL file logger with optional buffering.

    The default behavior is still immediate write-through. Callers on hot paths
    can raise ``flush_every`` to amortize open/write/flush costs while still
    bounding staleness with ``flush_interval_s``.
    """

    def __init__(
        self,
        file_path: str,
        *,
        flush_every: int = 1,
        flush_interval_s: float | None = None,
        enabled: bool = True,
    ) -> None:
        self.file_path = file_path
        self.enabled = enabled
        self.flush_every = max(1, int(flush_every))
        self.flush_interval_s = (
            None
            if flush_interval_s is None
            else max(0.0, float(flush_interval_s))
        )
        self._path = Path(file_path)
        self._buffer: list[str] = []
        self._handle: TextIO | None = None
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()

    def log(self, entry: dict) -> None:
        """Write a dict as a single JSON line to the log file."""
        if not self.enabled:
            return

        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock:
            self._buffer.append(line)
            should_flush = len(self._buffer) >= self.flush_every
            if (
                not should_flush
                and self.flush_interval_s is not None
                and self.flush_interval_s == 0.0
            ):
                should_flush = True
            if (
                not should_flush
                and self.flush_interval_s is not None
                and (time.monotonic() - self._last_flush) >= self.flush_interval_s
            ):
                should_flush = True
            if should_flush:
                self._flush_locked()

    def flush(self) -> None:
        """Flush buffered rows to disk."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        with self._lock:
            self._flush_locked()
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Best effort during interpreter teardown.
            pass

    def _flush_locked(self) -> None:
        if not self.enabled or not self._buffer:
            self._last_flush = time.monotonic()
            return
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8")
        self._handle.writelines(self._buffer)
        self._handle.flush()
        self._buffer.clear()
        self._last_flush = time.monotonic()
