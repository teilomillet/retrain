"""File-based counting semaphore for Tinker API throttling.

Limits concurrent Tinker API calls across independent campaign
subprocesses using filesystem locks (fcntl.flock).
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


class TinkerThrottle:
    """File-based counting semaphore using fcntl.flock().

    Creates ``max_concurrent`` numbered lock files in ``lock_dir``.
    Each process acquires one slot before making a Tinker API call
    and releases it after.

    Usage::

        throttle = TinkerThrottle("/tmp/locks", max_concurrent=4)
        with throttle:
            # only 4 processes reach here concurrently
            client.train_step(...)
    """

    def __init__(self, lock_dir: str, max_concurrent: int) -> None:
        self._lock_dir = Path(lock_dir)
        self._max_concurrent = max_concurrent
        self._held_fd: int | None = None
        self._held_slot: int | None = None
        # Ensure lock directory and files exist
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        for i in range(max_concurrent):
            lock_path = self._lock_dir / f"slot_{i}.lock"
            if not lock_path.exists():
                lock_path.touch()

    def acquire(self) -> None:
        """Acquire a slot. Tries non-blocking first, blocks on slot 0 if all busy."""
        # Try each slot with non-blocking lock
        for i in range(self._max_concurrent):
            lock_path = self._lock_dir / f"slot_{i}.lock"
            fd = os.open(str(lock_path), os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._held_fd = fd
                self._held_slot = i
                return
            except OSError:
                os.close(fd)

        # All slots busy — block on slot 0
        lock_path = self._lock_dir / "slot_0.lock"
        fd = os.open(str(lock_path), os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        self._held_fd = fd
        self._held_slot = 0

    def release(self) -> None:
        """Release the held slot."""
        if self._held_fd is not None:
            try:
                fcntl.flock(self._held_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._held_fd)
                self._held_fd = None
                self._held_slot = None

    def __enter__(self) -> TinkerThrottle:
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.release()


class NoOpThrottle:
    """Zero-cost opt-out throttle when no lock directory is configured."""

    def acquire(self) -> None:
        pass

    def release(self) -> None:
        pass

    def __enter__(self) -> NoOpThrottle:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        pass
