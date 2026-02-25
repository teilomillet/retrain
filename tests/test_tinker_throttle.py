"""Tests for retrain.tinker_throttle counting semaphore."""

import threading
import time

import pytest

from retrain.tinker_throttle import NoOpThrottle, TinkerThrottle


class TestTinkerThrottle:
    def test_creates_lock_files(self, tmp_path):
        lock_dir = str(tmp_path / "locks")
        throttle = TinkerThrottle(lock_dir, max_concurrent=3)
        for i in range(3):
            assert (tmp_path / "locks" / f"slot_{i}.lock").exists()

    def test_acquire_release(self, tmp_path):
        lock_dir = str(tmp_path / "locks")
        throttle = TinkerThrottle(lock_dir, max_concurrent=2)
        throttle.acquire()
        assert throttle._held_fd is not None
        assert throttle._held_slot is not None
        throttle.release()
        assert throttle._held_fd is None
        assert throttle._held_slot is None

    def test_context_manager(self, tmp_path):
        lock_dir = str(tmp_path / "locks")
        throttle = TinkerThrottle(lock_dir, max_concurrent=2)
        with throttle:
            assert throttle._held_fd is not None
        assert throttle._held_fd is None

    def test_context_manager_cleans_up_on_exception(self, tmp_path):
        lock_dir = str(tmp_path / "locks")
        throttle = TinkerThrottle(lock_dir, max_concurrent=2)
        with pytest.raises(ValueError):
            with throttle:
                assert throttle._held_fd is not None
                raise ValueError("boom")
        assert throttle._held_fd is None

    def test_limits_concurrent_access(self, tmp_path):
        """Multi-threaded test: max_concurrent=2 limits to 2 simultaneous holders."""
        lock_dir = str(tmp_path / "locks")
        max_concurrent = 2
        num_threads = 6

        current = threading.Semaphore(0)
        peak_lock = threading.Lock()
        peak = [0]
        current_count = [0]
        barrier = threading.Barrier(num_threads)

        def worker():
            throttle = TinkerThrottle(lock_dir, max_concurrent=max_concurrent)
            barrier.wait()
            with throttle:
                with peak_lock:
                    current_count[0] += 1
                    if current_count[0] > peak[0]:
                        peak[0] = current_count[0]
                time.sleep(0.05)
                with peak_lock:
                    current_count[0] -= 1

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert peak[0] <= max_concurrent

    def test_auto_creates_lock_dir(self, tmp_path):
        lock_dir = str(tmp_path / "nested" / "deep" / "locks")
        throttle = TinkerThrottle(lock_dir, max_concurrent=2)
        assert (tmp_path / "nested" / "deep" / "locks" / "slot_0.lock").exists()

    def test_works_when_lock_files_already_exist(self, tmp_path):
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir()
        (lock_dir / "slot_0.lock").write_text("pre-existing")
        (lock_dir / "slot_1.lock").write_text("pre-existing")

        throttle = TinkerThrottle(str(lock_dir), max_concurrent=2)
        with throttle:
            assert throttle._held_fd is not None


class TestNoOpThrottle:
    def test_is_zero_cost(self):
        throttle = NoOpThrottle()
        throttle.acquire()
        throttle.release()

    def test_context_manager(self):
        throttle = NoOpThrottle()
        with throttle:
            pass

    def test_context_manager_on_exception(self):
        throttle = NoOpThrottle()
        with pytest.raises(ValueError):
            with throttle:
                raise ValueError("boom")
