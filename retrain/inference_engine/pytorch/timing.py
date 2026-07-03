"""Low-overhead timing accumulation for PyTorch generation."""

from __future__ import annotations

import time

import torch

from retrain.backends.torch import is_cuda_device


class TimingAccumulator:
    """Accumulate CUDA timings without synchronizing every generated token."""

    def __init__(self, device: str):
        self.device = device
        self._cuda = is_cuda_device(device)
        self._events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._totals = {"prefill": 0.0, "decode": 0.0}

    def start(self):
        if self._cuda:
            with torch.cuda.device(torch.device(self.device)):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
            return event
        return time.perf_counter()

    def stop(self, start, bucket: str) -> None:
        if self._cuda:
            with torch.cuda.device(torch.device(self.device)):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
            self._events.append((bucket, start, end))
            return
        self._totals[bucket] = self._totals.get(bucket, 0.0) + (
            time.perf_counter() - start
        )

    def totals(self) -> dict[str, float]:
        if self._cuda and self._events:
            with torch.cuda.device(torch.device(self.device)):
                torch.cuda.synchronize()
            for bucket, start, end in self._events:
                self._totals[bucket] = self._totals.get(bucket, 0.0) + (
                    start.elapsed_time(end) / 1000.0
                )
            self._events.clear()
        return dict(self._totals)
