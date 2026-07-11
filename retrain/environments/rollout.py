"""Multi-turn environment rollout support."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from retrain.types import PromptLike

if TYPE_CHECKING:
    from retrain.backends import TrainHelper


@dataclass
class VerifiersTurnSample:
    """One assistant turn sampled by retrain for a rollout."""

    prompt_ids: list[int]
    completion_ids: list[int]
    completion_logprobs: list[float]
    completion_text: str
    observation_mask: list[int] | None = None
    echo_observation_capture_supported: bool = False
    post_observation_ids: list[int] | None = None
    post_observation_mask: list[int] | None = None
    post_observation_seen: bool = False
    post_observation_bridge_failed: bool = False
    echo_renderer_parity_failed: bool = False
    post_observation_terminal: bool = False


@dataclass
class VerifiersRolloutTiming:
    """Stage timings for a verifiers multi-turn rollout group."""

    init_state_s: float = 0.0
    prompt_render_s: float = 0.0
    prompt_encode_s: float = 0.0
    generation_s: float = 0.0
    decode_s: float = 0.0
    trajectory_step_s: float = 0.0
    render_completion_s: float = 0.0
    score_s: float = 0.0
    branch_s: float = 0.0
    scheduler_wait_s: float = 0.0
    scheduler_worker_s: float = 0.0
    scheduler_buffer_wait_s: float = 0.0
    total_s: float = 0.0
    turns: int = 0
    model_tokens: int = 0
    env_workers: int = 1
    buffer_size: int = 1
    env_timing_s: dict[str, float] | None = None

    def as_metrics(self, prefix: str = "rollout/") -> dict[str, float]:
        metrics = {
            f"{prefix}init_state_s": self.init_state_s,
            f"{prefix}prompt_render_s": self.prompt_render_s,
            f"{prefix}prompt_encode_s": self.prompt_encode_s,
            f"{prefix}generation_s": self.generation_s,
            f"{prefix}decode_s": self.decode_s,
            f"{prefix}trajectory_step_s": self.trajectory_step_s,
            f"{prefix}render_completion_s": self.render_completion_s,
            f"{prefix}score_s": self.score_s,
            f"{prefix}branch_s": self.branch_s,
            f"{prefix}scheduler_wait_s": self.scheduler_wait_s,
            f"{prefix}scheduler_worker_s": self.scheduler_worker_s,
            f"{prefix}scheduler_buffer_wait_s": self.scheduler_buffer_wait_s,
            f"{prefix}total_s": self.total_s,
            f"{prefix}turns": float(self.turns),
            f"{prefix}model_tokens": float(self.model_tokens),
            f"{prefix}env_workers": float(self.env_workers),
            f"{prefix}buffer_size": float(self.buffer_size),
        }
        if self.env_timing_s:
            for key, value in self.env_timing_s.items():
                metrics[f"{prefix}env/{key}"] = value
        return metrics


class RolloutScheduler:
    """Bound async environment work for one multi-turn rollout group."""

    def __init__(
        self,
        *,
        max_env_workers: int,
        max_buffered_rollouts: int,
    ) -> None:
        self.max_env_workers = max(1, int(max_env_workers))
        self.max_buffered_rollouts = max(1, int(max_buffered_rollouts))

    async def map_ordered(
        self,
        items: Sequence[object],
        worker: Callable[[object], Awaitable[object]],
        timing: VerifiersRolloutTiming,
    ) -> list[object]:
        """Run async item work with bounded concurrency and stable ordering."""
        if not items:
            return []

        if self.max_env_workers <= 1:
            results = []
            for item in items:
                started = time.perf_counter()
                result = await worker(item)
                timing.scheduler_worker_s += time.perf_counter() - started
                results.append(result)
            return results

        env_sem = asyncio.Semaphore(self.max_env_workers)
        buffer_sem = asyncio.Semaphore(min(self.max_buffered_rollouts, len(items)))

        async def run_one(pos: int, item: object) -> tuple[int, object]:
            buffer_wait_started = time.perf_counter()
            await buffer_sem.acquire()
            timing.scheduler_buffer_wait_s += time.perf_counter() - buffer_wait_started
            worker_wait_started = time.perf_counter()
            await env_sem.acquire()
            timing.scheduler_wait_s += time.perf_counter() - worker_wait_started
            try:
                started = time.perf_counter()
                result = await worker(item)
                timing.scheduler_worker_s += time.perf_counter() - started
                return pos, result
            finally:
                env_sem.release()
                buffer_sem.release()

        pending_results = await asyncio.gather(
            *(run_one(pos, item) for pos, item in enumerate(items)),
            return_exceptions=True,
        )
        for result in pending_results:
            if isinstance(result, BaseException):
                raise result

        slots: list[object | None] = [None] * len(items)
        for pos, result in cast(list[tuple[int, object]], pending_results):
            slots[pos] = result
        return [cast(object, result) for result in slots]


def rollout_temperatures(
    *,
    temperature: float,
    temperature_spread: float,
    num_rollouts: int,
) -> list[float]:
    """Return one sampling temperature per rollout."""
    if temperature_spread <= 0.0 or num_rollouts <= 1:
        return [temperature] * num_rollouts
    return [
        max(
            0.1,
            temperature + temperature_spread * (2 * i / (num_rollouts - 1) - 1),
        )
        for i in range(num_rollouts)
    ]


def sample_active_rollouts(
    *,
    helper: "TrainHelper",
    active: list[tuple[int, PromptLike, list[int], list[int] | None]],
    rollout_temps: list[float],
    max_tokens: int,
    top_p: float,
) -> list[list[tuple[list[int], list[float]]]]:
    """Sample active rollout prompts, batching only equal-temperature prompts."""
    sampled_groups: list[list[tuple[list[int], list[float]]]] = [[] for _ in active]
    batch_positions: list[int] = []
    batch_prompt_ids: list[list[int]] = []
    batch_temperature: float | None = None

    def flush_batch() -> None:
        nonlocal batch_positions, batch_prompt_ids, batch_temperature
        if batch_temperature is None:
            return
        grouped_positions: dict[tuple[int, ...], list[int]] = {}
        grouped_prompts: dict[tuple[int, ...], list[int]] = {}
        for position, prompt_ids in zip(batch_positions, batch_prompt_ids):
            key = tuple(prompt_ids)
            grouped_positions.setdefault(key, []).append(position)
            grouped_prompts.setdefault(key, prompt_ids)

        if all(len(positions) == 1 for positions in grouped_positions.values()):
            groups = helper.sample(
                batch_prompt_ids,
                1,
                max_tokens,
                batch_temperature,
                top_p,
            )
            for position, group in zip(batch_positions, groups):
                sampled_groups[position] = group
        else:
            for key, positions in grouped_positions.items():
                groups = helper.sample(
                    [grouped_prompts[key]],
                    len(positions),
                    max_tokens,
                    batch_temperature,
                    top_p,
                )
                samples = groups[0] if groups else []
                for offset, position in enumerate(positions):
                    sampled_groups[position] = (
                        [samples[offset]] if offset < len(samples) else []
                    )
        batch_positions = []
        batch_prompt_ids = []
        batch_temperature = None

    for position, (
        rollout_idx,
        _messages,
        prompt_ids,
        _observation_mask,
    ) in enumerate(active):
        rollout_temperature = rollout_temps[rollout_idx]
        if batch_temperature is None:
            batch_temperature = rollout_temperature
        elif rollout_temperature != batch_temperature:
            flush_batch()
            batch_temperature = rollout_temperature
        batch_positions.append(position)
        batch_prompt_ids.append(prompt_ids)
    flush_batch()

    return sampled_groups
