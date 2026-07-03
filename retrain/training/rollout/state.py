"""State accumulated while one training step samples rollout groups."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from retrain.advantages import AdvantageResult, EntropyStats
from retrain.config import TrainConfig
from retrain.training.echo import EchoBuildStats, EchoLimitStats, limit_echo_masks
from retrain.training.signals import RewardTieAccumulator
from retrain.training.telemetry import EchoStepPlan


def accumulate_metric_totals(
    totals: dict[str, float],
    metrics: Mapping[str, float],
) -> None:
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value)


def has_nonzero_advantage(rows: list[list[float]]) -> bool:
    return any(abs(value) > 0.0 for row in rows for value in row)


def _echo_allowed_tokens(
    *,
    rl_completion_tokens: int,
    max_tokens_per_step: int,
    max_token_ratio: float,
) -> int:
    """Compute the active ECHO cap for this step."""

    ratio_cap = int(rl_completion_tokens * max_token_ratio)
    return max(0, min(max_tokens_per_step, ratio_cap))


@dataclass
class RolloutAccumulator:
    """Everything one RL step's rollout phase produces for training/logging."""

    rewards: list[float] = field(default_factory=list)
    correct: int = 0
    max_token_hits: int = 0
    total_completions: int = 0
    ties: RewardTieAccumulator = field(default_factory=RewardTieAccumulator)
    surprisal_stats: list[EntropyStats] = field(default_factory=list)
    adv_results: list[AdvantageResult] = field(default_factory=list)
    logprobs_sepa: list[list[float]] = field(default_factory=list)
    planning_masks_sepa: list[list[int]] = field(default_factory=list)
    datum_tokens: list[list[int]] = field(default_factory=list)
    datum_logprobs: list[list[float]] = field(default_factory=list)
    datum_advantages: list[list[float]] = field(default_factory=list)
    datum_echo_advantages: list[list[float]] = field(default_factory=list)
    datum_echo_full_observation_counts: list[int] = field(default_factory=list)
    echo_build: EchoBuildStats = field(default_factory=EchoBuildStats)
    rl_completion_token_count: int = 0
    rl_completion_surprisal_sum: float = 0.0
    sampled_completion_token_count: int = 0
    sampled_completion_surprisal_sum: float = 0.0
    behavior_turns: int = 0
    behavior_invalid: int = 0
    behavior_actions: dict[str, int] = field(default_factory=dict)
    behavior_resp_lens: list[int] = field(default_factory=list)
    rollout_timing_metrics: dict[str, float] = field(default_factory=dict)
    sample_time_s: float = 0.0
    tl_grpo_ema: float | None = None


def prepare_echo_step_plan(
    config: TrainConfig,
    acc: RolloutAccumulator,
) -> EchoStepPlan:
    """Apply ECHO token limits and return the values logged for this step."""
    rl_completion_surprisal_mean = (
        acc.rl_completion_surprisal_sum / acc.rl_completion_token_count
        if acc.rl_completion_token_count > 0
        else 0.0
    )
    echo_completion_surprisal_mean = (
        acc.sampled_completion_surprisal_sum / acc.sampled_completion_token_count
        if acc.sampled_completion_token_count > 0
        else rl_completion_surprisal_mean
    )
    if not config.echo_enabled:
        return EchoStepPlan(
            limit=EchoLimitStats(),
            allowed_tokens=0,
            reference_completion_tokens=0,
            skipped_entropy_floor=False,
            rl_completion_surprisal_mean=rl_completion_surprisal_mean,
            echo_completion_surprisal_mean=echo_completion_surprisal_mean,
        )

    reference_completion_tokens = acc.sampled_completion_token_count
    allowed_tokens = _echo_allowed_tokens(
        rl_completion_tokens=reference_completion_tokens,
        max_tokens_per_step=config.echo_max_tokens_per_step,
        max_token_ratio=config.echo_max_token_ratio,
    )
    if echo_completion_surprisal_mean < config.echo_entropy_floor:
        acc.datum_echo_advantages = [
            [0.0] * len(row) for row in acc.datum_echo_advantages
        ]
        return EchoStepPlan(
            limit=EchoLimitStats(
                kept_datums=0,
                kept_tokens=0,
                truncated_tokens=acc.echo_build.candidate_tokens,
            ),
            allowed_tokens=allowed_tokens,
            reference_completion_tokens=reference_completion_tokens,
            skipped_entropy_floor=True,
            rl_completion_surprisal_mean=rl_completion_surprisal_mean,
            echo_completion_surprisal_mean=echo_completion_surprisal_mean,
        )

    acc.datum_echo_advantages, limit = limit_echo_masks(
        acc.datum_echo_advantages,
        max_positive_tokens=allowed_tokens,
    )
    return EchoStepPlan(
        limit=limit,
        allowed_tokens=allowed_tokens,
        reference_completion_tokens=reference_completion_tokens,
        skipped_entropy_floor=False,
        rl_completion_surprisal_mean=rl_completion_surprisal_mean,
        echo_completion_surprisal_mean=echo_completion_surprisal_mean,
    )
