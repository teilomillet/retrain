"""Per-step training telemetry builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from retrain.advantages import AdvantageResult, EntropyStats
from retrain.backend_definitions import BackendCapabilities
from retrain.backends import TrainHelper, collect_runtime_metrics
from retrain.backpressure import BackPressureDecision
from retrain.config import TrainConfig
from retrain.echo import EchoBuildStats, EchoLimitStats
from retrain.process_metrics import process_max_rss_mb
from retrain.runtime_support import RuntimeCounters


MetricValue = bool | int | float | str

_TIMING_EPS = 1e-9

_WANDB_ECHO_METRIC_KEYS = (
    "rl/train_time_s",
    "rl/completion_tokens",
    "rl/completion_surprisal_mean",
    "echo/enabled",
    "echo/loss",
    "echo/train_time_s",
    "echo/weight",
    "echo/allowed_tokens",
    "echo/reference_completion_tokens",
    "echo/completion_surprisal_mean",
    "echo/candidate_datums",
    "echo/candidate_tokens",
    "echo/observation_mask_datums",
    "echo/kept_datums",
    "echo/kept_tokens",
    "echo/truncated_tokens",
    "echo/token_ratio",
    "echo/skipped_low_overlap",
    "echo/skipped_bad_observation_mask",
    "echo/skipped_entropy_floor",
    "echo/entropy_floor",
    "echo/mode_collapse_guard",
)

_WANDB_BEHAVIOR_METRIC_KEYS = (
    "behavior/invalid_action_rate",
    "behavior/action_type_count",
    "behavior/action_dominance",
    "behavior/avg_response_chars",
)


class RewardTieTelemetry(Protocol):
    eligible_groups: int
    tie_groups: int
    uniform_groups: int
    tie_group_rate: float
    uniform_group_rate: float
    tie_pair_rate: float
    unique_fraction_mean: float


class RolloutTelemetry(Protocol):
    rewards: Sequence[float]
    correct: int
    max_token_hits: int
    total_completions: int
    ties: RewardTieTelemetry
    adv_results: Sequence[AdvantageResult]
    echo_build: EchoBuildStats
    rl_completion_token_count: int
    rollout_timing_metrics: Mapping[str, float]
    sample_time_s: float
    behavior_turns: int
    behavior_invalid: int
    behavior_actions: Mapping[str, int]
    behavior_resp_lens: Sequence[int]


@dataclass(frozen=True)
class EchoStepPlan:
    limit: EchoLimitStats
    allowed_tokens: int
    reference_completion_tokens: int
    skipped_entropy_floor: bool
    rl_completion_surprisal_mean: float
    echo_completion_surprisal_mean: float

    @property
    def token_ratio(self) -> float:
        if self.reference_completion_tokens <= 0:
            return 0.0
        return self.limit.kept_tokens / self.reference_completion_tokens


@dataclass(frozen=True)
class SurprisalSummary:
    has_values: bool = False
    exec_mean: float = 0.0
    exec_var: float = 0.0
    plan_mean: float = 0.0
    plan_var: float = 0.0
    post_exec_mean: float = 0.0
    post_exec_var: float = 0.0
    post_plan_mean: float = 0.0
    post_plan_var: float = 0.0


@dataclass(frozen=True)
class StepLogData:
    step: int
    condition_label: str
    loss_value: float
    echo_loss: float
    echo_joint_optimizer_step: bool
    mean_reward: float
    correct_rate: float
    running_correct_rate: float
    max_token_hit_rate: float
    num_datums: int
    step_time: float
    sample_time: float
    train_time: float
    rl_train_time: float
    echo_train_time: float
    bp_total_tokens: int
    batch_size: int
    group_size: int
    bp_warmup: bool
    sepa_lambda: float
    sepa_gate: bool
    clip_fraction: float
    policy_cov_fraction: float
    policy_abs_kl: float
    adv_cap_fraction: float
    adv_cap_magnitude: float
    tl_grpo_ema: float | None
    surprisal: SurprisalSummary


def summarize_surprisal_stats(
    stats: Sequence[EntropyStats],
) -> SurprisalSummary:
    if not stats:
        return SurprisalSummary()
    n_stats = len(stats)
    return SurprisalSummary(
        has_values=True,
        exec_mean=sum(s.exec_mean for s in stats) / n_stats,
        exec_var=sum(s.exec_var for s in stats) / n_stats,
        plan_mean=sum(s.plan_mean for s in stats) / n_stats,
        plan_var=sum(s.plan_var for s in stats) / n_stats,
        post_exec_mean=sum(s.post_exec_mean for s in stats) / n_stats,
        post_exec_var=sum(s.post_exec_var for s in stats) / n_stats,
        post_plan_mean=sum(s.post_plan_mean for s in stats) / n_stats,
        post_plan_var=sum(s.post_plan_var for s in stats) / n_stats,
    )


def _surprisal_metric_fields(
    summary: SurprisalSummary,
) -> dict[str, float]:
    return {
        "exec_entropy_mean": summary.exec_mean,
        "exec_entropy_var": summary.exec_var,
        "plan_entropy_mean": summary.plan_mean,
        "plan_entropy_var": summary.plan_var,
        "exec_surprisal_mean": summary.exec_mean,
        "exec_surprisal_var": summary.exec_var,
        "plan_surprisal_mean": summary.plan_mean,
        "plan_surprisal_var": summary.plan_var,
        "post_exec_surprisal_mean": summary.post_exec_mean,
        "post_exec_surprisal_var": summary.post_exec_var,
        "post_plan_surprisal_mean": summary.post_plan_mean,
        "post_plan_surprisal_var": summary.post_plan_var,
    }


def _advantage_extra_metric_names(
    results: Sequence[AdvantageResult],
) -> set[str]:
    return {key for result in results for key in result.extra_metrics}


def _average_advantage_extra_metrics(
    results: Sequence[AdvantageResult],
) -> dict[str, float]:
    averaged: dict[str, float] = {}
    for key in _advantage_extra_metric_names(results):
        values = [r.extra_metrics[key] for r in results if key in r.extra_metrics]
        averaged[key] = sum(values) / len(values)
    return averaged


def _add_rollout_timing_metrics(
    metrics: dict[str, MetricValue],
    rollout: RolloutTelemetry,
) -> None:
    if not rollout.rollout_timing_metrics:
        return
    metrics.update(rollout.rollout_timing_metrics)
    rollout_total = rollout.rollout_timing_metrics.get("rollout/total_s", 0.0)
    rollout_share = (
        rollout_total / rollout.sample_time_s
        if rollout.sample_time_s > _TIMING_EPS
        else 0.0
    )
    metrics["rollout/accounted_share_of_sample"] = min(
        max(rollout_share, 0.0),
        1.0,
    )


def _add_behavior_metrics(
    metrics: dict[str, MetricValue],
    rollout: RolloutTelemetry,
) -> None:
    if rollout.behavior_turns > 0:
        metrics["behavior/invalid_action_rate"] = (
            rollout.behavior_invalid / rollout.behavior_turns
        )
        metrics["behavior/action_type_count"] = len(rollout.behavior_actions)
        if rollout.behavior_actions:
            action_total = sum(rollout.behavior_actions.values())
            metrics["behavior/action_dominance"] = (
                max(rollout.behavior_actions.values()) / action_total
            )
    if rollout.behavior_resp_lens:
        metrics["behavior/avg_response_chars"] = (
            sum(rollout.behavior_resp_lens) / len(rollout.behavior_resp_lens)
        )


def build_step_metrics(
    data: StepLogData,
    *,
    config: TrainConfig,
    backend_caps: BackendCapabilities,
    rollout: RolloutTelemetry,
    echo_plan: EchoStepPlan,
    bp_decision: BackPressureDecision,
    batch_norm_metrics: Mapping[str, float],
    runtime_counters: RuntimeCounters,
    helper: TrainHelper,
) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {
        "step": data.step,
        "algorithm_mode": config.algorithm_mode,
        "advantage_mode": config.advantage_mode,
        "transform_mode": config.transform_mode,
        "uncertainty_kind": config.uncertainty_kind,
        "condition": data.condition_label,
        "backend_reports_sync_loss": backend_caps.reports_sync_loss,
        "backend_preserves_token_advantages": backend_caps.preserves_token_advantages,
        "loss_is_placeholder": not backend_caps.reports_sync_loss,
        "reported_loss": data.loss_value,
        "loss": data.loss_value,
        "mean_reward": data.mean_reward,
        "correct_rate": data.correct_rate,
        "running_correct_rate": data.running_correct_rate,
        "reward_tie_eligible_groups": rollout.ties.eligible_groups,
        "reward_tie_groups": rollout.ties.tie_groups,
        "reward_uniform_groups": rollout.ties.uniform_groups,
        "reward_tie_group_rate": rollout.ties.tie_group_rate,
        "reward_uniform_group_rate": rollout.ties.uniform_group_rate,
        "reward_tie_pair_rate": rollout.ties.tie_pair_rate,
        "reward_unique_fraction_mean": rollout.ties.unique_fraction_mean,
        "sepa_lambda": data.sepa_lambda,
        "sepa_gate_open": data.sepa_gate,
        "num_datums": data.num_datums,
        "max_token_hit_rate": data.max_token_hit_rate,
        "step_time_s": data.step_time,
        "sample_time_s": data.sample_time,
        "train_time_s": data.train_time,
        "batch_size": data.batch_size,
        "group_size": data.group_size,
        "bp_warmup": data.bp_warmup,
        "bp_action": bp_decision.action,
        "bp_regime": bp_decision.regime,
        "bp_p_star": bp_decision.p_star,
        "bp_sigma": bp_decision.sigma,
        "bp_kappa": bp_decision.kappa,
        "bp_utilization": bp_decision.utilization,
        "bp_throughput": bp_decision.throughput,
        "tokens_per_step": data.bp_total_tokens,
        "tokens_per_second": (
            data.bp_total_tokens / data.step_time
            if data.step_time > _TIMING_EPS
            else 0.0
        ),
        "sample_share": (
            data.sample_time / data.step_time
            if data.step_time > _TIMING_EPS
            else 0.0
        ),
        "train_share": (
            data.train_time / data.step_time
            if data.step_time > _TIMING_EPS
            else 0.0
        ),
        "rl/train_time_s": data.rl_train_time,
        "rl/completion_tokens": rollout.rl_completion_token_count,
        "rl/completion_surprisal_mean": (
            echo_plan.rl_completion_surprisal_mean
        ),
        "echo/enabled": int(config.echo_enabled),
        "echo/loss": data.echo_loss,
        "echo/train_time_s": data.echo_train_time,
        "echo/weight": config.echo_weight,
        "echo/allowed_tokens": echo_plan.allowed_tokens,
        "echo/reference_completion_tokens": echo_plan.reference_completion_tokens,
        "echo/completion_surprisal_mean": (
            echo_plan.echo_completion_surprisal_mean
        ),
        "echo/candidate_datums": rollout.echo_build.candidate_datums,
        "echo/candidate_tokens": rollout.echo_build.candidate_tokens,
        "echo/observation_mask_datums": rollout.echo_build.observation_mask_datums,
        "echo/kept_datums": echo_plan.limit.kept_datums,
        "echo/kept_tokens": echo_plan.limit.kept_tokens,
        "echo/truncated_tokens": echo_plan.limit.truncated_tokens,
        "echo/token_ratio": echo_plan.token_ratio,
        "echo/skipped_first_turns": rollout.echo_build.skipped_first_turns,
        "echo/skipped_no_suffix": rollout.echo_build.skipped_no_suffix,
        "echo/skipped_low_overlap": rollout.echo_build.skipped_low_overlap,
        "echo/skipped_bad_observation_mask": (
            rollout.echo_build.skipped_bad_observation_mask
        ),
        "echo/skipped_entropy_floor": int(echo_plan.skipped_entropy_floor),
        "echo/entropy_floor": config.echo_entropy_floor,
        "echo/mode_collapse_guard": int(echo_plan.skipped_entropy_floor),
        "echo/joint_optimizer_step": int(data.echo_joint_optimizer_step),
    }
    rss_mb = process_max_rss_mb()
    if rss_mb is not None:
        metrics["process_max_rss_mb"] = round(rss_mb, 3)
    metrics.update(runtime_counters.metrics())
    metrics.update(collect_runtime_metrics(helper))
    _add_rollout_timing_metrics(metrics, rollout)
    if data.surprisal.has_values:
        metrics.update(_surprisal_metric_fields(data.surprisal))
    metrics["clip_fraction"] = data.clip_fraction
    metrics["policy_loss_mode"] = config.policy_loss_mode
    metrics["policy/cov_fraction"] = data.policy_cov_fraction
    metrics["policy/abs_kl"] = data.policy_abs_kl
    metrics["adv_cap_fraction"] = data.adv_cap_fraction
    metrics["adv_cap_magnitude"] = data.adv_cap_magnitude
    metrics.update(batch_norm_metrics)
    metrics.update(_average_advantage_extra_metrics(rollout.adv_results))
    _add_behavior_metrics(metrics, rollout)
    return metrics


def build_wandb_metrics(
    data: StepLogData,
    *,
    config: TrainConfig,
    backend_caps: BackendCapabilities,
    rollout: RolloutTelemetry,
    bp_decision: BackPressureDecision,
    batch_norm_metrics: Mapping[str, float],
    metrics: Mapping[str, MetricValue],
) -> dict[str, MetricValue]:
    wandb_metrics: dict[str, MetricValue] = {
        "train/loss": data.loss_value,
        "train/reported_loss": data.loss_value,
        "train/uncertainty_kind": config.uncertainty_kind,
        "train/loss_is_placeholder": int(not backend_caps.reports_sync_loss),
        "train/rewards/mean_reward": data.mean_reward,
        "train/rewards/correct_rate": data.correct_rate,
        "train/rewards/running_correct_rate": data.running_correct_rate,
        "train/rewards/tie_eligible_groups": rollout.ties.eligible_groups,
        "train/rewards/tie_groups": rollout.ties.tie_groups,
        "train/rewards/uniform_groups": rollout.ties.uniform_groups,
        "train/rewards/tie_group_rate": rollout.ties.tie_group_rate,
        "train/rewards/uniform_group_rate": rollout.ties.uniform_group_rate,
        "train/rewards/tie_pair_rate": rollout.ties.tie_pair_rate,
        "train/rewards/unique_fraction_mean": rollout.ties.unique_fraction_mean,
        "train/backend/reports_sync_loss": int(backend_caps.reports_sync_loss),
        "train/backend/preserves_token_advantages": int(
            backend_caps.preserves_token_advantages
        ),
        "train/sepa_lambda": data.sepa_lambda,
        "train/sepa_gate_open": int(data.sepa_gate),
        "train/max_token_hit_rate": data.max_token_hit_rate,
        "train/num_datums": data.num_datums,
        "train/step_time_s": data.step_time,
        "train/batch_size": data.batch_size,
        "train/group_size": data.group_size,
        "train/entropy/exec_mean": data.surprisal.exec_mean,
        "train/entropy/exec_var": data.surprisal.exec_var,
        "train/entropy/plan_mean": data.surprisal.plan_mean,
        "train/entropy/plan_var": data.surprisal.plan_var,
        "train/surprisal/exec_mean": data.surprisal.exec_mean,
        "train/surprisal/exec_var": data.surprisal.exec_var,
        "train/surprisal/plan_mean": data.surprisal.plan_mean,
        "train/surprisal/plan_var": data.surprisal.plan_var,
        "train/surprisal/post_exec_mean": data.surprisal.post_exec_mean,
        "train/surprisal/post_exec_var": data.surprisal.post_exec_var,
        "train/surprisal/post_plan_mean": data.surprisal.post_plan_mean,
        "train/surprisal/post_plan_var": data.surprisal.post_plan_var,
        "train/clip_fraction": data.clip_fraction,
        "train/policy_cov_fraction": data.policy_cov_fraction,
        "train/policy_abs_kl": data.policy_abs_kl,
        "train/adv_cap_fraction": data.adv_cap_fraction,
        "train/adv_cap_magnitude": data.adv_cap_magnitude,
        "train/backpressure/action": bp_decision.action,
        "train/backpressure/regime": bp_decision.regime,
        "train/backpressure/p_star": bp_decision.p_star,
        "train/backpressure/sigma": bp_decision.sigma,
        "train/backpressure/kappa": bp_decision.kappa,
        "train/backpressure/utilization": bp_decision.utilization,
        "train/backpressure/throughput": bp_decision.throughput,
        "train/backpressure/warmup": int(data.bp_warmup),
    }
    for key, value in batch_norm_metrics.items():
        wandb_metrics[f"train/{key}"] = value
    for key in _advantage_extra_metric_names(rollout.adv_results):
        wandb_metrics[f"train/{key}"] = metrics.get(key, 0.0)
    for key in _WANDB_ECHO_METRIC_KEYS:
        wandb_metrics[f"train/{key}"] = metrics[key]
    if data.tl_grpo_ema is not None:
        wandb_metrics["train/tl_grpo_ema_baseline"] = data.tl_grpo_ema
    for key in _WANDB_BEHAVIOR_METRIC_KEYS:
        if key in metrics:
            wandb_metrics[f"train/{key}"] = metrics[key]
    return wandb_metrics


def build_emergence_step_entry(
    data: StepLogData,
    *,
    config: TrainConfig,
    rollout: RolloutTelemetry,
    echo_plan: EchoStepPlan,
    metrics: Mapping[str, MetricValue],
) -> dict[str, MetricValue]:
    step_entry: dict[str, MetricValue] = {
        "step": data.step,
        "mean_reward": data.mean_reward,
        "correct_count": rollout.correct,
        "total_count": len(rollout.rewards),
        "condition": data.condition_label,
        "uncertainty_kind": config.uncertainty_kind,
    }
    if data.surprisal.has_values:
        step_entry.update(_surprisal_metric_fields(data.surprisal))
    step_entry["clip_fraction"] = data.clip_fraction
    step_entry["policy_loss_mode"] = config.policy_loss_mode
    step_entry["policy/cov_fraction"] = data.policy_cov_fraction
    step_entry["policy/abs_kl"] = data.policy_abs_kl
    step_entry["adv_cap_fraction"] = data.adv_cap_fraction
    step_entry["adv_cap_magnitude"] = data.adv_cap_magnitude
    step_entry["echo/enabled"] = int(config.echo_enabled)
    step_entry["echo/kept_tokens"] = echo_plan.limit.kept_tokens
    step_entry["echo/token_ratio"] = echo_plan.token_ratio
    step_entry["echo/skipped_entropy_floor"] = int(echo_plan.skipped_entropy_floor)
    for key in _advantage_extra_metric_names(rollout.adv_results):
        step_entry[key] = metrics.get(key, 0.0)
    return step_entry
