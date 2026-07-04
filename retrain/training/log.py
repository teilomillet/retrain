"""Side effects for recording one RL training step."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from retrain.advantages import EntropyStats
from retrain.backends import TrainHelper
from retrain.backends.catalog import BackendCapabilities
from retrain.training.backpressure import BackPressureDecision
from retrain.config import TrainConfig
from retrain.training.rollouts import RuntimeCounters
from retrain.training.recoverability import checkpoint_recoverability_wandb_metrics
from retrain.training.telemetry import (
    EchoStepPlan,
    MetricValue,
    RolloutTelemetry,
    StepLogData,
    build_emergence_step_entry,
    build_step_metrics,
    build_wandb_metrics,
    format_step_log_summary,
    summarize_surprisal_stats,
)


class WandbRunLike(Protocol):
    def log(
        self,
        data: Mapping[str, object],
        *,
        step: int | None = None,
    ) -> object: ...

    def finish(self) -> object: ...


class WandbModuleLike(Protocol):
    def init(
        self,
        *,
        project: str,
        name: str,
        config: Mapping[str, str | int | float],
        entity: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> WandbRunLike: ...


class StepLoggerLike(Protocol):
    def log(self, entry: dict) -> object: ...


@dataclass(frozen=True)
class StepLoggingContext:
    step: int
    condition_label: str
    loss_value: float
    echo_loss: float
    echo_joint_optimizer_step: bool
    num_datums: int
    total_correct: int
    total_completions: int
    step_time: float
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
    surprisal_stats: Sequence[EntropyStats]


@dataclass(frozen=True)
class StepLoggingResult:
    metrics: Mapping[str, MetricValue]
    delight_eta_ema: float | None


def init_wandb(
    config: TrainConfig,
    *,
    condition_label: str,
) -> WandbRunLike | None:
    """Start a wandb run mirroring the config, or None when unconfigured."""
    if not config.wandb_project:
        return None
    import wandb as wandb_module

    wandb = cast(WandbModuleLike, wandb_module)
    run_name = config.wandb_run_name or Path(config.log_dir).name or condition_label
    wandb_tags = (
        [t.strip() for t in config.wandb_tags.split(",") if t.strip()]
        if config.wandb_tags
        else None
    )
    wandb_config: dict[str, str | int | float] = {
        "algorithm_mode": config.algorithm_mode,
        "advantage_mode": config.advantage_mode,
        "transform_mode": config.transform_mode,
        "uncertainty_kind": config.uncertainty_kind,
        "condition": condition_label,
        "model": config.model,
        "lora_rank": config.lora_rank,
        "lr": config.lr,
        "batch_size": config.batch_size,
        "group_size": config.group_size,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "gtpo_beta": config.gtpo_beta,
        "hicra_alpha": config.hicra_alpha,
        "sepa_steps": config.sepa_steps,
        "sepa_delay_steps": config.sepa_delay_steps,
        "sepa_correct_rate_gate": config.sepa_correct_rate_gate,
        "max_steps": config.max_steps,
        "backend": config.backend,
        "seed": config.seed,
        "batch_advantage_norm": int(config.batch_advantage_norm),
        "clip_eps": config.clip_eps,
        "clip_eps_high": config.clip_eps_high,
        "policy_loss_mode": config.policy_loss_mode,
        "kl_cov_percent": config.kl_cov_percent,
        "kl_cov_coef": config.kl_cov_coef,
        "clip_cov_ratio": config.clip_cov_ratio,
        "clip_cov_min": config.clip_cov_min,
        "clip_cov_max": config.clip_cov_max,
        "adv_clip_max": config.adv_clip_max,
        "sft_warmup_steps": config.sft_warmup_steps,
        "tl_grpo": int(config.tl_grpo),
        "echo_enabled": int(config.echo_enabled),
        "echo_weight": config.echo_weight,
        "echo_max_tokens_per_step": config.echo_max_tokens_per_step,
        "echo_max_token_ratio": config.echo_max_token_ratio,
        "echo_entropy_floor": config.echo_entropy_floor,
        "checkpoint_artifacts": config.checkpoint_artifacts,
    }
    run = wandb.init(
        project=config.wandb_project,
        name=run_name,
        config=wandb_config,
        entity=config.wandb_entity or None,
        group=config.wandb_group or None,
        tags=wandb_tags,
    )
    print(f"Wandb initialized: {config.wandb_project}/{run_name}")
    return run


def record_training_step(
    context: StepLoggingContext,
    *,
    config: TrainConfig,
    backend_caps: BackendCapabilities,
    rollout: RolloutTelemetry,
    echo_plan: EchoStepPlan,
    bp_decision: BackPressureDecision,
    batch_norm_metrics: Mapping[str, float],
    runtime_counters: RuntimeCounters,
    helper: TrainHelper,
    metrics_logger: StepLoggerLike,
    steps_logger: StepLoggerLike,
    wandb_run: WandbRunLike | None,
) -> StepLoggingResult:
    """Persist all logs for one RL step and return state derived from metrics."""
    step_log = StepLogData(
        step=context.step,
        condition_label=context.condition_label,
        loss_value=context.loss_value,
        echo_loss=context.echo_loss,
        echo_joint_optimizer_step=context.echo_joint_optimizer_step,
        mean_reward=(
            sum(rollout.rewards) / len(rollout.rewards) if rollout.rewards else 0.0
        ),
        correct_rate=(
            rollout.correct / len(rollout.rewards) if rollout.rewards else 0.0
        ),
        running_correct_rate=(
            context.total_correct / context.total_completions
            if context.total_completions > 0
            else 0.0
        ),
        max_token_hit_rate=(
            rollout.max_token_hits / rollout.total_completions
            if rollout.total_completions > 0
            else 0.0
        ),
        num_datums=context.num_datums,
        step_time=context.step_time,
        sample_time=rollout.sample_time_s,
        train_time=context.train_time,
        rl_train_time=context.rl_train_time,
        echo_train_time=context.echo_train_time,
        bp_total_tokens=context.bp_total_tokens,
        batch_size=context.batch_size,
        group_size=context.group_size,
        bp_warmup=context.bp_warmup,
        sepa_lambda=context.sepa_lambda,
        sepa_gate=context.sepa_gate,
        clip_fraction=context.clip_fraction,
        policy_cov_fraction=context.policy_cov_fraction,
        policy_abs_kl=context.policy_abs_kl,
        adv_cap_fraction=context.adv_cap_fraction,
        adv_cap_magnitude=context.adv_cap_magnitude,
        tl_grpo_ema=context.tl_grpo_ema,
        surprisal=summarize_surprisal_stats(context.surprisal_stats),
    )
    metrics = build_step_metrics(
        step_log,
        config=config,
        backend_caps=backend_caps,
        rollout=rollout,
        echo_plan=echo_plan,
        bp_decision=bp_decision,
        batch_norm_metrics=batch_norm_metrics,
        runtime_counters=runtime_counters,
        helper=helper,
    )
    metrics_logger.log(metrics)
    print(
        format_step_log_summary(
            step_log,
            backend_caps=backend_caps,
            rollout=rollout,
        )
    )

    if wandb_run is not None:
        wandb_metrics: dict[str, object] = dict(
            build_wandb_metrics(
                step_log,
                adv_results=rollout.adv_results,
                batch_norm_metrics=batch_norm_metrics,
                metrics=metrics,
            )
        )
        wandb_metrics.update(
            checkpoint_recoverability_wandb_metrics(config, wandb_run)
        )
        wandb_run.log(
            wandb_metrics,
            step=context.step,
        )

    steps_logger.log(
        build_emergence_step_entry(
            step_log,
            config=config,
            rollout=rollout,
            echo_plan=echo_plan,
            metrics=metrics,
        )
    )
    return StepLoggingResult(
        metrics=metrics,
        delight_eta_ema=(float(metrics["dg_eta"]) if "dg_eta" in metrics else None),
    )
