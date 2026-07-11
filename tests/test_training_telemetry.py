from __future__ import annotations

from types import SimpleNamespace

import pytest

import retrain.training.telemetry as telemetry
from retrain.advantages import AdvantageResult, EntropyStats
from retrain.backends.catalog import BackendCapabilities
from retrain.training.backpressure import BackPressureDecision
from retrain.config import TrainConfig
from retrain.training.echo import EchoBuildStats, EchoLimitStats
from retrain.training.rollouts import RuntimeCounters
from retrain.training.rollout import RolloutAccumulator
from retrain.training.telemetry import (
    EchoStepPlan,
    StepLogData,
    build_emergence_step_entry,
    build_runtime_wandb_metrics,
    build_step_metrics,
    build_wandb_metrics,
    format_step_log_summary,
    summarize_surprisal_stats,
)


class _RuntimeMetricsHelper:
    def checkpoint(self, name: str) -> None:
        _ = name

    def sample(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[tuple[list[int], list[float]]]]:
        _ = prompt_ids_list, num_samples, max_tokens, temperature, top_p
        return []

    def train_step(
        self,
        all_tokens: list[list[int]],
        all_logprobs: list[list[float]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float:
        _ = all_tokens, all_logprobs, all_advantages, lr, weight_decay
        return 0.0

    def save_adapter(self, path: str, name: str) -> str:
        return f"{path}/{name}"

    def load_state(self, name: str) -> None:
        _ = name

    def runtime_metrics(self) -> dict[str, float | str]:
        return {
            "backend/tokens_s": 12.0,
            "local_train_wall_s": 3.0,
            "local_train_microbatches": 5,
            "local_train_padding_avoidance_fraction": 0.4,
            "local_train_attention_proxy_avoidance_fraction": 0.6,
            "local_train_sequence_length_p95": 512,
            "optimizer/local_effective_rows_sha256": "e" * 64,
        }


def test_optimizer_signal_count_is_refreshed_from_final_advantages() -> None:
    acc = RolloutAccumulator(
        pre_optimizer_nonzero_advantage_token_count=4,
        datum_advantages=[[0.0, 1.0, 0.0], [0.0, -2.0]],
    )

    acc.refresh_optimizer_advantage_token_count()

    assert acc.pre_optimizer_nonzero_advantage_token_count == 4
    assert acc.optimizer_nonzero_advantage_token_count == 2


def test_runtime_wandb_projection_cannot_override_step_loss() -> None:
    wandb = build_runtime_wandb_metrics(
        {
            "loss": 99.0,
            "local_train_wall_s": 0.5,
            "optimizer/local_effective_rows_sha256": "e" * 64,
        }
    )

    assert "train/loss" not in wandb
    assert wandb["train/backend/local/wall_s"] == pytest.approx(0.5)
    assert wandb["train/optimizer/local_effective_rows_sha256"] == "e" * 64


def test_step_telemetry_builders_share_one_metric_contract(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "max_rss_mb", lambda: 12.34567)
    surprisal = summarize_surprisal_stats(
        [
            EntropyStats(
                exec_mean=2.0,
                exec_var=0.2,
                plan_mean=4.0,
                plan_var=0.4,
                post_exec_mean=6.0,
                post_exec_var=0.6,
                post_plan_mean=8.0,
                post_plan_var=0.8,
            ),
            EntropyStats(
                exec_mean=4.0,
                exec_var=0.4,
                plan_mean=6.0,
                plan_var=0.6,
                post_exec_mean=8.0,
                post_exec_var=0.8,
                post_plan_mean=10.0,
                post_plan_var=1.0,
            ),
        ]
    )
    step = StepLogData(
        step=3,
        condition_label="grpo+none",
        loss_value=0.125,
        echo_loss=0.25,
        echo_joint_optimizer_step=True,
        mean_reward=0.75,
        correct_rate=0.5,
        running_correct_rate=0.625,
        max_token_hit_rate=0.25,
        num_datums=5,
        step_time=10.0,
        sample_time=2.0,
        train_time=3.0,
        rl_train_time=3.0,
        echo_train_time=3.0,
        bp_total_tokens=100,
        batch_size=2,
        group_size=4,
        bp_warmup=False,
        sepa_lambda=0.1,
        sepa_gate=True,
        clip_fraction=0.2,
        policy_cov_fraction=0.3,
        policy_abs_kl=0.4,
        adv_cap_fraction=0.5,
        adv_cap_magnitude=0.6,
        tl_grpo_ema=0.7,
        surprisal=surprisal,
    )
    rollout = SimpleNamespace(
        rewards=[1.0, 0.5],
        correct=1,
        max_token_hits=1,
        total_completions=4,
        ties=SimpleNamespace(
            eligible_groups=2,
            tie_groups=1,
            uniform_groups=0,
            tie_group_rate=0.5,
            uniform_group_rate=0.0,
            tie_pair_rate=0.25,
            unique_fraction_mean=0.75,
        ),
        adv_results=[
            AdvantageResult(extra_metrics={"dg_eta": 0.2}),
            AdvantageResult(extra_metrics={"dg_eta": 0.4, "aux": 1.0}),
        ],
        echo_build=EchoBuildStats(
            candidate_datums=3,
            candidate_tokens=7,
            observation_mask_datums=2,
            skipped_low_overlap=1,
            observation_responses=4,
            bridged_transition_datums=3,
            bridge_failures=1,
            renderer_parity_failures=1,
            terminal_candidate_tokens=2,
            explicit_transition_rollouts=2,
        ),
        sampled_completion_token_count=16,
        eligible_completion_token_count=12,
        pre_optimizer_nonzero_advantage_token_count=8,
        optimizer_nonzero_advantage_token_count=6,
        rl_completion_token_count=12,
        echo_eligible_rollout_count=2,
        optimizer_logical_batch_sha256="a" * 64,
        rollout_timing_metrics={"rollout/total_s": 3.0},
        sample_time_s=2.0,
        behavior_turns=4,
        behavior_invalid=1,
        behavior_actions={"query": 3, "edit": 1},
        behavior_resp_lens=[10, 14],
    )
    echo_plan = EchoStepPlan(
        limit=EchoLimitStats(
            kept_datums=2,
            kept_tokens=5,
            kept_terminal_tokens=1,
            truncated_tokens=1,
        ),
        allowed_tokens=8,
        reference_completion_tokens=10,
        skipped_entropy_floor=False,
        rl_completion_surprisal_mean=1.5,
        echo_completion_surprisal_mean=1.75,
    )
    config = TrainConfig(
        echo_enabled=True,
        echo_weight=0.25,
        echo_entropy_floor=0.05,
    )
    backend_caps = BackendCapabilities(
        reports_sync_loss=True,
        preserves_token_advantages=True,
        supports_checkpoint_resume=True,
        resume_runtime_dependent=False,
    )
    bp_decision = BackPressureDecision(
        action="hold",
        regime="stable",
        p_star=2.0,
        sigma=0.1,
        kappa=0.2,
        utilization=0.3,
        throughput=4.0,
    )

    metrics = build_step_metrics(
        step,
        config=config,
        backend_caps=backend_caps,
        rollout=rollout,
        echo_plan=echo_plan,
        bp_decision=bp_decision,
        batch_norm_metrics={"adv_batch_mean": 0.0},
        runtime_counters=RuntimeCounters(prompt_encode_calls=2),
        helper=_RuntimeMetricsHelper(),
    )
    wandb = build_wandb_metrics(
        step,
        adv_results=rollout.adv_results,
        batch_norm_metrics={"adv_batch_mean": 0.0},
        metrics=metrics,
    )
    emergence = build_emergence_step_entry(
        step,
        config=config,
        rollout=rollout,
        echo_plan=echo_plan,
        metrics=metrics,
    )

    assert metrics["exec_surprisal_mean"] == pytest.approx(3.0)
    assert metrics["post_plan_surprisal_var"] == pytest.approx(0.9)
    assert metrics["rollout/accounted_share_of_sample"] == pytest.approx(1.0)
    assert metrics["behavior/action_dominance"] == pytest.approx(0.75)
    assert metrics["behavior/avg_response_chars"] == pytest.approx(12.0)
    assert metrics["dg_eta"] == pytest.approx(0.3)
    assert metrics["aux"] == pytest.approx(1.0)
    assert metrics["backend/tokens_s"] == pytest.approx(12.0)
    assert metrics["process_max_rss_mb"] == pytest.approx(12.346)
    assert metrics["echo/token_ratio"] == pytest.approx(0.5)
    assert metrics["echo/observation_responses"] == 4
    assert metrics["echo/bridged_transition_datums"] == 3
    assert metrics["echo/bridge_failures"] == 1
    assert metrics["echo/renderer_parity_failures"] == 1
    assert metrics["echo/terminal_candidate_tokens"] == 2
    assert metrics["echo/terminal_kept_tokens"] == 1
    assert metrics["echo/explicit_transition_rollouts"] == 2
    assert metrics["rl/action_token_datumization_ratio"] == pytest.approx(1.0)
    assert metrics["rl/sampled_action_tokens"] == 16
    assert metrics["rl/pre_optimizer_nonzero_advantage_action_tokens"] == 8
    assert metrics["rl/optimizer_nonzero_advantage_action_tokens"] == 6
    assert metrics["rl/nonzero_advantage_action_tokens"] == 6
    assert metrics["optimizer/logical_batch_sha256"] == "a" * 64
    assert metrics["optimizer/batch_sha256"] == "a" * 64
    assert metrics["optimizer/local_effective_rows_sha256"] == "e" * 64

    assert wandb["train/dg_eta"] == pytest.approx(0.3)
    assert wandb["train/echo/kept_tokens"] == 5
    assert wandb["train/rl/datumized_action_tokens"] == 12
    assert wandb["train/optimizer/logical_batch_sha256"] == "a" * 64
    assert wandb["train/optimizer/batch_sha256"] == "a" * 64
    assert wandb["train/optimizer/local_effective_rows_sha256"] == "e" * 64
    assert wandb["train/behavior/action_dominance"] == pytest.approx(0.75)
    assert wandb["train/tl_grpo_ema_baseline"] == pytest.approx(0.7)
    assert wandb["train/rewards/mean_reward"] == metrics["mean_reward"]
    assert wandb["train/sample_time_s"] == pytest.approx(2.0)
    assert wandb["train/train_time_s"] == pytest.approx(3.0)
    assert wandb["train/tokens_per_step"] == 100
    assert wandb["train/tokens_per_second"] == pytest.approx(10.0)
    assert wandb["train/train_share"] == pytest.approx(0.3)
    assert wandb["train/train_time_semantics"] == "synchronous_optimizer_update"
    assert wandb["train/backpressure/warmup"] == 0
    assert type(wandb["train/backpressure/warmup"]) is int
    assert wandb["train/backend/reports_sync_loss"] == 1
    assert type(wandb["train/backend/reports_sync_loss"]) is int
    assert wandb["train/backend/local/wall_s"] == pytest.approx(3.0)
    assert wandb["train/backend/local/microbatches"] == 5
    assert wandb["train/backend/local/padding_avoidance_fraction"] == pytest.approx(0.4)
    assert wandb[
        "train/backend/local/attention_proxy_avoidance_fraction"
    ] == pytest.approx(0.6)
    assert wandb["train/backend/local/sequence_length_p95"] == 512

    prime_config = TrainConfig(backend="prime_rl")
    prime_caps = BackendCapabilities(
        reports_sync_loss=False,
        preserves_token_advantages=False,
        supports_checkpoint_resume=True,
        resume_runtime_dependent=False,
    )
    prime_metrics = build_step_metrics(
        step,
        config=prime_config,
        backend_caps=prime_caps,
        rollout=rollout,
        echo_plan=echo_plan,
        bp_decision=bp_decision,
        batch_norm_metrics={},
        runtime_counters=RuntimeCounters(),
        helper=_RuntimeMetricsHelper(),
    )
    prime_wandb = build_wandb_metrics(
        step,
        adv_results=rollout.adv_results,
        batch_norm_metrics={},
        metrics=prime_metrics,
    )
    assert prime_metrics["train_time_semantics"] == "submit_enqueue_latency"
    assert prime_metrics["train_submit_enqueue_time_s"] == pytest.approx(3.0)
    assert prime_metrics["train_submit_enqueue_share"] == pytest.approx(0.3)
    assert "train_share" not in prime_metrics
    assert prime_wandb["train/train_time_semantics"] == "submit_enqueue_latency"
    assert prime_wandb["train/train_submit_enqueue_time_s"] == pytest.approx(3.0)
    assert "train/train_share" not in prime_wandb

    assert emergence["correct_count"] == 1
    assert emergence["total_count"] == 2
    assert emergence["dg_eta"] == pytest.approx(0.3)
    assert emergence["post_plan_surprisal_var"] == pytest.approx(0.9)
    assert format_step_log_summary(
        step,
        backend_caps=backend_caps,
        rollout=rollout,
    ) == (
        "Step 3 [grpo+none] | loss=0.1250 | reward=0.750"
        " | correct=50.0% | datums=5 | bs=2 | gs=4"
        " | tie_g=50.0% | sepa_l=0.1000 | time=10.0s"
    )


def test_wandb_keeps_zero_surprisal_defaults_when_jsonl_omits_them() -> None:
    data = StepLogData(
        step=1,
        condition_label="grpo+none",
        loss_value=0.0,
        echo_loss=0.0,
        echo_joint_optimizer_step=False,
        mean_reward=0.0,
        correct_rate=0.0,
        running_correct_rate=0.0,
        max_token_hit_rate=0.0,
        num_datums=0,
        step_time=1.0,
        sample_time=0.0,
        train_time=0.0,
        rl_train_time=0.0,
        echo_train_time=0.0,
        bp_total_tokens=0,
        batch_size=1,
        group_size=1,
        bp_warmup=True,
        sepa_lambda=0.0,
        sepa_gate=False,
        clip_fraction=0.0,
        policy_cov_fraction=0.0,
        policy_abs_kl=0.0,
        adv_cap_fraction=0.0,
        adv_cap_magnitude=0.0,
        tl_grpo_ema=None,
        surprisal=summarize_surprisal_stats([]),
    )
    metrics = {
        "loss": 0.0,
        "reported_loss": 0.0,
        "uncertainty_kind": "surprisal",
        "loss_is_placeholder": False,
        "bp_warmup": True,
        "echo/enabled": 0,
        "echo/loss": 0.0,
        "echo/train_time_s": 0.0,
        "echo/weight": 0.0,
        "echo/allowed_tokens": 0,
        "echo/reference_completion_tokens": 0,
        "echo/completion_surprisal_mean": 0.0,
        "echo/candidate_datums": 0,
        "echo/candidate_tokens": 0,
        "echo/observation_mask_datums": 0,
        "echo/observation_responses": 0,
        "echo/bridged_transition_datums": 0,
        "echo/bridge_failures": 0,
        "echo/renderer_parity_failures": 0,
        "echo/terminal_candidate_tokens": 0,
        "echo/terminal_kept_tokens": 0,
        "echo/explicit_transition_rollouts": 0,
        "echo/kept_datums": 0,
        "echo/kept_tokens": 0,
        "echo/truncated_tokens": 0,
        "echo/token_ratio": 0.0,
        "echo/skipped_low_overlap": 0,
        "echo/skipped_bad_observation_mask": 0,
        "echo/skipped_entropy_floor": 0,
        "echo/entropy_floor": 0.0,
        "echo/mode_collapse_guard": 0,
        "rl/train_time_s": 0.0,
        "rl/train_time_semantics": "synchronous_optimizer_update",
        "rl/completion_tokens": 0,
        "rl/sampled_action_tokens": 0,
        "rl/eligible_action_tokens": 0,
        "rl/datumized_action_tokens": 0,
        "rl/pre_optimizer_nonzero_advantage_action_tokens": 0,
        "rl/optimizer_nonzero_advantage_action_tokens": 0,
        "rl/nonzero_advantage_action_tokens": 0,
        "rl/action_token_datumization_ratio": 0.0,
        "rl/completion_surprisal_mean": 0.0,
        "echo/split_non_prefix": 0,
        "echo/eligible_rollouts": 0,
    }

    wandb = build_wandb_metrics(
        data,
        adv_results=[],
        batch_norm_metrics={},
        metrics=metrics,
    )

    assert "exec_surprisal_mean" not in metrics
    assert wandb["train/surprisal/exec_mean"] == 0.0
    assert wandb["train/surprisal/post_plan_var"] == 0.0
    assert wandb["train/backpressure/warmup"] == 1
    assert type(wandb["train/backpressure/warmup"]) is int
