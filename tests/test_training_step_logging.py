from __future__ import annotations

import sys
from collections.abc import Mapping
from types import SimpleNamespace

import pytest

from retrain.advantages import AdvantageResult
from retrain.backends.catalog import BackendCapabilities
from retrain.training.backpressure import BackPressureDecision
from retrain.config import TrainConfig
from retrain.training.echo import EchoBuildStats, EchoLimitStats
from retrain.training.rollouts import RuntimeCounters
from retrain.training.log import (
    StepLoggingContext,
    init_wandb,
    record_training_step,
)
from retrain.training.recoverability import (
    announce_checkpoint_recoverability,
    checkpoint_recoverability_wandb_metrics,
    upload_checkpoint_artifact,
)
from retrain.training.telemetry import EchoStepPlan


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

    def runtime_metrics(self) -> dict[str, float]:
        return {"backend/tokens_s": 9.0}


class _ListLogger:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def log(self, entry: dict[str, object]) -> None:
        self.entries.append(entry)


class _WandbRun:
    def __init__(self) -> None:
        self.id = "test-run"
        self.entries: list[tuple[dict[str, object], int | None]] = []
        self.artifacts: list[tuple[object, list[str] | None]] = []

    def log(
        self,
        data: Mapping[str, object],
        *,
        step: int | None = None,
    ) -> None:
        self.entries.append((dict(data), step))

    def log_artifact(
        self,
        artifact: object,
        *,
        aliases: list[str] | None = None,
    ) -> None:
        self.artifacts.append((artifact, aliases))

    def finish(self) -> None:
        pass


class _FailingArtifactWandbRun(_WandbRun):
    def log_artifact(
        self,
        artifact: object,
        *,
        aliases: list[str] | None = None,
    ) -> None:
        _ = artifact, aliases
        raise RuntimeError("upload unavailable")


class _FakeArtifact:
    def __init__(
        self,
        name: str,
        artifact_type: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.name = name
        self.type = artifact_type
        self.metadata = metadata or {}
        self.files: list[tuple[str, str | None]] = []
        self.dirs: list[tuple[str, str | None]] = []

    def add_file(self, local_path: str, *, name: str | None = None) -> None:
        self.files.append((local_path, name))

    def add_dir(self, local_path: str, *, name: str | None = None) -> None:
        self.dirs.append((local_path, name))


class _FakeWandbModule:
    def __init__(self) -> None:
        self.created: list[_FakeArtifact] = []

    def Artifact(
        self,
        name: str,
        artifact_type: str,
        metadata: dict[str, object] | None = None,
    ) -> _FakeArtifact:
        artifact = _FakeArtifact(name, artifact_type, metadata)
        self.created.append(artifact)
        return artifact


def test_init_wandb_returns_none_when_disabled() -> None:
    assert init_wandb(TrainConfig(), condition_label="grpo+none") is None


def test_record_training_step_writes_all_step_outputs(capsys) -> None:
    metrics_logger = _ListLogger()
    steps_logger = _ListLogger()
    wandb_run = _WandbRun()
    rollout = SimpleNamespace(
        rewards=[1.0, 0.0],
        correct=1,
        max_token_hits=1,
        total_completions=2,
        ties=SimpleNamespace(
            eligible_groups=1,
            tie_groups=0,
            uniform_groups=0,
            tie_group_rate=0.0,
            uniform_group_rate=0.0,
            tie_pair_rate=0.0,
            unique_fraction_mean=1.0,
        ),
        adv_results=[AdvantageResult(extra_metrics={"dg_eta": 0.6})],
        echo_build=EchoBuildStats(candidate_datums=1, candidate_tokens=4),
        sampled_completion_token_count=4,
        eligible_completion_token_count=4,
        pre_optimizer_nonzero_advantage_token_count=4,
        optimizer_nonzero_advantage_token_count=4,
        rl_completion_token_count=4,
        echo_eligible_rollout_count=1,
        optimizer_logical_batch_sha256="b" * 64,
        rollout_timing_metrics={},
        sample_time_s=1.25,
        behavior_turns=0,
        behavior_invalid=0,
        behavior_actions={},
        behavior_resp_lens=[],
        datum_tokens=[[1, 2, 3], [4]],
        surprisal_stats=[],
    )
    echo_plan = EchoStepPlan(
        limit=EchoLimitStats(kept_datums=1, kept_tokens=2),
        allowed_tokens=4,
        reference_completion_tokens=4,
        skipped_entropy_floor=False,
        rl_completion_surprisal_mean=1.5,
        echo_completion_surprisal_mean=1.75,
    )
    result = record_training_step(
        StepLoggingContext(
            step=7,
            condition_label="grpo+none",
            loss_value=0.25,
            echo_loss=0.1,
            echo_joint_optimizer_step=True,
            num_datums=2,
            total_correct=3,
            total_completions=6,
            step_time=4.0,
            train_time=2.0,
            rl_train_time=2.0,
            echo_train_time=2.0,
            bp_total_tokens=16,
            batch_size=2,
            group_size=2,
            bp_warmup=False,
            sepa_lambda=0.3,
            sepa_gate=True,
            clip_fraction=0.4,
            policy_cov_fraction=0.5,
            policy_abs_kl=0.6,
            adv_cap_fraction=0.7,
            adv_cap_magnitude=0.8,
            tl_grpo_ema=0.9,
            surprisal_stats=[],
        ),
        config=TrainConfig(
            echo_enabled=True,
            echo_weight=0.25,
            echo_target_retention="bounded",
        ),
        backend_caps=BackendCapabilities(
            reports_sync_loss=True,
            preserves_token_advantages=True,
            supports_checkpoint_resume=True,
            resume_runtime_dependent=False,
        ),
        rollout=rollout,
        echo_plan=echo_plan,
        bp_decision=BackPressureDecision(
            action="hold",
            regime="stable",
            p_star=2.0,
            sigma=0.1,
            kappa=0.2,
            utilization=0.3,
            throughput=4.0,
        ),
        batch_norm_metrics={},
        runtime_counters=RuntimeCounters(),
        helper=_RuntimeMetricsHelper(),
        metrics_logger=metrics_logger,
        steps_logger=steps_logger,
        wandb_run=wandb_run,
    )

    assert result.delight_eta_ema == pytest.approx(0.6)
    assert result.metrics is metrics_logger.entries[0]
    assert metrics_logger.entries[0]["num_datums"] == 2
    assert metrics_logger.entries[0]["running_correct_rate"] == pytest.approx(0.5)
    assert metrics_logger.entries[0]["backend/tokens_s"] == pytest.approx(9.0)
    assert (
        metrics_logger.entries[0]["optimizer/logical_batch_sha256"] == "b" * 64
    )
    assert metrics_logger.entries[0]["optimizer/batch_sha256"] == "b" * 64
    assert steps_logger.entries[0]["correct_count"] == 1
    assert steps_logger.entries[0]["total_count"] == 2
    assert steps_logger.entries[0]["dg_eta"] == pytest.approx(0.6)
    assert wandb_run.entries[0][1] == 7
    assert wandb_run.entries[0][0]["train/dg_eta"] == pytest.approx(0.6)
    assert (
        wandb_run.entries[0][0]["train/optimizer/logical_batch_sha256"]
        == "b" * 64
    )
    assert (
        wandb_run.entries[0][0]["train/optimizer/batch_sha256"] == "b" * 64
    )
    assert (
        wandb_run.entries[0][0][
            "train/recoverability/checkpoint_artifacts_enabled"
        ]
        == 1
    )
    assert wandb_run.entries[0][0]["train/recoverability/local_only"] == 0
    assert "Step 7 [grpo+none] | loss=0.2500" in capsys.readouterr().out


def test_recoverability_metrics_show_preemption_not_ready_without_periodic_checkpoints() -> None:
    with pytest.warns(UserWarning, match="save_every=0"):
        config = TrainConfig(wandb_project="proj", save_every=0)

    metrics = checkpoint_recoverability_wandb_metrics(config, _WandbRun())

    assert metrics["train/recoverability/checkpoint_artifacts_live"] == 1
    assert metrics["train/recoverability/periodic_checkpoints_enabled"] == 0
    assert metrics["train/recoverability/preemption_resume_ready"] == 0


def test_auto_checkpoint_artifacts_warn_when_periodic_checkpoints_disabled(
    capsys,
) -> None:
    with pytest.warns(UserWarning, match="save_every=0"):
        config = TrainConfig(wandb_project="proj", save_every=0)

    announce_checkpoint_recoverability(config, _WandbRun())

    out = capsys.readouterr().out
    assert "save_every=0 means only the final adapter will be uploaded" in out
    assert "Spot/preemptible runs cannot resume mid-run" in out


def test_upload_checkpoint_artifact_adds_adapter_and_state(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "adapters" / "checkpoint_step_2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_text("fake")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "trainer_state.json").write_text('{"step": 1}\n')
    (log_dir / "latest_sampler_path.txt").write_text(f"{checkpoint}\n")

    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    wandb_run = _WandbRun()
    config = TrainConfig(
        wandb_project="proj",
        log_dir=str(log_dir),
        adapter_path=str(tmp_path / "adapters"),
    )

    result = upload_checkpoint_artifact(
        config,
        wandb_run,
        checkpoint_name="checkpoint_step_2",
        checkpoint_path=str(checkpoint),
        step=1,
    )

    assert result.uploaded is True
    assert result.artifact_name == "retrain-test-run-checkpoints"
    assert len(wandb_run.artifacts) == 1
    artifact, aliases = wandb_run.artifacts[0]
    assert isinstance(artifact, _FakeArtifact)
    assert artifact.name == "retrain-test-run-checkpoints"
    assert artifact.metadata["checkpoint_name"] == "checkpoint_step_2"
    assert (str(checkpoint), "adapter") in artifact.dirs
    assert ("trainer_state.json" in [name for _, name in artifact.files])
    assert ("latest_sampler_path.txt" in [name for _, name in artifact.files])
    assert aliases == ["latest", "checkpoint_step_2"]
    assert (
        wandb_run.entries[-1][0][
            "train/recoverability/latest_checkpoint_uploaded"
        ]
        == 1
    )


def test_upload_checkpoint_artifact_auto_warns_on_failure(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    wandb_run = _FailingArtifactWandbRun()
    config = TrainConfig(
        wandb_project="proj",
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapters"),
    )

    result = upload_checkpoint_artifact(
        config,
        wandb_run,
        checkpoint_name="checkpoint_step_2",
        checkpoint_path="tinker://checkpoint_step_2",
        step=1,
    )

    assert result.uploaded is False
    assert result.reason == "RuntimeError"
    assert "WARNING: W&B checkpoint artifact upload failed" in capsys.readouterr().out
    assert (
        wandb_run.entries[-1][0][
            "train/recoverability/latest_checkpoint_uploaded"
        ]
        == 0
    )


def test_upload_checkpoint_artifact_required_raises_on_failure(
    tmp_path,
    monkeypatch,
) -> None:
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = TrainConfig(
        wandb_project="proj",
        checkpoint_artifacts="wandb",
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapters"),
    )

    with pytest.raises(RuntimeError, match="upload failed"):
        upload_checkpoint_artifact(
            config,
            _FailingArtifactWandbRun(),
            checkpoint_name="checkpoint_step_2",
            checkpoint_path="tinker://checkpoint_step_2",
            step=1,
        )


def test_required_checkpoint_artifacts_reject_offline_wandb(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    config = TrainConfig(wandb_project="proj", checkpoint_artifacts="wandb")

    with pytest.raises(RuntimeError, match="not live"):
        announce_checkpoint_recoverability(config, _WandbRun())
