"""Tests for retrain.status — log scanning and status reporting."""

import json
import time
from pathlib import Path

import pytest

from retrain.status import (
    CampaignSummary,
    RunSummary,
    format_campaign,
    format_run,
    format_time,
    scan_all,
    scan_campaign,
    scan_run,
)


def _write_metrics(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _write_trainer_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))


class TestFormatTime:
    def test_seconds(self):
        assert format_time(42) == "42s"

    def test_minutes(self):
        assert format_time(125) == "2m05s"

    def test_hours(self):
        assert format_time(3661) == "1h01m"


class TestScanRun:
    def test_no_metrics(self, tmp_path):
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        assert scan_run(run_dir) is None

    def test_basic_run(self, tmp_path):
        run_dir = tmp_path / "run1"
        _write_metrics(
            run_dir / "metrics.jsonl",
            [
                {"step": 0, "condition": "grpo+none", "loss": 1.0, "mean_reward": 0.1, "correct_rate": 0.05, "step_time_s": 10.0},
                {"step": 1, "condition": "grpo+none", "loss": 0.8, "mean_reward": 0.3, "correct_rate": 0.15, "step_time_s": 12.0},
            ],
        )
        result = scan_run(run_dir)
        assert result is not None
        assert result.step == 1
        assert result.condition == "grpo+none"
        assert result.loss == 0.8
        assert result.correct_rate == 0.15
        assert result.wall_time_s == pytest.approx(22.0)
        assert not result.completed

    def test_completed_run(self, tmp_path):
        run_dir = tmp_path / "run1"
        _write_metrics(
            run_dir / "metrics.jsonl",
            [{"step": 0, "condition": "grpo+none", "loss": 0.5, "correct_rate": 0.3, "mean_reward": 0.5, "step_time_s": 5.0}],
        )
        _write_trainer_state(
            run_dir / "trainer_state.json",
            {"step": 0, "checkpoint_name": "final"},
        )
        result = scan_run(run_dir)
        assert result is not None
        assert result.completed is True
        assert not result.stale

    def test_stale_run(self, tmp_path):
        run_dir = tmp_path / "run1"
        metrics_path = run_dir / "metrics.jsonl"
        _write_metrics(
            metrics_path,
            [{"step": 0, "condition": "test", "loss": 1.0, "correct_rate": 0.0, "mean_reward": 0.0, "step_time_s": 1.0}],
        )
        # Set mtime to 10 minutes ago
        old_time = time.time() - 600
        import os
        os.utime(metrics_path, (old_time, old_time))

        result = scan_run(run_dir)
        assert result is not None
        assert result.stale is True

    def test_to_dict(self, tmp_path):
        run_dir = tmp_path / "run1"
        _write_metrics(
            run_dir / "metrics.jsonl",
            [{"step": 0, "condition": "test", "loss": 1.0, "correct_rate": 0.0, "mean_reward": 0.0, "step_time_s": 1.0}],
        )
        result = scan_run(run_dir)
        d = result.to_dict()
        assert d["step"] == 0
        assert d["condition"] == "test"
        assert isinstance(d, dict)


class TestScanCampaign:
    def test_no_manifest(self, tmp_path):
        assert scan_campaign(tmp_path) is None

    def test_basic_campaign(self, tmp_path):
        campaign_dir = tmp_path / "campaign_001"
        campaign_dir.mkdir()

        # Write manifest
        manifest = {
            "conditions": ["grpo+none", "maxrl+gtpo_sepa"],
            "seeds": [42, 101],
            "max_steps": 100,
            "num_runs": 4,
            "runs": [
                {"condition": "grpo+none", "seed": 42, "run_name": "grpo+none_s42", "log_dir": str(campaign_dir / "runs" / "grpo+none_s42")},
                {"condition": "grpo+none", "seed": 101, "run_name": "grpo+none_s101", "log_dir": str(campaign_dir / "runs" / "grpo+none_s101")},
                {"condition": "maxrl+gtpo_sepa", "seed": 42, "run_name": "maxrl+gtpo_sepa_s42", "log_dir": str(campaign_dir / "runs" / "maxrl+gtpo_sepa_s42")},
                {"condition": "maxrl+gtpo_sepa", "seed": 101, "run_name": "maxrl+gtpo_sepa_s101", "log_dir": str(campaign_dir / "runs" / "maxrl+gtpo_sepa_s101")},
            ],
        }
        (campaign_dir / "manifest.json").write_text(json.dumps(manifest))

        # Write metrics for 2 of 4 runs
        _write_metrics(
            campaign_dir / "runs" / "grpo+none_s42" / "metrics.jsonl",
            [{"step": 99, "condition": "grpo+none", "loss": 0.3, "correct_rate": 0.45, "mean_reward": 0.6, "step_time_s": 5.0}],
        )
        _write_trainer_state(
            campaign_dir / "runs" / "grpo+none_s42" / "trainer_state.json",
            {"step": 99, "checkpoint_name": "final"},
        )
        _write_metrics(
            campaign_dir / "runs" / "grpo+none_s101" / "metrics.jsonl",
            [{"step": 50, "condition": "grpo+none", "loss": 0.5, "correct_rate": 0.30, "mean_reward": 0.4, "step_time_s": 3.0}],
        )

        result = scan_campaign(campaign_dir)
        assert result is not None
        assert result.num_runs == 4
        assert result.completed == 1
        assert len(result.runs) == 2
        assert result.matrix["grpo+none"][42] == pytest.approx(0.45)
        assert result.matrix["grpo+none"][101] == pytest.approx(0.30)
        assert result.matrix["maxrl+gtpo_sepa"][42] is None  # no metrics yet


class TestScanAll:
    def test_empty_dir(self, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        runs, campaigns = scan_all(logs)
        assert runs == []
        assert campaigns == []

    def test_nonexistent_dir(self, tmp_path):
        runs, campaigns = scan_all(tmp_path / "nope")
        assert runs == []
        assert campaigns == []

    def test_standalone_run(self, tmp_path):
        logs = tmp_path / "logs"
        run_dir = logs / "train"
        _write_metrics(
            run_dir / "metrics.jsonl",
            [{"step": 5, "condition": "grpo+none", "loss": 0.7, "correct_rate": 0.2, "mean_reward": 0.3, "step_time_s": 2.0}],
        )
        runs, campaigns = scan_all(logs)
        assert len(runs) == 1
        assert runs[0].step == 5
        assert campaigns == []


class TestFormatters:
    def test_format_run(self):
        run = RunSummary(
            path="logs/train",
            condition="grpo+none",
            step=10,
            correct_rate=0.25,
            loss=0.5,
            wall_time_s=120.0,
            completed=False,
        )
        text = format_run(run)
        assert "grpo+none" in text
        assert "running" in text

    def test_format_run_completed(self):
        run = RunSummary(
            path="logs/train",
            condition="grpo+none",
            step=99,
            correct_rate=0.5,
            loss=0.3,
            wall_time_s=600.0,
            completed=True,
        )
        text = format_run(run)
        assert "done" in text

    def test_format_campaign(self):
        camp = CampaignSummary(
            path="logs/campaign_001",
            conditions=["grpo+none"],
            seeds=[42, 101],
            max_steps=100,
            num_runs=2,
            completed=1,
            matrix={"grpo+none": {42: 0.45, 101: None}},
        )
        text = format_campaign(camp)
        assert "campaign_001" in text
        assert "grpo+none" in text
        assert "45.0%" in text
