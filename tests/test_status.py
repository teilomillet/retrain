"""Tests for retrain.status — log scanning and status reporting."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from retrain.status import (
    CampaignSummary,
    RunSummary,
    _run_cell,
    _truncate_condition,
    campaign_status,
    format_campaign,
    format_run,
    format_time,
    is_pid_alive,
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


class TestIsPidAlive:
    def test_current_process(self):
        assert is_pid_alive(os.getpid()) is True

    def test_zero_pid(self):
        assert is_pid_alive(0) is False

    def test_negative_pid(self):
        assert is_pid_alive(-1) is False

    def test_nonexistent_pid(self):
        # Use a very high PID that's extremely unlikely to exist
        assert is_pid_alive(4_000_000) is False


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
        assert "pid" in d
        assert "alive" in d


class TestScanCampaign:
    def test_no_manifest(self, tmp_path):
        assert scan_campaign(tmp_path) is None

    def test_basic_campaign(self, tmp_path):
        campaign_dir = tmp_path / "campaign_001"
        campaign_dir.mkdir()

        # Write manifest
        manifest = {
            "timestamp": "20260225_194813",
            "campaign_toml": "campaigns/test.toml",
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
        # Matrix now stores RunSummary objects
        run_42 = result.matrix["grpo+none"][42]
        assert run_42 is not None
        assert run_42.correct_rate == pytest.approx(0.45)
        run_101 = result.matrix["grpo+none"][101]
        assert run_101 is not None
        assert run_101.correct_rate == pytest.approx(0.30)
        assert result.matrix["maxrl+gtpo_sepa"][42] is None  # no metrics yet
        assert result.campaign_toml == "campaigns/test.toml"
        assert result.timestamp == "20260225_194813"
        assert result.status in ("running", "partial", "dead", "done")


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

    def test_newest_first_sort(self, tmp_path):
        """Campaigns should be sorted newest-first by directory name."""
        logs = tmp_path / "logs"
        for name in ["campaign_20260101_000000", "campaign_20260225_120000", "campaign_20260115_060000"]:
            d = logs / name
            d.mkdir(parents=True)
            manifest = {"conditions": [], "seeds": [], "max_steps": 10, "num_runs": 0, "runs": []}
            (d / "manifest.json").write_text(json.dumps(manifest))

        _, campaigns = scan_all(logs)
        paths = [c.path for c in campaigns]
        assert "20260225" in paths[0]
        assert "20260115" in paths[1]
        assert "20260101" in paths[2]


class TestCampaignStatus:
    def test_done_when_all_completed(self):
        runs = [
            RunSummary(path="r1", completed=True, step=99),
            RunSummary(path="r2", completed=True, step=99),
        ]
        assert campaign_status(runs, num_runs=2) == "done"

    def test_running_when_active_runs(self):
        runs = [
            RunSummary(path="r1", completed=False, stale=False, step=10),
            RunSummary(path="r2", completed=True, step=99),
        ]
        assert campaign_status(runs, num_runs=2) == "running"

    def test_partial_when_some_done_rest_stale(self):
        runs = [
            RunSummary(path="r1", completed=True, step=99),
            RunSummary(path="r2", completed=False, stale=True, step=50),
        ]
        assert campaign_status(runs, num_runs=2) == "partial"

    def test_dead_when_all_stale(self):
        runs = [
            RunSummary(path="r1", completed=False, stale=True, step=10),
            RunSummary(path="r2", completed=False, stale=True, step=20),
        ]
        assert campaign_status(runs, num_runs=2) == "dead"

    def test_dead_when_no_runs(self):
        assert campaign_status([], num_runs=4) == "dead"


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
        r42 = RunSummary(path="r1", step=99, completed=True, max_steps=100, correct_rate=0.45, wall_time_s=30.0)
        camp = CampaignSummary(
            path="logs/campaign_001",
            conditions=["grpo+none"],
            seeds=[42, 101],
            max_steps=100,
            num_runs=2,
            completed=1,
            matrix={"grpo+none": {42: r42, 101: None}},
            status="partial",
        )
        text = format_campaign(camp)
        assert "campaign_001" in text
        assert "grpo+none" in text
        assert "(partial)" in text


class TestCampaignWithStepProgress:
    """Matrix should show step/max_steps instead of correct_rate."""

    def test_matrix_shows_steps(self, tmp_path):
        campaign_dir = tmp_path / "campaign_step"
        campaign_dir.mkdir()

        manifest = {
            "conditions": ["cond_a"],
            "seeds": [42, 101],
            "max_steps": 100,
            "num_runs": 2,
            "runs": [
                {"condition": "cond_a", "seed": 42, "log_dir": str(campaign_dir / "runs" / "cond_a_s42")},
                {"condition": "cond_a", "seed": 101, "log_dir": str(campaign_dir / "runs" / "cond_a_s101")},
            ],
        }
        (campaign_dir / "manifest.json").write_text(json.dumps(manifest))

        # Run at step 41 (0-indexed), active
        _write_metrics(
            campaign_dir / "runs" / "cond_a_s42" / "metrics.jsonl",
            [{"step": i, "condition": "cond_a", "loss": 0.5, "correct_rate": 0.1, "mean_reward": 0.2, "step_time_s": 1.0} for i in range(42)],
        )
        # Run completed
        _write_metrics(
            campaign_dir / "runs" / "cond_a_s101" / "metrics.jsonl",
            [{"step": i, "condition": "cond_a", "loss": 0.3, "correct_rate": 0.4, "mean_reward": 0.5, "step_time_s": 1.0} for i in range(100)],
        )
        _write_trainer_state(
            campaign_dir / "runs" / "cond_a_s101" / "trainer_state.json",
            {"step": 99, "checkpoint_name": "final"},
        )

        result = scan_campaign(campaign_dir)
        text = format_campaign(result)
        # Active run shows step/max_steps
        assert "42/100" in text
        # Completed run shows step + checkmark
        assert "100 \u2713" in text

    def test_dead_run_shows_cross(self, tmp_path):
        campaign_dir = tmp_path / "campaign_dead"
        campaign_dir.mkdir()

        manifest = {
            "conditions": ["cond_a"],
            "seeds": [42],
            "max_steps": 100,
            "num_runs": 1,
            "runs": [
                {"condition": "cond_a", "seed": 42, "log_dir": str(campaign_dir / "runs" / "cond_a_s42")},
            ],
        }
        (campaign_dir / "manifest.json").write_text(json.dumps(manifest))

        metrics_path = campaign_dir / "runs" / "cond_a_s42" / "metrics.jsonl"
        _write_metrics(
            metrics_path,
            [{"step": i, "condition": "cond_a", "loss": 0.5, "correct_rate": 0.1, "mean_reward": 0.2, "step_time_s": 1.0} for i in range(13)],
        )
        # Make it stale
        old_time = time.time() - 600
        os.utime(metrics_path, (old_time, old_time))

        result = scan_campaign(campaign_dir)
        text = format_campaign(result)
        assert "13 \u2717" in text

    def test_not_started_shows_dash(self, tmp_path):
        campaign_dir = tmp_path / "campaign_notstarted"
        campaign_dir.mkdir()

        manifest = {
            "conditions": ["cond_a"],
            "seeds": [42],
            "max_steps": 100,
            "num_runs": 1,
            "runs": [
                {"condition": "cond_a", "seed": 42, "log_dir": str(campaign_dir / "runs" / "cond_a_s42")},
            ],
        }
        (campaign_dir / "manifest.json").write_text(json.dumps(manifest))
        # Don't create any metrics
        result = scan_campaign(campaign_dir)
        text = format_campaign(result)
        assert "\u2014" in text  # em-dash


class TestFormatTruncatesLongConditions:
    def test_short_condition_unchanged(self):
        assert _truncate_condition("grpo+none") == "grpo+none"

    def test_long_condition_truncated(self):
        label = "predictive_variance~surprisal~some_extra_long_label_here"
        result = _truncate_condition(label, max_width=30)
        assert len(result) <= 30
        assert result.startswith("...")
        assert result.endswith("_here")

    def test_exact_width_unchanged(self):
        label = "a" * 30
        assert _truncate_condition(label, max_width=30) == label

    def test_in_format_campaign(self):
        long_cond = "predictive_variance~surprisal~extra_long_condition_name"
        r = RunSummary(path="r1", step=10, max_steps=100, wall_time_s=5.0)
        camp = CampaignSummary(
            path="logs/campaign_trunc",
            conditions=[long_cond],
            seeds=[42],
            max_steps=100,
            num_runs=1,
            matrix={long_cond: {42: r}},
            status="running",
        )
        text = format_campaign(camp)
        assert "..." in text
        # Original long label should NOT appear as-is
        assert long_cond not in text


class TestActiveFilter:
    """The --active default should only show running/partial campaigns."""

    def test_filter_active_campaigns(self, tmp_path):
        logs = tmp_path / "logs"

        # Campaign 1: done (all completed)
        c1 = logs / "campaign_done"
        c1.mkdir(parents=True)
        m1 = {
            "conditions": ["cond_a"],
            "seeds": [42],
            "max_steps": 10,
            "num_runs": 1,
            "runs": [{"condition": "cond_a", "seed": 42, "log_dir": str(c1 / "runs" / "cond_a_s42")}],
        }
        (c1 / "manifest.json").write_text(json.dumps(m1))
        _write_metrics(
            c1 / "runs" / "cond_a_s42" / "metrics.jsonl",
            [{"step": i, "condition": "cond_a", "loss": 0.5, "correct_rate": 0.1, "mean_reward": 0.2, "step_time_s": 1.0} for i in range(10)],
        )
        _write_trainer_state(
            c1 / "runs" / "cond_a_s42" / "trainer_state.json",
            {"step": 9, "checkpoint_name": "final"},
        )

        # Campaign 2: running (active metrics)
        c2 = logs / "campaign_running"
        c2.mkdir(parents=True)
        m2 = {
            "conditions": ["cond_b"],
            "seeds": [42],
            "max_steps": 100,
            "num_runs": 1,
            "runs": [{"condition": "cond_b", "seed": 42, "log_dir": str(c2 / "runs" / "cond_b_s42")}],
        }
        (c2 / "manifest.json").write_text(json.dumps(m2))
        _write_metrics(
            c2 / "runs" / "cond_b_s42" / "metrics.jsonl",
            [{"step": i, "condition": "cond_b", "loss": 0.5, "correct_rate": 0.1, "mean_reward": 0.2, "step_time_s": 1.0} for i in range(10)],
        )

        _, all_campaigns = scan_all(logs)
        assert len(all_campaigns) == 2

        active = [c for c in all_campaigns if c.status in ("running", "partial")]
        assert len(active) == 1
        assert "running" in active[0].path

    def test_dead_campaign_filtered_out(self, tmp_path):
        logs = tmp_path / "logs"

        c1 = logs / "campaign_dead"
        c1.mkdir(parents=True)
        m1 = {
            "conditions": ["cond_a"],
            "seeds": [42],
            "max_steps": 100,
            "num_runs": 1,
            "runs": [{"condition": "cond_a", "seed": 42, "log_dir": str(c1 / "runs" / "cond_a_s42")}],
        }
        (c1 / "manifest.json").write_text(json.dumps(m1))
        metrics_path = c1 / "runs" / "cond_a_s42" / "metrics.jsonl"
        _write_metrics(
            metrics_path,
            [{"step": 5, "condition": "cond_a", "loss": 0.5, "correct_rate": 0.1, "mean_reward": 0.2, "step_time_s": 1.0}],
        )
        # Make stale
        old_time = time.time() - 600
        os.utime(metrics_path, (old_time, old_time))

        _, all_campaigns = scan_all(logs)
        assert len(all_campaigns) == 1
        assert all_campaigns[0].status == "dead"

        active = [c for c in all_campaigns if c.status in ("running", "partial")]
        assert len(active) == 0


class TestRunCell:
    def test_none_shows_dash(self):
        assert _run_cell(None, 100) == "\u2014"

    def test_completed_shows_checkmark(self):
        r = RunSummary(path="r", step=99, completed=True, max_steps=100)
        cell = _run_cell(r, 100)
        assert "\u2713" in cell
        assert "100" in cell

    def test_stale_shows_cross(self):
        r = RunSummary(path="r", step=12, stale=True, max_steps=-1)
        cell = _run_cell(r, 100)
        assert "\u2717" in cell
        assert "13" in cell

    def test_running_shows_fraction(self):
        r = RunSummary(path="r", step=41, max_steps=-1)
        cell = _run_cell(r, 100)
        assert cell == "42/100"
