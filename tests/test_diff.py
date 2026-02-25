"""Tests for retrain.diff — run comparison and sparkline rendering."""

import json
from pathlib import Path

import pytest

from retrain.diff import (
    DiffResult,
    MetricsEntry,
    _sparkline,
    _winner,
    diff_conditions,
    diff_runs,
    format_diff,
    load_metrics,
)


def _write_metrics(run_dir: Path, entries: list[dict]) -> None:
    """Write a synthetic metrics.jsonl file."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.jsonl", "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _make_entries(n: int, loss_start: float = 1.0, cr_end: float = 0.5) -> list[dict]:
    """Generate n synthetic metric entries with linear trends."""
    entries = []
    for i in range(n):
        t = i / max(n - 1, 1)
        entries.append(
            {
                "step": i,
                "loss": loss_start * (1 - 0.8 * t),
                "correct_rate": cr_end * t,
                "mean_reward": cr_end * t * 2,
                "step_time_s": 1.0,
            }
        )
    return entries


class TestLoadMetrics:
    def test_load_basic(self, tmp_path):
        entries = _make_entries(5)
        _write_metrics(tmp_path / "run_a", entries)

        loaded = load_metrics(tmp_path / "run_a")
        assert len(loaded) == 5
        assert loaded[0].step == 0
        assert loaded[-1].step == 4

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_metrics(tmp_path / "nonexistent")

    def test_load_skips_bad_lines(self, tmp_path):
        run_dir = tmp_path / "run_a"
        run_dir.mkdir()
        with open(run_dir / "metrics.jsonl", "w") as f:
            f.write('{"step": 0, "loss": 1.0, "correct_rate": 0.0, "mean_reward": 0.0, "step_time_s": 1.0}\n')
            f.write("not json\n")
            f.write('{"step": 1, "loss": 0.5, "correct_rate": 0.3, "mean_reward": 0.6, "step_time_s": 1.0}\n')
            f.write("\n")  # empty line

        loaded = load_metrics(run_dir)
        assert len(loaded) == 2
        assert loaded[0].step == 0
        assert loaded[1].step == 1


class TestSparkline:
    def test_empty(self):
        assert _sparkline([]) == ""

    def test_constant(self):
        result = _sparkline([1.0, 1.0, 1.0])
        assert len(result) == 3
        # All same value → all same char (mid-level)
        assert len(set(result)) == 1

    def test_increasing(self):
        result = _sparkline([0.0, 0.5, 1.0], width=3)
        # First char should be lowest, last should be highest
        assert result[0] == " "
        assert result[-1] == "█"

    def test_width_resampling(self):
        values = list(range(100))
        result = _sparkline([float(v) for v in values], width=10)
        assert len(result) == 10


class TestWinner:
    def test_loss_lower_wins(self):
        assert _winner("loss", 0.1, 0.5) == "<"
        assert _winner("loss", 0.5, 0.1) == ">"

    def test_correct_rate_higher_wins(self):
        assert _winner("correct_rate", 0.8, 0.5) == ">"
        assert _winner("correct_rate", 0.3, 0.7) == "<"

    def test_mean_reward_higher_wins(self):
        assert _winner("mean_reward", 1.0, 0.5) == ">"

    def test_equal(self):
        assert _winner("loss", 0.5, 0.5) == "="
        assert _winner("correct_rate", 0.5, 0.5) == "="


class TestDiffRuns:
    def test_basic_diff(self, tmp_path):
        _write_metrics(tmp_path / "a", _make_entries(10, loss_start=1.0, cr_end=0.5))
        _write_metrics(tmp_path / "b", _make_entries(10, loss_start=0.8, cr_end=0.7))

        result = diff_runs(tmp_path / "a", tmp_path / "b")
        assert result.steps_a == 10
        assert result.steps_b == 10
        assert result.final_a["correct_rate"] < result.final_b["correct_rate"]
        assert len(result.curve_a) == 10
        assert len(result.curve_b) == 10

    def test_diff_missing_run(self, tmp_path):
        _write_metrics(tmp_path / "a", _make_entries(5))
        with pytest.raises(FileNotFoundError):
            diff_runs(tmp_path / "a", tmp_path / "nonexistent")

    def test_wall_time(self, tmp_path):
        _write_metrics(tmp_path / "a", _make_entries(10))
        _write_metrics(tmp_path / "b", _make_entries(5))

        result = diff_runs(tmp_path / "a", tmp_path / "b")
        assert result.wall_time_a == pytest.approx(10.0)
        assert result.wall_time_b == pytest.approx(5.0)


class TestDiffConditions:
    def _setup_campaign(self, tmp_path, seeds=(42, 101)):
        """Create a campaign dir with two conditions and given seeds."""
        campaign_dir = tmp_path / "campaign"
        campaign_dir.mkdir()

        runs = []
        for cond in ("grpo+none", "maxrl+gtpo_sepa"):
            for seed in seeds:
                run_dir = campaign_dir / f"{cond}_s{seed}"
                cr_end = 0.5 if cond == "grpo+none" else 0.7
                _write_metrics(run_dir, _make_entries(10, cr_end=cr_end))
                runs.append(
                    {
                        "condition": cond,
                        "seed": seed,
                        "log_dir": str(run_dir),
                    }
                )

        manifest = {
            "conditions": ["grpo+none", "maxrl+gtpo_sepa"],
            "seeds": list(seeds),
            "max_steps": 10,
            "num_runs": len(runs),
            "runs": runs,
        }
        (campaign_dir / "manifest.json").write_text(json.dumps(manifest))
        return campaign_dir

    def test_basic_condition_diff(self, tmp_path):
        campaign_dir = self._setup_campaign(tmp_path)
        result = diff_conditions(campaign_dir, "grpo+none", "maxrl+gtpo_sepa")
        assert result.label_a == "grpo+none"
        assert result.label_b == "maxrl+gtpo_sepa"
        # maxrl+gtpo_sepa has higher correct_rate (0.7 vs 0.5)
        assert result.final_b["correct_rate"] > result.final_a["correct_rate"]

    def test_missing_condition(self, tmp_path):
        campaign_dir = self._setup_campaign(tmp_path)
        with pytest.raises(FileNotFoundError, match="No runs found"):
            diff_conditions(campaign_dir, "grpo+none", "nonexistent")

    def test_missing_manifest(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            diff_conditions(tmp_path, "a", "b")


class TestFormatDiff:
    def test_basic_format(self, tmp_path):
        _write_metrics(tmp_path / "a", _make_entries(10, loss_start=1.0, cr_end=0.5))
        _write_metrics(tmp_path / "b", _make_entries(10, loss_start=0.8, cr_end=0.7))

        result = diff_runs(tmp_path / "a", tmp_path / "b")
        text = format_diff(result)

        assert "A =" in text
        assert "B =" in text
        assert "loss" in text
        assert "correct_rate" in text
        assert "mean_reward" in text
        assert "wall_time" in text
        assert "steps" in text
        assert "correct_rate curves:" in text

    def test_format_shows_winner(self, tmp_path):
        _write_metrics(tmp_path / "a", _make_entries(10, loss_start=1.0, cr_end=0.3))
        _write_metrics(tmp_path / "b", _make_entries(10, loss_start=0.5, cr_end=0.8))

        result = diff_runs(tmp_path / "a", tmp_path / "b")
        text = format_diff(result)

        # B has lower loss and higher correct_rate → B wins both
        lines = text.splitlines()
        loss_line = [l for l in lines if "loss" in l and "correct" not in l][0]
        cr_line = [l for l in lines if "correct_rate" in l and "curves" not in l][0]
        assert ">" in loss_line  # A has higher loss → B wins → ">"
        assert "<" in cr_line  # A has lower cr → B wins → "<"


class TestDiffCli:
    def test_diff_two_runs(self, tmp_path, capsys):
        from retrain.cli import _run_diff

        _write_metrics(tmp_path / "a", _make_entries(5))
        _write_metrics(tmp_path / "b", _make_entries(5))

        _run_diff([str(tmp_path / "a"), str(tmp_path / "b")])
        out = capsys.readouterr().out
        assert "loss" in out
        assert "correct_rate" in out

    def test_diff_json(self, tmp_path, capsys):
        from retrain.cli import _run_diff

        _write_metrics(tmp_path / "a", _make_entries(5))
        _write_metrics(tmp_path / "b", _make_entries(5))

        _run_diff(["--json", str(tmp_path / "a"), str(tmp_path / "b")])
        payload = json.loads(capsys.readouterr().out)
        assert "final_a" in payload
        assert "final_b" in payload
        assert "curve_a" in payload

    def test_diff_bad_args(self, capsys):
        from retrain.cli import _run_diff

        with pytest.raises(SystemExit) as exc_info:
            _run_diff(["only_one_arg"])
        assert exc_info.value.code == 1

    def test_diff_unknown_flag(self, capsys):
        from retrain.cli import _run_diff

        with pytest.raises(SystemExit) as exc_info:
            _run_diff(["--bogus", "a", "b"])
        assert exc_info.value.code == 1
        assert "Unknown diff flag" in capsys.readouterr().err

    def test_diff_missing_run(self, tmp_path, capsys):
        from retrain.cli import _run_diff

        _write_metrics(tmp_path / "a", _make_entries(5))
        with pytest.raises(SystemExit) as exc_info:
            _run_diff([str(tmp_path / "a"), str(tmp_path / "nonexistent")])
        assert exc_info.value.code == 1
