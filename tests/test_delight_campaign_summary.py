"""Tests for Delight campaign summary generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrain.delight_campaign_summary import (
    SUMMARY_JSON_NAME,
    SUMMARY_MD_NAME,
    summarize_delight_campaign,
    write_delight_summary,
)


def _write_metrics(run_dir: Path, entries: list[dict[str, float]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.jsonl", "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_summarize_delight_campaign_writes_reports(tmp_path):
    campaign_dir = tmp_path / "logs" / "campaign_test"
    campaign_dir.mkdir(parents=True)

    cond_a_run = campaign_dir / "runs" / "grpo+none_s42"
    cond_b_run = campaign_dir / "runs" / "grpo+delight_s42"
    _write_metrics(
        cond_a_run,
        [
            {"step": 0, "loss": 1.0, "correct_rate": 0.30, "mean_reward": 0.30, "step_time_s": 1.0},
            {"step": 1, "loss": 0.9, "correct_rate": 0.40, "mean_reward": 0.40, "step_time_s": 1.0},
        ],
    )
    _write_metrics(
        cond_b_run,
        [
            {
                "step": 0,
                "loss": 0.9,
                "correct_rate": 0.35,
                "mean_reward": 0.35,
                "step_time_s": 1.0,
                "dg_neutral_frac": 0.82,
                "dg_breakthrough_frac": 0.03,
                "dg_gate_ordering_gap": 0.11,
                "dg_eta": 1.4,
            },
            {
                "step": 1,
                "loss": 0.7,
                "correct_rate": 0.55,
                "mean_reward": 0.55,
                "step_time_s": 1.0,
                "dg_neutral_frac": 0.68,
                "dg_breakthrough_frac": 0.08,
                "dg_gate_ordering_gap": 0.21,
                "dg_eta": 0.9,
            },
        ],
    )

    manifest = {
        "campaign_toml": "campaigns/delight-gate.toml",
        "conditions": ["grpo+none", "grpo+delight"],
        "runs": [
            {
                "condition": "grpo+none",
                "seed": 42,
                "run_name": "grpo+none_s42",
                "log_dir": "logs/campaign_test/runs/grpo+none_s42",
            },
            {
                "condition": "grpo+delight",
                "seed": 42,
                "run_name": "grpo+delight_s42",
                "log_dir": "logs/campaign_test/runs/grpo+delight_s42",
            },
        ],
    }
    (campaign_dir / "manifest.json").write_text(json.dumps(manifest) + "\n")

    summary = summarize_delight_campaign(campaign_dir)
    assert summary["baseline_condition"] == "grpo+none"
    assert summary["best_final_condition"]["condition_id"] == "C2"

    conditions = summary["conditions"]
    assert conditions[1]["final_correct_rate_delta_vs_baseline"] == pytest.approx(0.15)
    assert conditions[1]["final_dg_gate_ordering_gap_mean"] == pytest.approx(0.21)
    assert conditions[1]["final_dg_eta_mean"] == pytest.approx(0.9)

    json_path, md_path = write_delight_summary(campaign_dir, summary)
    assert json_path.name == SUMMARY_JSON_NAME
    assert md_path.name == SUMMARY_MD_NAME
    md_text = md_path.read_text()
    assert "C2" in md_text
    assert "Delta vs C1" in md_text


def test_summarize_delight_campaign_skips_bad_lines_and_computes_std(tmp_path):
    campaign_dir = tmp_path / "logs" / "campaign_stats"
    campaign_dir.mkdir(parents=True)

    cond_run_a = campaign_dir / "runs" / "grpo+delight_s1"
    cond_run_b = campaign_dir / "runs" / "grpo+delight_s2"
    cond_run_a.mkdir(parents=True, exist_ok=True)
    cond_run_b.mkdir(parents=True, exist_ok=True)

    with (cond_run_a / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 0, "correct_rate": 0.20, "loss": 1.0}) + "\n")
        handle.write("not json\n")
        handle.write(json.dumps({"step": 1, "correct_rate": 0.40, "loss": 0.8}) + "\n")

    with (cond_run_b / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 0, "correct_rate": 0.40, "loss": 0.9}) + "\n")
        handle.write(json.dumps({"step": 1, "correct_rate": 0.60, "loss": 0.7}) + "\n")

    manifest = {
        "campaign_toml": "campaigns/delight-stats.toml",
        "conditions": ["grpo+delight"],
        "runs": [
            {
                "condition": "grpo+delight",
                "seed": 1,
                "run_name": "grpo+delight_s1",
                "log_dir": "logs/campaign_stats/runs/grpo+delight_s1",
            },
            {
                "condition": "grpo+delight",
                "seed": 2,
                "run_name": "grpo+delight_s2",
                "log_dir": "logs/campaign_stats/runs/grpo+delight_s2",
            },
        ],
    }
    (campaign_dir / "manifest.json").write_text(json.dumps(manifest) + "\n")

    summary = summarize_delight_campaign(campaign_dir)
    condition = summary["conditions"][0]
    run_a = condition["runs"][0]

    assert run_a["num_steps"] == 2
    assert run_a["mean_correct_rate"] == pytest.approx(0.30)
    assert run_a["peak_correct_rate"] == pytest.approx(0.40)
    assert condition["final_correct_rate_mean"] == pytest.approx(0.50)
    assert condition["final_correct_rate_std"] == pytest.approx(0.10)
