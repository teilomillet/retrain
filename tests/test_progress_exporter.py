"""Tests for retrain.progress_exporter."""

from __future__ import annotations

import json
import time
from pathlib import Path

from retrain.progress_exporter import collect_run_snapshots, render_prometheus_text


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_collect_run_snapshots_reads_metrics_and_diagnostics(tmp_path: Path) -> None:
    run_dir = tmp_path / "energy-rl-demo"
    run_dir.mkdir()
    (run_dir / "run_meta.json").write_text('{"trainer":"retrain"}', encoding="utf-8")
    _write_jsonl(
        run_dir / "metrics.jsonl",
        [
            {
                "step": 12,
                "mean_reward": 0.42,
                "loss": -0.3,
                "max_token_hit_rate": 0.125,
                "behavior/invalid_action_rate": 0.25,
                "behavior/avg_response_chars": 612.0,
                "step_time_s": 91.5,
                "tokens_per_second": 128.0,
                "sample_share": 0.7,
                "train_share": 0.2,
                "process_max_rss_mb": 2048.0,
            }
        ],
    )
    now = time.time()
    _write_jsonl(
        run_dir / "tinker_sample_diagnostics.jsonl",
        [
            {"event": "helper_initialized", "ts": now - 5},
            {"event": "dispatch", "prompt_idx": 0, "prompt_tokens": 8123, "ts": now - 4},
            {
                "event": "result",
                "prompt_idx": 0,
                "prompt_tokens": 8123,
                "completion_tokens": [640],
                "result_latency_s": 12.5,
                "status": "ok",
                "ts": now - 3,
            },
            {"event": "dispatch", "prompt_idx": 1, "prompt_tokens": 16600, "ts": now - 2},
            {
                "event": "result",
                "prompt_idx": 1,
                "prompt_tokens": 16600,
                "completion_tokens": [],
                "result_latency_s": 300.0,
                "status": "error",
                "error_type": "TimeoutError",
                "ts": now - 1,
            },
        ],
    )

    snapshots = collect_run_snapshots(tmp_path)
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.run == "energy-rl-demo"
    assert snapshot.trainer == "retrain"
    assert snapshot.metrics_present is True
    assert snapshot.active is True
    assert snapshot.latest_step == 12
    assert snapshot.latest_mean_reward == 0.42
    assert snapshot.latest_loss == -0.3
    assert snapshot.latest_max_token_hit_rate == 0.125
    assert snapshot.latest_invalid_action_rate == 0.25
    assert snapshot.latest_avg_response_chars == 612.0
    assert snapshot.latest_tokens_per_second == 128.0
    assert snapshot.latest_sample_share == 0.7
    assert snapshot.latest_train_share == 0.2
    assert snapshot.latest_process_max_rss_mb == 2048.0
    assert snapshot.recent_dispatch_count == 2
    assert snapshot.recent_result_count == 2
    assert snapshot.recent_error_count == 1
    assert snapshot.recent_timeout_count == 1
    assert snapshot.recent_prompt_tokens_last == 16600
    assert snapshot.recent_prompt_tokens_max == 16600
    assert snapshot.recent_completion_tokens_max == 640
    assert snapshot.recent_result_latency_max_s == 300.0


def test_render_prometheus_text_includes_expected_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "energy-sft-demo"
    run_dir.mkdir()
    (run_dir / "run_meta.json").write_text('{"trainer":"retrain"}', encoding="utf-8")
    _write_jsonl(
        run_dir / "tinker_sample_diagnostics.jsonl",
        [
            {"event": "helper_initialized", "ts": time.time() - 1},
            {"event": "dispatch", "prompt_idx": 0, "prompt_tokens": 4096, "ts": time.time() - 1},
            {
                "event": "result",
                "prompt_idx": 0,
                "prompt_tokens": 4096,
                "completion_tokens": [512],
                "result_latency_s": 22.0,
                "status": "ok",
                "ts": time.time() - 1,
            },
        ],
    )

    text = render_prometheus_text(collect_run_snapshots(tmp_path))
    assert "soma_retrain_run_info" in text
    assert 'run="energy-sft-demo"' in text
    assert "soma_retrain_recent_completion_tokens_max" in text
    assert "soma_retrain_sample_result_age_seconds" in text
    assert "soma_retrain_latest_tokens_per_second" in text
