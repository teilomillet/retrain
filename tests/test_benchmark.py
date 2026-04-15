"""Tests for retrain.benchmark helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass

from retrain.benchmark import run_benchmark_suite, summarize_run, summarize_suite
from retrain.config import TrainConfig


@dataclass
class _FakeRunResult:
    policy_ref: str
    ok: bool = True
    failure_status: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_ref": self.policy_ref,
            "status": "succeeded",
        }


class _FakeRunner:
    def run(self, config: TrainConfig) -> _FakeRunResult:
        run_dir = config.log_dir
        emergence_dir = f"{run_dir}/emergence"
        from pathlib import Path

        Path(emergence_dir).mkdir(parents=True, exist_ok=True)
        metrics_path = Path(run_dir) / "metrics.jsonl"
        metrics_path.write_text(
            json.dumps(
                {
                    "step": 0,
                    "loss": 0.5,
                    "mean_reward": 0.25,
                    "correct_rate": 0.5,
                    "step_time_s": 2.0,
                    "sample_time_s": 1.0,
                    "train_time_s": 0.7,
                    "sample_share": 0.5,
                    "train_share": 0.35,
                    "tokens_per_step": 80,
                    "tokens_per_second": 40.0,
                    "process_max_rss_mb": 256.0,
                    "prompt_encode_calls": 1,
                    "prompt_preview_calls": 1,
                    "token_lookup_requests": 8,
                    "token_lookup_convert_calls": 1,
                    "token_lookup_cache_misses": 4,
                    "batch_decode_calls": 1,
                    "batch_decoded_sequences": 2,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (Path(emergence_dir) / "generations.jsonl").write_text(
            '{"step": 0, "completion": "hi"}\n',
            encoding="utf-8",
        )
        assert config.wandb_project == ""
        return _FakeRunResult(policy_ref=config.adapter_path)


def test_summarize_run_reads_perf_fields(tmp_path) -> None:
    run_dir = tmp_path / "run"
    emergence_dir = run_dir / "emergence"
    emergence_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 0,
                "loss": 0.9,
                "mean_reward": 0.1,
                "correct_rate": 0.25,
                "step_time_s": 4.0,
                "sample_time_s": 3.0,
                "train_time_s": 0.5,
                "sample_share": 0.75,
                "train_share": 0.125,
                "tokens_per_step": 100,
                "tokens_per_second": 25.0,
                "process_max_rss_mb": 512.0,
                "engine_prompt_decode_calls": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (emergence_dir / "generations.jsonl").write_text("abc", encoding="utf-8")

    summary = summarize_run(run_dir)

    assert summary.steps == 1
    assert summary.wall_time_s == 4.0
    assert summary.generations_bytes == 3
    assert summary.mean_tokens_per_second == 25.0
    assert summary.peak_process_max_rss_mb == 512.0
    assert summary.engine_prompt_decode_calls == 3


def test_run_benchmark_suite_creates_repeated_runs(tmp_path) -> None:
    config = TrainConfig(
        log_dir=str(tmp_path / "base-log"),
        adapter_path=str(tmp_path / "base-adapter"),
        wandb_project="should_be_disabled",
    )

    suite = run_benchmark_suite(
        config,
        repeats=2,
        output_dir=tmp_path / "bench",
        runner_factory=lambda cfg: _FakeRunner(),
        disable_wandb=True,
    )

    assert suite.repeats == 2
    assert suite.wandb_disabled is True
    assert len(suite.runs) == 2
    assert (tmp_path / "bench" / "benchmark_summary.json").is_file()
    assert (tmp_path / "bench" / "repeat_01" / "metrics.jsonl").is_file()
    assert suite.aggregates["mean_step_time_s"].mean == 2.0


def test_summarize_suite_reads_repeat_directories(tmp_path) -> None:
    for idx in (1, 2):
        run_dir = tmp_path / f"repeat_{idx:02d}"
        emergence_dir = run_dir / "emergence"
        emergence_dir.mkdir(parents=True)
        (run_dir / "metrics.jsonl").write_text(
            json.dumps({"step": 0, "step_time_s": float(idx)}) + "\n",
            encoding="utf-8",
        )
        (emergence_dir / "generations.jsonl").write_text("x", encoding="utf-8")

    suite = summarize_suite(tmp_path)

    assert suite.repeats == 2
    assert suite.aggregates["wall_time_s"].mean == 1.5
