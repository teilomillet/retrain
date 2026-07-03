"""Tests for benchmark summaries, execution, and rendering."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import cast

import pytest

from retrain.benchmark.format import format_run_summary, format_suite_summary
from retrain.benchmark.run import run_benchmark_suite
from retrain.benchmark.summary import (
    BenchmarkSuiteSummary,
    RunBenchmarkSummary,
    SummaryStat,
    summarize_run,
    summarize_suite,
)
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


@dataclass
class _NonCallableToDictRunResult:
    policy_ref: str
    ok: bool = True
    failure_status: str = ""
    error_message: str = ""
    to_dict: int = 1


class _FakeRunner:
    def __init__(self, *, result_has_callable_to_dict: bool = True) -> None:
        self._result_has_callable_to_dict = result_has_callable_to_dict

    def run(
        self,
        config: TrainConfig,
    ) -> _FakeRunResult | _NonCallableToDictRunResult:
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
                    "engine_adapter_reload_calls": 1,
                    "engine_adapter_reload_failures": 0,
                    "engine_adapter_reload_skips": 0,
                    "engine_generation_wall_s": 0.8,
                    "engine_prompt_prefill_s": 0.2,
                    "engine_decode_s": 0.6,
                    "engine_generation_tokens_per_s": 50.0,
                    "local_sample_wall_s": 0.9,
                    "local_sample_generation_tokens_per_s": 44.0,
                    "local_train_forward_s": 0.3,
                    "local_train_backward_s": 0.2,
                    "local_train_optimizer_s": 0.05,
                    "local_adapter_sync_s": 0.01,
                    "rollout/total_s": 1.0,
                    "rollout/trajectory_step_s": 0.3,
                    "rollout/scheduler_worker_s": 0.4,
                    "rollout/env/dbt_total_s": 0.25,
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
        if not self._result_has_callable_to_dict:
            return _NonCallableToDictRunResult(policy_ref=config.adapter_path)
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
                "engine_adapter_reload_calls": 2,
                "engine_adapter_reload_failures": 0,
                "engine_adapter_reload_skips": 0,
                "engine_generation_wall_s": 2.5,
                "engine_prompt_prefill_s": 0.5,
                "engine_decode_s": 2.0,
                "engine_generation_tokens_per_s": 32.0,
                "local_train_forward_s": 0.2,
                "local_train_backward_s": 0.25,
                "local_train_optimizer_s": 0.05,
                "local_adapter_sync_s": 0.01,
                "local_train_gpu_peak_memory_allocated_mb": 1024.0,
                "rollout/total_s": 3.0,
                "rollout/trajectory_step_s": 1.25,
                "rollout/scheduler_worker_s": 1.5,
                "rollout/env/dbt_total_s": 0.75,
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
    assert summary.engine_adapter_reload_calls == 2
    assert summary.engine_adapter_reload_failures == 0
    assert summary.engine_adapter_reload_skips == 0
    assert summary.mean_engine_prompt_prefill_s == 0.5
    assert summary.mean_engine_decode_s == 2.0
    assert summary.mean_local_train_forward_s == 0.2
    assert summary.peak_local_train_gpu_peak_memory_allocated_mb == 1024.0
    assert summary.mean_rollout_total_s == 3.0
    assert summary.mean_rollout_trajectory_step_s == 1.25
    assert summary.mean_rollout_env_dbt_total_s == 0.75

    assert "peak_process_max_rss_mb: 512.000" in format_run_summary(summary)


def test_summarize_run_aggregates_multiple_rows(tmp_path) -> None:
    run_dir = tmp_path / "run"
    emergence_dir = run_dir / "emergence"
    emergence_dir.mkdir(parents=True)
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "step": 0,
                    "loss": 0.9,
                    "mean_reward": 0.1,
                    "correct_rate": 0.2,
                    "step_time_s": 2.0,
                    "sample_time_s": 1.0,
                    "train_time_s": 0.5,
                    "sample_share": 0.5,
                    "train_share": 0.25,
                    "tokens_per_step": 100,
                    "tokens_per_second": 50.0,
                    "process_max_rss_mb": 256.0,
                    "engine_prompt_prefill_s": 0.2,
                    "engine_decode_s": 0.8,
                    "local_train_forward_s": 0.3,
                    "local_train_backward_s": 0.2,
                    "rollout/total_s": 2.0,
                    "rollout/env/dbt_total_s": 0.4,
                }
            )
            + "\n"
        )
        handle.write("not json\n")
        handle.write(
            json.dumps(
                {
                    "step": 1,
                    "loss": 0.4,
                    "mean_reward": 0.7,
                    "correct_rate": 0.8,
                    "step_time_s": 4.0,
                    "sample_time_s": 3.0,
                    "train_time_s": 1.5,
                    "sample_share": 0.75,
                    "train_share": 0.2,
                    "tokens_per_step": 140,
                    "tokens_per_second": 70.0,
                    "process_max_rss_mb": 768.0,
                    "prompt_encode_calls": 8,
                    "engine_prompt_prefill_s": 0.4,
                    "engine_decode_s": 1.6,
                    "local_train_forward_s": 0.6,
                    "local_train_backward_s": 0.4,
                    "rollout/total_s": 4.0,
                    "rollout/env/dbt_total_s": 0.8,
                }
            )
            + "\n"
        )
    (emergence_dir / "generations.jsonl").write_text("abcd", encoding="utf-8")

    summary = summarize_run(run_dir)

    assert summary.steps == 2
    assert summary.wall_time_s == 6.0
    assert summary.mean_step_time_s == 3.0
    assert summary.median_step_time_s == 3.0
    assert summary.mean_sample_time_s == 2.0
    assert summary.mean_train_time_s == 1.0
    assert summary.mean_sample_share == 0.625
    assert summary.mean_train_share == 0.225
    assert summary.mean_tokens_per_step == 120.0
    assert summary.mean_tokens_per_second == 60.0
    assert summary.peak_process_max_rss_mb == 768.0
    assert summary.final_loss == 0.4
    assert summary.final_correct_rate == 0.8
    assert summary.prompt_encode_calls == 8
    assert summary.mean_engine_prompt_prefill_s == pytest.approx(0.3)
    assert summary.mean_engine_decode_s == pytest.approx(1.2)
    assert summary.mean_local_train_forward_s == pytest.approx(0.45)
    assert summary.mean_local_train_backward_s == pytest.approx(0.3)
    assert summary.mean_rollout_total_s == pytest.approx(3.0)
    assert summary.mean_rollout_env_dbt_total_s == pytest.approx(0.6)


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
    assert suite.aggregates["engine_adapter_reload_calls"].mean == 1.0
    assert suite.aggregates["engine_adapter_reload_failures"].mean == 0.0
    assert suite.aggregates["mean_engine_prompt_prefill_s"].mean == 0.2
    assert suite.aggregates["mean_rollout_env_dbt_total_s"].mean == 0.25

    payload = suite.to_dict()
    summary_json = json.loads((tmp_path / "bench" / "benchmark_summary.json").read_text())
    runs_payload = cast(list[dict[str, object]], payload["runs"])
    aggregates_payload = cast(dict[str, dict[str, object]], payload["aggregates"])
    assert payload == asdict(suite)
    assert summary_json == payload
    assert set(payload) == {f.name for f in fields(BenchmarkSuiteSummary)}
    assert set(runs_payload[0]) == {f.name for f in fields(RunBenchmarkSummary)}
    assert set(aggregates_payload["mean_step_time_s"]) == {
        f.name for f in fields(SummaryStat)
    }


def test_run_benchmark_suite_tolerates_non_callable_to_dict(tmp_path) -> None:
    config = TrainConfig(
        log_dir=str(tmp_path / "base-log"),
        adapter_path=str(tmp_path / "base-adapter"),
    )

    suite = run_benchmark_suite(
        config,
        repeats=1,
        output_dir=tmp_path / "bench",
        runner_factory=lambda cfg: _FakeRunner(result_has_callable_to_dict=False),
        disable_wandb=True,
    )

    assert suite.repeats == 1
    meta = json.loads((tmp_path / "bench" / "repeat_01" / "run_meta.json").read_text())
    assert meta == {
        "trainer": "retrain",
        "run_id": "repeat_01",
        "status": "running",
    }


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
    assert "rss=n/a" in format_suite_summary(suite)
