"""Execute a benchmark suite: repeated runs of one config."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from retrain.config import TrainConfig
from retrain.benchmark.summary import (
    BenchmarkSuiteSummary,
    RunBenchmarkSummary,
    summarize_run,
    summarize_suite,
)


class _RunnerLike(Protocol):
    def run(self, config: TrainConfig) -> object: ...


def default_benchmark_output_dir(config_path: str, config: TrainConfig) -> Path:
    """Choose a deterministic benchmark suite root near the configured log dir."""
    stem = Path(config_path).stem or "benchmark"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base_log_dir = Path(config.log_dir)
    return base_log_dir.parent / f"{base_log_dir.name}-{stem}-benchmark-{ts}"


def run_benchmark_suite(
    config: TrainConfig,
    *,
    repeats: int,
    output_dir: Path,
    runner_factory,
    disable_wandb: bool = True,
) -> BenchmarkSuiteSummary:
    """Run a config repeatedly and summarize the resulting benchmark suite."""
    if repeats <= 0:
        raise ValueError("repeats must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_summaries: list[RunBenchmarkSummary] = []

    suite_meta = {
        "repeats": repeats,
        "base_log_dir": config.log_dir,
        "base_adapter_path": config.adapter_path,
        "wandb_disabled": disable_wandb,
    }
    (output_dir / "benchmark_meta.json").write_text(
        json.dumps(suite_meta, indent=2),
        encoding="utf-8",
    )

    for repeat_idx in range(repeats):
        run_dir = output_dir / f"repeat_{repeat_idx + 1:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = run_dir / "adapter"

        run_config = replace(
            config,
            log_dir=str(run_dir),
            adapter_path=str(adapter_dir),
        )
        if disable_wandb:
            run_config = replace(
                run_config,
                wandb_project="",
                wandb_run_name="",
                wandb_entity="",
                wandb_group="",
                wandb_tags="",
            )

        meta_path = run_dir / "run_meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "trainer": run_config.trainer,
                    "run_id": run_dir.name,
                    "status": "running",
                }
            ),
            encoding="utf-8",
        )

        runner: _RunnerLike = runner_factory(run_config)
        result = runner.run(run_config)
        if not getattr(result, "ok", False):
            failure_status = getattr(result, "failure_status", "")
            error_message = getattr(result, "error_message", "")
            raise RuntimeError(
                f"Benchmark run {run_dir.name} failed ({failure_status}): {error_message}"
            )
        result_to_dict = getattr(result, "to_dict", None)
        if callable(result_to_dict):
            meta = {"trainer": run_config.trainer}
            meta.update(result_to_dict())
            meta_path.write_text(json.dumps(meta), encoding="utf-8")

        run_summaries.append(summarize_run(run_dir))

    suite = summarize_suite(output_dir)
    suite.wandb_disabled = disable_wandb
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(suite.to_dict(), indent=2),
        encoding="utf-8",
    )
    return suite
