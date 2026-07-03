"""Benchmark helpers for repeated retrain runs and run-level summaries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Protocol

from retrain.config import TrainConfig
from retrain.metrics_scan import float_or_none, int_or_none, scan_metrics_file


class _RunnerLike(Protocol):
    def run(self, config: TrainConfig) -> object: ...


@dataclass
class SummaryStat:
    mean: float
    stdev: float
    minimum: float
    maximum: float


@dataclass
class RunBenchmarkSummary:
    label: str
    path: str
    steps: int
    wall_time_s: float
    mean_step_time_s: float
    median_step_time_s: float
    p95_step_time_s: float
    mean_sample_time_s: float
    mean_train_time_s: float
    mean_sample_share: float
    mean_train_share: float
    mean_tokens_per_step: float
    mean_tokens_per_second: float
    peak_process_max_rss_mb: float | None
    generations_bytes: int
    final_loss: float | None
    final_mean_reward: float | None
    final_correct_rate: float | None
    prompt_encode_calls: int | None
    prompt_preview_calls: int | None
    token_lookup_requests: int | None
    token_lookup_convert_calls: int | None
    token_lookup_cache_misses: int | None
    batch_decode_calls: int | None
    batch_decoded_sequences: int | None
    engine_prompt_decode_calls: int | None
    engine_prompt_cache_hits: int | None
    engine_prompt_cache_size: int | None
    engine_token_prompt_calls: int | None
    engine_token_prompt_fallbacks: int | None
    engine_token_native_prompt_enabled: int | None
    engine_adapter_reload_calls: int | None
    engine_adapter_reload_failures: int | None
    engine_adapter_reload_skips: int | None
    engine_prefix_cache_hits: int | None
    engine_prefix_cache_misses: int | None
    engine_prefix_cache_fallbacks: int | None
    engine_prefix_cache_reused_tokens: int | None
    engine_prefix_cache_entries: int | None
    mean_engine_generation_wall_s: float | None = None
    mean_engine_prompt_prefill_s: float | None = None
    mean_engine_decode_s: float | None = None
    mean_engine_generation_tokens_per_s: float | None = None
    mean_local_sample_wall_s: float | None = None
    mean_local_sample_generation_tokens_per_s: float | None = None
    mean_local_train_forward_s: float | None = None
    mean_local_train_backward_s: float | None = None
    mean_local_train_optimizer_s: float | None = None
    mean_local_adapter_sync_s: float | None = None
    mean_rollout_total_s: float | None = None
    mean_rollout_generation_s: float | None = None
    mean_rollout_trajectory_step_s: float | None = None
    mean_rollout_score_s: float | None = None
    mean_rollout_scheduler_worker_s: float | None = None
    mean_rollout_scheduler_wait_s: float | None = None
    mean_rollout_scheduler_buffer_wait_s: float | None = None
    mean_rollout_env_dbt_total_s: float | None = None
    peak_local_sample_gpu_peak_memory_allocated_mb: float | None = None
    peak_local_sample_gpu_peak_memory_reserved_mb: float | None = None
    peak_local_train_gpu_peak_memory_allocated_mb: float | None = None
    peak_local_train_gpu_peak_memory_reserved_mb: float | None = None


@dataclass
class BenchmarkSuiteSummary:
    path: str
    repeats: int
    wandb_disabled: bool
    runs: list[RunBenchmarkSummary]
    aggregates: dict[str, SummaryStat]

def _summary_stat(values: list[float]) -> SummaryStat | None:
    if not values:
        return None
    return SummaryStat(
        mean=mean(values),
        stdev=pstdev(values) if len(values) > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
    )


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(0.95 * (len(ordered) - 1))))
    return ordered[index]


def summarize_run(run_dir: Path) -> RunBenchmarkSummary:
    """Summarize one completed run directory."""
    metrics_summary = scan_metrics_file(
        run_dir / "metrics.jsonl",
        mean_fields=(
            "sample_time_s",
            "train_time_s",
            "sample_share",
            "train_share",
            "tokens_per_step",
            "tokens_per_second",
            "engine_generation_wall_s",
            "engine_prompt_prefill_s",
            "engine_decode_s",
            "engine_generation_tokens_per_s",
            "local_sample_wall_s",
            "local_sample_generation_tokens_per_s",
            "local_train_forward_s",
            "local_train_backward_s",
            "local_train_optimizer_s",
            "local_adapter_sync_s",
            "rollout/total_s",
            "rollout/generation_s",
            "rollout/trajectory_step_s",
            "rollout/score_s",
            "rollout/scheduler_worker_s",
            "rollout/scheduler_wait_s",
            "rollout/scheduler_buffer_wait_s",
            "rollout/env/dbt_total_s",
        ),
        max_fields=(
            "process_max_rss_mb",
            "local_sample_gpu_peak_memory_allocated_mb",
            "local_sample_gpu_peak_memory_reserved_mb",
            "local_train_gpu_peak_memory_allocated_mb",
            "local_train_gpu_peak_memory_reserved_mb",
        ),
        collect_step_times=True,
    )
    if metrics_summary.rows <= 0 or metrics_summary.last is None:
        raise FileNotFoundError(f"No metrics.jsonl in {run_dir}")

    last = metrics_summary.last
    generations_path = run_dir / "emergence" / "generations.jsonl"
    generations_bytes = generations_path.stat().st_size if generations_path.is_file() else 0

    return RunBenchmarkSummary(
        label=run_dir.name,
        path=str(run_dir),
        steps=metrics_summary.rows,
        wall_time_s=metrics_summary.wall_time_s,
        mean_step_time_s=(
            metrics_summary.wall_time_s / metrics_summary.step_time_count
            if metrics_summary.step_time_count > 0 else 0.0
        ),
        median_step_time_s=median(metrics_summary.step_times) if metrics_summary.step_times else 0.0,
        p95_step_time_s=_p95(metrics_summary.step_times),
        mean_sample_time_s=metrics_summary.mean("sample_time_s") or 0.0,
        mean_train_time_s=metrics_summary.mean("train_time_s") or 0.0,
        mean_sample_share=metrics_summary.mean("sample_share") or 0.0,
        mean_train_share=metrics_summary.mean("train_share") or 0.0,
        mean_tokens_per_step=metrics_summary.mean("tokens_per_step") or 0.0,
        mean_tokens_per_second=metrics_summary.mean("tokens_per_second") or 0.0,
        peak_process_max_rss_mb=metrics_summary.maximum("process_max_rss_mb"),
        generations_bytes=generations_bytes,
        final_loss=float_or_none(last.get("loss")),
        final_mean_reward=float_or_none(last.get("mean_reward")),
        final_correct_rate=float_or_none(last.get("correct_rate")),
        prompt_encode_calls=int_or_none(last.get("prompt_encode_calls")),
        prompt_preview_calls=int_or_none(last.get("prompt_preview_calls")),
        token_lookup_requests=int_or_none(last.get("token_lookup_requests")),
        token_lookup_convert_calls=int_or_none(last.get("token_lookup_convert_calls")),
        token_lookup_cache_misses=int_or_none(last.get("token_lookup_cache_misses")),
        batch_decode_calls=int_or_none(last.get("batch_decode_calls")),
        batch_decoded_sequences=int_or_none(last.get("batch_decoded_sequences")),
        engine_prompt_decode_calls=int_or_none(last.get("engine_prompt_decode_calls")),
        engine_prompt_cache_hits=int_or_none(last.get("engine_prompt_cache_hits")),
        engine_prompt_cache_size=int_or_none(last.get("engine_prompt_cache_size")),
        engine_token_prompt_calls=int_or_none(last.get("engine_token_prompt_calls")),
        engine_token_prompt_fallbacks=int_or_none(
            last.get("engine_token_prompt_fallbacks")
        ),
        engine_token_native_prompt_enabled=int_or_none(
            last.get("engine_token_native_prompt_enabled")
        ),
        engine_adapter_reload_calls=int_or_none(
            last.get("engine_adapter_reload_calls")
        ),
        engine_adapter_reload_failures=int_or_none(
            last.get("engine_adapter_reload_failures")
        ),
        engine_adapter_reload_skips=int_or_none(
            last.get("engine_adapter_reload_skips")
        ),
        engine_prefix_cache_hits=int_or_none(last.get("engine_prefix_cache_hits")),
        engine_prefix_cache_misses=int_or_none(last.get("engine_prefix_cache_misses")),
        engine_prefix_cache_fallbacks=int_or_none(
            last.get("engine_prefix_cache_fallbacks")
        ),
        engine_prefix_cache_reused_tokens=int_or_none(
            last.get("engine_prefix_cache_reused_tokens")
        ),
        engine_prefix_cache_entries=int_or_none(last.get("engine_prefix_cache_entries")),
        mean_engine_generation_wall_s=metrics_summary.mean("engine_generation_wall_s"),
        mean_engine_prompt_prefill_s=metrics_summary.mean("engine_prompt_prefill_s"),
        mean_engine_decode_s=metrics_summary.mean("engine_decode_s"),
        mean_engine_generation_tokens_per_s=metrics_summary.mean(
            "engine_generation_tokens_per_s"
        ),
        mean_local_sample_wall_s=metrics_summary.mean("local_sample_wall_s"),
        mean_local_sample_generation_tokens_per_s=metrics_summary.mean(
            "local_sample_generation_tokens_per_s"
        ),
        mean_local_train_forward_s=metrics_summary.mean("local_train_forward_s"),
        mean_local_train_backward_s=metrics_summary.mean("local_train_backward_s"),
        mean_local_train_optimizer_s=metrics_summary.mean("local_train_optimizer_s"),
        mean_local_adapter_sync_s=metrics_summary.mean("local_adapter_sync_s"),
        mean_rollout_total_s=metrics_summary.mean("rollout/total_s"),
        mean_rollout_generation_s=metrics_summary.mean("rollout/generation_s"),
        mean_rollout_trajectory_step_s=metrics_summary.mean(
            "rollout/trajectory_step_s"
        ),
        mean_rollout_score_s=metrics_summary.mean("rollout/score_s"),
        mean_rollout_scheduler_worker_s=metrics_summary.mean(
            "rollout/scheduler_worker_s"
        ),
        mean_rollout_scheduler_wait_s=metrics_summary.mean(
            "rollout/scheduler_wait_s"
        ),
        mean_rollout_scheduler_buffer_wait_s=metrics_summary.mean(
            "rollout/scheduler_buffer_wait_s"
        ),
        mean_rollout_env_dbt_total_s=metrics_summary.mean("rollout/env/dbt_total_s"),
        peak_local_sample_gpu_peak_memory_allocated_mb=metrics_summary.maximum(
            "local_sample_gpu_peak_memory_allocated_mb"
        ),
        peak_local_sample_gpu_peak_memory_reserved_mb=metrics_summary.maximum(
            "local_sample_gpu_peak_memory_reserved_mb"
        ),
        peak_local_train_gpu_peak_memory_allocated_mb=metrics_summary.maximum(
            "local_train_gpu_peak_memory_allocated_mb"
        ),
        peak_local_train_gpu_peak_memory_reserved_mb=metrics_summary.maximum(
            "local_train_gpu_peak_memory_reserved_mb"
        ),
    )


def summarize_suite(root: Path) -> BenchmarkSuiteSummary:
    """Summarize a benchmark suite directory containing repeated runs."""
    if (root / "metrics.jsonl").is_file():
        runs = [summarize_run(root)]
    else:
        runs = [
            summarize_run(child)
            for child in sorted(root.iterdir())
            if child.is_dir() and (child / "metrics.jsonl").is_file()
        ]
    if not runs:
        raise FileNotFoundError(f"No run directories with metrics.jsonl in {root}")

    aggregates: dict[str, SummaryStat] = {}
    for field_name in (
        "wall_time_s",
        "mean_step_time_s",
        "mean_sample_time_s",
        "mean_train_time_s",
        "mean_sample_share",
        "mean_train_share",
        "mean_tokens_per_step",
        "mean_tokens_per_second",
        "generations_bytes",
        "mean_engine_generation_wall_s",
        "mean_engine_prompt_prefill_s",
        "mean_engine_decode_s",
        "mean_engine_generation_tokens_per_s",
        "mean_local_sample_wall_s",
        "mean_local_sample_generation_tokens_per_s",
        "mean_local_train_forward_s",
        "mean_local_train_backward_s",
        "mean_local_train_optimizer_s",
        "mean_local_adapter_sync_s",
        "mean_rollout_total_s",
        "mean_rollout_generation_s",
        "mean_rollout_trajectory_step_s",
        "mean_rollout_score_s",
        "mean_rollout_scheduler_worker_s",
        "mean_rollout_scheduler_wait_s",
        "mean_rollout_scheduler_buffer_wait_s",
        "mean_rollout_env_dbt_total_s",
    ):
        values = [
            float(value)
            for value in (getattr(run, field_name) for run in runs)
            if value is not None
        ]
        stat = _summary_stat(values)
        if stat is not None:
            aggregates[field_name] = stat

    for field_name in (
        "peak_process_max_rss_mb",
        "prompt_encode_calls",
        "prompt_preview_calls",
        "token_lookup_requests",
        "token_lookup_convert_calls",
        "token_lookup_cache_misses",
        "batch_decode_calls",
        "batch_decoded_sequences",
        "engine_prompt_decode_calls",
        "engine_prompt_cache_hits",
        "engine_prompt_cache_size",
        "engine_token_prompt_calls",
        "engine_token_prompt_fallbacks",
        "engine_token_native_prompt_enabled",
        "engine_adapter_reload_calls",
        "engine_adapter_reload_failures",
        "engine_adapter_reload_skips",
        "engine_prefix_cache_hits",
        "engine_prefix_cache_misses",
        "engine_prefix_cache_fallbacks",
        "engine_prefix_cache_reused_tokens",
        "engine_prefix_cache_entries",
        "peak_local_sample_gpu_peak_memory_allocated_mb",
        "peak_local_sample_gpu_peak_memory_reserved_mb",
        "peak_local_train_gpu_peak_memory_allocated_mb",
        "peak_local_train_gpu_peak_memory_reserved_mb",
    ):
        values = [
            float(value)
            for value in (getattr(run, field_name) for run in runs)
            if value is not None
        ]
        stat = _summary_stat(values)
        if stat is not None:
            aggregates[field_name] = stat

    return BenchmarkSuiteSummary(
        path=str(root),
        repeats=len(runs),
        wandb_disabled=False,
        runs=runs,
        aggregates=aggregates,
    )


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
        if hasattr(result, "to_dict"):
            meta = {"trainer": run_config.trainer}
            meta.update(result.to_dict())
            meta_path.write_text(json.dumps(meta), encoding="utf-8")

        run_summaries.append(summarize_run(run_dir))

    suite = summarize_suite(output_dir)
    suite.wandb_disabled = disable_wandb
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(asdict(suite), indent=2),
        encoding="utf-8",
    )
    return suite


def format_run_summary(summary: RunBenchmarkSummary) -> str:
    peak_rss = (
        f"{summary.peak_process_max_rss_mb:.3f}"
        if summary.peak_process_max_rss_mb is not None
        else "n/a"
    )
    lines = [
        f"run: {summary.label}",
        f"path: {summary.path}",
        f"steps: {summary.steps}",
        f"wall_time_s: {summary.wall_time_s:.3f}",
        f"mean_step_time_s: {summary.mean_step_time_s:.3f}",
        f"mean_sample_time_s: {summary.mean_sample_time_s:.3f}",
        f"mean_train_time_s: {summary.mean_train_time_s:.3f}",
        f"mean_sample_share: {summary.mean_sample_share:.3f}",
        f"mean_train_share: {summary.mean_train_share:.3f}",
        f"mean_tokens_per_second: {summary.mean_tokens_per_second:.3f}",
        f"peak_process_max_rss_mb: {peak_rss}",
        f"generations_bytes: {summary.generations_bytes}",
    ]
    for key in (
        "mean_engine_generation_wall_s",
        "mean_engine_prompt_prefill_s",
        "mean_engine_decode_s",
        "mean_engine_generation_tokens_per_s",
        "mean_local_sample_wall_s",
        "mean_local_sample_generation_tokens_per_s",
        "mean_local_train_forward_s",
        "mean_local_train_backward_s",
        "mean_local_train_optimizer_s",
        "mean_local_adapter_sync_s",
        "mean_rollout_total_s",
        "mean_rollout_generation_s",
        "mean_rollout_trajectory_step_s",
        "mean_rollout_score_s",
        "mean_rollout_scheduler_worker_s",
        "mean_rollout_scheduler_wait_s",
        "mean_rollout_scheduler_buffer_wait_s",
        "mean_rollout_env_dbt_total_s",
        "peak_local_sample_gpu_peak_memory_allocated_mb",
        "peak_local_sample_gpu_peak_memory_reserved_mb",
        "peak_local_train_gpu_peak_memory_allocated_mb",
        "peak_local_train_gpu_peak_memory_reserved_mb",
    ):
        value = getattr(summary, key)
        if value is not None:
            lines.append(f"{key}: {value:.3f}")
    for key in (
        "prompt_encode_calls",
        "prompt_preview_calls",
        "token_lookup_requests",
        "token_lookup_convert_calls",
        "token_lookup_cache_misses",
        "batch_decode_calls",
        "batch_decoded_sequences",
        "engine_prompt_decode_calls",
        "engine_prompt_cache_hits",
        "engine_prompt_cache_size",
        "engine_token_prompt_calls",
        "engine_token_prompt_fallbacks",
        "engine_token_native_prompt_enabled",
        "engine_adapter_reload_calls",
        "engine_adapter_reload_failures",
        "engine_adapter_reload_skips",
        "engine_prefix_cache_hits",
        "engine_prefix_cache_misses",
        "engine_prefix_cache_fallbacks",
        "engine_prefix_cache_reused_tokens",
        "engine_prefix_cache_entries",
    ):
        value = getattr(summary, key)
        if value is not None:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def format_suite_summary(summary: BenchmarkSuiteSummary) -> str:
    lines = [
        f"benchmark: {summary.path}",
        f"repeats: {summary.repeats}",
        f"wandb_disabled: {summary.wandb_disabled}",
        "",
        "aggregate metrics:",
    ]
    for key in sorted(summary.aggregates):
        stat = summary.aggregates[key]
        lines.append(
            f"  {key}: mean={stat.mean:.3f} stdev={stat.stdev:.3f} "
            f"min={stat.minimum:.3f} max={stat.maximum:.3f}"
        )
    lines.append("")
    lines.append("runs:")
    for run in summary.runs:
        rss = (
            f"{run.peak_process_max_rss_mb:.3f}"
            if run.peak_process_max_rss_mb is not None
            else "n/a"
        )
        lines.append(
            f"  {run.label}: step={run.mean_step_time_s:.3f}s "
            f"sample={run.mean_sample_share:.3f} train={run.mean_train_share:.3f} "
            f"tok/s={run.mean_tokens_per_second:.3f} rss={rss}"
        )
    return "\n".join(lines)
