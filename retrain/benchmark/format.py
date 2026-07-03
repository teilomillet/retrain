"""Render benchmark summaries for the terminal."""

from __future__ import annotations

from retrain.benchmark.summary import (
    BenchmarkSuiteSummary,
    RunBenchmarkSummary,
    SummaryStat,
)


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
