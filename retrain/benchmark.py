"""Benchmark helpers for repeated retrain runs and run-level summaries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Protocol

from retrain.config import TrainConfig


JsonDict = dict[str, object]


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


@dataclass
class BenchmarkSuiteSummary:
    path: str
    repeats: int
    wandb_disabled: bool
    runs: list[RunBenchmarkSummary]
    aggregates: dict[str, SummaryStat]


def _load_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _float_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _metric_list(rows: list[JsonDict], key: str, fallback_key: str | None = None) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _float_or_none(row.get(key))
        if value is None and fallback_key is not None:
            value = _float_or_none(row.get(fallback_key))
        if value is not None:
            values.append(value)
    return values


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
    metrics_rows = _load_jsonl(run_dir / "metrics.jsonl")
    if not metrics_rows:
        raise FileNotFoundError(f"No metrics.jsonl in {run_dir}")

    step_times = _metric_list(metrics_rows, "step_time_s", fallback_key="time_s")
    sample_times = _metric_list(metrics_rows, "sample_time_s")
    train_times = _metric_list(metrics_rows, "train_time_s")
    sample_shares = _metric_list(metrics_rows, "sample_share")
    train_shares = _metric_list(metrics_rows, "train_share")
    tokens_per_step = _metric_list(metrics_rows, "tokens_per_step")
    tokens_per_second = _metric_list(metrics_rows, "tokens_per_second")
    rss_values = _metric_list(metrics_rows, "process_max_rss_mb")
    last = metrics_rows[-1]
    generations_path = run_dir / "emergence" / "generations.jsonl"
    generations_bytes = generations_path.stat().st_size if generations_path.is_file() else 0

    return RunBenchmarkSummary(
        label=run_dir.name,
        path=str(run_dir),
        steps=len(metrics_rows),
        wall_time_s=sum(step_times),
        mean_step_time_s=mean(step_times) if step_times else 0.0,
        median_step_time_s=median(step_times) if step_times else 0.0,
        p95_step_time_s=_p95(step_times),
        mean_sample_time_s=mean(sample_times) if sample_times else 0.0,
        mean_train_time_s=mean(train_times) if train_times else 0.0,
        mean_sample_share=mean(sample_shares) if sample_shares else 0.0,
        mean_train_share=mean(train_shares) if train_shares else 0.0,
        mean_tokens_per_step=mean(tokens_per_step) if tokens_per_step else 0.0,
        mean_tokens_per_second=mean(tokens_per_second) if tokens_per_second else 0.0,
        peak_process_max_rss_mb=max(rss_values) if rss_values else None,
        generations_bytes=generations_bytes,
        final_loss=_float_or_none(last.get("loss")),
        final_mean_reward=_float_or_none(last.get("mean_reward")),
        final_correct_rate=_float_or_none(last.get("correct_rate")),
        prompt_encode_calls=_int_or_none(last.get("prompt_encode_calls")),
        prompt_preview_calls=_int_or_none(last.get("prompt_preview_calls")),
        token_lookup_requests=_int_or_none(last.get("token_lookup_requests")),
        token_lookup_convert_calls=_int_or_none(last.get("token_lookup_convert_calls")),
        token_lookup_cache_misses=_int_or_none(last.get("token_lookup_cache_misses")),
        batch_decode_calls=_int_or_none(last.get("batch_decode_calls")),
        batch_decoded_sequences=_int_or_none(last.get("batch_decoded_sequences")),
        engine_prompt_decode_calls=_int_or_none(last.get("engine_prompt_decode_calls")),
        engine_prompt_cache_hits=_int_or_none(last.get("engine_prompt_cache_hits")),
        engine_prompt_cache_size=_int_or_none(last.get("engine_prompt_cache_size")),
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
    ):
        values = [float(getattr(run, field_name)) for run in runs]
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
        f"peak_process_max_rss_mb: "
        f"{summary.peak_process_max_rss_mb:.3f}" if summary.peak_process_max_rss_mb is not None
        else "peak_process_max_rss_mb: n/a",
        f"generations_bytes: {summary.generations_bytes}",
    ]
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
        lines.append(
            f"  {run.label}: step={run.mean_step_time_s:.3f}s "
            f"sample={run.mean_sample_share:.3f} train={run.mean_train_share:.3f} "
            f"tok/s={run.mean_tokens_per_second:.3f} rss="
            f"{run.peak_process_max_rss_mb:.3f}" if run.peak_process_max_rss_mb is not None
            else f"  {run.label}: step={run.mean_step_time_s:.3f}s "
            f"sample={run.mean_sample_share:.3f} train={run.mean_train_share:.3f} "
            f"tok/s={run.mean_tokens_per_second:.3f} rss=n/a"
        )
    return "\n".join(lines)
