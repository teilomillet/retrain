"""Compare two training runs or campaign conditions.

Loads metrics.jsonl from run directories and produces structured
comparisons of final metrics, wall times, and score curves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from retrain.metrics.scan import float_or_none, int_or_none, iter_jsonl_objects


@dataclass
class MetricsEntry:
    """A single step from metrics.jsonl."""

    step: int
    loss: float
    correct_rate: float
    mean_reward: float
    step_time_s: float
    sample_time_s: float | None = None
    train_time_s: float | None = None
    tokens_per_second: float | None = None
    sample_share: float | None = None
    train_share: float | None = None
    process_max_rss_mb: float | None = None


@dataclass
class DiffResult:
    """Comparison between two runs or conditions."""

    label_a: str
    label_b: str
    final_a: dict[str, float] = field(default_factory=dict)
    final_b: dict[str, float] = field(default_factory=dict)
    perf_a: dict[str, float] = field(default_factory=dict)
    perf_b: dict[str, float] = field(default_factory=dict)
    wall_time_a: float = 0.0
    wall_time_b: float = 0.0
    steps_a: int = 0
    steps_b: int = 0
    curve_a: list[float] = field(default_factory=list)
    curve_b: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return the JSON payload without recursive dataclass deep-copying."""
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "final_a": dict(self.final_a),
            "final_b": dict(self.final_b),
            "perf_a": dict(self.perf_a),
            "perf_b": dict(self.perf_b),
            "wall_time_a": self.wall_time_a,
            "wall_time_b": self.wall_time_b,
            "steps_a": self.steps_a,
            "steps_b": self.steps_b,
            "curve_a": list(self.curve_a),
            "curve_b": list(self.curve_b),
        }


def _float_metric(row: dict[str, object], key: str, default: float = 0.0) -> float:
    """Read a required numeric metric for tolerant run comparisons."""
    value = float_or_none(row.get(key))
    return default if value is None else value


def _int_metric(row: dict[str, object], key: str, default: int = 0) -> int:
    """Read a required integer metric for tolerant run comparisons."""
    value = int_or_none(row.get(key))
    return default if value is None else value


def load_metrics(run_dir: Path) -> list[MetricsEntry]:
    """Read metrics.jsonl from a run directory into a list of MetricsEntry."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"No metrics.jsonl in {run_dir}")

    entries: list[MetricsEntry] = []
    for d in iter_jsonl_objects(metrics_path):
        entries.append(
            MetricsEntry(
                step=_int_metric(d, "step"),
                loss=_float_metric(d, "loss"),
                correct_rate=_float_metric(d, "correct_rate"),
                mean_reward=_float_metric(d, "mean_reward"),
                step_time_s=_float_metric(d, "step_time_s"),
                sample_time_s=float_or_none(d.get("sample_time_s")),
                train_time_s=float_or_none(d.get("train_time_s")),
                tokens_per_second=float_or_none(d.get("tokens_per_second")),
                sample_share=float_or_none(d.get("sample_share")),
                train_share=float_or_none(d.get("train_share")),
                process_max_rss_mb=float_or_none(d.get("process_max_rss_mb")),
            )
        )
    return entries


def _final_metrics(entries: list[MetricsEntry]) -> dict[str, float]:
    """Extract the last entry's metrics as a flat dict."""
    if not entries:
        return {"loss": 0.0, "correct_rate": 0.0, "mean_reward": 0.0}
    last = entries[-1]
    return {
        "loss": last.loss,
        "correct_rate": last.correct_rate,
        "mean_reward": last.mean_reward,
    }


def _wall_time(entries: list[MetricsEntry]) -> float:
    return sum(e.step_time_s for e in entries)


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _max_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present)


def _perf_metrics(entries: list[MetricsEntry]) -> dict[str, float]:
    if not entries:
        return {}
    perf: dict[str, float] = {
        "mean_step_time_s": _wall_time(entries) / len(entries),
    }
    optional_means = {
        "mean_sample_time_s": _mean_optional([e.sample_time_s for e in entries]),
        "mean_train_time_s": _mean_optional([e.train_time_s for e in entries]),
        "mean_tokens_per_second": _mean_optional([e.tokens_per_second for e in entries]),
        "mean_sample_share": _mean_optional([e.sample_share for e in entries]),
        "mean_train_share": _mean_optional([e.train_share for e in entries]),
    }
    for key, value in optional_means.items():
        if value is not None:
            perf[key] = value
    peak_rss = _max_optional([e.process_max_rss_mb for e in entries])
    if peak_rss is not None:
        perf["peak_process_max_rss_mb"] = peak_rss
    return perf


def diff_runs(dir_a: Path, dir_b: Path) -> DiffResult:
    """Compare two individual run directories."""
    entries_a = load_metrics(dir_a)
    entries_b = load_metrics(dir_b)

    return DiffResult(
        label_a=str(dir_a),
        label_b=str(dir_b),
        final_a=_final_metrics(entries_a),
        final_b=_final_metrics(entries_b),
        perf_a=_perf_metrics(entries_a),
        perf_b=_perf_metrics(entries_b),
        wall_time_a=_wall_time(entries_a),
        wall_time_b=_wall_time(entries_b),
        steps_a=len(entries_a),
        steps_b=len(entries_b),
        curve_a=[e.correct_rate for e in entries_a],
        curve_b=[e.correct_rate for e in entries_b],
    )


def diff_conditions(
    campaign_dir: Path, cond_a: str, cond_b: str
) -> DiffResult:
    """Compare two conditions in a campaign, averaging across seeds."""
    manifest_path = campaign_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No manifest.json in {campaign_dir}")

    manifest = json.loads(manifest_path.read_text())
    runs_meta = manifest.get("runs", [])

    def _collect(cond: str) -> list[list[MetricsEntry]]:
        result = []
        for run_meta in runs_meta:
            if run_meta.get("condition", "") != cond:
                continue
            log_dir = run_meta.get("log_dir", "")
            if not log_dir:
                continue
            try:
                result.append(load_metrics(Path(log_dir)))
            except FileNotFoundError:
                continue
        return result

    runs_a = _collect(cond_a)
    runs_b = _collect(cond_b)

    if not runs_a:
        raise FileNotFoundError(f"No runs found for condition '{cond_a}'")
    if not runs_b:
        raise FileNotFoundError(f"No runs found for condition '{cond_b}'")

    def _avg_finals(runs: list[list[MetricsEntry]]) -> dict[str, float]:
        finals = [_final_metrics(r) for r in runs]
        keys = finals[0].keys()
        return {k: sum(f[k] for f in finals) / len(finals) for k in keys}

    def _avg_wall(runs: list[list[MetricsEntry]]) -> float:
        return sum(_wall_time(r) for r in runs) / len(runs)

    def _avg_perf(runs: list[list[MetricsEntry]]) -> dict[str, float]:
        per_run = [_perf_metrics(run) for run in runs]
        keys = {key for metrics in per_run for key in metrics}
        result: dict[str, float] = {}
        for key in keys:
            vals = [metrics[key] for metrics in per_run if key in metrics]
            if vals:
                result[key] = sum(vals) / len(vals)
        return result

    def _avg_curve(runs: list[list[MetricsEntry]]) -> list[float]:
        max_len = max(len(r) for r in runs)
        curve = []
        for i in range(max_len):
            vals = [r[i].correct_rate for r in runs if i < len(r)]
            curve.append(sum(vals) / len(vals) if vals else 0.0)
        return curve

    def _avg_steps(runs: list[list[MetricsEntry]]) -> int:
        return sum(len(r) for r in runs) // len(runs)

    return DiffResult(
        label_a=cond_a,
        label_b=cond_b,
        final_a=_avg_finals(runs_a),
        final_b=_avg_finals(runs_b),
        perf_a=_avg_perf(runs_a),
        perf_b=_avg_perf(runs_b),
        wall_time_a=_avg_wall(runs_a),
        wall_time_b=_avg_wall(runs_b),
        steps_a=_avg_steps(runs_a),
        steps_b=_avg_steps(runs_b),
        curve_a=_avg_curve(runs_a),
        curve_b=_avg_curve(runs_b),
    )
