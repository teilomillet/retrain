"""Compare two training runs or campaign conditions.

Loads metrics.jsonl from run directories and produces structured
comparison tables with final metrics, wall times, and ASCII sparklines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from retrain.status import format_time


@dataclass
class MetricsEntry:
    """A single step from metrics.jsonl."""

    step: int
    loss: float
    correct_rate: float
    mean_reward: float
    step_time_s: float


@dataclass
class DiffResult:
    """Comparison between two runs or conditions."""

    label_a: str
    label_b: str
    final_a: dict[str, float] = field(default_factory=dict)
    final_b: dict[str, float] = field(default_factory=dict)
    wall_time_a: float = 0.0
    wall_time_b: float = 0.0
    steps_a: int = 0
    steps_b: int = 0
    curve_a: list[float] = field(default_factory=list)
    curve_b: list[float] = field(default_factory=list)


_SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 20) -> str:
    """Render a list of floats as an ASCII sparkline string."""
    if not values:
        return ""
    # Resample to width buckets
    n = len(values)
    if n <= width:
        sampled = values
    else:
        sampled = []
        for i in range(width):
            start = i * n // width
            end = (i + 1) * n // width
            bucket = values[start:end]
            sampled.append(sum(bucket) / len(bucket) if bucket else 0.0)

    lo = min(sampled)
    hi = max(sampled)
    span = hi - lo
    chars = []
    for v in sampled:
        if span == 0:
            idx = 4  # mid-level
        else:
            idx = int((v - lo) / span * (len(_SPARKLINE_CHARS) - 1))
            idx = min(idx, len(_SPARKLINE_CHARS) - 1)
        chars.append(_SPARKLINE_CHARS[idx])
    return "".join(chars)


def _winner(metric: str, val_a: float, val_b: float) -> str:
    """Return '<', '>', or '=' indicating which side is better.

    Loss: lower is better. All others: higher is better.
    """
    if abs(val_a - val_b) < 1e-9:
        return "="
    if metric == "loss":
        return "<" if val_a < val_b else ">"
    return ">" if val_a > val_b else "<"


def load_metrics(run_dir: Path) -> list[MetricsEntry]:
    """Read metrics.jsonl from a run directory into a list of MetricsEntry."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"No metrics.jsonl in {run_dir}")

    entries: list[MetricsEntry] = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append(
                MetricsEntry(
                    step=d.get("step", 0),
                    loss=d.get("loss", 0.0),
                    correct_rate=d.get("correct_rate", 0.0),
                    mean_reward=d.get("mean_reward", 0.0),
                    step_time_s=d.get("step_time_s", 0.0),
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


def diff_runs(dir_a: Path, dir_b: Path) -> DiffResult:
    """Compare two individual run directories."""
    entries_a = load_metrics(dir_a)
    entries_b = load_metrics(dir_b)

    return DiffResult(
        label_a=str(dir_a),
        label_b=str(dir_b),
        final_a=_final_metrics(entries_a),
        final_b=_final_metrics(entries_b),
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
        wall_time_a=_avg_wall(runs_a),
        wall_time_b=_avg_wall(runs_b),
        steps_a=_avg_steps(runs_a),
        steps_b=_avg_steps(runs_b),
        curve_a=_avg_curve(runs_a),
        curve_b=_avg_curve(runs_b),
    )


def format_diff(result: DiffResult) -> str:
    """Format a DiffResult as a human-readable comparison table."""
    lines: list[str] = []
    w_label = max(len(result.label_a), len(result.label_b), 8)
    header = f"  {'metric':>15s}  {'A':>{w_label}s}  {'B':>{w_label}s}  win"
    lines.append(f"  A = {result.label_a}")
    lines.append(f"  B = {result.label_b}")
    lines.append("")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    metrics_order = ["loss", "correct_rate", "mean_reward"]
    for m in metrics_order:
        va = result.final_a.get(m, 0.0)
        vb = result.final_b.get(m, 0.0)
        w = _winner(m, va, vb)
        if m == "correct_rate":
            fmt_a, fmt_b = f"{va:.1%}", f"{vb:.1%}"
        else:
            fmt_a, fmt_b = f"{va:.4f}", f"{vb:.4f}"
        lines.append(f"  {m:>15s}  {fmt_a:>{w_label}s}  {fmt_b:>{w_label}s}   {w}")

    # Wall time
    ta = format_time(result.wall_time_a)
    tb = format_time(result.wall_time_b)
    lines.append(f"  {'wall_time':>15s}  {ta:>{w_label}s}  {tb:>{w_label}s}")

    # Steps
    lines.append(f"  {'steps':>15s}  {str(result.steps_a):>{w_label}s}  {str(result.steps_b):>{w_label}s}")

    # Sparklines
    if result.curve_a or result.curve_b:
        lines.append("")
        lines.append("  correct_rate curves:")
        lines.append(f"    A: {_sparkline(result.curve_a)}")
        lines.append(f"    B: {_sparkline(result.curve_b)}")

    return "\n".join(lines)
