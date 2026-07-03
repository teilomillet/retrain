"""Render a run comparison as a terminal table."""

from __future__ import annotations

from retrain.diff.compute import DiffResult
from retrain.status.format import format_time


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
    if metric in {
        "loss",
        "mean_step_time_s",
        "mean_sample_time_s",
        "mean_train_time_s",
        "peak_process_max_rss_mb",
    }:
        return "<" if val_a < val_b else ">"
    return ">" if val_a > val_b else "<"


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

    perf_order = [
        ("mean_step_time_s", "s"),
        ("mean_sample_time_s", "s"),
        ("mean_train_time_s", "s"),
        ("mean_tokens_per_second", ""),
        ("mean_sample_share", "%"),
        ("mean_train_share", "%"),
        ("peak_process_max_rss_mb", "MB"),
    ]
    perf_lines: list[str] = []
    for metric, unit in perf_order:
        if metric not in result.perf_a and metric not in result.perf_b:
            continue
        va = result.perf_a.get(metric)
        vb = result.perf_b.get(metric)
        if va is None or vb is None:
            continue
        win = _winner(metric, va, vb)
        if unit == "%":
            fmt_a, fmt_b = f"{va:.1%}", f"{vb:.1%}"
        elif unit == "MB":
            fmt_a, fmt_b = f"{va:.1f}MB", f"{vb:.1f}MB"
        elif unit == "s":
            fmt_a, fmt_b = f"{va:.3f}s", f"{vb:.3f}s"
        else:
            fmt_a, fmt_b = f"{va:.3f}", f"{vb:.3f}"
        perf_lines.append(
            f"  {metric:>15s}  {fmt_a:>{w_label}s}  {fmt_b:>{w_label}s}   {win}"
        )
    if perf_lines:
        lines.append("")
        lines.append("  performance:")
        lines.extend(perf_lines)

    # Sparklines
    if result.curve_a or result.curve_b:
        lines.append("")
        lines.append("  correct_rate curves:")
        lines.append(f"    A: {_sparkline(result.curve_a)}")
        lines.append(f"    B: {_sparkline(result.curve_b)}")

    return "\n".join(lines)
