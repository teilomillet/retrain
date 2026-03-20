"""Summarize Delight campaign runs into machine- and human-readable reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import pstdev

SUMMARY_JSON_NAME = "delight-summary.json"
SUMMARY_MD_NAME = "delight-summary.md"
_FINAL_EXTRA_FIELDS = (
    "dg_neutral_frac",
    "dg_breakthrough_frac",
    "dg_gate_ordering_gap",
    "dg_eta",
    "dg_eta_raw",
)


def _safe_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pstdev(values))


def _resolve_run_dir(campaign_dir: Path, log_dir: object) -> Path:
    if not isinstance(log_dir, str) or not log_dir:
        return campaign_dir
    run_dir = Path(log_dir)
    if run_dir.is_absolute():
        return run_dir

    candidates = [
        campaign_dir.parent.parent / run_dir,
        campaign_dir.parent / run_dir,
        campaign_dir / run_dir,
        Path.cwd() / run_dir,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_manifest(campaign_dir: Path) -> dict[str, object]:
    manifest_path = campaign_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No manifest.json in {campaign_dir}")
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid manifest.json in {campaign_dir}: expected JSON object"
        )
    return payload


def _load_metrics_entries(run_dir: Path) -> list[dict[str, object]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return []

    entries: list[dict[str, object]] = []
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def _summarize_run(run_meta: dict[str, object], campaign_dir: Path) -> dict[str, object] | None:
    run_dir = _resolve_run_dir(campaign_dir, run_meta.get("log_dir"))
    entries = _load_metrics_entries(run_dir)
    if not entries:
        return None

    final_entry = entries[-1]
    correct_rate_series = [
        value
        for entry in entries
        if (value := _safe_float(entry.get("correct_rate"))) is not None
    ]
    final_correct_rate = _safe_float(final_entry.get("correct_rate"))
    final_mean_reward = _safe_float(final_entry.get("mean_reward"))
    final_loss = _safe_float(final_entry.get("loss"))
    if final_correct_rate is None:
        return None

    summary: dict[str, object] = {
        "run_name": run_meta.get("run_name", run_dir.name),
        "seed": run_meta.get("seed"),
        "log_dir": str(run_dir),
        "num_steps": len(entries),
        "final_correct_rate": final_correct_rate,
        "peak_correct_rate": (
            max(correct_rate_series) if correct_rate_series else final_correct_rate
        ),
        "mean_correct_rate": _mean(correct_rate_series),
    }
    if final_mean_reward is not None:
        summary["final_mean_reward"] = final_mean_reward
    if final_loss is not None:
        summary["final_loss"] = final_loss
    for key in _FINAL_EXTRA_FIELDS:
        value = _safe_float(final_entry.get(key))
        if value is not None:
            summary[f"final_{key}"] = value
    return summary


def summarize_delight_campaign(campaign_dir: Path) -> dict[str, object]:
    campaign_dir = campaign_dir.resolve()
    manifest = _load_manifest(campaign_dir)
    raw_conditions = manifest.get("conditions", [])
    raw_runs = manifest.get("runs", [])
    conditions = [c for c in raw_conditions if isinstance(c, str)]
    runs = [r for r in raw_runs if isinstance(r, dict)]

    condition_rows: list[dict[str, object]] = []
    for idx, condition in enumerate(conditions, start=1):
        matching_runs = [
            run_meta for run_meta in runs
            if run_meta.get("condition") == condition
        ]
        run_summaries = [
            run_summary
            for run_meta in matching_runs
            if (run_summary := _summarize_run(run_meta, campaign_dir)) is not None
        ]

        row: dict[str, object] = {
            "condition_id": f"C{idx}",
            "condition": condition,
            "configured_runs": len(matching_runs),
            "completed_runs": len(run_summaries),
            "runs": run_summaries,
        }

        final_correct_rates = [
            float(run_summary["final_correct_rate"])
            for run_summary in run_summaries
        ]
        peak_correct_rates = [
            float(run_summary["peak_correct_rate"])
            for run_summary in run_summaries
        ]
        mean_correct_rates = [
            float(run_summary["mean_correct_rate"])
            for run_summary in run_summaries
        ]
        if final_correct_rates:
            row["final_correct_rate_mean"] = _mean(final_correct_rates)
            row["final_correct_rate_std"] = _std(final_correct_rates)
            row["peak_correct_rate_mean"] = _mean(peak_correct_rates)
            row["mean_correct_rate_mean"] = _mean(mean_correct_rates)

        final_mean_rewards = [
            float(run_summary["final_mean_reward"])
            for run_summary in run_summaries
            if "final_mean_reward" in run_summary
        ]
        if final_mean_rewards:
            row["final_mean_reward_mean"] = _mean(final_mean_rewards)

        final_losses = [
            float(run_summary["final_loss"])
            for run_summary in run_summaries
            if "final_loss" in run_summary
        ]
        if final_losses:
            row["final_loss_mean"] = _mean(final_losses)

        for key in _FINAL_EXTRA_FIELDS:
            values = [
                float(run_summary[f"final_{key}"])
                for run_summary in run_summaries
                if f"final_{key}" in run_summary
            ]
            if values:
                row[f"final_{key}_mean"] = _mean(values)

        condition_rows.append(row)

    baseline_final = _safe_float(
        condition_rows[0].get("final_correct_rate_mean") if condition_rows else None
    )
    for row in condition_rows:
        final_mean = _safe_float(row.get("final_correct_rate_mean"))
        if baseline_final is not None and final_mean is not None:
            row["final_correct_rate_delta_vs_baseline"] = final_mean - baseline_final

    completed_rows = [
        row for row in condition_rows
        if int(row.get("completed_runs", 0)) > 0 and "final_correct_rate_mean" in row
    ]
    best_final = (
        max(
            completed_rows,
            key=lambda row: float(row["final_correct_rate_mean"]),
        )
        if completed_rows
        else None
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_dir": str(campaign_dir),
        "campaign_toml": manifest.get("campaign_toml", ""),
        "baseline_condition": condition_rows[0]["condition"] if condition_rows else "",
        "conditions": condition_rows,
        "best_final_condition": {
            "condition_id": best_final["condition_id"],
            "condition": best_final["condition"],
            "final_correct_rate_mean": best_final["final_correct_rate_mean"],
        } if best_final is not None else None,
    }


def _fmt_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _fmt_percent_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "n/a"
    if std is None or std == 0.0:
        return _fmt_percent(mean)
    return f"{mean * 100:.1f}% ± {std * 100:.1f}%"


def _fmt_delta_pp(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.1f}pp"


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_delight_summary(summary: dict[str, object]) -> str:
    condition_rows = [
        row for row in summary.get("conditions", [])
        if isinstance(row, dict)
    ]
    lines = [
        "# Delight Campaign Summary",
        "",
        f"Campaign: `{summary.get('campaign_dir', '')}`",
        f"Source: `{summary.get('campaign_toml', '')}`",
    ]
    baseline_condition = summary.get("baseline_condition")
    if isinstance(baseline_condition, str) and baseline_condition:
        lines.append(f"Baseline: `{baseline_condition}`")

    best_final = summary.get("best_final_condition")
    if isinstance(best_final, dict):
        best_rate = _safe_float(best_final.get("final_correct_rate_mean"))
        lines.append(
            "Leader: "
            f"{best_final.get('condition_id', '')} "
            f"`{best_final.get('condition', '')}` "
            f"at {_fmt_percent(best_rate)} final correct rate"
        )

    lines.extend([
        "",
        "| ID | Condition | Runs | Final CR | Delta vs C1 | Peak CR | Mean CR | Neutral | Breakthrough | Ordering | Eta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for row in condition_rows:
        final_mean = _safe_float(row.get("final_correct_rate_mean"))
        final_std = _safe_float(row.get("final_correct_rate_std"))
        peak_mean = _safe_float(row.get("peak_correct_rate_mean"))
        mean_cr = _safe_float(row.get("mean_correct_rate_mean"))
        neutral = _safe_float(row.get("final_dg_neutral_frac_mean"))
        breakthrough = _safe_float(row.get("final_dg_breakthrough_frac_mean"))
        ordering = _safe_float(row.get("final_dg_gate_ordering_gap_mean"))
        eta = _safe_float(row.get("final_dg_eta_mean"))
        lines.append(
            "| "
            f"{row.get('condition_id', '')} | "
            f"`{row.get('condition', '')}` | "
            f"{row.get('completed_runs', 0)}/{row.get('configured_runs', 0)} | "
            f"{_fmt_percent_mean_std(final_mean, final_std)} | "
            f"{_fmt_delta_pp(_safe_float(row.get('final_correct_rate_delta_vs_baseline')))} | "
            f"{_fmt_percent(peak_mean)} | "
            f"{_fmt_percent(mean_cr)} | "
            f"{_fmt_percent(neutral)} | "
            f"{_fmt_percent(breakthrough)} | "
            f"{_fmt_float(ordering)} | "
            f"{_fmt_float(eta)} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_delight_summary(
    campaign_dir: Path,
    summary: dict[str, object],
) -> tuple[Path, Path]:
    campaign_dir = campaign_dir.resolve()
    json_path = campaign_dir / SUMMARY_JSON_NAME
    md_path = campaign_dir / SUMMARY_MD_NAME
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    md_path.write_text(render_delight_summary(summary) + "\n")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize a Delight campaign directory."
    )
    parser.add_argument("campaign_dir", help="Path to campaign log directory")
    args = parser.parse_args(argv)

    campaign_dir = Path(args.campaign_dir)
    summary = summarize_delight_campaign(campaign_dir)
    json_path, md_path = write_delight_summary(campaign_dir, summary)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
