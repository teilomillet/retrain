"""Terminal rendering of run and campaign status."""

from __future__ import annotations

from pathlib import Path

from retrain.status.scan import CampaignSummary, RunSummary

_COND_MAX_WIDTH = 30  # max display width for condition labels


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m{s:02d}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def _truncate_condition(label: str, max_width: int = _COND_MAX_WIDTH) -> str:
    """Truncate a long condition label from the left, preserving the end."""
    if len(label) <= max_width:
        return label
    return "..." + label[-(max_width - 3):]


def _run_cell(
    run: RunSummary | None, max_steps: int, runner_alive: bool = False,
) -> str:
    """Format a single matrix cell showing step progress.

    When *runner_alive* is True (or the run's own PID is alive), a stale
    run is shown as ``step/max`` instead of ``step ✗`` because the process
    is still running — the step just takes longer than the staleness window.
    """
    if run is None:
        return "\u2014"  # em-dash
    effective_max = run.max_steps if run.max_steps > 0 else max_steps
    step = run.step + 1 if run.step >= 0 else 0  # step is 0-indexed in metrics
    if run.completed:
        return f"{step} \u2713"
    if run.stale and not run.alive and not runner_alive:
        return f"{step} \u2717"
    if effective_max > 0:
        return f"{step}/{effective_max}"
    return str(step)


def format_summary_banner(campaigns: list[CampaignSummary]) -> str:
    """One-line aggregate summary of all campaigns and runs.

    Counts pending runs from matrix (None entries).  Run classification:
    - done:    run.completed
    - active:  not completed and not stale; OR stale but alive (own or runner)
    - dead:    stale, not alive, not completed
    - pending: no metrics yet (None in matrix)
    """
    # Campaign-level counts
    camp_counts: dict[str, int] = {}
    for c in campaigns:
        camp_counts[c.status] = camp_counts.get(c.status, 0) + 1

    # Run-level counts across all campaigns
    # Only count pending from non-dead campaigns — dead campaigns' empty
    # slots are debris, not genuinely waiting runs.
    r_done = 0
    r_active = 0
    r_dead = 0
    r_pending = 0
    for c in campaigns:
        for cond, seeds in c.matrix.items():
            for seed, run in seeds.items():
                if run is None:
                    if c.status != "dead":
                        r_pending += 1
                elif run.completed:
                    r_done += 1
                elif run.stale and not run.alive and not c.runner_alive:
                    r_dead += 1
                else:
                    r_active += 1

    # Build campaign part — show non-zero counts in a fixed order
    camp_parts: list[str] = []
    for label in ("running", "partial", "done", "dead"):
        n = camp_counts.get(label, 0)
        if n:
            camp_parts.append(f"{n} {label}")

    # Build run part — always show in fixed order, skip zeros
    run_parts: list[str] = []
    for n, label in [(r_active, "active"), (r_dead, "dead"), (r_done, "done"), (r_pending, "pending")]:
        if n:
            run_parts.append(f"{n} {label}")

    camp_str = ", ".join(camp_parts) if camp_parts else "0 campaigns"
    run_str = ", ".join(run_parts) if run_parts else "0 runs"
    return f"Campaigns: {camp_str}  |  Runs: {run_str}"


def format_run(run: RunSummary) -> str:
    """Format a single run summary as a text line."""
    status = "done" if run.completed else ("stale" if run.stale else "running")
    cond = run.condition or "unknown"
    t = format_time(run.wall_time_s)
    perf_parts: list[str] = []
    if run.tokens_per_second > 0:
        perf_parts.append(f"tok/s={run.tokens_per_second:.1f}")
    if run.sample_share > 0:
        perf_parts.append(f"sample={run.sample_share:.0%}")
    if run.latest_step_time_s > 0:
        perf_parts.append(f"step_t={run.latest_step_time_s:.1f}s")
    if run.process_max_rss_mb > 0:
        perf_parts.append(f"rss={run.process_max_rss_mb:.0f}MB")
    perf_tag = f"  {'  '.join(perf_parts)}" if perf_parts else ""
    trainer_tag = f"  trainer={run.trainer}" if run.trainer else ""
    return (
        f"  {run.path:40s}  {cond:20s}  step={run.step:>4d}  "
        f"cr={run.correct_rate:.1%}  loss={run.loss:.4f}  "
        f"time={t:>8s}  [{status}]{perf_tag}{trainer_tag}"
    )


def format_campaign(camp: CampaignSummary) -> str:
    """Format a campaign summary as a multi-line text block."""
    # Header: campaign name + status + path
    name = Path(camp.path).name
    status_tag = f" ({camp.status})" if camp.status else ""
    lines = [f"Campaign: {name}{status_tag}  {camp.path}"]

    # Summary line
    elapsed = format_time(sum(r.wall_time_s for r in camp.runs))
    parts = [
        f"{len(camp.conditions)} conditions x {len(camp.seeds)} seeds = {camp.num_runs} runs",
    ]
    if camp.max_steps > 0:
        parts.append(f"max_steps={camp.max_steps}")
    parts.append(f"elapsed: {elapsed}")
    lines.append("  " + "  ".join(parts))

    if camp.matrix:
        # Determine column width from cell contents
        cell_width = 8
        seed_strs = [f"{'s' + str(s):>{cell_width}s}" for s in camp.seeds]
        cond_width = min(
            max(len(_truncate_condition(c)) for c in camp.conditions),
            _COND_MAX_WIDTH,
        )
        cond_width = max(cond_width, len("condition"))

        header = f"  {'condition':>{cond_width}s}  " + "  ".join(seed_strs)
        lines.append(header)
        lines.append("  " + "-" * (cond_width + 2 + (cell_width + 2) * len(camp.seeds)))

        for cond in camp.conditions:
            cells = []
            for s in camp.seeds:
                run = camp.matrix.get(cond, {}).get(s)
                cell = _run_cell(run, camp.max_steps, camp.runner_alive)
                cells.append(f"{cell:>{cell_width}s}")
            display_cond = _truncate_condition(cond)
            lines.append(f"  {display_cond:>{cond_width}s}  " + "  ".join(cells))

    return "\n".join(lines)
