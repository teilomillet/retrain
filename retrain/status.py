"""Log scanning and status reporting for retrain runs and campaigns.

Scans a root directory for training runs (metrics.jsonl) and campaigns
(manifest.json), producing structured summaries suitable for terminal
display or JSON output.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


_STALE_SECONDS = 300  # 5 minutes without progress → stale


@dataclass
class RunSummary:
    """Status summary for a single training run."""

    path: str
    condition: str = ""
    step: int = -1
    max_steps: int = -1
    correct_rate: float = 0.0
    loss: float = 0.0
    mean_reward: float = 0.0
    wall_time_s: float = 0.0
    completed: bool = False
    stale: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CampaignSummary:
    """Status summary for a campaign (multiple runs)."""

    path: str
    conditions: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    max_steps: int = -1
    num_runs: int = 0
    completed: int = 0
    failed: int = 0
    runs: list[RunSummary] = field(default_factory=list)
    # condition -> seed -> correct_rate (for matrix display)
    matrix: dict[str, dict[int, float | None]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["runs"] = [r.to_dict() for r in self.runs]
        return d


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


def scan_run(run_dir: Path) -> RunSummary | None:
    """Scan a single run directory for status information.

    Returns None if the directory has no metrics.jsonl.
    """
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return None

    summary = RunSummary(path=str(run_dir))

    # Read metrics.jsonl
    wall_time = 0.0
    last_entry: dict | None = None
    try:
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                last_entry = entry
                wall_time += entry.get("step_time_s", 0.0)
    except OSError:
        return None

    if last_entry is not None:
        summary.step = last_entry.get("step", -1)
        summary.condition = last_entry.get("condition", "")
        summary.correct_rate = last_entry.get("correct_rate", 0.0)
        summary.loss = last_entry.get("loss", 0.0)
        summary.mean_reward = last_entry.get("mean_reward", 0.0)
    summary.wall_time_s = wall_time

    # Check trainer_state.json for completion
    state_path = run_dir / "trainer_state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
            if state.get("checkpoint_name") == "final":
                summary.completed = True
            summary.max_steps = state.get("step", -1) + 1 if summary.completed else -1
        except (json.JSONDecodeError, OSError):
            pass

    # Staleness: metrics file not updated for > 5 min and not completed
    if not summary.completed:
        try:
            mtime = metrics_path.stat().st_mtime
            if time.time() - mtime > _STALE_SECONDS:
                summary.stale = True
        except OSError:
            pass

    return summary


def scan_campaign(campaign_dir: Path) -> CampaignSummary | None:
    """Scan a campaign directory for status information.

    Returns None if the directory has no manifest.json.
    """
    manifest_path = campaign_dir / "manifest.json"
    if not manifest_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    summary = CampaignSummary(
        path=str(campaign_dir),
        conditions=manifest.get("conditions", []),
        seeds=manifest.get("seeds", []),
        max_steps=manifest.get("max_steps", -1),
        num_runs=manifest.get("num_runs", 0),
    )

    # Build matrix and scan individual runs
    matrix: dict[str, dict[int, float | None]] = {}
    for cond in summary.conditions:
        matrix[cond] = {s: None for s in summary.seeds}

    runs_meta = manifest.get("runs", [])
    completed = 0
    failed = 0
    for run_meta in runs_meta:
        log_dir = run_meta.get("log_dir", "")
        if not log_dir:
            continue
        run_path = Path(log_dir)
        run_summary = scan_run(run_path)
        if run_summary is not None:
            summary.runs.append(run_summary)
            if run_summary.completed:
                completed += 1
            cond = run_meta.get("condition", "")
            seed = run_meta.get("seed", 0)
            if cond in matrix and seed in matrix[cond]:
                matrix[cond][seed] = run_summary.correct_rate
        else:
            # No metrics at all — count as not started
            pass

    summary.completed = completed
    summary.failed = failed
    summary.matrix = matrix
    return summary


def scan_all(root: Path) -> tuple[list[RunSummary], list[CampaignSummary]]:
    """Scan a root directory for all runs and campaigns.

    Campaigns are directories containing manifest.json.
    Standalone runs are directories containing metrics.jsonl but no
    manifest.json in their parent.
    """
    runs: list[RunSummary] = []
    campaigns: list[CampaignSummary] = []

    if not root.is_dir():
        return runs, campaigns

    # Track campaign run dirs so we don't double-count
    campaign_run_dirs: set[str] = set()

    # First pass: find campaigns
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        camp = scan_campaign(candidate)
        if camp is not None:
            campaigns.append(camp)
            for r in camp.runs:
                campaign_run_dirs.add(r.path)
            continue

    # Second pass: find standalone runs (not part of a campaign)
    for candidate in sorted(root.rglob("metrics.jsonl")):
        run_dir = candidate.parent
        if str(run_dir) in campaign_run_dirs:
            continue
        # Skip if this is inside a campaign runs/ subdir
        if any(str(run_dir).startswith(cd) for cd in campaign_run_dirs):
            continue
        run = scan_run(run_dir)
        if run is not None:
            runs.append(run)

    return runs, campaigns


def format_run(run: RunSummary) -> str:
    """Format a single run summary as a text line."""
    status = "done" if run.completed else ("stale" if run.stale else "running")
    cond = run.condition or "unknown"
    t = format_time(run.wall_time_s)
    return (
        f"  {run.path:40s}  {cond:20s}  step={run.step:>4d}  "
        f"cr={run.correct_rate:.1%}  loss={run.loss:.4f}  "
        f"time={t:>8s}  [{status}]"
    )


def format_campaign(camp: CampaignSummary) -> str:
    """Format a campaign summary as a multi-line text block."""
    lines = [
        f"Campaign: {camp.path}",
        f"  {len(camp.conditions)} conditions x {len(camp.seeds)} seeds = {camp.num_runs} runs",
        f"  completed: {camp.completed}/{camp.num_runs}  max_steps: {camp.max_steps}",
    ]

    if camp.matrix:
        # Header row
        seed_strs = [f"{'s' + str(s):>8s}" for s in camp.seeds]
        lines.append(f"  {'condition':>20s}  " + "  ".join(seed_strs))
        lines.append("  " + "-" * (22 + 10 * len(camp.seeds)))
        for cond in camp.conditions:
            cells = []
            for s in camp.seeds:
                val = camp.matrix.get(cond, {}).get(s)
                if val is not None:
                    cells.append(f"{val:>7.1%}")
                else:
                    cells.append(f"{'—':>8s}")
            lines.append(f"  {cond:>20s}  " + "  ".join(cells))

    return "\n".join(lines)
