"""Log scanning and status reporting for retrain runs and campaigns.

Scans a root directory for training runs (metrics.jsonl) and campaigns
(manifest.json), producing structured summaries suitable for terminal
display or JSON output.
"""

from __future__ import annotations

import json
import os
import re
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


_STALE_SECONDS = 300  # 5 minutes without progress → stale
_COND_MAX_WIDTH = 30  # max display width for condition labels


def is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still alive.
        return True
    except OSError:
        return False


def campaign_status(
    runs: list[RunSummary],
    num_runs: int,
) -> str:
    """Derive a campaign-level status from its runs.

    Returns one of: ``"running"``, ``"done"``, ``"dead"``, ``"partial"``.
    """
    completed = sum(1 for r in runs if r.completed)
    active = sum(1 for r in runs if not r.completed and not r.stale and r.step >= 0)

    if completed == num_runs and num_runs > 0:
        return "done"
    if active > 0:
        return "running"
    if completed > 0:
        return "partial"
    # No completed, no active — but runs may exist with stale metrics
    if runs:
        return "dead"
    return "dead"


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
    pid: int = 0
    alive: bool = False

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
    # condition -> seed -> RunSummary | None (for matrix display)
    matrix: dict[str, dict[int, RunSummary | None]] = field(default_factory=dict)
    status: str = ""
    campaign_toml: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["runs"] = [r.to_dict() for r in self.runs]
        # Flatten matrix to serialisable form (step values)
        flat_matrix: dict[str, dict[int, float | None]] = {}
        for cond, seeds in self.matrix.items():
            flat_matrix[cond] = {}
            for seed, run in seeds.items():
                flat_matrix[cond][seed] = run.correct_rate if run is not None else None
        d["matrix"] = flat_matrix
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


def _truncate_condition(label: str, max_width: int = _COND_MAX_WIDTH) -> str:
    """Truncate a long condition label from the left, preserving the end."""
    if len(label) <= max_width:
        return label
    return "..." + label[-(max_width - 3):]


def _read_pid_from_stdout(run_dir: Path) -> int:
    """Try to extract a child PID from the run's stdout.log.

    The campaign runner prints lines like:
        [1/6] cond_s42 started (pid=12345)
    but that goes to the campaign runner's own stdout, not the run's logs.
    Instead, we look for the run's own PID from its stderr.log header if present.
    """
    for log_name in ("stderr.log", "stdout.log"):
        log_path = run_dir / log_name
        if not log_path.is_file():
            continue
        try:
            # Only read the first few KB to find a PID line
            with open(log_path) as f:
                head = f.read(4096)
            m = re.search(r"\(pid=(\d+)\)", head)
            if m:
                return int(m.group(1))
        except OSError:
            pass
    return 0


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

    # Try to read PID and check liveness
    pid = _read_pid_from_stdout(run_dir)
    if pid:
        summary.pid = pid
        summary.alive = is_pid_alive(pid)

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
        campaign_toml=manifest.get("campaign_toml", ""),
        timestamp=manifest.get("timestamp", ""),
    )

    # Build matrix and scan individual runs
    matrix: dict[str, dict[int, RunSummary | None]] = {}
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
            # Propagate campaign-level max_steps to the run when not completed
            if run_summary.max_steps < 0 and summary.max_steps > 0:
                run_summary.max_steps = summary.max_steps
            summary.runs.append(run_summary)
            if run_summary.completed:
                completed += 1
            cond = run_meta.get("condition", "")
            seed = run_meta.get("seed", 0)
            if cond in matrix and seed in matrix[cond]:
                matrix[cond][seed] = run_summary

    summary.completed = completed
    summary.failed = failed
    summary.matrix = matrix
    summary.status = campaign_status(summary.runs, summary.num_runs)
    return summary


def scan_all(root: Path) -> tuple[list[RunSummary], list[CampaignSummary]]:
    """Scan a root directory for all runs and campaigns.

    Campaigns are directories containing manifest.json.
    Standalone runs are directories containing metrics.jsonl but no
    manifest.json in their parent.

    Campaigns are sorted newest-first (by directory name, descending).
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

    # Sort campaigns newest-first (directory names contain timestamps)
    campaigns.sort(key=lambda c: c.path, reverse=True)

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


def _run_cell(run: RunSummary | None, max_steps: int) -> str:
    """Format a single matrix cell showing step progress."""
    if run is None:
        return "\u2014"  # em-dash
    effective_max = run.max_steps if run.max_steps > 0 else max_steps
    step = run.step + 1 if run.step >= 0 else 0  # step is 0-indexed in metrics
    if run.completed:
        return f"{step} \u2713"
    if run.stale:
        return f"{step} \u2717"
    if effective_max > 0:
        return f"{step}/{effective_max}"
    return str(step)


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
                cell = _run_cell(run, camp.max_steps)
                cells.append(f"{cell:>{cell_width}s}")
            display_cond = _truncate_condition(cond)
            lines.append(f"  {display_cond:>{cond_width}s}  " + "  ".join(cells))

    return "\n".join(lines)
