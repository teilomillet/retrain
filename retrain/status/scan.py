"""Scan run and campaign directories into status summaries."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from retrain.metrics.scan import (
    float_metric as _metric_float,
    float_or_none,
    int_metric as _metric_int,
    scan_metrics_file,
)
from retrain.training.state import TRAINER_STATE_FILE


_STALE_SECONDS = 300  # 5 minutes without progress -> stale


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
    runner_alive: bool = False,
) -> str:
    """Derive a campaign-level status from its runs.

    Returns one of: ``"running"``, ``"done"``, ``"dead"``, ``"partial"``.

    When *runner_alive* is True, non-completed runs that have metrics are
    considered active even if their metrics file is stale (slow steps).
    """
    completed = sum(1 for r in runs if r.completed)
    active = sum(1 for r in runs if not r.completed and not r.stale and r.step >= 0)

    # If the campaign runner is alive, stale-but-not-completed runs with
    # metrics are still active (slow steps on large models).
    if runner_alive:
        active += sum(1 for r in runs if not r.completed and r.stale and r.step >= 0)

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
    trainer: str = ""
    resume_mode: str = ""
    resume_warning: str = ""
    latest_step_time_s: float = 0.0
    tokens_per_second: float = 0.0
    sample_share: float = 0.0
    train_share: float | None = None
    train_time_semantics: str = ""
    train_submit_enqueue_time_s: float | None = None
    train_submit_enqueue_share: float | None = None
    process_max_rss_mb: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "condition": self.condition,
            "step": self.step,
            "max_steps": self.max_steps,
            "correct_rate": self.correct_rate,
            "loss": self.loss,
            "mean_reward": self.mean_reward,
            "wall_time_s": self.wall_time_s,
            "completed": self.completed,
            "stale": self.stale,
            "pid": self.pid,
            "alive": self.alive,
            "trainer": self.trainer,
            "resume_mode": self.resume_mode,
            "resume_warning": self.resume_warning,
            "latest_step_time_s": self.latest_step_time_s,
            "tokens_per_second": self.tokens_per_second,
            "sample_share": self.sample_share,
            "train_share": self.train_share,
            "train_time_semantics": self.train_time_semantics,
            "train_submit_enqueue_time_s": self.train_submit_enqueue_time_s,
            "train_submit_enqueue_share": self.train_submit_enqueue_share,
            "process_max_rss_mb": self.process_max_rss_mb,
        }


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
    runner_pid: int = 0
    runner_alive: bool = False
    last_activity: float = 0.0  # epoch timestamp of most recent metrics write

    def to_dict(self) -> dict[str, object]:
        # Matrix cells contain RunSummary objects for display; JSON output keeps
        # only the compact score view used by the CLI status API.
        flat_matrix: dict[str, dict[int, float | None]] = {}
        for cond, seeds in self.matrix.items():
            flat_matrix[cond] = {}
            for seed, run in seeds.items():
                flat_matrix[cond][seed] = run.correct_rate if run is not None else None
        return {
            "path": self.path,
            "conditions": list(self.conditions),
            "seeds": list(self.seeds),
            "max_steps": self.max_steps,
            "num_runs": self.num_runs,
            "completed": self.completed,
            "failed": self.failed,
            "runs": [r.to_dict() for r in self.runs],
            "matrix": flat_matrix,
            "status": self.status,
            "campaign_toml": self.campaign_toml,
            "timestamp": self.timestamp,
            "runner_pid": self.runner_pid,
            "runner_alive": self.runner_alive,
            "last_activity": self.last_activity,
        }


def _metric_str(row: dict[str, object], key: str, default: str = "") -> str:
    value = row.get(key)
    return value if isinstance(value, str) else default


def _read_run_pid(run_dir: Path) -> int:
    """Read the run's PID, preferring ``run.pid`` file over log parsing.

    Falls back to scanning stderr/stdout logs for ``(pid=NNNN)`` patterns
    so that old campaigns (before ``run.pid`` was written) still work.
    """
    # Preferred: dedicated PID file written by campaign runner after Popen
    pid_path = run_dir / "run.pid"
    if pid_path.is_file():
        try:
            return int(pid_path.read_text().strip())
        except (ValueError, OSError):
            pass

    # Fallback: parse logs for old campaigns
    for log_name in ("stderr.log", "stdout.log"):
        log_path = run_dir / log_name
        if not log_path.is_file():
            continue
        try:
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

    try:
        metrics = scan_metrics_file(metrics_path)
    except OSError:
        return None

    last_entry = metrics.last
    if last_entry is not None:
        summary.step = _metric_int(last_entry, "step", default=-1)
        summary.condition = _metric_str(last_entry, "condition")
        summary.correct_rate = _metric_float(last_entry, "correct_rate")
        summary.loss = _metric_float(last_entry, "loss")
        summary.mean_reward = _metric_float(last_entry, "mean_reward")
        summary.latest_step_time_s = _metric_float(last_entry, "step_time_s")
        summary.tokens_per_second = _metric_float(last_entry, "tokens_per_second")
        summary.sample_share = _metric_float(last_entry, "sample_share")
        summary.train_share = float_or_none(last_entry.get("train_share"))
        summary.train_time_semantics = _metric_str(
            last_entry,
            "train_time_semantics",
        )
        summary.train_submit_enqueue_time_s = float_or_none(
            last_entry.get("train_submit_enqueue_time_s")
        )
        summary.train_submit_enqueue_share = float_or_none(
            last_entry.get("train_submit_enqueue_share")
        )
        summary.process_max_rss_mb = _metric_float(last_entry, "process_max_rss_mb")
    summary.wall_time_s = metrics.wall_time_s

    # Check trainer_state.json for completion
    state_path = run_dir / TRAINER_STATE_FILE
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
            if state.get("checkpoint_name") == "final":
                summary.completed = True
            summary.max_steps = state.get("step", -1) + 1 if summary.completed else -1
            resume_mode = state.get("resume_mode")
            if isinstance(resume_mode, str):
                summary.resume_mode = resume_mode
            resume_warning = state.get("resume_warning")
            if isinstance(resume_warning, str):
                summary.resume_warning = resume_warning
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
    pid = _read_run_pid(run_dir)
    if pid:
        summary.pid = pid
        summary.alive = is_pid_alive(pid)

    # Read run metadata (trainer info)
    meta_path = run_dir / "run_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            summary.trainer = meta.get("trainer", "")
        except (json.JSONDecodeError, OSError):
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

    runner_pid = int(manifest.get("runner_pid", 0))
    runner_alive = is_pid_alive(runner_pid) if runner_pid else False

    summary = CampaignSummary(
        path=str(campaign_dir),
        conditions=manifest.get("conditions", []),
        seeds=manifest.get("seeds", []),
        max_steps=manifest.get("max_steps", -1),
        num_runs=manifest.get("num_runs", 0),
        campaign_toml=manifest.get("campaign_toml", ""),
        timestamp=manifest.get("timestamp", ""),
        runner_pid=runner_pid,
        runner_alive=runner_alive,
    )

    # Build matrix and scan individual runs
    matrix: dict[str, dict[int, RunSummary | None]] = {}
    for cond in summary.conditions:
        matrix[cond] = {s: None for s in summary.seeds}

    runs_meta = manifest.get("runs", [])
    completed = 0
    failed = 0
    last_activity = 0.0
    for run_meta in runs_meta:
        log_dir = run_meta.get("log_dir", "")
        if not log_dir:
            continue
        run_path = Path(log_dir)
        # Track latest metrics mtime for last_activity
        metrics_path = run_path / "metrics.jsonl"
        if metrics_path.is_file():
            try:
                mt = metrics_path.stat().st_mtime
                if mt > last_activity:
                    last_activity = mt
            except OSError:
                pass
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
    summary.last_activity = last_activity
    summary.status = campaign_status(summary.runs, summary.num_runs, runner_alive)
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
    campaign_dirs: set[Path] = set()
    campaign_run_dirs: set[Path] = set()

    # First pass: find campaigns
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        camp = scan_campaign(candidate)
        if camp is not None:
            campaigns.append(camp)
            campaign_dirs.add(candidate)
            for r in camp.runs:
                campaign_run_dirs.add(Path(r.path))
            continue

    # Sort campaigns newest-first (directory names contain timestamps)
    campaigns.sort(key=lambda c: c.path, reverse=True)

    # Second pass: find standalone runs (not part of a campaign)
    for candidate in sorted(root.rglob("metrics.jsonl")):
        run_dir = candidate.parent
        if run_dir in campaign_run_dirs:
            continue
        # Skip if this is inside a campaign runs/ subdir
        if any(parent in campaign_dirs for parent in run_dir.parents):
            continue
        run = scan_run(run_dir)
        if run is not None:
            runs.append(run)

    return runs, campaigns
