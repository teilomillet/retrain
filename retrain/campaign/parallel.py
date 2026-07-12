"""Parallel campaign execution via subprocesses."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import IO

from retrain.campaign.model import CampaignRun


def run_parallel(
    runs: list[CampaignRun],
    max_workers: int,
    stagger_seconds: float = 0.0,
) -> list[CampaignRun]:
    """Execute runs as parallel subprocesses.

    ``max_workers`` limits concurrency **within this campaign only**.
    Multiple campaigns can run simultaneously as independent processes,
    each managing its own worker pool.  To run two campaigns in parallel,
    launch them in separate terminals / background processes.

    Args:
        stagger_seconds: Delay between launching consecutive subprocesses
            to desynchronize their backend API calls and reduce contention.

    Returns runs list with ``returncode`` added to each entry.
    """
    total = len(runs)
    pending = list(runs)
    active: list[
        tuple[subprocess.Popen[bytes], CampaignRun, float, IO[str], IO[str]]
    ] = []
    finished = 0

    try:
        while pending or active:
            # Fill slots
            while pending and len(active) < max_workers:
                run = pending.pop(0)
                finished += 1
                log_path = Path(run["log_dir"])
                log_path.mkdir(parents=True, exist_ok=True)
                stdout_f = open(log_path / "stdout.log", "w")
                stderr_f = open(log_path / "stderr.log", "w")
                config_path = run.get("config_path")
                if not config_path:
                    raise ValueError(f"{run['run_name']} is missing config_path")
                proc = subprocess.Popen(
                    [sys.executable, "-m", "retrain.cli", config_path],
                    stdout=stdout_f,
                    stderr=stderr_f,
                )
                (log_path / "run.pid").write_text(str(proc.pid))
                print(
                    f"[{finished}/{total}] {run['run_name']} started (pid={proc.pid})"
                )
                active.append((proc, run, time.monotonic(), stdout_f, stderr_f))
                if stagger_seconds > 0 and pending and len(active) < max_workers:
                    time.sleep(stagger_seconds)

            # Poll active processes
            still_active = []
            for proc, run, start_t, stdout_f, stderr_f in active:
                ret = proc.poll()
                if ret is not None:
                    elapsed = time.monotonic() - start_t
                    stdout_f.close()
                    stderr_f.close()
                    run["returncode"] = ret
                    status = "ok" if ret == 0 else f"FAILED (exit {ret})"
                    print(f"  {run['run_name']} {status} ({elapsed:.1f}s)")
                else:
                    still_active.append((proc, run, start_t, stdout_f, stderr_f))
            active = still_active

            if active:
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrupted — terminating active runs...")
        for proc, run, start_t, stdout_f, stderr_f in active:
            proc.terminate()
            proc.wait(timeout=10)
            stdout_f.close()
            stderr_f.close()
            run["returncode"] = -1
        for run in pending:
            run["returncode"] = -1
        print(f"Terminated {len(active)} active + skipped {len(pending)} pending runs.")

    return runs
