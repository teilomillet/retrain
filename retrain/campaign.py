"""Campaign orchestrator: generates and optionally runs a full sweep.

Usage:
    python -m retrain.campaign                           # dry-run
    python -m retrain.campaign --execute                 # run all
    python -m retrain.campaign --seeds 101,102,103,104   # custom seeds
    python -m retrain.campaign --wandb-project sepa-deep # set wandb project
    python -m retrain.campaign --max-steps 100           # override max steps
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# 5 conditions from the SEPA paper
CONDITIONS: list[tuple[str, str]] = [
    ("grpo", "none"),
    ("maxrl", "none"),
    ("maxrl", "gtpo"),
    ("maxrl", "gtpo_hicra"),
    ("maxrl", "gtpo_sepa"),
]

DEFAULT_SEEDS: list[int] = [42, 101, 202, 303, 404, 505, 606, 707]


def build_commands(
    *,
    seeds: list[int],
    wandb_project: str,
    max_steps: int,
    config_path: str | None,
    campaign_dir: Path,
) -> list[dict[str, str | list[str]]]:
    """Build the list of run commands for all conditions × seeds."""
    runs: list[dict[str, str | list[str]]] = []

    for adv_mode, tx_mode in CONDITIONS:
        condition = f"{adv_mode}+{tx_mode}"
        for seed in seeds:
            run_name = f"{condition}_s{seed}"
            log_dir = str(campaign_dir / "runs" / run_name)

            cmd = [sys.executable, "-m", "retrain"]
            if config_path:
                cmd.append(config_path)
            cmd.extend([
                "--advantage-mode", adv_mode,
                "--transform-mode", tx_mode,
                "--seed", str(seed),
                "--max-steps", str(max_steps),
                "--log-dir", log_dir,
            ])
            if wandb_project:
                cmd.extend([
                    "--wandb-project", wandb_project,
                    "--wandb-group", condition,
                    "--wandb-run-name", run_name,
                    "--wandb-tags", f"{condition},seed{seed}",
                ])

            runs.append({
                "condition": condition,
                "seed": str(seed),
                "run_name": run_name,
                "log_dir": log_dir,
                "cmd": cmd,
            })

    return runs


def main() -> None:
    """CLI entry point for campaign orchestrator."""
    args = sys.argv[1:]

    # Parse args
    execute = False
    seeds = list(DEFAULT_SEEDS)
    wandb_project = ""
    max_steps = 100
    config_path: str | None = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--execute":
            execute = True
            i += 1
        elif arg == "--seeds" and i + 1 < len(args):
            seeds = [int(s.strip()) for s in args[i + 1].split(",")]
            i += 2
        elif arg == "--wandb-project" and i + 1 < len(args):
            wandb_project = args[i + 1]
            i += 2
        elif arg == "--max-steps" and i + 1 < len(args):
            max_steps = int(args[i + 1])
            i += 2
        elif arg == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        elif arg in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            print(f"Unknown argument: {arg}")
            sys.exit(1)

    # Create campaign directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    campaign_dir = Path("logs") / f"campaign_{timestamp}"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    runs = build_commands(
        seeds=seeds,
        wandb_project=wandb_project,
        max_steps=max_steps,
        config_path=config_path,
        campaign_dir=campaign_dir,
    )

    # Write manifest
    manifest = {
        "timestamp": timestamp,
        "conditions": [f"{a}+{t}" for a, t in CONDITIONS],
        "seeds": seeds,
        "wandb_project": wandb_project,
        "max_steps": max_steps,
        "num_runs": len(runs),
        "runs": runs,
    }
    manifest_path = campaign_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Write shell script
    script_path = campaign_dir / "run_all.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for run in runs:
        cmd_str = " ".join(run["cmd"])  # type: ignore[arg-type]
        lines.append(f"echo '>>> {run['run_name']}'")
        lines.append(cmd_str)
        lines.append("")
    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)

    # Summary
    n_conditions = len(CONDITIONS)
    n_seeds = len(seeds)
    print(f"Campaign: {n_conditions} conditions x {n_seeds} seeds = {len(runs)} runs")
    print(f"  manifest: {manifest_path}")
    print(f"  script:   {script_path}")
    print()

    for run in runs:
        cmd_str = " ".join(run["cmd"])  # type: ignore[arg-type]
        print(f"  {cmd_str}")
    print()

    if not execute:
        print("Dry run. Pass --execute to run all commands.")
        return

    # Execute sequentially
    print("Executing campaign...")
    for idx, run in enumerate(runs):
        print(f"\n[{idx + 1}/{len(runs)}] {run['run_name']}")
        result = subprocess.run(run["cmd"], check=False)  # type: ignore[arg-type]
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})")
        else:
            print(f"  OK")


if __name__ == "__main__":
    main()
