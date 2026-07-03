"""Campaign orchestration: manifest, execution, and summary hook."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from retrain.campaign.configs import write_run_configs
from retrain.campaign.model import CampaignRun
from retrain.campaign.parallel import run_parallel
from retrain.campaign.parse import (
    campaign_float,
    campaign_int,
    campaign_seeds,
    optional_object_dict,
    parse_campaign_conditions,
)
from retrain.campaign.sequential import run_sequential
from retrain.campaign.squeeze import auto_squeeze
from retrain.config import TrainConfig, load_config


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Write campaign manifest atomically enough for local workflows."""
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def _run_campaign_summary(
    campaign_cfg: dict[str, object],
    campaign_dir: Path,
    campaign_path: str,
) -> dict[str, object] | None:
    """Run an optional campaign summary script.

    Contract:
      - `campaign.summary_script` is a Python script path.
      - The campaign dir is passed as the first positional argument.
      - Optional `campaign.summary_args` appends extra string args.
    """
    raw_script = campaign_cfg.get("summary_script")
    if not raw_script:
        return None
    if not isinstance(raw_script, str):
        raise ValueError("campaign.summary_script must be a string")

    raw_args = campaign_cfg.get("summary_args", [])
    if raw_args is None:
        raw_args = []
    if not isinstance(raw_args, list) or not all(isinstance(a, str) for a in raw_args):
        raise ValueError("campaign.summary_args must be a list of strings")
    summary_args = cast(list[str], raw_args)

    script_path = Path(raw_script)
    if not script_path.is_absolute():
        script_path = Path(campaign_path).resolve().parent / script_path
    if not script_path.is_file():
        raise ValueError(
            f"campaign.summary_script does not exist: {script_path}"
        )

    cmd = [sys.executable, str(script_path), str(campaign_dir), *summary_args]
    print(f"Running summary: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return {
        "script": str(script_path),
        "args": list(summary_args),
        "returncode": proc.returncode,
    }


def run_campaign(campaign_path: str) -> str:
    """Load a campaign TOML and execute all runs (sequential or parallel)."""
    with open(campaign_path, "rb") as f:
        data = tomllib.load(f)

    campaign = optional_object_dict(data.get("campaign"), "campaign") or {}

    # Campaign-level settings
    seeds = campaign_seeds(campaign)
    max_steps = campaign_int(campaign, "max_steps", TrainConfig().max_steps)
    parallel = bool(campaign.get("parallel", False))
    max_workers = campaign_int(campaign, "max_workers", 0)
    stagger_seconds = campaign_float(campaign, "stagger_seconds", 0.0)

    # Conditions
    raw_conditions = campaign.get("conditions", None)
    conditions = parse_campaign_conditions(raw_conditions, campaign_path)

    # Load the same TOML as a base training config (non-campaign sections)
    base_config = load_config(campaign_path)

    # Campaign directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    campaign_dir = Path("logs") / f"campaign_{timestamp}"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    # Build run list
    runs: list[CampaignRun] = []
    for cond in conditions:
        condition = cond.label
        for seed in seeds:
            run_name = f"{condition}_s{seed}"
            log_dir = str(campaign_dir / "runs" / run_name)
            runs.append({
                "condition": condition,
                "advantage_mode": cond.advantage_mode,
                "transform_mode": cond.transform_mode,
                "overrides": cond.overrides,
                "seed": seed,
                "run_name": run_name,
                "log_dir": log_dir,
            })

    # Write manifest
    manifest: dict[str, object] = {
        "timestamp": timestamp,
        "campaign_toml": campaign_path,
        "conditions": [c.label for c in conditions],
        "seeds": seeds,
        "max_steps": max_steps,
        "parallel": parallel,
        "num_runs": len(runs),
        "runner_pid": os.getpid(),
        "runs": runs,
    }
    manifest_path = campaign_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)

    # Summary
    mode_str = "parallel" if parallel else "sequential"
    print(f"Campaign: {len(conditions)} conditions x {len(seeds)} seeds = {len(runs)} runs ({mode_str})")
    print(f"  conditions: {', '.join(c.label for c in conditions)}")
    print(f"  seeds:      {seeds}")
    print(f"  max_steps:  {max_steps}")
    print(f"  output:     {campaign_dir}")
    if parallel:
        effective_workers = max_workers if max_workers > 0 else len(runs)
        print(f"  workers:    {effective_workers}")
    print()

    # Squeeze config (optional)
    squeeze_cfg = optional_object_dict(data.get("squeeze"), "squeeze")

    # Execute
    failed = 0
    recommended_rank: int | None = None

    if parallel:
        # Write per-run config files
        config_dir = campaign_dir / "configs"
        throttle_dir = ""
        if base_config.backend == "tinker":
            throttle_path = campaign_dir / "tinker_throttle"
            throttle_path.mkdir(parents=True, exist_ok=True)
            throttle_dir = str(throttle_path)
        write_run_configs(runs, base_config, max_steps, config_dir, throttle_dir)
        _write_manifest(manifest_path, manifest)

        effective_workers = max_workers if max_workers > 0 else len(runs)
        runs = run_parallel(runs, effective_workers, stagger_seconds)
        _write_manifest(manifest_path, manifest)

        failed = sum(1 for r in runs if r.get("returncode", -1) != 0)

        # Auto-squeeze after ALL parallel runs complete
        if squeeze_cfg:
            first_ok = next(
                (r for r in runs if r.get("returncode") == 0), None
            )
            if first_ok:
                adapter_dir = Path(first_ok["log_dir"])
                # Find adapter path in log dir (convention: adapters subdir or log_dir itself)
                adapter_path = str(adapter_dir)
                try:
                    recommended_rank = auto_squeeze(
                        adapter_path,
                        squeeze_cfg,
                        base_config.lora_rank,
                        wandb_project=base_config.wandb_project,
                        wandb_entity=base_config.wandb_entity,
                    )
                except Exception as e:
                    print(f"  Squeeze failed (non-fatal): {e}")
    else:
        failed, recommended_rank = run_sequential(
            runs, base_config, max_steps, squeeze_cfg
        )
        _write_manifest(manifest_path, manifest)

    # Update manifest with squeeze result
    if recommended_rank is not None:
        manifest["squeeze"] = {"recommended_rank": recommended_rank}
        _write_manifest(manifest_path, manifest)

    summary_result = _run_campaign_summary(campaign, campaign_dir, campaign_path)
    if summary_result is not None:
        manifest["summary"] = summary_result
        _write_manifest(manifest_path, manifest)
        if summary_result["returncode"] != 0:
            print(
                "Summary script failed (non-fatal): "
                f"exit {summary_result['returncode']}"
            )

    if failed:
        print(f"\n{failed}/{len(runs)} runs failed.")
    else:
        print(f"\nAll {len(runs)} runs completed.")
    if recommended_rank is not None:
        print(f"Squeeze recommended rank: {recommended_rank}")
    print(f"Results in {campaign_dir}")
    return str(campaign_dir)
