"""Campaign runner: executes a sweep defined in a TOML file.

A campaign TOML has a [campaign] section with conditions and seeds.
All other sections (model, training, inference, etc.) serve as the
base config for each run.

Example:

    [campaign]
    seeds = [42, 101, 202, 303]
    max_steps = 50

    [[campaign.conditions]]
    advantage_mode = "grpo"
    transform_mode = "none"

    [[campaign.conditions]]
    advantage_mode = "maxrl"
    transform_mode = "gtpo_sepa"

    # Everything below is the base training config
    [model]
    model = "Qwen/Qwen3-4B-Instruct-2507"

    [training]
    batch_size = 8

    [logging]
    wandb_project = "sepa-pilot"
"""

from __future__ import annotations

import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from retrain.config import TrainConfig, load_config

DEFAULT_CONDITIONS: list[tuple[str, str]] = [
    ("grpo", "none"),
    ("maxrl", "none"),
    ("maxrl", "gtpo"),
    ("maxrl", "gtpo_hicra"),
    ("maxrl", "gtpo_sepa"),
]

DEFAULT_SEEDS: list[int] = [42, 101, 202, 303, 404, 505, 606, 707]


def run_campaign(campaign_path: str) -> None:
    """Load a campaign TOML and execute all runs sequentially."""
    with open(campaign_path, "rb") as f:
        data = tomllib.load(f)

    campaign = data.get("campaign", {})

    # Campaign-level settings
    seeds = campaign.get("seeds", DEFAULT_SEEDS)
    max_steps = campaign.get("max_steps", TrainConfig().max_steps)

    # Conditions
    raw_conditions = campaign.get("conditions", None)
    if raw_conditions:
        conditions = [
            (c["advantage_mode"], c["transform_mode"])
            for c in raw_conditions
        ]
    else:
        conditions = list(DEFAULT_CONDITIONS)

    # Load the same TOML as a base training config (non-campaign sections)
    base_config = load_config(campaign_path)

    # Campaign directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    campaign_dir = Path("logs") / f"campaign_{timestamp}"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    # Build run list
    runs: list[dict] = []
    for adv_mode, tx_mode in conditions:
        condition = f"{adv_mode}+{tx_mode}"
        for seed in seeds:
            run_name = f"{condition}_s{seed}"
            log_dir = str(campaign_dir / "runs" / run_name)
            runs.append({
                "condition": condition,
                "advantage_mode": adv_mode,
                "transform_mode": tx_mode,
                "seed": seed,
                "run_name": run_name,
                "log_dir": log_dir,
            })

    # Write manifest
    manifest = {
        "timestamp": timestamp,
        "campaign_toml": campaign_path,
        "conditions": [f"{a}+{t}" for a, t in conditions],
        "seeds": seeds,
        "max_steps": max_steps,
        "num_runs": len(runs),
        "runs": runs,
    }
    manifest_path = campaign_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Summary
    print(f"Campaign: {len(conditions)} conditions x {len(seeds)} seeds = {len(runs)} runs")
    print(f"  conditions: {', '.join(f'{a}+{t}' for a, t in conditions)}")
    print(f"  seeds:      {seeds}")
    print(f"  max_steps:  {max_steps}")
    print(f"  output:     {campaign_dir}")
    print()

    # Execute: each run is a train() call with overridden fields
    from retrain.trainer import train
    import copy

    failed = 0
    for idx, run in enumerate(runs):
        print(f"[{idx + 1}/{len(runs)}] {run['run_name']}")
        try:
            cfg = copy.deepcopy(base_config)
            cfg.advantage_mode = run["advantage_mode"]
            cfg.transform_mode = run["transform_mode"]
            cfg.seed = run["seed"]
            cfg.max_steps = max_steps
            cfg.log_dir = run["log_dir"]

            # Set wandb fields if configured
            condition = run["condition"]
            if cfg.wandb_project:
                cfg.wandb_group = condition
                cfg.wandb_run_name = run["run_name"]
                cfg.wandb_tags = f"{condition},seed{run['seed']}"

            train(cfg)
            print(f"  OK")
        except RuntimeError as e:
            # Fatal errors (missing backend, bad config) — abort campaign
            print(f"  FATAL: {e}")
            print(f"\nAborting campaign.")
            sys.exit(1)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    if failed:
        print(f"\n{failed}/{len(runs)} runs failed.")
    else:
        print(f"\nAll {len(runs)} runs completed.")
    print(f"Results in {campaign_dir}")
