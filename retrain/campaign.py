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


def _parse_campaign_conditions(
    raw_conditions: object, campaign_path: str
) -> list[tuple[str, str]]:
    """Parse campaign conditions and fail fast on malformed entries."""
    if not raw_conditions:
        return list(DEFAULT_CONDITIONS)
    if not isinstance(raw_conditions, list):
        raise ValueError(
            f"campaign.conditions must be a list in {campaign_path}"
        )

    conditions: list[tuple[str, str]] = []
    for idx, condition in enumerate(raw_conditions):
        if not isinstance(condition, dict):
            raise ValueError(
                f"campaign.conditions[{idx}] must be a table in {campaign_path}"
            )

        adv_mode = condition.get("advantage_mode")
        tx_mode = condition.get("transform_mode")
        if not isinstance(adv_mode, str) or not adv_mode:
            raise ValueError(
                f"campaign.conditions[{idx}].advantage_mode must be a non-empty string in {campaign_path}"
            )
        if not isinstance(tx_mode, str) or not tx_mode:
            raise ValueError(
                f"campaign.conditions[{idx}].transform_mode must be a non-empty string in {campaign_path}"
            )

        try:
            TrainConfig(advantage_mode=adv_mode, transform_mode=tx_mode)
        except ValueError as exc:
            raise ValueError(
                f"Invalid campaign condition at index {idx} in {campaign_path}: {exc}"
            ) from exc

        conditions.append((adv_mode, tx_mode))

    return conditions


def _auto_squeeze(
    adapter_path: str,
    squeeze_cfg: dict,
    lora_rank: int,
    wandb_project: str = "",
    wandb_entity: str = "",
) -> int:
    """Run squeeze analysis, print results, log to wandb. Returns recommended rank."""
    from retrain.squeeze import analyze_adapter

    min_var = float(squeeze_cfg.get("min_variance_retention", 0.95))
    source_rank = int(squeeze_cfg.get("source_rank", 0)) or lora_rank

    print(f"\n{'=' * 60}")
    print(f"Auto-squeeze: analyzing {adapter_path}")
    print(f"  source_rank={source_rank}, min_variance_retention={min_var}")

    analysis = analyze_adapter(
        adapter_path=adapter_path,
        source_rank=source_rank,
        min_variance_retention=min_var,
    )

    # Print variance table
    print(f"\nSource rank: {analysis.layers[0].source_rank}")
    print(f"Layers analyzed: {len(analysis.layers)}\n")

    header = f"{'Rank':>6}  {'Mean Var%':>9}  {'Min Var%':>9}  {'Max Var%':>9}"
    print(header)
    print("-" * len(header))

    for k in analysis.target_ranks:
        vals = [layer.variance_at_rank[k] for layer in analysis.layers]
        mean_v = analysis.mean_variance[k]
        min_v = min(vals)
        max_v = max(vals)
        marker = " <--" if k == analysis.recommended_rank else ""
        print(
            f"{k:>6}  {mean_v * 100:>8.2f}%  {min_v * 100:>8.2f}%  {max_v * 100:>8.2f}%{marker}"
        )

    print(
        f"\nRecommended rank: {analysis.recommended_rank} "
        f"(>= {min_var * 100:.0f}% variance retained)"
    )
    print(f"{'=' * 60}\n")

    # Log to wandb
    if wandb_project:
        _log_squeeze_to_wandb(analysis, wandb_project, wandb_entity)

    return analysis.recommended_rank


def _log_squeeze_to_wandb(
    analysis: "SqueezeAnalysis",  # noqa: F821
    wandb_project: str,
    wandb_entity: str = "",
) -> None:
    """Log squeeze analysis to wandb as a dedicated run with table + summary."""
    try:
        import wandb
    except ImportError:
        print("wandb not installed, skipping squeeze logging")
        return

    wandb_kwargs: dict = {
        "project": wandb_project,
        "name": "squeeze-analysis",
        "job_type": "squeeze",
        "tags": ["squeeze", f"rank-{analysis.recommended_rank}"],
        "config": {
            "source_rank": analysis.layers[0].source_rank,
            "recommended_rank": analysis.recommended_rank,
            "min_variance_retention": analysis.min_variance_retention,
            "num_layers": len(analysis.layers),
        },
    }
    if wandb_entity:
        wandb_kwargs["entity"] = wandb_entity

    run = wandb.init(**wandb_kwargs)

    # Variance table
    columns = ["rank", "mean_variance", "min_variance", "max_variance", "recommended"]
    table = wandb.Table(columns=columns)

    for k in analysis.target_ranks:
        vals = [layer.variance_at_rank[k] for layer in analysis.layers]
        mean_v = analysis.mean_variance[k]
        min_v = min(vals)
        max_v = max(vals)
        table.add_data(k, mean_v, min_v, max_v, k == analysis.recommended_rank)

    run.log({"squeeze/variance_table": table})

    # Line chart data: log each rank as a step for a clean variance curve
    for k in analysis.target_ranks:
        vals = [layer.variance_at_rank[k] for layer in analysis.layers]
        run.log({
            "squeeze/mean_variance": analysis.mean_variance[k],
            "squeeze/min_variance": min(vals),
            "squeeze/max_variance": max(vals),
            "squeeze/rank": k,
        })

    # Summary metrics
    run.summary["squeeze/recommended_rank"] = analysis.recommended_rank
    run.summary["squeeze/source_rank"] = analysis.layers[0].source_rank
    run.summary["squeeze/min_variance_retention"] = analysis.min_variance_retention
    run.summary["squeeze/num_layers"] = len(analysis.layers)

    run.finish()
    print(f"Squeeze results logged to wandb: {wandb_project}/squeeze-analysis")


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
    conditions = _parse_campaign_conditions(raw_conditions, campaign_path)

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

    # Squeeze config (optional)
    squeeze_cfg = data.get("squeeze", None)

    # Execute: each run is a train() call with overridden fields
    from retrain.trainer import train
    import copy

    failed = 0
    recommended_rank: int | None = None

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

            adapter_path = train(cfg)
            print(f"  OK")

            # Auto-squeeze after first run (errors here don't fail the run)
            if idx == 0 and squeeze_cfg and adapter_path:
                try:
                    recommended_rank = _auto_squeeze(
                        adapter_path,
                        squeeze_cfg,
                        base_config.lora_rank,
                        wandb_project=base_config.wandb_project,
                        wandb_entity=base_config.wandb_entity,
                    )
                except Exception as e:
                    print(f"  Squeeze failed (non-fatal): {e}")
        except RuntimeError as e:
            # Fatal errors (missing backend, bad config) — abort campaign
            print(f"  FATAL: {e}")
            print(f"\nAborting campaign.")
            sys.exit(1)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    # Update manifest with squeeze result
    if recommended_rank is not None:
        manifest["squeeze"] = {"recommended_rank": recommended_rank}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if failed:
        print(f"\n{failed}/{len(runs)} runs failed.")
    else:
        print(f"\nAll {len(runs)} runs completed.")
    if recommended_rank is not None:
        print(f"Squeeze recommended rank: {recommended_rank}")
    print(f"Results in {campaign_dir}")
