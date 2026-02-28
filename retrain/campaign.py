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

import copy
import json
import os
import subprocess
import sys
import time
import tomllib
from dataclasses import MISSING, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from retrain.config import TrainConfig, _TOML_MAP, load_config

DEFAULT_CONDITIONS: list[tuple[str, str]] = [
    ("grpo", "none"),
    ("maxrl", "none"),
    ("maxrl", "gtpo"),
    ("maxrl", "gtpo_hicra"),
    ("maxrl", "gtpo_sepa"),
]

DEFAULT_SEEDS: list[int] = [42, 101, 202, 303, 404, 505, 606, 707]


# ---------------------------------------------------------------------------
# TOML serialization helpers
# ---------------------------------------------------------------------------

def _toml_value(value: object) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        # Escape backslashes and double quotes
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    return repr(value)


def _config_to_toml(cfg: TrainConfig) -> str:
    """Serialize a fully-resolved TrainConfig to TOML.

    Uses the reverse of ``config._TOML_MAP``. Emits sections in ``_TOML_MAP``
    order, skipping fields that are still at their default value.
    """
    defaults = TrainConfig.__new__(TrainConfig)
    for f in fields(TrainConfig):
        if f.default is not MISSING:
            setattr(defaults, f.name, f.default)
        elif f.default_factory is not MISSING:
            setattr(defaults, f.name, f.default_factory())

    lines: list[str] = []
    for section, mapping in _TOML_MAP.items():
        section_lines: list[str] = []
        for toml_key, field_name in mapping.items():
            val = getattr(cfg, field_name)
            default_val = getattr(defaults, field_name)
            if val == default_val:
                continue
            section_lines.append(f"{toml_key} = {_toml_value(val)}")
        if section_lines:
            lines.append(f"[{section}]")
            lines.extend(section_lines)
            lines.append("")

    return "\n".join(lines) + "\n" if lines else "\n"


_CONDITION_EXTRA_KEYS = frozenset({"advantage_mode", "transform_mode"})


@dataclass(frozen=True)
class CampaignCondition:
    """A single campaign condition: required modes + optional overrides."""

    advantage_mode: str
    transform_mode: str
    overrides: dict[str, object] = field(default_factory=dict)

    @property
    def label(self) -> str:
        base = f"{self.advantage_mode}+{self.transform_mode}"
        if not self.overrides:
            return base
        suffix = ",".join(
            f"{k}={v}" for k, v in sorted(self.overrides.items())
        )
        return f"{base}~{suffix}"

    def as_legacy_tuple(self) -> tuple[str, str]:
        return (self.advantage_mode, self.transform_mode)


def _parse_campaign_conditions(
    raw_conditions: object, campaign_path: str
) -> list[CampaignCondition]:
    """Parse campaign conditions and fail fast on malformed entries."""
    if not raw_conditions:
        return [
            CampaignCondition(advantage_mode=a, transform_mode=t)
            for a, t in DEFAULT_CONDITIONS
        ]
    if not isinstance(raw_conditions, list):
        raise ValueError(
            f"campaign.conditions must be a list in {campaign_path}"
        )

    conditions: list[CampaignCondition] = []
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

        overrides = {
            k: v for k, v in condition.items()
            if k not in _CONDITION_EXTRA_KEYS
        }

        try:
            TrainConfig(
                advantage_mode=adv_mode,
                transform_mode=tx_mode,
                **overrides,
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid campaign condition at index {idx} in {campaign_path}: {exc}"
            ) from exc

        conditions.append(
            CampaignCondition(
                advantage_mode=adv_mode,
                transform_mode=tx_mode,
                overrides=overrides,
            )
        )

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


def _write_run_configs(
    runs: list[dict],
    base_config: TrainConfig,
    max_steps: int,
    config_dir: Path,
    throttle_dir: str = "",
) -> None:
    """Write per-run TOML config files for parallel execution."""
    config_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        cfg = copy.deepcopy(base_config)
        cfg.advantage_mode = run["advantage_mode"]
        cfg.transform_mode = run["transform_mode"]
        cfg.seed = run["seed"]
        cfg.max_steps = max_steps
        cfg.log_dir = run["log_dir"]
        if throttle_dir:
            cfg.tinker_throttle_dir = throttle_dir
        for key, value in run.get("overrides", {}).items():
            setattr(cfg, key, value)

        condition = run["condition"]
        if cfg.wandb_project:
            cfg.wandb_group = condition
            cfg.wandb_run_name = run["run_name"]
            cfg.wandb_tags = f"{condition},seed{run['seed']}"

        config_path = config_dir / f"{run['run_name']}.toml"
        config_path.write_text(_config_to_toml(cfg))
        run["config_path"] = str(config_path)


def _run_parallel(
    runs: list[dict],
    config_dir: Path,
    max_workers: int,
    stagger_seconds: float = 0.0,
) -> list[dict]:
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
    active: list[tuple[subprocess.Popen, dict, float, "IO", "IO"]] = []  # type: ignore[name-defined]
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
                proc = subprocess.Popen(
                    [sys.executable, "-m", "retrain.cli", run["config_path"]],
                    stdout=stdout_f,
                    stderr=stderr_f,
                )
                (log_path / "run.pid").write_text(str(proc.pid))
                print(f"[{finished}/{total}] {run['run_name']} started (pid={proc.pid})")
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


def _run_sequential(
    runs: list[dict],
    base_config: TrainConfig,
    max_steps: int,
    squeeze_cfg: dict | None,
) -> tuple[int, int | None]:
    """Execute runs sequentially in-process.

    Returns ``(failed_count, recommended_rank)``.
    """
    from retrain.registry import get_registry

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
            for key, value in run.get("overrides", {}).items():
                setattr(cfg, key, value)

            # Set wandb fields if configured
            condition = run["condition"]
            if cfg.wandb_project:
                cfg.wandb_group = condition
                cfg.wandb_run_name = run["run_name"]
                cfg.wandb_tags = f"{condition},seed{run['seed']}"

            runner = get_registry("trainer").create(cfg.trainer, cfg)
            adapter_path = runner.run(cfg)
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

    return failed, recommended_rank


def run_campaign(campaign_path: str) -> str:
    """Load a campaign TOML and execute all runs (sequential or parallel)."""
    with open(campaign_path, "rb") as f:
        data = tomllib.load(f)

    campaign = data.get("campaign", {})

    # Campaign-level settings
    seeds = campaign.get("seeds", DEFAULT_SEEDS)
    max_steps = campaign.get("max_steps", TrainConfig().max_steps)
    parallel = bool(campaign.get("parallel", False))
    max_workers = int(campaign.get("max_workers", 0))
    stagger_seconds = float(campaign.get("stagger_seconds", 0))

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
    manifest = {
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
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

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
    squeeze_cfg = data.get("squeeze", None)

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
        _write_run_configs(runs, base_config, max_steps, config_dir, throttle_dir)

        effective_workers = max_workers if max_workers > 0 else len(runs)
        runs = _run_parallel(runs, config_dir, effective_workers, stagger_seconds)

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
                    recommended_rank = _auto_squeeze(
                        adapter_path,
                        squeeze_cfg,
                        base_config.lora_rank,
                        wandb_project=base_config.wandb_project,
                        wandb_entity=base_config.wandb_entity,
                    )
                except Exception as e:
                    print(f"  Squeeze failed (non-fatal): {e}")
    else:
        failed, recommended_rank = _run_sequential(
            runs, base_config, max_steps, squeeze_cfg
        )

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
    return str(campaign_dir)
