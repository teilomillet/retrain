"""Materialize per-run TOML configs for parallel campaign execution."""

from __future__ import annotations

import copy
from dataclasses import MISSING, fields
from pathlib import Path

from retrain.campaign.model import CampaignRun
from retrain.config import TrainConfig, _TOML_MAP


def toml_value(value: object) -> str:
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


def config_to_toml(cfg: TrainConfig) -> str:
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
            section_lines.append(f"{toml_key} = {toml_value(val)}")
        if section_lines:
            lines.append(f"[{section}]")
            lines.extend(section_lines)
            lines.append("")

    return "\n".join(lines) + "\n" if lines else "\n"


def write_run_configs(
    runs: list[CampaignRun],
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
        config_path.write_text(config_to_toml(cfg))
        run["config_path"] = str(config_path)
