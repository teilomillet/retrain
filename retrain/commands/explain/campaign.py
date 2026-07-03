"""Dry-run preview for a campaign."""

from __future__ import annotations

import json
import tomllib

from retrain.commands.backends.capability import payload as capability_payload
from retrain.commands.backends.capability import summary as capability_summary


def explain_campaign(config_path: str, fmt: str) -> None:
    """Explain what a campaign would do."""
    from retrain.campaign.parse import DEFAULT_SEEDS, parse_campaign_conditions

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    campaign = data.get("campaign", {})
    seeds = campaign.get("seeds", DEFAULT_SEEDS)
    max_steps = campaign.get("max_steps", 500)
    raw_conditions = campaign.get("conditions", None)
    conditions = parse_campaign_conditions(raw_conditions, config_path)
    condition_labels = [c.label for c in conditions]
    total_runs = len(conditions) * len(seeds)
    backend_sec = data.get("backend", {})
    backend_name = "local"
    backend_options: dict[str, object] = {}
    if isinstance(backend_sec, dict):
        backend_name = str(backend_sec.get("backend", "local") or "local")
        raw_options = backend_sec.get("options", {})
        if isinstance(raw_options, dict):
            backend_options = dict(raw_options)
    backend_capabilities = capability_payload(
        backend_name,
        backend_options,
    )

    info = {
        "mode": "campaign",
        "config": config_path,
        "backend": backend_name,
        "backend_capabilities": backend_capabilities,
        "conditions": condition_labels,
        "seeds": seeds,
        "max_steps": max_steps,
        "total_runs": total_runs,
    }

    if fmt == "json":
        print(json.dumps(info, indent=2))
        return

    print("retrain explain — campaign dry-run preview")
    print(f"  config        : {config_path}")
    print(f"  backend       : {backend_name}")
    print(f"  backend caps  : {capability_summary(backend_capabilities)}")
    print(f"  conditions    : {', '.join(condition_labels)}")
    print(f"  seeds         : {seeds}")
    print(f"  max_steps     : {max_steps}")
    print(f"  total runs    : {total_runs}")
