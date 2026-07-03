"""Dry-run preview for a squeeze run."""

from __future__ import annotations

import json


def explain_squeeze(config_path: str, fmt: str) -> None:
    """Explain what a squeeze run would do."""
    from retrain.config import load_squeeze_config

    cfg = load_squeeze_config(config_path)

    info = {
        "mode": "squeeze",
        "config": config_path,
        "adapter_path": cfg.adapter_path,
        "source_rank": cfg.source_rank,
        "min_variance_retention": cfg.min_variance_retention,
    }
    if cfg.output_path:
        info["output_path"] = cfg.output_path
    if cfg.compress_to > 0:
        info["compress_to"] = cfg.compress_to

    if fmt == "json":
        print(json.dumps(info, indent=2))
        return

    print("retrain explain — squeeze dry-run preview")
    print(f"  config                : {config_path}")
    print(f"  adapter_path          : {cfg.adapter_path}")
    print(f"  source_rank           : {cfg.source_rank}")
    print(f"  min_variance_retention: {cfg.min_variance_retention}")
    if cfg.output_path:
        print(f"  output_path           : {cfg.output_path}")
    if cfg.compress_to > 0:
        print(f"  compress_to           : {cfg.compress_to}")
