"""`retrain explain` command."""

from __future__ import annotations

import sys
from pathlib import Path

from retrain.commands.explain.campaign import explain_campaign
from retrain.commands.explain.single import explain_single
from retrain.commands.explain.squeeze import explain_squeeze


def run(args: list[str]) -> None:
    """Dry-run: show what a config would do without running it."""
    fmt = "text"
    config_path: str | None = None
    for arg in args:
        if arg == "--json":
            fmt = "json"
        elif arg.startswith("--"):
            print(f"Unknown explain flag: {arg}", file=sys.stderr)
            sys.exit(1)
        else:
            config_path = arg

    # Resolve config path
    if config_path is None:
        if Path("retrain.toml").is_file():
            config_path = "retrain.toml"
        else:
            print("No config file specified and no retrain.toml in cwd.")
            sys.exit(1)

    if not Path(config_path).is_file():
        print(f"File not found: {config_path}")
        sys.exit(1)

    # Route by config type
    from retrain.config import config_kind

    kind = config_kind(config_path)
    if kind == "campaign":
        explain_campaign(config_path, fmt)
    elif kind == "squeeze":
        explain_squeeze(config_path, fmt)
    else:
        explain_single(config_path, fmt)
