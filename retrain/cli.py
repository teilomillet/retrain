"""Single entry point for retrain.

Usage:
    retrain                  # loads retrain.toml from cwd
    retrain config.toml      # single training run
    retrain campaign.toml    # campaign (if TOML has [campaign] section)

A TOML with a [campaign] section runs multiple conditions × seeds.
A TOML without it runs a single training job. Same command either way.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env file if present. Sets vars into os.environ."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        eq = line.find("=")
        if eq == -1:
            continue
        key = line[:eq].strip()
        val = line[eq + 1 :].strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        os.environ[key] = val
    print("Loaded .env")


def _is_squeeze(path: str) -> bool:
    """Check if a TOML file has a [squeeze] section."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return "squeeze" in data


def _is_campaign(path: str) -> bool:
    """Check if a TOML file has a [campaign] section."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return "campaign" in data


def main() -> None:
    """Single entry point: retrain config.toml"""
    _load_dotenv()

    args = sys.argv[1:]

    if args and args[0] in ("-h", "--help", "help"):
        print(__doc__)
        sys.exit(0)

    # Find the TOML
    if args and not args[0].startswith("--"):
        config_path = args[0]
    elif Path("retrain.toml").is_file():
        config_path = "retrain.toml"
    else:
        print("No retrain.toml found. Create one or pass a path:")
        print("  retrain path/to/config.toml")
        sys.exit(1)

    if not Path(config_path).is_file():
        print(f"File not found: {config_path}")
        sys.exit(1)

    # Route: campaign | squeeze | single run
    # Campaign checked first — a campaign TOML may also have [squeeze] for auto-squeeze.
    if _is_campaign(config_path):
        from retrain.campaign import run_campaign
        run_campaign(config_path)
    elif _is_squeeze(config_path):
        from retrain.squeeze import run_squeeze
        run_squeeze(config_path)
    else:
        from retrain.config import load_config
        from retrain.trainer import train
        config = load_config(config_path)
        train(config)


if __name__ == "__main__":
    main()
