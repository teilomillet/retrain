"""Single entry point for retrain.

Usage:
    retrain                  # loads retrain.toml from cwd
    retrain config.toml      # single training run
    retrain campaign.toml    # campaign (if TOML has [campaign] section)
    retrain doctor           # check installed dependencies for all components

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


def _run_doctor() -> None:
    """Print dependency status for all known components."""
    from retrain.registry import check_environment

    print("retrain doctor — checking component dependencies\n")
    results = check_environment(config=None)
    all_ok = True
    for name, import_name, hint, available in results:
        status = "OK" if available else "MISSING"
        if not available:
            all_ok = False
        print(f"  {name:20s} {import_name:25s} {status}")
        if not available:
            print(f"  {'':20s} -> {hint}")
    print()
    if all_ok:
        print("All optional dependencies are installed.")
    else:
        print("Some optional dependencies are missing (see above).")


def _check_environment(config: "TrainConfig") -> None:  # noqa: F821
    """Warn if the config references components whose deps are missing."""
    from retrain.registry import check_environment

    results = check_environment(config=config)
    for name, import_name, hint, available in results:
        if not available:
            print(
                f"WARNING: component '{name}' requires '{import_name}' "
                f"which is not installed.\n  -> {hint}"
            )


def main() -> None:
    """Single entry point: retrain config.toml"""
    _load_dotenv()

    args = sys.argv[1:]

    if args and args[0] in ("-h", "--help", "help"):
        print(__doc__)
        sys.exit(0)

    if args and args[0] == "doctor":
        _run_doctor()
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
        _check_environment(config)
        train(config)


if __name__ == "__main__":
    main()
