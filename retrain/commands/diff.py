"""`retrain diff` command."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def run(args: list[str]) -> None:
    """Compare two runs or campaign conditions."""
    from retrain.diff import diff_conditions, diff_runs, format_diff

    fmt = "text"
    positional: list[str] = []
    for arg in args:
        if arg == "--json":
            fmt = "json"
        elif arg.startswith("--"):
            print(f"Unknown diff flag: {arg}", file=sys.stderr)
            sys.exit(1)
        else:
            positional.append(arg)

    if len(positional) == 2:
        dir_a, dir_b = Path(positional[0]), Path(positional[1])
        try:
            result = diff_runs(dir_a, dir_b)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
    elif len(positional) == 3:
        campaign_dir = Path(positional[0])
        cond_a, cond_b = positional[1], positional[2]
        try:
            result = diff_conditions(campaign_dir, cond_a, cond_b)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
    else:
        print("Usage:", file=sys.stderr)
        print("  retrain diff <run_a> <run_b>", file=sys.stderr)
        print("  retrain diff <campaign_dir> <cond_a> <cond_b>", file=sys.stderr)
        sys.exit(1)

    if fmt == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_diff(result))
