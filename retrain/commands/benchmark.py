"""`retrain benchmark` command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from retrain.commands.doctor.warn import warn_missing


def run(args: list[str]) -> None:
    """Run or summarize a benchmark suite."""
    from retrain.benchmark import (
        default_benchmark_output_dir,
        format_run_summary,
        format_suite_summary,
        run_benchmark_suite,
        summarize_run,
        summarize_suite,
    )
    from retrain.config import load_config, parse_cli_overrides
    from retrain.registry import get_registry

    fmt = "text"
    repeats = 1
    output_dir: str | None = None
    passthrough: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--json":
            fmt = "json"
        elif arg.startswith("--repeat="):
            repeats = int(arg.split("=", 1)[1])
        elif arg == "--repeat":
            i += 1
            if i >= len(args):
                print("Flag --repeat requires a value.", file=sys.stderr)
                sys.exit(1)
            repeats = int(args[i])
        elif arg.startswith("--output-dir="):
            output_dir = arg.split("=", 1)[1]
        elif arg == "--output-dir":
            i += 1
            if i >= len(args):
                print("Flag --output-dir requires a value.", file=sys.stderr)
                sys.exit(1)
            output_dir = args[i]
        else:
            passthrough.append(arg)
        i += 1

    config_path, overrides = parse_cli_overrides(passthrough)
    if config_path is None:
        print("Usage:", file=sys.stderr)
        print(
            "  retrain benchmark <config.toml> [--repeat N] [--output-dir DIR] [--json]",
            file=sys.stderr,
        )
        print(
            "  retrain benchmark <run_dir|suite_dir> [--json]",
            file=sys.stderr,
        )
        sys.exit(1)

    target = Path(config_path)
    if target.is_dir():
        try:
            if (target / "metrics.jsonl").is_file():
                run_summary = summarize_run(target)
                if fmt == "json":
                    print(json.dumps(run_summary.to_dict(), indent=2))
                else:
                    print(format_run_summary(run_summary))
                return
            suite_summary = summarize_suite(target)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        if fmt == "json":
            print(json.dumps(suite_summary.to_dict(), indent=2))
        else:
            print(format_suite_summary(suite_summary))
        return

    if not target.is_file():
        print(f"File not found: {target}", file=sys.stderr)
        sys.exit(1)

    config = load_config(str(target), overrides=overrides)
    warn_missing(config)
    suite_dir = Path(output_dir) if output_dir else default_benchmark_output_dir(
        str(target),
        config,
    )
    suite_summary = run_benchmark_suite(
        config,
        repeats=repeats,
        output_dir=suite_dir,
        runner_factory=lambda cfg: get_registry("trainer").create(cfg.trainer, cfg),
        disable_wandb=True,
    )
    if fmt == "json":
        print(json.dumps(suite_summary.to_dict(), indent=2))
    else:
        print(format_suite_summary(suite_summary))
