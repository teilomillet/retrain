"""Single entry point for retrain.

Usage:
    retrain                  # loads retrain.toml from cwd
    retrain config.toml      # single training run
    retrain campaign.toml    # campaign (if TOML has [campaign] section)
    retrain backends         # list backend capabilities/schema metadata
    retrain init             # generate a starter retrain.toml
    retrain doctor           # check installed dependencies for all components
    retrain migrate-config config.toml   # migrate legacy backend keys
    retrain man              # human/agent-friendly manual
    retrain --seed 42 --lr 1e-4   # override config values from CLI

A TOML with a [campaign] section runs multiple conditions × seeds.
A TOML without it runs a single training job. Same command either way.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable

from retrain.commands import manual as manual_command
from retrain.commands.benchmark import run as run_benchmark
from retrain.commands.backends.run import run as run_backends
from retrain.commands.doctor.run import run as run_doctor
from retrain.commands.diff import run as run_diff
from retrain.commands.doctor.warn import warn_missing
from retrain.commands.explain.run import run as run_explain
from retrain.commands.init.run import run as run_init
from retrain.commands.migrate import run as run_migrate_config
from retrain.commands.name import resolve as resolve_cli_name
from retrain.commands.plugins.run import run as run_plugins
from retrain.commands.plugins.scaffold import run as run_init_plugin
from retrain.commands.help import print_help
from retrain.commands.resume_check import run as run_resume_check
from retrain.commands.status.run import run as run_status
from retrain.commands.status.top import run as run_top
from retrain.commands.trace.run import run as run_trace
from retrain.commands.tree.run import run as run_tree


def _manual_path() -> Path:
    """Location of the editable bundled manual file."""
    return Path(__file__).with_name("retrain.man")


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
    print("Loaded .env", file=sys.stderr)


def _commands(cli_name: str) -> dict[str, Callable[[list[str]], None]]:
    """Named commands; anything else is a config path or override flags."""
    return {
        "help": lambda rest: print_help(cli_name),
        "-h": lambda rest: print_help(cli_name),
        "--help": lambda rest: print_help(cli_name),
        "man": lambda rest: manual_command.run(rest, cli_name=cli_name, manual_path=_manual_path),
        "manual": lambda rest: manual_command.run(rest, cli_name=cli_name, manual_path=_manual_path),
        "backends": run_backends,
        "doctor": lambda rest: run_doctor(),
        "migrate-config": run_migrate_config,
        "init": lambda rest: run_init(args=rest, cli_name=cli_name),
        "init-plugin": lambda rest: run_init_plugin(args=rest, cli_name=cli_name),
        "plugins": run_plugins,
        "resume-check": run_resume_check,
        "status": run_status,
        "top": run_top,
        "explain": run_explain,
        "diff": run_diff,
        "benchmark": run_benchmark,
        "trace": run_trace,
        "tree": run_tree,
    }


def main() -> None:
    """Single entry point: retrain config.toml"""
    _load_dotenv()

    args = sys.argv[1:]
    cli_name = resolve_cli_name()

    if args:
        command = _commands(cli_name).get(args[0])
        if command is not None:
            command(args[1:])
            sys.exit(0)

    # Parse CLI overrides
    from retrain.config import parse_cli_overrides

    config_path, overrides = parse_cli_overrides(args)

    # Resolve config path
    if config_path is None:
        if Path("retrain.toml").is_file():
            config_path = "retrain.toml"
        elif not overrides:
            if sys.stdin.isatty():
                answer = input("No retrain.toml found. Create one now? [Y/n]: ").strip().lower()
                if answer in ("", "y", "yes"):
                    run_init(cli_name=cli_name)
                    sys.exit(0)
            print("No retrain.toml found. Create one with:")
            print(f"  {cli_name} init")
            print("Or pass a path:")
            print(f"  {cli_name} path/to/config.toml")
            print("Manual:")
            print(f"  {cli_name} man")
            sys.exit(1)
        # else: overrides-only mode, use defaults

    if config_path is not None and not Path(config_path).is_file():
        print(f"File not found: {config_path}")
        sys.exit(1)

    # Route: campaign | squeeze | single run
    # Campaign/squeeze only when a TOML file is provided (CLI overrides don't apply)
    if config_path is not None:
        from retrain.config import config_kind

        kind = config_kind(config_path)
        if kind == "campaign":
            from retrain.campaign.run import run_campaign
            run_campaign(config_path)
            return
        if kind == "squeeze":
            from retrain.squeeze.run import run_squeeze
            run_squeeze(config_path)
            return
    from retrain.config import load_config
    from retrain.registry.builtin import get_registry

    config = load_config(config_path, overrides=overrides)
    warn_missing(config)
    meta_dir = Path(config.log_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "run_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "trainer": config.trainer,
                "run_id": meta_dir.name or "run",
                "status": "running",
            }
        )
    )
    runner = get_registry("trainer").create(config.trainer, config)
    result = runner.run(config)
    meta: dict[str, object] = {"trainer": config.trainer}
    meta.update(result.to_dict())
    meta_path.write_text(json.dumps(meta))
    if not result.ok:
        print(
            f"Training failed ({result.failure_status}): {result.error_message}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
