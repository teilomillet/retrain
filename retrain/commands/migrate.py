"""`retrain migrate-config` command."""

from __future__ import annotations

import difflib
import json
import sys
import tomllib
from pathlib import Path


def run(args: list[str]) -> None:
    """Migrate legacy backend config keys to [backend.options] format."""
    from retrain.config import migrate_legacy_backend_keys_toml_text

    check_only = False
    write_in_place = False
    backup = False
    stdin_mode = False
    stdout_mode = False
    json_mode = False
    output_path: str | None = None
    positional: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--check":
            check_only = True
        elif arg == "--write":
            write_in_place = True
        elif arg == "--backup":
            backup = True
        elif arg == "--stdin":
            stdin_mode = True
        elif arg == "--stdout":
            stdout_mode = True
        elif arg == "--json":
            json_mode = True
        elif arg in ("--output", "-o"):
            i += 1
            if i >= len(args):
                print("Flag --output requires a path.", file=sys.stderr)
                sys.exit(1)
            output_path = args[i]
        elif arg.startswith("--output="):
            output_path = arg.split("=", 1)[1]
        elif arg.startswith("--"):
            print(f"Unknown migrate-config flag: {arg}", file=sys.stderr)
            sys.exit(1)
        else:
            positional.append(arg)
        i += 1

    if stdin_mode and positional:
        print("Use either a config path or --stdin, not both.", file=sys.stderr)
        sys.exit(1)
    if not stdin_mode and len(positional) != 1:
        print(
            "Usage: retrain migrate-config <config.toml> "
            "[--check|--write|--output PATH] [--backup] [--stdin|--stdout] [--json]",
            file=sys.stderr,
        )
        sys.exit(1)
    if check_only and (write_in_place or output_path or stdout_mode or backup):
        print(
            "Flag --check cannot be combined with --write, --output, --stdout, or --backup.",
            file=sys.stderr,
        )
        sys.exit(1)
    if stdin_mode and write_in_place:
        print(
            "Flag --write requires a file path input (cannot be used with --stdin).",
            file=sys.stderr,
        )
        sys.exit(1)
    if write_in_place and output_path:
        print("Use either --write or --output, not both.", file=sys.stderr)
        sys.exit(1)
    if stdout_mode and (write_in_place or output_path):
        print(
            "Flag --stdout cannot be combined with --write or --output.",
            file=sys.stderr,
        )
        sys.exit(1)
    if backup and not write_in_place:
        print("Flag --backup can only be used with --write.", file=sys.stderr)
        sys.exit(1)

    config_path: Path | None = None
    source_label = "<stdin>"
    if stdin_mode:
        original_text = sys.stdin.read()
        if not original_text:
            print("No TOML content received on stdin.", file=sys.stderr)
            sys.exit(1)
    else:
        config_path = Path(positional[0])
        if not config_path.is_file():
            print(f"File not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        source_label = str(config_path)
        original_text = config_path.read_text()

    try:
        migrated = migrate_legacy_backend_keys_toml_text(original_text)
    except tomllib.TOMLDecodeError as exc:
        print(f"Invalid TOML in {source_label}: {exc}", file=sys.stderr)
        sys.exit(1)

    needs_migration = bool(migrated.legacy_keys)
    diff_text = "\n".join(
        difflib.unified_diff(
            original_text.splitlines(),
            migrated.output_text.splitlines(),
            fromfile=source_label,
            tofile=f"{source_label}.migrated",
            lineterm="",
        )
    )

    mode = "preview"
    if check_only:
        mode = "check"
    elif write_in_place:
        mode = "write"
    elif output_path:
        mode = "output"
    elif stdout_mode:
        mode = "stdout"

    payload: dict[str, object] = {
        "config": source_label,
        "mode": mode,
        "needs_migration": needs_migration,
        "changed": migrated.changed,
        "legacy_keys": list(migrated.legacy_keys),
        "merged_backend_options": migrated.merged_backend_options,
        "diff": diff_text,
        "written": False,
        "output_path": None,
        "backup_path": None,
    }

    if check_only:
        if json_mode:
            print(json.dumps(payload, indent=2))
        elif needs_migration:
            keys = ", ".join(migrated.legacy_keys)
            print(f"Migration required in {source_label} (legacy keys: {keys}).")
        else:
            print(f"No migration needed: {source_label}")
        if needs_migration:
            sys.exit(1)
        return

    if write_in_place:
        assert config_path is not None
        if backup:
            backup_path = Path(str(config_path) + ".bak")
            backup_path.write_text(original_text)
            payload["backup_path"] = str(backup_path)
        config_path.write_text(migrated.output_text)
        payload["written"] = True
        payload["output_path"] = str(config_path)
    elif output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(migrated.output_text)
        payload["written"] = True
        payload["output_path"] = str(out_path)

    if json_mode:
        print(json.dumps(payload, indent=2))
        return

    if stdout_mode:
        print(migrated.output_text, end="")
        return

    if payload["written"]:
        if needs_migration:
            print(f"Migrated config written to {payload['output_path']}")
        else:
            print(
                f"No migration required. Wrote unchanged config to {payload['output_path']}"
            )
        if payload["backup_path"]:
            print(f"Backup written to {payload['backup_path']}")
        return

    if needs_migration:
        print(diff_text)
    else:
        print(f"No migration needed: {source_label}")
