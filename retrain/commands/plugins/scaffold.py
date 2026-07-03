"""`retrain init-plugin` command."""

from __future__ import annotations

import sys
from pathlib import Path

from retrain.commands.name import resolve as resolve_cli_name
from retrain.commands.plugins.kinds import KINDS
from retrain.commands.plugins.name import sanitize
from retrain.commands.plugins.template import render


def run(args: list[str], cli_name: str | None = None) -> None:
    """Scaffold a plugin module for students."""
    if not cli_name:
        cli_name = resolve_cli_name()
    kind = ""
    name = ""
    output_dir = "plugins"
    with_test = False

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--with-test":
            with_test = True
        elif arg in ("--kind", "-k"):
            i += 1
            if i >= len(args):
                print("Flag --kind requires a value.", file=sys.stderr)
                sys.exit(1)
            kind = args[i].strip().lower()
        elif arg.startswith("--kind="):
            kind = arg.split("=", 1)[1].strip().lower()
        elif arg in ("--name", "-n"):
            i += 1
            if i >= len(args):
                print("Flag --name requires a value.", file=sys.stderr)
                sys.exit(1)
            name = args[i].strip()
        elif arg.startswith("--name="):
            name = arg.split("=", 1)[1].strip()
        elif arg in ("--output-dir", "-o"):
            i += 1
            if i >= len(args):
                print("Flag --output-dir requires a value.", file=sys.stderr)
                sys.exit(1)
            output_dir = args[i].strip()
        elif arg.startswith("--output-dir="):
            output_dir = arg.split("=", 1)[1].strip()
        else:
            print(f"Unknown init-plugin flag: {arg}", file=sys.stderr)
            sys.exit(1)
        i += 1

    if kind not in KINDS:
        print(
            f"Invalid --kind '{kind}'. "
            f"Choose one of: {sorted(KINDS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not name:
        print("Flag --name is required.", file=sys.stderr)
        sys.exit(1)

    module_name = sanitize(name)
    module_content, snippet = render(kind, module_name)
    out_dir = Path(output_dir)
    plugin_path = out_dir / f"{module_name}.py"
    test_path = Path("tests") / f"test_{module_name}_plugin.py"

    if plugin_path.exists():
        print(f"{plugin_path} already exists — refusing to overwrite.")
        sys.exit(1)
    if with_test and test_path.exists():
        print(f"{test_path} already exists — refusing to overwrite.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    plugin_path.write_text(module_content)
    print(f"Created {plugin_path}")

    if with_test:
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.write_text(
            "import importlib\n\n"
            f"def test_{module_name}_importable():\n"
            f"    mod = importlib.import_module('plugins.{module_name}')\n"
            f"    assert hasattr(mod, '{module_name}')\n"
        )
        print(f"Created {test_path}")

    print("\nTOML snippet:")
    print(snippet.rstrip())
    print(f"\nRun it with: {cli_name} retrain.toml")
