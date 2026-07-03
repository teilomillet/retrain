"""`retrain init` command."""

from __future__ import annotations

import sys
from pathlib import Path

from retrain.commands.init.interactive import run as run_interactive
from retrain.commands.init.templates import INIT_TEMPLATES
from retrain.commands.name import resolve as resolve_cli_name


def run(args: list[str] | None = None, cli_name: str | None = None) -> None:
    """Generate a starter config file in the current directory."""
    if not cli_name:
        cli_name = resolve_cli_name()
    args = args or []

    template_name = "default"
    list_templates = False
    interactive_mode = False
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--interactive", "-i"):
            interactive_mode = True
        elif arg in ("--list", "-l"):
            list_templates = True
        elif arg in ("--template", "-t"):
            i += 1
            if i >= len(args):
                print("Flag --template requires a value.", file=sys.stderr)
                sys.exit(1)
            template_name = args[i]
        elif arg.startswith("--template="):
            template_name = arg.split("=", 1)[1]
        else:
            print(f"Unknown init flag: {arg}", file=sys.stderr)
            sys.exit(1)
        i += 1

    if list_templates:
        print("Available templates:")
        for name, (_, filename) in sorted(INIT_TEMPLATES.items()):
            print(f"  {name:12s} -> {filename}")
        return

    if interactive_mode:
        run_interactive(cli_name)
        return

    if template_name not in INIT_TEMPLATES:
        print(
            f"Unknown template '{template_name}'. "
            f"Available: {sorted(INIT_TEMPLATES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    content, filename = INIT_TEMPLATES[template_name]
    dest = Path(filename)
    if dest.exists():
        print(f"{filename} already exists — refusing to overwrite.")
        sys.exit(1)
    dest.write_text(content)
    print(f"Created {filename} (template: {template_name})")
    print(f"Edit it, then run: {cli_name}")
    print(f"Need guidance? Run: {cli_name} man")
