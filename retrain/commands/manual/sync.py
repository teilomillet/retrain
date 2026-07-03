"""Manual auto-block synchronization."""

from __future__ import annotations

from pathlib import Path

from retrain.commands.manual.render import (
    render_commands,
    render_environment,
    render_options,
    render_quickstart,
)
from retrain.commands.manual.text import ManualPath, load_text

AUTO_BLOCK_NAMES = (
    "COMMANDS",
    "OPTIONS",
    "QUICKSTART",
    "ENVIRONMENT",
)


def replace_auto_block(text: str, name: str, rendered_lines: list[str]) -> str:
    start = f"<<AUTO:{name}>>"
    end = f"<<END:AUTO:{name}>>"
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(lines):
        line = lines[i]
        if line.strip() == start:
            replaced = True
            out.append(line)
            out.extend(rendered_lines)
            i += 1
            while i < len(lines) and lines[i].strip() != end:
                i += 1
            if i >= len(lines):
                raise ValueError(f"Missing block end marker: {end}")
            out.append(lines[i])
            i += 1
            continue
        out.append(line)
        i += 1

    if not replaced:
        raise ValueError(f"Missing block markers for {name}: {start} ... {end}")

    return "\n".join(out).rstrip() + "\n"


def _rendered_blocks(cli_name: str) -> dict[str, list[str]]:
    return {
        "COMMANDS": render_commands(cli_name),
        "OPTIONS": render_options(),
        "QUICKSTART": render_quickstart(cli_name),
        "ENVIRONMENT": render_environment(cli_name),
    }


def sync_file(cli_name: str, manual_path: ManualPath) -> tuple[Path, bool]:
    """Refresh auto-generated blocks in the editable manual."""
    path = manual_path()
    original = path.read_text() if path.is_file() else load_text(cli_name, manual_path)

    updated = original
    rendered = _rendered_blocks(cli_name)
    for name in AUTO_BLOCK_NAMES:
        updated = replace_auto_block(updated, name, rendered[name])

    changed = updated != original
    if changed:
        path.write_text(updated)
    return path, changed


def check_file(cli_name: str, manual_path: ManualPath) -> tuple[Path, bool]:
    """Check whether auto-generated manual blocks are up to date."""
    path = manual_path()
    original = path.read_text() if path.is_file() else load_text(cli_name, manual_path)

    updated = original
    rendered = _rendered_blocks(cli_name)
    for name in AUTO_BLOCK_NAMES:
        updated = replace_auto_block(updated, name, rendered[name])
    return path, updated == original
