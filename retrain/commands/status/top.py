"""`retrain top` command."""

from __future__ import annotations

from retrain.commands.status.run import run as run_status


def run(args: list[str]) -> None:
    """Live dashboard: alias for ``retrain status --watch --active``."""
    status_args = ["--watch", "--active"]
    for arg in args:
        if not arg.startswith("-"):
            status_args.append(arg)
            break
    run_status(status_args)
