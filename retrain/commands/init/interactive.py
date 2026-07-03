"""Interactive `retrain init` flow."""

from __future__ import annotations

import sys
from pathlib import Path

from retrain.commands.init.customize import customize
from retrain.commands.init.templates import INIT_TEMPLATES


GOALS: dict[str, tuple[str, int]] = {
    "1": ("quickstart", 20),
    "2": ("experiment", 500),
    "3": ("campaign", 200),
}


def run(cli_name: str) -> None:
    """Interactively build a retrain config file."""
    if not sys.stdin.isatty():
        print("Interactive init requires a terminal (TTY).", file=sys.stderr)
        sys.exit(1)

    print(f"{cli_name} init — interactive setup\n")
    print("What would you like to do?")
    print("  1) quickstart  — 20-step smoke test")
    print("  2) experiment  — reproducible training run")
    print("  3) campaign    — sweep across conditions and seeds")
    choice = input("\nChoice [1]: ").strip() or "1"
    if choice not in GOALS:
        print(f"Invalid choice '{choice}'. Using quickstart.", file=sys.stderr)
        choice = "1"

    goal_name, default_steps = GOALS[choice]

    steps_input = input(f"Training steps [{default_steps}]: ").strip()
    if steps_input:
        try:
            max_steps = int(steps_input)
        except ValueError:
            print(f"Not a number, using default ({default_steps}).", file=sys.stderr)
            max_steps = default_steps
    else:
        max_steps = default_steps

    seed_input = input("Reproducible seed [42]: ").strip()
    if seed_input:
        try:
            seed = int(seed_input)
        except ValueError:
            print("Not a number, using default (42).", file=sys.stderr)
            seed = 42
    else:
        seed = 42

    wandb_project = input("Wandb project name (empty to skip): ").strip()

    content, filename = INIT_TEMPLATES[goal_name]
    content = customize(content, max_steps=max_steps, seed=seed, wandb_project=wandb_project or None)

    dest = Path(filename)
    if dest.exists():
        print(f"{filename} already exists — refusing to overwrite.")
        sys.exit(1)
    dest.write_text(content)
    print(f"\nCreated {filename} (template: {goal_name})")
    print(f"Edit it, then run: {cli_name}")
