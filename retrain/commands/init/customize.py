"""TOML template customization for `retrain init`."""

from __future__ import annotations

import re


def customize(
    content: str,
    max_steps: int | None = None,
    seed: int | None = None,
    wandb_project: str | None = None,
) -> str:
    """Apply CLI customizations to a TOML template string."""
    if max_steps is not None:
        content = re.sub(
            r"^(max_steps\s*=\s*)\d+", rf"\g<1>{max_steps}", content, flags=re.MULTILINE
        )
    if seed is not None:
        content = re.sub(
            r"^(seed\s*=\s*)-?\d+", rf"\g<1>{seed}", content, flags=re.MULTILINE
        )
    if wandb_project is not None and wandb_project:
        content = re.sub(
            r"^#\s*wandb_project\s*=.*$",
            f'wandb_project = "{wandb_project}"',
            content,
            flags=re.MULTILINE,
        )
        content = re.sub(
            r'^wandb_project\s*=\s*"[^"]*"',
            f'wandb_project = "{wandb_project}"',
            content,
            flags=re.MULTILINE,
        )
    return content
