"""Dependency warnings for config-backed commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrain.config import TrainConfig


def warn_missing(config: "TrainConfig") -> None:
    """Warn if the config references components whose deps are missing."""
    from retrain.registry import check_environment

    results = check_environment(config=config)
    for name, import_name, hint, available in results:
        if not available:
            print(
                f"WARNING: component '{name}' requires '{import_name}' "
                f"which is not installed.\n  -> {hint}"
            )
