"""Non-fatal TrainConfig warnings."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from retrain.config.constants import _DEFAULT_ADAPTER_PATH

if TYPE_CHECKING:
    from retrain.config.schema import TrainConfig


def emit_warnings(config: TrainConfig) -> None:
    # --- Warnings (non-fatal) ---
    if (
        config.adapter_path.startswith("/tmp")
        and config.adapter_path != _DEFAULT_ADAPTER_PATH
    ):
        warnings.warn(
            "adapter_path starts with /tmp — checkpoints may be lost on reboot.",
            stacklevel=2,
        )
    if config.temperature > 2.0:
        warnings.warn(
            f"temperature={config.temperature} is unusually high.",
            stacklevel=2,
        )
    if config.save_every == 0:
        warnings.warn(
            "save_every=0 disables periodic checkpoints.",
            stacklevel=2,
        )
        if config.wandb_project and config.checkpoint_artifacts != "off":
            warnings.warn(
                "checkpoint_artifacts is enabled but save_every=0, so W&B "
                "will only receive the final adapter. Spot/preemptible runs "
                "cannot resume mid-run; set save_every > 0.",
                stacklevel=2,
            )
    if config.weight_decay < 0:
        warnings.warn(
            f"weight_decay={config.weight_decay} is negative — this is unusual.",
            stacklevel=2,
        )
    if config.clip_eps > 0 and config.backend not in ("local", "tinker"):
        warnings.warn(
            f"clip_eps={config.clip_eps} is set but backend='{config.backend}' — "
            "ratio clipping is only implemented in the local and tinker backends.",
            stacklevel=2,
        )
