"""Runtime identity checks for externally readiness-bound training configs."""

from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrain.config.schema import TrainConfig


CONFIG_PATH_ENV = "RETRAIN_CONFIG_PATH"


def readiness_config_path(config: TrainConfig) -> str | None:
    """Return the environment's non-empty readiness self-binding, if any."""

    raw = config.environment_args
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    readiness = value.get("readiness_config")
    return readiness if isinstance(readiness, str) and readiness.strip() else None


def assert_readiness_runtime_matches_file(config: TrainConfig) -> None:
    """Fail before model/environment setup if runtime config differs from its TOML."""

    runtime_binding = readiness_config_path(config)
    launched_raw = os.environ.get(CONFIG_PATH_ENV)
    if not launched_raw:
        if runtime_binding is not None:
            raise RuntimeError(
                "Readiness-bound training requires RETRAIN_CONFIG_PATH from the "
                "Retrain CLI."
            )
        return
    launched = Path(launched_raw).expanduser().resolve()
    if not launched.is_file():
        if runtime_binding is not None:
            raise RuntimeError(
                f"Readiness-bound training config does not exist: {launched}"
            )
        return

    from retrain.config.load import load_config

    pristine = load_config(str(launched))
    pristine_binding = readiness_config_path(pristine)
    if runtime_binding is None and pristine_binding is None:
        return
    differing_fields = [
        field.name
        for field in fields(config)
        if getattr(config, field.name) != getattr(pristine, field.name)
    ]
    if differing_fields:
        raise RuntimeError(
            "Readiness-bound runtime config differs from the launched TOML in: "
            + ", ".join(differing_fields)
        )
