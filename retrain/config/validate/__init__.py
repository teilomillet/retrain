"""TrainConfig validation orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from retrain.config.validate.bounds import collect_bounds_errors
from retrain.config.validate.defaults import apply_env_defaults
from retrain.config.validate.mode import collect_mode_errors
from retrain.config.validate.runtime import collect_runtime_errors
from retrain.config.validate.warn import emit_warnings
from retrain.plugins.resolve import set_plugin_runtime

if TYPE_CHECKING:
    from retrain.config.schema import TrainConfig


def validate_train_config(config: TrainConfig) -> None:
    apply_env_defaults(config)

    errors: list[str] = []
    collect_bounds_errors(config, errors)
    collect_mode_errors(config, errors)
    collect_runtime_errors(config, errors)

    if errors:
        raise ValueError("\n".join(errors))

    # Keep plugin runtime config synchronized for dotted-path resolution.
    set_plugin_runtime(config.plugins_search_paths, config.plugins_strict)
    emit_warnings(config)
