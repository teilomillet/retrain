"""Algorithm, advantage, transform, and uncertainty checks."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

from retrain.advantages import (
    canonicalize_uncertainty_kind,
    get_builtin_advantage_modes,
    get_builtin_algorithm_modes,
    get_builtin_transform_modes,
    is_valid_advantage_mode_name,
    is_valid_algorithm_mode_name,
    is_valid_transform_mode_name,
)
from retrain.config.uncertainty import (
    _canonicalize_uncertainty_override,
    _first_uncertainty_override,
)

if TYPE_CHECKING:
    from retrain.config.schema import TrainConfig


def collect_mode_errors(config: TrainConfig, errors: list[str]) -> None:
    try:
        config.uncertainty_kind = canonicalize_uncertainty_kind(
            config.uncertainty_kind
        )
    except ValueError as exc:
        errors.append(str(exc))

    valid_algorithm_modes = set(get_builtin_algorithm_modes())
    if config.algorithm_mode:
        if config.algorithm_mode not in valid_algorithm_modes:
            if not is_valid_algorithm_mode_name(config.algorithm_mode):
                errors.append(
                    f"Invalid algorithm_mode '{config.algorithm_mode}'. "
                    f"Must be one of: {sorted(valid_algorithm_modes)} "
                    "or a dotted plugin path (e.g. 'my_module.my_algorithm')."
                )

    valid_advantage_modes = set(get_builtin_advantage_modes())
    if config.advantage_mode not in valid_advantage_modes:
        if not is_valid_advantage_mode_name(config.advantage_mode):
            errors.append(
                f"Invalid advantage_mode '{config.advantage_mode}'. "
                f"Must be one of: {sorted(valid_advantage_modes)} "
                "or a dotted plugin path (e.g. 'my_module.my_advantage')."
            )

    valid_transform_modes = set(get_builtin_transform_modes())
    if config.transform_mode not in valid_transform_modes:
        if not is_valid_transform_mode_name(config.transform_mode):
            errors.append(
                f"Invalid transform_mode '{config.transform_mode}'. "
                f"Must be one of: {sorted(valid_transform_modes)} "
                "or a dotted plugin path (e.g. 'my_module.make_transform_spec')."
            )

    for field_name in ("algorithm_params", "advantage_params", "transform_params"):
        value = getattr(config, field_name)
        if not isinstance(value, dict):
            errors.append(
                f"{field_name} must be a mapping table."
            )

    uncertainty_overrides: list[tuple[str, str]] = []
    if isinstance(config.transform_params, dict):
        override = _canonicalize_uncertainty_override(
            config.transform_params,
            label="transform_params",
            errors=errors,
        )
        if override is not None:
            uncertainty_overrides.append(override)
    if config.algorithm_mode and isinstance(config.algorithm_params, dict):
        override = _canonicalize_uncertainty_override(
            config.algorithm_params,
            label="algorithm_params",
            errors=errors,
        )
        if override is not None:
            uncertainty_overrides.append(override)

        raw_transform_params = config.algorithm_params.get("transform_params")
        nested_transform_params: dict[str, object] | None
        if isinstance(raw_transform_params, dict):
            nested_transform_params = typing.cast(
                dict[str, object],
                raw_transform_params,
            )
        elif isinstance(raw_transform_params, typing.Mapping):
            nested_transform_params = typing.cast(
                dict[str, object],
                dict(raw_transform_params),
            )
            config.algorithm_params["transform_params"] = nested_transform_params
        elif raw_transform_params is not None:
            nested_transform_params = None
            errors.append(
                "algorithm_params.transform_params must be a mapping table."
            )
        else:
            nested_transform_params = None
        if nested_transform_params is not None:
            override = _canonicalize_uncertainty_override(
                nested_transform_params,
                label="algorithm_params.transform_params",
                errors=errors,
            )
            if override is not None:
                uncertainty_overrides.append(override)

    override = _first_uncertainty_override(
        uncertainty_overrides,
        errors=errors,
    )
    if override is not None:
        label, value = override
        # The documented transform-param override may replace the default.
        # Any explicit non-default top-level selector must still agree.
        if config.uncertainty_kind == "surprisal":
            config.uncertainty_kind = value
        elif config.uncertainty_kind != value:
            errors.append(
                "Conflicting uncertainty_kind settings: "
                f"uncertainty_kind={config.uncertainty_kind!r} but "
                f"{label}={value!r}."
            )
