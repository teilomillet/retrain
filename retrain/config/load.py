"""TOML config loader."""

from __future__ import annotations

import json
import tomllib
from dataclasses import MISSING, fields
from pathlib import Path

from retrain.config.migrate import _extract_backend_options
from retrain.config.override import _coerce_mapping_value, _coerce_value
from retrain.config.schema import TrainConfig
from retrain.config.sections import _FIELD_TYPES, _MAPPING_OVERRIDE_FIELDS, _TOML_MAP


def load_config(
    path: str | None = None,
    overrides: dict[str, object] | None = None,
) -> TrainConfig:
    """Load config from a TOML file.

    If path is None, looks for retrain.toml in cwd.
    Returns TrainConfig with TOML values overlaid on defaults.

    Empty-string TOML values are ignored for string fields
    (keeps the default).

    *overrides* (from CLI flags) are applied after TOML loading
    but before validation.
    """
    config = TrainConfig.__new__(TrainConfig)
    # Initialise with defaults (skip __post_init__ until the end)
    for f in fields(TrainConfig):
        if f.default is not MISSING:
            setattr(config, f.name, f.default)
        elif f.default_factory is not MISSING:
            setattr(config, f.name, f.default_factory())

    if path is None:
        if Path("retrain.toml").is_file():
            path = "retrain.toml"

    if path is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f)

        backend_options = _extract_backend_options(data.get("backend"))
        if backend_options is not None:
            setattr(config, "backend_options", backend_options)

        plugins_sec = data.get("plugins")
        if isinstance(plugins_sec, dict) and "search_paths" in plugins_sec:
            raw_paths = plugins_sec["search_paths"]
            if isinstance(raw_paths, list):
                setattr(config, "plugins_search_paths", [str(v) for v in raw_paths])
            elif isinstance(raw_paths, str):
                setattr(
                    config,
                    "plugins_search_paths",
                    [p.strip() for p in raw_paths.split(",") if p.strip()],
                )
            else:
                raise ValueError(
                    "Invalid [plugins].search_paths value. "
                    "Use a TOML list of strings."
                )

        optimizer_batch_sec = data.get("optimizer_batch")
        if isinstance(optimizer_batch_sec, dict) and (
            "allow_config_differences" in optimizer_batch_sec
        ):
            raw_differences = optimizer_batch_sec["allow_config_differences"]
            if not isinstance(raw_differences, list) or not all(
                isinstance(value, str) for value in raw_differences
            ):
                raise ValueError(
                    "Invalid [optimizer_batch].allow_config_differences value. "
                    "Use a TOML list of strings."
                )
            setattr(
                config,
                "optimizer_batch_allow_config_differences",
                list(raw_differences),
            )

        for section, mapping in _TOML_MAP.items():
            sec = data.get(section)
            if sec is None:
                continue
            for toml_key, field_name in mapping.items():
                if toml_key not in sec:
                    continue
                val = sec[toml_key]
                ftype = _FIELD_TYPES[field_name]
                if ftype is bool:
                    setattr(config, field_name, bool(val))
                elif ftype is int:
                    # Preserve the seed's raw TOML type so validation can reject
                    # floats, booleans, and strings instead of silently truncating
                    # or coercing them into an accepted integer.
                    setattr(
                        config,
                        field_name,
                        val if field_name == "seed" else int(val),
                    )
                elif ftype is float:
                    setattr(config, field_name, float(val))
                else:
                    if field_name == "environment_args" and isinstance(
                        val, (dict, list, tuple)
                    ):
                        setattr(config, field_name, json.dumps(val))
                        continue
                    # Empty-string values keep the field's default
                    s = str(val)
                    if s:
                        setattr(config, field_name, s)

        algorithm_sec = data.get("algorithm")
        if isinstance(algorithm_sec, dict):
            for key, field_name in (
                ("params", "algorithm_params"),
                ("advantage_params", "advantage_params"),
                ("transform_params", "transform_params"),
            ):
                if key not in algorithm_sec:
                    continue
                raw_map = algorithm_sec[key]
                if not isinstance(raw_map, dict):
                    raise ValueError(
                        f"Invalid [algorithm].{key} value. "
                        "Use a TOML table."
                    )
                setattr(config, field_name, dict(raw_map))

    # Apply CLI overrides
    if overrides:
        for field_name, raw_value in overrides.items():
            if field_name in _MAPPING_OVERRIDE_FIELDS:
                merged = dict(getattr(config, "backend_options", {}))
                if field_name != "backend_options":
                    merged = dict(getattr(config, field_name, {}))
                merged.update(_coerce_mapping_value(field_name, raw_value))
                setattr(config, field_name, merged)
                continue
            setattr(config, field_name, _coerce_value(field_name, raw_value))

    # Validate after all overrides
    config.__post_init__()
    return config
