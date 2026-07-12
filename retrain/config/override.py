"""Command-line override parsing for train configs."""

from __future__ import annotations

import difflib
import json
import sys
import typing

from retrain.config.sections import (
    _CLI_FLAG_MAP,
    _FIELD_TYPES,
    _MAPPING_OVERRIDE_FIELDS,
)
from retrain.config.table import as_object_table


def _coerce_mapping_value(field_name: str, raw: object) -> dict[str, object]:
    """Coerce one mapping-style override."""
    mapping = as_object_table(raw)
    if mapping is None:
        raise ValueError(f"{field_name} override must be a mapping of key=value options.")
    return dict(mapping)


def _coerce_value(field_name: str, raw: object) -> object:
    """Coerce a CLI string value to the type expected by *field_name*."""
    if field_name in _MAPPING_OVERRIDE_FIELDS:
        return _coerce_mapping_value(field_name, raw)

    if field_name in {
        "plugins_search_paths",
        "optimizer_batch_allow_config_differences",
    }:
        label = field_name
        if isinstance(raw, list):
            return [str(v) for v in raw]
        if isinstance(raw, str):
            if not raw.strip():
                return []
            if raw.strip().startswith("["):
                loaded = json.loads(raw)
                if not isinstance(loaded, list):
                    raise ValueError(f"{label} JSON override must decode to a list.")
                return [str(v) for v in loaded]
            return [p.strip() for p in raw.split(",") if p.strip()]
        raise ValueError(f"{label} override must be list or comma string.")

    ftype = _FIELD_TYPES[field_name]
    if ftype is bool:
        if isinstance(raw, bool):
            return raw
        if not isinstance(raw, str):
            raise ValueError(f"Expected string for {field_name}, got {type(raw).__name__}")
        return raw.lower() in ("1", "true", "yes")
    if ftype is int:
        if field_name == "seed":
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, int):
                return raw
            if isinstance(raw, str):
                try:
                    return int(raw)
                except ValueError:
                    return raw
            return raw
        if isinstance(raw, int):
            return raw
        if isinstance(raw, (str, bytes, bytearray, typing.SupportsInt)):
            return int(raw)
        raise ValueError(
            f"Expected int-coercible value for {field_name}, got {type(raw).__name__}"
        )
    if ftype is float:
        if isinstance(raw, float):
            return raw
        if isinstance(raw, (str, bytes, bytearray, typing.SupportsFloat)):
            return float(raw)
        raise ValueError(
            f"Expected float-coercible value for {field_name}, got {type(raw).__name__}"
        )
    return raw


def _parse_backend_opt(raw_value: str) -> tuple[str, str]:
    """Parse one backend option override from CLI key=value format."""
    if "=" not in raw_value:
        raise ValueError(
            "Flag --backend-opt requires key=value (example: --backend-opt transport=zmq)."
        )
    key, value = raw_value.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("Flag --backend-opt requires a non-empty key (key=value).")
    return key, value


def _parse_param_opt(raw_value: str, flag_name: str) -> tuple[str, object]:
    """Parse one repeatable parameter override from key=value format."""
    if "=" not in raw_value:
        raise ValueError(
            f"Flag {flag_name} requires key=value "
            f"(example: {flag_name} alpha=0.2)."
        )
    key, value = raw_value.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Flag {flag_name} requires a non-empty key (key=value).")

    v = value.strip()
    if not v:
        return key, ""
    if v.lower() in {"true", "false"}:
        return key, v.lower() == "true"
    if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
        try:
            return key, json.loads(v)
        except json.JSONDecodeError:
            pass
    try:
        if "." in v:
            return key, float(v)
        return key, int(v)
    except ValueError:
        return key, v


def parse_cli_overrides(argv: list[str]) -> tuple[str | None, dict[str, object]]:
    """Parse CLI args into (config_path, overrides).

    Supports ``--kebab-case value`` and ``--kebab-case=value``.
    The first positional argument (not starting with ``--``) is the config path.
    Unknown flags produce a helpful error with close-match suggestions.
    """
    config_path: str | None = None
    overrides: dict[str, object] = {}
    backend_opt_overrides: dict[str, object] = {}
    algorithm_param_overrides: dict[str, object] = {}
    advantage_param_overrides: dict[str, object] = {}
    transform_param_overrides: dict[str, object] = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith("--"):
            if config_path is None:
                config_path = arg
            i += 1
            continue

        # Handle --flag=value
        if "=" in arg:
            flag, value = arg.split("=", 1)
        else:
            flag = arg
            value = None

        if flag == "--backend-opt":
            if value is None:
                i += 1
                if i >= len(argv):
                    print("Flag --backend-opt requires a value.", file=sys.stderr)
                    sys.exit(1)
                value = argv[i]
            try:
                key, opt_value = _parse_backend_opt(value)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                sys.exit(1)
            backend_opt_overrides[key] = opt_value
            i += 1
            continue

        if flag in ("--algorithm-param", "--advantage-param", "--transform-param"):
            if value is None:
                i += 1
                if i >= len(argv):
                    print(f"Flag {flag} requires a value.", file=sys.stderr)
                    sys.exit(1)
                value = argv[i]
            try:
                key, param_value = _parse_param_opt(value, flag)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                sys.exit(1)
            if flag == "--algorithm-param":
                algorithm_param_overrides[key] = param_value
            elif flag == "--advantage-param":
                advantage_param_overrides[key] = param_value
            else:
                transform_param_overrides[key] = param_value
            i += 1
            continue

        if flag not in _CLI_FLAG_MAP:
            close = difflib.get_close_matches(
                flag,
                list(_CLI_FLAG_MAP.keys())
                + [
                    "--backend-opt",
                    "--algorithm-param",
                    "--advantage-param",
                    "--transform-param",
                ],
                n=1,
                cutoff=0.6,
            )
            hint = f" Did you mean: {close[0]}?" if close else ""
            print(f"Unknown flag: {flag}.{hint}", file=sys.stderr)
            sys.exit(1)

        if value is None:
            i += 1
            if i >= len(argv):
                print(f"Flag {flag} requires a value.", file=sys.stderr)
                sys.exit(1)
            value = argv[i]

        overrides[_CLI_FLAG_MAP[flag]] = value
        i += 1

    if backend_opt_overrides:
        overrides["backend_options"] = backend_opt_overrides
    if algorithm_param_overrides:
        overrides["algorithm_params"] = algorithm_param_overrides
    if advantage_param_overrides:
        overrides["advantage_params"] = advantage_param_overrides
    if transform_param_overrides:
        overrides["transform_params"] = transform_param_overrides

    return config_path, overrides
