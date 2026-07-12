"""Backend option schemas, validation, and coercion."""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from typing import cast

from retrain.backends.option_schemas import (
    BackendOptionSpec as BackendOptionSpec,
    OptionValidator as OptionValidator,
    local_option_schema as local_option_schema,
    prime_rl_option_schema as prime_rl_option_schema,
    unsloth_option_schema as unsloth_option_schema,
)


def coerce_plugin_option_schema(
    backend_name: str,
    raw: object,
) -> dict[str, BackendOptionSpec] | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        schema: dict[str, BackendOptionSpec] = {}
        for key, spec in raw.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Invalid backend option schema for '{backend_name}': non-string key {key!r}."
                )
            schema[key] = _coerce_plugin_option_spec(key, spec)
        return schema
    raise ValueError(
        f"Invalid backend option schema for '{backend_name}'. Expected mapping."
    )


def normalize_option_schema(
    backend: str,
    options: Mapping[str, object],
    schema: Mapping[str, BackendOptionSpec],
) -> dict[str, object]:
    unknown = sorted(k for k in options if k not in schema)
    if unknown:
        bad = unknown[0]
        close = difflib.get_close_matches(bad, schema.keys(), n=1, cutoff=0.6)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(
            f"Unknown [backend.options] key '{bad}' for backend '{backend}'.{hint}"
        )

    normalized = {k: spec.default for k, spec in schema.items()}
    for key, raw in options.items():
        spec = schema[key]
        normalized[key] = coerce_option_value(backend, key, raw, spec)
    return normalized


def schema_to_payload(
    schema: Mapping[str, BackendOptionSpec],
) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for key, spec in schema.items():
        payload[key] = {
            "type": option_type_name(spec.value_type),
            "default": spec.default,
            "choices": list(spec.choices) if spec.choices else None,
            "has_validator": spec.validator is not None,
        }
    return payload


def option_type_name(value_type: type) -> str:
    if value_type is bool:
        return "bool"
    if value_type is int:
        return "int"
    if value_type is float:
        return "float"
    if value_type is str:
        return "str"
    return getattr(value_type, "__name__", str(value_type))


def coerce_option_value(
    backend: str,
    key: str,
    raw: object,
    spec: BackendOptionSpec,
) -> object:
    value: object

    if spec.value_type is bool:
        if isinstance(raw, bool):
            value = raw
        elif isinstance(raw, str):
            s = raw.strip().lower()
            if s in {"1", "true", "yes", "on"}:
                value = True
            elif s in {"0", "false", "no", "off"}:
                value = False
            else:
                raise ValueError(
                    f"Invalid [backend.options] {key}={raw!r} for backend '{backend}': "
                    "expected a boolean."
                )
        else:
            raise ValueError(
                f"Invalid [backend.options] {key}={raw!r} for backend '{backend}': "
                "expected a boolean."
            )
    elif spec.value_type is int:
        if isinstance(raw, bool):
            raise ValueError(
                f"Invalid [backend.options] {key}={raw!r} for backend '{backend}': "
                "expected an integer."
            )
        try:
            value = int(cast(str | int | float, raw))
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid [backend.options] {key}={raw!r} for backend '{backend}': "
                "expected an integer."
            ) from None
    elif spec.value_type is float:
        try:
            value = float(cast(str | int | float, raw))
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid [backend.options] {key}={raw!r} for backend '{backend}': "
                "expected a float."
            ) from None
    elif spec.value_type is str:
        value = str(raw)
    else:
        value = raw

    if spec.choices and value not in spec.choices:
        raise ValueError(
            f"Invalid [backend.options] {key}={value!r} for backend '{backend}': "
            f"must be one of {list(spec.choices)}."
        )

    if spec.validator:
        err = spec.validator(value)
        if err:
            raise ValueError(
                f"Invalid [backend.options] {key}={value!r} for backend '{backend}': {err}"
            )

    return value


def _coerce_option_type(raw_type: object) -> type:
    if isinstance(raw_type, type):
        return raw_type
    if isinstance(raw_type, str):
        key = raw_type.strip().lower()
        if key in {"bool", "boolean"}:
            return bool
        if key in {"int", "integer"}:
            return int
        if key == "float":
            return float
        if key in {"str", "string"}:
            return str
    raise ValueError(
        "Invalid backend option type. "
        "Expected one of bool/int/float/str (or corresponding python type)."
    )


def _coerce_plugin_option_spec(key: str, raw: object) -> BackendOptionSpec:
    if isinstance(raw, BackendOptionSpec):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"Invalid backend option schema entry for '{key}'. "
            "Expected BackendOptionSpec or mapping."
        )
    raw_map = cast(Mapping[str, object], raw)
    raw_type = raw_map.get("value_type", raw_map.get("type", str))
    value_type = _coerce_option_type(raw_type)
    default = raw_map.get("default")
    choices_raw = raw_map.get("choices")
    choices: tuple[object, ...] | None = None
    if choices_raw is not None:
        if not isinstance(choices_raw, (list, tuple)):
            raise ValueError(
                f"Invalid backend option schema choices for '{key}': expected list/tuple."
            )
        choices = tuple(choices_raw)
    validator_raw = raw_map.get("validator")
    if validator_raw is not None and not callable(validator_raw):
        raise ValueError(
            f"Invalid backend option validator for '{key}': expected callable."
        )
    validator = cast(OptionValidator | None, validator_raw)
    return BackendOptionSpec(
        value_type=value_type,
        default=default,
        choices=choices,
        validator=validator,
    )
