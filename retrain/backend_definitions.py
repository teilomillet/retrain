"""Built-in backend definitions and backend-option normalization."""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class BackendOptionSpec:
    """Schema entry for one built-in backend option."""

    value_type: type
    default: object
    choices: tuple[object, ...] | None = None
    validator: Callable[[object], str | None] | None = None


@dataclass(frozen=True)
class BackendDefinition:
    """Single source of truth for a built-in backend."""

    name: str
    factory: Callable[[Any], Any]
    dependency_import: str
    dependency_hint: str
    option_schema: dict[str, BackendOptionSpec] = field(default_factory=dict)


def _create_local(config: Any) -> Any:
    try:
        from retrain.local_train_helper import LocalTrainHelper
    except ImportError:
        raise RuntimeError(
            "Backend 'local' requires PyTorch.\n"
            "Install it with: pip install retrain[local]"
        ) from None
    return LocalTrainHelper(
        config.model,
        config.adapter_path,
        config.devices,
        config.lora_rank,
        config.inference_engine,
        config.inference_url,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        optim_beta1=config.optim_beta1,
        optim_beta2=config.optim_beta2,
        optim_eps=config.optim_eps,
    )


def _create_tinker(config: Any) -> Any:
    try:
        from retrain.tinker_backend import TinkerTrainHelper
    except ImportError:
        raise RuntimeError(
            "Backend 'tinker' requires the tinker SDK.\n"
            "Install it with: pip install retrain[tinker]"
        ) from None

    tinker_url = config.inference_url or config.base_url
    return TinkerTrainHelper(
        config.model,
        tinker_url,
        config.lora_rank,
        optim_beta1=config.optim_beta1,
        optim_beta2=config.optim_beta2,
        optim_eps=config.optim_eps,
    )


def _create_prime_rl(config: Any) -> Any:
    try:
        from retrain.prime_rl_backend import PrimeRLTrainHelper
    except ImportError:
        raise RuntimeError(
            "Backend 'prime_rl' requires PRIME-RL.\n"
            "Install it with: pip install prime-rl"
        ) from None

    options = normalize_backend_options("prime_rl", config.backend_options)
    inference_url = config.inference_url or config.base_url or "http://localhost:8000"

    return PrimeRLTrainHelper(
        model_name=config.model,
        output_dir=config.adapter_path,
        inference_url=inference_url,
        transport_type=options["transport"],
        zmq_host=options["zmq_host"],
        zmq_port=options["zmq_port"],
        zmq_hwm=options["zmq_hwm"],
        strict_advantages=options["strict_advantages"],
        sync_wait_s=options["sync_wait_s"],
        sync_poll_s=options["sync_poll_s"],
    )


def _validate_port(value: object) -> str | None:
    v = int(value)
    if v <= 0 or v > 65535:
        return "must be in [1, 65535]. Try: 5555"
    return None


def _validate_positive_int(value: object) -> str | None:
    if int(value) <= 0:
        return "must be > 0"
    return None


def _validate_non_negative_int(value: object) -> str | None:
    if int(value) < 0:
        return "must be >= 0"
    return None


def _validate_positive_float(value: object) -> str | None:
    if float(value) <= 0:
        return "must be > 0"
    return None


_BUILTIN_BACKENDS: dict[str, BackendDefinition] = {
    "local": BackendDefinition(
        name="local",
        factory=_create_local,
        dependency_import="torch",
        dependency_hint="pip install retrain[local]",
        option_schema={},
    ),
    "tinker": BackendDefinition(
        name="tinker",
        factory=_create_tinker,
        dependency_import="tinker",
        dependency_hint="pip install retrain[tinker]",
        option_schema={},
    ),
    "prime_rl": BackendDefinition(
        name="prime_rl",
        factory=_create_prime_rl,
        dependency_import="prime_rl",
        dependency_hint="pip install prime-rl",
        option_schema={
            "transport": BackendOptionSpec(
                value_type=str,
                default="filesystem",
                choices=("filesystem", "zmq"),
            ),
            "zmq_host": BackendOptionSpec(value_type=str, default="localhost"),
            "zmq_port": BackendOptionSpec(
                value_type=int,
                default=5555,
                validator=_validate_port,
            ),
            "zmq_hwm": BackendOptionSpec(
                value_type=int,
                default=10,
                validator=_validate_positive_int,
            ),
            "strict_advantages": BackendOptionSpec(value_type=bool, default=True),
            "sync_wait_s": BackendOptionSpec(
                value_type=int,
                default=30,
                validator=_validate_non_negative_int,
            ),
            "sync_poll_s": BackendOptionSpec(
                value_type=float,
                default=0.2,
                validator=_validate_positive_float,
            ),
        },
    ),
}


def get_builtin_backend_definitions() -> dict[str, BackendDefinition]:
    """Return all built-in backend definitions keyed by backend name."""
    return dict(_BUILTIN_BACKENDS)


def get_backend_dependency_map() -> dict[str, tuple[str, str]]:
    """Return dependency metadata for built-in backends."""
    return {
        name: (definition.dependency_import, definition.dependency_hint)
        for name, definition in _BUILTIN_BACKENDS.items()
    }


def _coerce_option_value(backend: str, key: str, raw: object, spec: BackendOptionSpec) -> object:
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
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid [backend.options] {key}={raw!r} for backend '{backend}': "
                "expected an integer."
            ) from None
    elif spec.value_type is float:
        try:
            value = float(raw)
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


def normalize_backend_options(
    backend: str,
    raw_options: Mapping[str, object] | None,
) -> dict[str, object]:
    """Normalize backend options for built-in backends; pass through plugins."""
    options = {} if raw_options is None else dict(raw_options)
    if not isinstance(options, dict):
        raise ValueError("backend_options must be a mapping.")

    # For dotted plugins, options are backend-defined; keep as-is.
    if backend not in _BUILTIN_BACKENDS:
        if "." in backend:
            return options
        return options

    schema = _BUILTIN_BACKENDS[backend].option_schema
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
        normalized[key] = _coerce_option_value(backend, key, raw, spec)

    return normalized


def backend_options_to_json(raw_options: Mapping[str, object]) -> str:
    """Serialize backend options for CLI override transport."""
    return json.dumps(dict(raw_options), sort_keys=True)
