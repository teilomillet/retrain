"""Built-in backend definitions and backend-option normalization."""

from __future__ import annotations

import difflib
import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from retrain.backends import TrainHelper
    from retrain.config import TrainConfig


BackendFactory = Callable[["TrainConfig"], "TrainHelper"]
OptionValidator = Callable[[object], str | None]


class PrimeRLOptions(TypedDict):
    transport: str
    zmq_host: str
    zmq_port: int
    zmq_hwm: int
    strict_advantages: bool
    sync_wait_s: int
    sync_poll_s: float


@dataclass(frozen=True)
class BackendOptionSpec:
    """Schema entry for one built-in backend option."""

    value_type: type
    default: object
    choices: tuple[object, ...] | None = None
    validator: OptionValidator | None = None


@dataclass(frozen=True)
class BackendCapabilities:
    """Core runtime capability metadata exposed by a backend."""

    reports_sync_loss: bool
    preserves_token_advantages: bool
    supports_checkpoint_resume: bool
    resume_runtime_dependent: bool


@dataclass(frozen=True)
class BackendDefinition:
    """Single source of truth for a built-in backend."""

    name: str
    factory: BackendFactory
    dependency_import: str
    dependency_hint: str
    capabilities: BackendCapabilities
    option_schema: dict[str, BackendOptionSpec] = field(default_factory=dict)


def _create_local(config: "TrainConfig") -> "TrainHelper":
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


def _create_tinker(config: "TrainConfig") -> "TrainHelper":
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


def _normalize_prime_rl_options(raw_options: Mapping[str, object]) -> PrimeRLOptions:
    options = normalize_backend_options("prime_rl", raw_options)
    return {
        "transport": cast(str, options["transport"]),
        "zmq_host": cast(str, options["zmq_host"]),
        "zmq_port": cast(int, options["zmq_port"]),
        "zmq_hwm": cast(int, options["zmq_hwm"]),
        "strict_advantages": cast(bool, options["strict_advantages"]),
        "sync_wait_s": cast(int, options["sync_wait_s"]),
        "sync_poll_s": cast(float, options["sync_poll_s"]),
    }


def _create_prime_rl(config: "TrainConfig") -> "TrainHelper":
    try:
        from retrain.prime_rl_backend import PrimeRLTrainHelper
    except ImportError:
        raise RuntimeError(
            "Backend 'prime_rl' requires PRIME-RL.\n"
            "Install it with: pip install prime-rl"
        ) from None

    options = _normalize_prime_rl_options(config.backend_options)
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
    v = cast(int, value)
    if v <= 0 or v > 65535:
        return "must be in [1, 65535]. Try: 5555"
    return None


def _validate_positive_int(value: object) -> str | None:
    if cast(int, value) <= 0:
        return "must be > 0"
    return None


def _validate_non_negative_int(value: object) -> str | None:
    if cast(int, value) < 0:
        return "must be >= 0"
    return None


def _validate_positive_float(value: object) -> str | None:
    if cast(float, value) <= 0:
        return "must be > 0"
    return None


_BUILTIN_BACKENDS: dict[str, BackendDefinition] = {
    "local": BackendDefinition(
        name="local",
        factory=_create_local,
        dependency_import="torch",
        dependency_hint="pip install retrain[local]",
        capabilities=BackendCapabilities(
            reports_sync_loss=True,
            preserves_token_advantages=True,
            supports_checkpoint_resume=True,
            resume_runtime_dependent=False,
        ),
        option_schema={},
    ),
    "tinker": BackendDefinition(
        name="tinker",
        factory=_create_tinker,
        dependency_import="tinker",
        dependency_hint="pip install retrain[tinker]",
        capabilities=BackendCapabilities(
            reports_sync_loss=True,
            preserves_token_advantages=True,
            supports_checkpoint_resume=True,
            resume_runtime_dependent=True,
        ),
        option_schema={},
    ),
    "prime_rl": BackendDefinition(
        name="prime_rl",
        factory=_create_prime_rl,
        dependency_import="prime_rl",
        dependency_hint="pip install prime-rl",
        capabilities=BackendCapabilities(
            reports_sync_loss=False,
            preserves_token_advantages=False,
            supports_checkpoint_resume=True,
            resume_runtime_dependent=False,
        ),
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

_PLUGIN_DEFAULT_CAPABILITIES = BackendCapabilities(
    reports_sync_loss=True,
    preserves_token_advantages=True,
    supports_checkpoint_resume=True,
    resume_runtime_dependent=False,
)
_PLUGIN_CAPABILITIES_HOOKS = (
    "retrain_backend_capabilities",
    "RETRAIN_BACKEND_CAPABILITIES",
)
_PLUGIN_OPTION_SCHEMA_HOOKS = (
    "retrain_backend_option_schema",
    "RETRAIN_BACKEND_OPTION_SCHEMA",
)


def get_builtin_backend_definitions() -> dict[str, BackendDefinition]:
    """Return all built-in backend definitions keyed by backend name."""
    return dict(_BUILTIN_BACKENDS)


def get_backend_dependency_map() -> dict[str, tuple[str, str]]:
    """Return dependency metadata for built-in backends."""
    return {
        name: (definition.dependency_import, definition.dependency_hint)
        for name, definition in _BUILTIN_BACKENDS.items()
    }


def _coerce_backend_capabilities(raw: object) -> BackendCapabilities | None:
    if isinstance(raw, BackendCapabilities):
        return raw
    if isinstance(raw, Mapping):
        payload = cast(Mapping[str, object], raw)
        try:
            return BackendCapabilities(
                reports_sync_loss=bool(payload["reports_sync_loss"]),
                preserves_token_advantages=bool(payload["preserves_token_advantages"]),
                supports_checkpoint_resume=bool(payload["supports_checkpoint_resume"]),
                resume_runtime_dependent=bool(payload["resume_runtime_dependent"]),
            )
        except KeyError:
            return None
    return None


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


def _coerce_plugin_option_schema(
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


def _resolve_hook_value(raw: object, backend_options: Mapping[str, object]) -> object:
    if not callable(raw):
        return raw
    callback = cast(Callable[..., object], raw)
    try:
        return callback(dict(backend_options))
    except TypeError:
        return callback()


def _import_plugin_target(backend_name: str) -> tuple[object, object] | tuple[None, None]:
    module_path, _, attr_name = backend_name.rpartition(".")
    if not module_path or not attr_name:
        return None, None
    try:
        module = importlib.import_module(module_path)
    except Exception:
        return None, None
    target = getattr(module, attr_name, None)
    if target is None:
        return None, None
    return module, target


def _plugin_hook_value(
    backend_name: str,
    hook_names: tuple[str, ...],
    backend_options: Mapping[str, object],
) -> object | None:
    module, target = _import_plugin_target(backend_name)
    if module is None or target is None:
        return None
    for hook_name in hook_names:
        for holder in (target, module):
            raw = getattr(holder, hook_name, None)
            if raw is None:
                continue
            return _resolve_hook_value(raw, backend_options)
    return None


def resolve_backend_capabilities(
    backend_name: str,
    backend_options: Mapping[str, object] | None = None,
) -> BackendCapabilities:
    """Resolve capability metadata for built-in and dotted plugin backends."""
    options = {} if backend_options is None else dict(backend_options)
    definition = _BUILTIN_BACKENDS.get(backend_name)
    if definition is not None:
        return definition.capabilities
    if "." in backend_name:
        raw = _plugin_hook_value(
            backend_name,
            _PLUGIN_CAPABILITIES_HOOKS,
            options,
        )
        caps = _coerce_backend_capabilities(raw)
        if caps is not None:
            return caps
        return _PLUGIN_DEFAULT_CAPABILITIES
    return _PLUGIN_DEFAULT_CAPABILITIES


def backend_capability_source(
    backend_name: str,
    backend_options: Mapping[str, object] | None = None,
) -> str:
    """Human-readable source for capability resolution diagnostics."""
    options = {} if backend_options is None else dict(backend_options)
    if backend_name in _BUILTIN_BACKENDS:
        return "builtin"
    if "." in backend_name:
        raw = _plugin_hook_value(
            backend_name,
            _PLUGIN_CAPABILITIES_HOOKS,
            options,
        )
        if _coerce_backend_capabilities(raw) is not None:
            return "plugin/hook"
    return "plugin/default"


def _option_type_name(value_type: type) -> str:
    if value_type is bool:
        return "bool"
    if value_type is int:
        return "int"
    if value_type is float:
        return "float"
    if value_type is str:
        return "str"
    return getattr(value_type, "__name__", str(value_type))


def _capabilities_to_payload(caps: BackendCapabilities) -> dict[str, bool]:
    return {
        "reports_sync_loss": caps.reports_sync_loss,
        "preserves_token_advantages": caps.preserves_token_advantages,
        "supports_checkpoint_resume": caps.supports_checkpoint_resume,
        "resume_runtime_dependent": caps.resume_runtime_dependent,
    }


def _schema_to_payload(
    schema: Mapping[str, BackendOptionSpec],
) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for key, spec in schema.items():
        payload[key] = {
            "type": _option_type_name(spec.value_type),
            "default": spec.default,
            "choices": list(spec.choices) if spec.choices else None,
            "has_validator": spec.validator is not None,
        }
    return payload


def describe_backends_catalog() -> dict[str, object]:
    """Machine-readable catalog for built-in backends + plugin hook contract."""
    builtins: list[dict[str, object]] = []
    for name, definition in sorted(_BUILTIN_BACKENDS.items()):
        builtins.append(
            {
                "name": name,
                "dependency": {
                    "import": definition.dependency_import,
                    "hint": definition.dependency_hint,
                },
                "capabilities": _capabilities_to_payload(definition.capabilities),
                "option_schema": _schema_to_payload(definition.option_schema),
            }
        )
    return {
        "builtins": builtins,
        "plugin": {
            "dotted_path_supported": True,
            "default_capabilities": _capabilities_to_payload(
                _PLUGIN_DEFAULT_CAPABILITIES
            ),
            "capability_hooks": list(_PLUGIN_CAPABILITIES_HOOKS),
            "option_schema_hooks": list(_PLUGIN_OPTION_SCHEMA_HOOKS),
            "option_schema_format": (
                "mapping[str, BackendOptionSpec | "
                "{type|value_type, default, choices?, validator?}]"
            ),
        },
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


def normalize_backend_options(
    backend: str,
    raw_options: Mapping[str, object] | None,
) -> dict[str, object]:
    """Normalize backend options for built-in backends; pass through plugins."""
    options = {} if raw_options is None else dict(raw_options)
    if not isinstance(options, dict):
        raise ValueError("backend_options must be a mapping.")

    if backend not in _BUILTIN_BACKENDS:
        if "." in backend:
            raw_schema = _plugin_hook_value(
                backend,
                _PLUGIN_OPTION_SCHEMA_HOOKS,
                options,
            )
            schema = _coerce_plugin_option_schema(backend, raw_schema)
            if not schema:
                return options
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
