"""Built-in backend definitions and backend-option normalization."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from retrain.backends import TrainHelper
    from retrain.config import TrainConfig

from retrain.backends.options import (
    BackendOptionSpec,
    coerce_plugin_option_schema,
    local_option_schema,
    normalize_option_schema,
    prime_rl_option_schema,
    schema_to_payload,
    unsloth_option_schema,
)
from retrain.backends.create import (
    create_local,
    create_prime_rl,
    create_tinker,
    create_unsloth,
)


BackendFactory = Callable[["TrainConfig"], "TrainHelper"]


@dataclass(frozen=True)
class BackendCapabilities:
    """Core runtime capability metadata exposed by a backend."""

    reports_sync_loss: bool
    preserves_token_advantages: bool
    supports_checkpoint_resume: bool
    resume_runtime_dependent: bool
    supports_echo_shared_forward: bool = False


@dataclass(frozen=True)
class BackendDefinition:
    """Single source of truth for a built-in backend."""

    name: str
    factory: BackendFactory
    dependency_import: str
    dependency_hint: str
    capabilities: BackendCapabilities
    option_schema: dict[str, BackendOptionSpec] = field(default_factory=dict)


_BUILTIN_BACKENDS: dict[str, BackendDefinition] = {
    "local": BackendDefinition(
        name="local",
        factory=create_local,
        dependency_import="torch",
        dependency_hint="pip install retrain[local]",
        capabilities=BackendCapabilities(
            reports_sync_loss=True,
            preserves_token_advantages=True,
            supports_checkpoint_resume=True,
            resume_runtime_dependent=False,
            supports_echo_shared_forward=True,
        ),
        option_schema=local_option_schema(),
    ),
    "unsloth": BackendDefinition(
        name="unsloth",
        factory=create_unsloth,
        dependency_import="unsloth",
        dependency_hint="uv pip install unsloth --torch-backend=auto",
        capabilities=BackendCapabilities(
            reports_sync_loss=True,
            preserves_token_advantages=True,
            supports_checkpoint_resume=True,
            resume_runtime_dependent=False,
            supports_echo_shared_forward=True,
        ),
        option_schema=unsloth_option_schema(),
    ),
    "tinker": BackendDefinition(
        name="tinker",
        factory=create_tinker,
        dependency_import="tinker",
        dependency_hint="pip install retrain[tinker]",
        capabilities=BackendCapabilities(
            reports_sync_loss=True,
            preserves_token_advantages=True,
            supports_checkpoint_resume=True,
            resume_runtime_dependent=True,
            supports_echo_shared_forward=False,
        ),
        option_schema={},
    ),
    "prime_rl": BackendDefinition(
        name="prime_rl",
        factory=create_prime_rl,
        dependency_import="prime_rl",
        dependency_hint="pip install prime-rl",
        capabilities=BackendCapabilities(
            reports_sync_loss=False,
            preserves_token_advantages=False,
            supports_checkpoint_resume=True,
            resume_runtime_dependent=False,
            supports_echo_shared_forward=False,
        ),
        option_schema=prime_rl_option_schema(),
    ),
}

_PLUGIN_DEFAULT_CAPABILITIES = BackendCapabilities(
    reports_sync_loss=True,
    preserves_token_advantages=False,
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
                supports_echo_shared_forward=bool(
                    payload.get("supports_echo_shared_forward", False)
                ),
            )
        except KeyError:
            return None
    return None


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


def _capabilities_to_payload(caps: BackendCapabilities) -> dict[str, object]:
    return {
        "reports_sync_loss": caps.reports_sync_loss,
        "preserves_token_advantages": caps.preserves_token_advantages,
        "supports_checkpoint_resume": caps.supports_checkpoint_resume,
        "resume_runtime_dependent": caps.resume_runtime_dependent,
        "supports_echo_shared_forward": caps.supports_echo_shared_forward,
    }


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
                "option_schema": schema_to_payload(definition.option_schema),
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
            schema = coerce_plugin_option_schema(backend, raw_schema)
            if not schema:
                return options
            return normalize_option_schema(backend, options, schema)
        return options

    schema = _BUILTIN_BACKENDS[backend].option_schema
    return normalize_option_schema(backend, options, schema)
