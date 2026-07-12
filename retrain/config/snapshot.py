"""Secret-safe resolved configuration snapshots."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

if TYPE_CHECKING:
    from retrain.config.schema import TrainConfig

REDACTED_CONFIG_VALUE = "<redacted>"

_SENSITIVE_CONFIG_KEYS = {
    "access_key",
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "bearer_token",
    "credential",
    "credentials",
    "hf_token",
    "openai_api_key",
    "password",
    "passwd",
    "refresh_token",
    "secret",
}
_AMBIGUOUS_SENSITIVE_KEYS = {
    "auth",
    "key",
    "sig",
    "signature",
    "token",
}
_SENSITIVE_URL_QUERY_KEYS = _SENSITIVE_CONFIG_KEYS | _AMBIGUOUS_SENSITIVE_KEYS
_OPAQUE_CONFIG_FIELDS = {"trainer_command"}
_JSON_STRING_CONFIG_FIELDS = {"environment_args"}


def is_sensitive_config_key(
    key: object,
    *,
    include_ambiguous: bool = True,
) -> bool:
    """Return whether a config key conventionally contains a credential."""

    normalized = str(key).lower()
    return (
        normalized in _SENSITIVE_CONFIG_KEYS
        or (include_ambiguous and normalized in _AMBIGUOUS_SENSITIVE_KEYS)
        or normalized.endswith("_api_key")
        or normalized.endswith("_password")
        or normalized.endswith("_secret")
        or normalized.endswith("_token")
        or "credential" in normalized
    )


def sanitize_config_value(
    value: object,
    *,
    key: object | None = None,
    redact_ambiguous_keys: bool = True,
) -> object:
    """Return JSON-shaped config provenance without persisted credentials."""

    if key is not None and is_sensitive_config_key(
        key,
        include_ambiguous=redact_ambiguous_keys,
    ):
        return REDACTED_CONFIG_VALUE
    if key in _OPAQUE_CONFIG_FIELDS and value:
        return REDACTED_CONFIG_VALUE
    if key in _JSON_STRING_CONFIG_FIELDS and isinstance(value, str) and value:
        return _sanitize_json_string(
            value,
            redact_ambiguous_keys=redact_ambiguous_keys,
        )
    if isinstance(value, Mapping):
        return {
            str(nested_key): sanitize_config_value(
                nested_value,
                key=nested_key,
                redact_ambiguous_keys=redact_ambiguous_keys,
            )
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            sanitize_config_value(
                item,
                redact_ambiguous_keys=redact_ambiguous_keys,
            )
            for item in value
        ]
    if isinstance(value, str):
        return _sanitize_url(value)
    return value


def config_snapshot(config: TrainConfig) -> dict[str, object]:
    """Return a secret-safe, fully resolved TrainConfig snapshot."""

    return {
        field.name: sanitize_config_value(
            getattr(config, field.name),
            key=field.name,
        )
        for field in fields(config)
    }


def _sanitize_json_string(
    value: str,
    *,
    redact_ambiguous_keys: bool,
) -> str:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return REDACTED_CONFIG_VALUE
    sanitized = sanitize_config_value(
        parsed,
        redact_ambiguous_keys=redact_ambiguous_keys,
    )
    return json.dumps(sanitized, sort_keys=True, separators=(",", ":"))


def _sanitize_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return REDACTED_CONFIG_VALUE if "://" in value else value
    if not parts.scheme or not parts.netloc:
        return value

    hostname = parts.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    host = f"{hostname}:{port}" if port is not None else hostname
    userinfo = (
        f"{REDACTED_CONFIG_VALUE}@"
        if parts.username is not None or parts.password is not None
        else ""
    )
    query = urlencode(
        [
            (
                query_key,
                (
                    REDACTED_CONFIG_VALUE
                    if (
                        query_key.lower() in _SENSITIVE_URL_QUERY_KEYS
                        or is_sensitive_config_key(query_key)
                    )
                    else query_value
                ),
            )
            for query_key, query_value in parse_qsl(
                parts.query,
                keep_blank_values=True,
            )
        ]
    )
    fragment = REDACTED_CONFIG_VALUE if parts.fragment else ""
    return urlunsplit((parts.scheme, f"{userinfo}{host}", parts.path, query, fragment))
