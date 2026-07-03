"""Parse model completions into OpenEnv gym actions.

Mirrors the action-parsing semantics of verifiers' OpenEnv integration so a
model trained through either path sees the same action contract: strip one
optional code fence, parse a JSON object, and fall back to wrapping plain
text in the schema's single required string field when there is one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from retrain.types import JSONObject


class ActionParseError(ValueError):
    """Completion text could not be converted into an action object."""


def strip_code_fence(text: str) -> str:
    if text.startswith("```") and text.endswith("```"):
        return "\n".join(text.split("\n")[1:-1]).strip()
    return text


def single_string_field(schema: Mapping[str, object]) -> str | None:
    """Name of the schema's lone string field, if the schema has exactly one.

    Lets bare-text completions drive single-field envs (e.g. a ``tool``
    command field) without demanding JSON, matching upstream behavior.
    """
    props = schema.get("properties")
    if not isinstance(props, dict):
        return None
    props_obj = cast(JSONObject, props)
    required = schema.get("required")
    if isinstance(required, list):
        required_names = [name for name in required if isinstance(name, str)]
        if len(required_names) == 1:
            spec = props_obj.get(required_names[0])
            if isinstance(spec, dict) and spec.get("type") == "string":
                return required_names[0]
    if len(props_obj) == 1:
        field_name, spec = next(iter(props_obj.items()))
        if isinstance(spec, dict) and spec.get("type") == "string":
            return field_name
    return None


def parse_action(text: str, schema: Mapping[str, object]) -> JSONObject:
    """Convert completion text into an action dict for ``step``."""
    cleaned = strip_code_fence(text.strip())
    try:
        action = json.loads(cleaned)
    except json.JSONDecodeError:
        action = None
    if isinstance(action, dict):
        return cast(JSONObject, action)

    field = single_string_field(schema)
    if field:
        return {field: cleaned}
    raise ActionParseError(
        "Failed to parse action JSON. "
        "Provide a single JSON object matching the action schema."
    )
