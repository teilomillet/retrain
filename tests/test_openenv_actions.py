"""Action-parsing tests: same contract as verifiers' OpenEnv integration."""

from __future__ import annotations

import pytest

from retrain.environments.openenv.actions import (
    ActionParseError,
    parse_action,
    single_string_field,
    strip_code_fence,
)

_SINGLE_FIELD_SCHEMA = {
    "properties": {"tool": {"type": "string"}},
    "required": ["tool"],
}
_MULTI_FIELD_SCHEMA = {
    "properties": {
        "tool": {"type": "string"},
        "args": {"type": "object"},
    },
    "required": ["tool", "args"],
}


class TestParseAction:
    def test_json_object_passes_through(self):
        assert parse_action('{"tool": "ls"}', {}) == {"tool": "ls"}

    def test_code_fence_is_stripped(self):
        text = '```json\n{"tool": "ls"}\n```'
        assert parse_action(text, {}) == {"tool": "ls"}

    def test_bare_text_wraps_into_single_string_field(self):
        assert parse_action("ls -la", _SINGLE_FIELD_SCHEMA) == {"tool": "ls -la"}

    def test_bare_text_without_single_field_schema_raises(self):
        with pytest.raises(ActionParseError, match="JSON object"):
            parse_action("not json", _MULTI_FIELD_SCHEMA)

    def test_json_array_is_not_an_action(self):
        with pytest.raises(ActionParseError):
            parse_action('["a", "b"]', _MULTI_FIELD_SCHEMA)


class TestSingleStringField:
    def test_lone_required_string_field(self):
        assert single_string_field(_SINGLE_FIELD_SCHEMA) == "tool"

    def test_multiple_required_fields_disqualify(self):
        assert single_string_field(_MULTI_FIELD_SCHEMA) is None

    def test_single_property_without_required(self):
        schema = {"properties": {"cmd": {"type": "string"}}}
        assert single_string_field(schema) == "cmd"

    def test_non_string_single_field_disqualifies(self):
        schema = {"properties": {"n": {"type": "integer"}}}
        assert single_string_field(schema) is None


class TestStripCodeFence:
    def test_unfenced_text_unchanged(self):
        assert strip_code_fence('{"a": 1}') == '{"a": 1}'

    def test_fence_with_language_tag(self):
        assert strip_code_fence("```json\n{}\n```") == "{}"
