"""Unit tests for retrain.verifiers_bridge helper functions."""

from __future__ import annotations

import pytest

from retrain.verifiers_bridge import (
    encode_prompt_for_sampling,
    parse_environment_args,
    prompt_preview,
)


class _DummyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True):
        # Stable synthetic IDs for tests.
        size = len(messages)
        return [10 + size, 20 + size]

    def encode(self, text):
        return [len(text), len(text) + 1]


class TestParseEnvironmentArgs:
    def test_empty_args(self):
        assert parse_environment_args("") == {}
        assert parse_environment_args("   ") == {}
        assert parse_environment_args(None) == {}

    def test_dict_args_passthrough(self):
        assert parse_environment_args({"game": "Wordle-v0", "seed": 42}) == {
            "game": "Wordle-v0",
            "seed": 42,
        }

    def test_valid_json_object(self):
        parsed = parse_environment_args('{"game":"Wordle-v0","seed":42}')
        assert parsed == {"game": "Wordle-v0", "seed": 42}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="must be valid JSON"):
            parse_environment_args("{bad-json")

    def test_non_object_json_raises(self):
        with pytest.raises(ValueError, match="JSON object"):
            parse_environment_args('["not","object"]')

    def test_non_str_non_dict_raises(self):
        with pytest.raises(ValueError, match="JSON string/object"):
            parse_environment_args(123)  # type: ignore[arg-type]


class TestPromptHelpers:
    def test_prompt_preview_string(self):
        assert prompt_preview("hello world", max_chars=5) == "hello"

    def test_prompt_preview_messages(self):
        preview = prompt_preview(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Solve 2+2"},
            ]
        )
        assert "system: You are helpful." in preview
        assert "user: Solve 2+2" in preview

    def test_encode_prompt_string(self):
        tok = _DummyTokenizer()
        ids = encode_prompt_for_sampling(tok, "test prompt")
        assert ids == [11, 21]

    def test_encode_prompt_messages(self):
        tok = _DummyTokenizer()
        ids = encode_prompt_for_sampling(
            tok,
            [{"role": "user", "content": "test prompt"}],
        )
        assert ids == [11, 21]
