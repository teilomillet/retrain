"""Tests for retrain.io.json."""

import pytest

import retrain.io.json as json_codec


def test_dumps_jsonl_appends_newline_and_preserves_unicode():
    row = json_codec.dumps_jsonl({"text": "π", "step": 1})

    assert row.endswith("\n")
    assert json_codec.loads(row) == {"text": "π", "step": 1}


def test_loads_accepts_memoryview_bytes():
    assert json_codec.loads(memoryview(b'{"ok": true}')) == {"ok": True}


def test_loads_raises_exported_decode_error():
    with pytest.raises(json_codec.JSONDecodeError):
        json_codec.loads("{")
