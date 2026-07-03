"""Tests for retrain.json_utils."""

import pytest

from retrain import json_utils


def test_dumps_jsonl_appends_newline_and_preserves_unicode():
    row = json_utils.dumps_jsonl({"text": "π", "step": 1})

    assert row.endswith("\n")
    assert json_utils.loads(row) == {"text": "π", "step": 1}


def test_loads_accepts_memoryview_bytes():
    assert json_utils.loads(memoryview(b'{"ok": true}')) == {"ok": True}


def test_loads_raises_exported_decode_error():
    with pytest.raises(json_utils.JSONDecodeError):
        json_utils.loads("{")
