"""Tests for retrain.logging_utils — JsonlLogger."""

import json

import pytest

from retrain.logging_utils import JsonlLogger


class TestJsonlLogger:
    def test_writes_jsonl(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        logger = JsonlLogger(path)
        logger.log({"step": 0, "loss": 1.5})
        logger.log({"step": 1, "loss": 0.8})

        with open(path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert json.loads(lines[0]) == {"step": 0, "loss": 1.5}
        assert json.loads(lines[1]) == {"step": 1, "loss": 0.8}

    def test_appends(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        logger = JsonlLogger(path)
        logger.log({"a": 1})

        # New logger instance, same file
        logger2 = JsonlLogger(path)
        logger2.log({"b": 2})

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_unicode(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        logger = JsonlLogger(path)
        logger.log({"text": "π ≈ 3.14, ∑ℕ"})

        with open(path) as f:
            data = json.loads(f.readline())
        assert data["text"] == "π ≈ 3.14, ∑ℕ"

    def test_nested_dict(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        logger = JsonlLogger(path)
        logger.log({"metrics": {"loss": 0.5, "acc": 0.9}})

        with open(path) as f:
            data = json.loads(f.readline())
        assert data["metrics"]["loss"] == 0.5
