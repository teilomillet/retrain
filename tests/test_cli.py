"""Tests for retrain.cli — init subcommand and flag suggestions."""

from pathlib import Path

import pytest

from retrain.cli import _run_init
from retrain.config import parse_cli_overrides


class TestInit:
    def test_init_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _run_init()
        assert (tmp_path / "retrain.toml").is_file()
        content = (tmp_path / "retrain.toml").read_text()
        assert "[model]" in content
        assert "[training]" in content

    def test_init_refuses_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "retrain.toml").write_text("existing")
        with pytest.raises(SystemExit) as exc_info:
            _run_init()
        assert exc_info.value.code == 1
        # Original content preserved
        assert (tmp_path / "retrain.toml").read_text() == "existing"


class TestFlagSuggestion:
    def test_unknown_flag_shows_suggestion(self, capsys):
        with pytest.raises(SystemExit):
            parse_cli_overrides(["--sead", "42"])
        captured = capsys.readouterr()
        assert "Unknown flag" in captured.err
        assert "--seed" in captured.err
