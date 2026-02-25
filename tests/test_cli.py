"""Tests for retrain.cli — init subcommand and flag suggestions."""

import json
from pathlib import Path

import pytest

from retrain.cli import _print_top_help, _run_init, _run_man
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


class TestTopHelp:
    def test_help_mentions_manual(self, capsys):
        _print_top_help("retrain")
        captured = capsys.readouterr()
        assert "retrain man" in captured.out
        assert "retrain man --path" in captured.out
        assert "retrain man --sync" in captured.out
        assert "retrain man --check" in captured.out


class TestManCommand:
    def test_man_text_default(self, capsys):
        _run_man([])
        captured = capsys.readouterr()
        assert "RETRAIN(1)" in captured.out
        assert "COMMANDS" in captured.out
        assert "CONFIGURATION" in captured.out

    def test_man_json_format(self, capsys):
        _run_man(["--json"])
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["tool"] == "retrain"
        assert payload["path"].endswith("retrain/retrain.man")
        assert "manual" in payload

    def test_man_topic_json(self, capsys):
        _run_man(["--json", "--topic", "environments"])
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["topic"] == "environments"
        assert payload["section"] == "ENVIRONMENT"
        assert "primeintellect/gsm8k" in payload["content"]

    def test_man_path(self, capsys):
        _run_man(["--path"])
        captured = capsys.readouterr()
        assert captured.out.strip().endswith("retrain/retrain.man")

    def test_man_list_topics(self, capsys):
        _run_man(["--list-topics"])
        captured = capsys.readouterr()
        assert "environments" in captured.out
        assert "quickstart" in captured.out
        assert "configuration" in captured.out
        assert "campaign" in captured.out
        assert "options" in captured.out
        assert "squeeze" in captured.out
        assert "inference" in captured.out
        assert "validation" in captured.out

    def test_man_topic_configuration(self, capsys):
        _run_man(["--topic", "configuration"])
        captured = capsys.readouterr()
        assert "batch_size" in captured.out
        assert "transform_mode" in captured.out
        assert "[algorithm]" in captured.out

    def test_man_invalid_topic_exits(self):
        with pytest.raises(SystemExit):
            _run_man(["--topic", "nope"])

    def test_man_invalid_format_exits(self):
        with pytest.raises(SystemExit):
            _run_man(["--format", "yaml"])

    def test_man_sync_updates_auto_blocks(self, tmp_path, monkeypatch, capsys):
        import retrain.cli as cli

        manual = tmp_path / "retrain.man"
        manual.write_text(
            "RETRAIN(1)\n\n"
            "COMMANDS\n"
            "<<AUTO:COMMANDS>>\nold\n<<END:AUTO:COMMANDS>>\n\n"
            "OPTIONS\n"
            "<<AUTO:OPTIONS>>\nold\n<<END:AUTO:OPTIONS>>\n\n"
            "QUICKSTART\n"
            "<<AUTO:QUICKSTART>>\nold\n<<END:AUTO:QUICKSTART>>\n\n"
            "ENVIRONMENT\n"
            "<<AUTO:ENVIRONMENT>>\nold\n<<END:AUTO:ENVIRONMENT>>\n"
        )
        monkeypatch.setattr(cli, "_manual_path", lambda: manual)

        _run_man(["--sync"])
        captured = capsys.readouterr()
        assert str(manual) in captured.out

        synced = manual.read_text()
        assert "--sync            refreshes auto-generated manual blocks." in synced
        assert "--check           exits non-zero if auto blocks are stale." in synced
        assert "primeintellect/gsm8k" in synced

    def test_man_check_passes_when_up_to_date(self, tmp_path, monkeypatch, capsys):
        import retrain.cli as cli

        manual = tmp_path / "retrain.man"
        manual.write_text(
            "RETRAIN(1)\n\n"
            "COMMANDS\n"
            "<<AUTO:COMMANDS>>\nold\n<<END:AUTO:COMMANDS>>\n\n"
            "OPTIONS\n"
            "<<AUTO:OPTIONS>>\nold\n<<END:AUTO:OPTIONS>>\n\n"
            "QUICKSTART\n"
            "<<AUTO:QUICKSTART>>\nold\n<<END:AUTO:QUICKSTART>>\n\n"
            "ENVIRONMENT\n"
            "<<AUTO:ENVIRONMENT>>\nold\n<<END:AUTO:ENVIRONMENT>>\n"
        )
        monkeypatch.setattr(cli, "_manual_path", lambda: manual)

        _run_man(["--sync"])
        _run_man(["--check"])
        captured = capsys.readouterr()
        assert "(up to date)" in captured.out

    def test_man_check_fails_when_outdated(self, tmp_path, monkeypatch, capsys):
        import retrain.cli as cli

        manual = tmp_path / "retrain.man"
        manual.write_text(
            "RETRAIN(1)\n\n"
            "COMMANDS\n"
            "<<AUTO:COMMANDS>>\nold\n<<END:AUTO:COMMANDS>>\n\n"
            "OPTIONS\n"
            "<<AUTO:OPTIONS>>\nold\n<<END:AUTO:OPTIONS>>\n\n"
            "QUICKSTART\n"
            "<<AUTO:QUICKSTART>>\nold\n<<END:AUTO:QUICKSTART>>\n\n"
            "ENVIRONMENT\n"
            "<<AUTO:ENVIRONMENT>>\nold\n<<END:AUTO:ENVIRONMENT>>\n"
        )
        monkeypatch.setattr(cli, "_manual_path", lambda: manual)

        with pytest.raises(SystemExit) as exc_info:
            _run_man(["--check"])
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "(out of date). Run: retrain man --sync" in captured.err

    def test_man_sync_fails_when_markers_missing(
        self, tmp_path, monkeypatch, capsys
    ):
        import retrain.cli as cli

        manual = tmp_path / "retrain.man"
        manual.write_text("RETRAIN(1)\n\nCOMMANDS\nno auto markers\n")
        monkeypatch.setattr(cli, "_manual_path", lambda: manual)

        with pytest.raises(SystemExit) as exc_info:
            _run_man(["--sync"])
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Manual sync failed" in captured.err
        assert "Missing block markers for COMMANDS" in captured.err

    def test_man_check_fails_when_markers_missing(
        self, tmp_path, monkeypatch, capsys
    ):
        import retrain.cli as cli

        manual = tmp_path / "retrain.man"
        manual.write_text("RETRAIN(1)\n\nCOMMANDS\nno auto markers\n")
        monkeypatch.setattr(cli, "_manual_path", lambda: manual)

        with pytest.raises(SystemExit) as exc_info:
            _run_man(["--check"])
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Manual check failed" in captured.err
        assert "Missing block markers for COMMANDS" in captured.err

    def test_man_sync_and_check_together_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            _run_man(["--sync", "--check"])
        assert exc_info.value.code == 1
