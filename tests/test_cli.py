"""Tests for retrain.cli — init, explain, and flag suggestions."""

import json
from pathlib import Path

import pytest

from retrain.cli import (
    _INIT_TEMPLATES,
    _customize_toml,
    _print_top_help,
    _run_diff,
    _run_explain,
    _run_init,
    _run_init_interactive,
    _run_man,
)
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
        assert "plugins" in captured.out
        assert "glossary" in captured.out

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
        assert "--backend-opt K=V    backend-specific option override" in synced
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

    def test_man_troff_format(self, capsys):
        _run_man(["--format", "troff"])
        out = capsys.readouterr().out
        assert ".TH" in out
        assert ".SH" in out

    def test_man_html_format(self, capsys):
        _run_man(["--format", "html"])
        out = capsys.readouterr().out
        assert "<!DOCTYPE html>" in out
        assert "<h2" in out

    def test_man_topic_html(self, capsys):
        _run_man(["--format", "html", "--topic", "quickstart"])
        out = capsys.readouterr().out
        assert "<!DOCTYPE html>" in out
        assert "QUICKSTART" in out

    def test_man_topic_plugins(self, capsys):
        _run_man(["--topic", "plugins"])
        out = capsys.readouterr().out
        assert "PLUGINS" in out
        assert "TransformSpec" in out

    def test_man_topic_glossary(self, capsys):
        _run_man(["--topic", "glossary"])
        out = capsys.readouterr().out
        assert "GLOSSARY" in out
        assert "RLVR" in out
        assert "GRPO" in out

    def test_man_sync_and_check_together_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            _run_man(["--sync", "--check"])
        assert exc_info.value.code == 1


class TestInitTemplates:
    def test_list_templates(self, capsys, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _run_init(args=["--list"])
        captured = capsys.readouterr()
        assert "default" in captured.out
        assert "quickstart" in captured.out
        assert "experiment" in captured.out
        assert "campaign" in captured.out

    def test_quickstart_template(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _run_init(args=["--template", "quickstart"])
        content = (tmp_path / "retrain.toml").read_text()
        assert "max_steps = 20" in content
        assert 'transform_mode = "none"' in content

    def test_experiment_template(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _run_init(args=["--template", "experiment"])
        content = (tmp_path / "retrain.toml").read_text()
        assert "max_steps = 500" in content
        assert "seed = 42" in content
        assert "[sepa]" in content

    def test_campaign_template(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _run_init(args=["--template", "campaign"])
        content = (tmp_path / "campaign.toml").read_text()
        assert "[campaign]" in content
        assert "[[campaign.conditions]]" in content

    def test_unknown_template_exits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            _run_init(args=["--template", "nope"])
        assert exc_info.value.code == 1

    def test_default_template_unchanged(self, tmp_path, monkeypatch):
        """Default template produces same output as before."""
        monkeypatch.chdir(tmp_path)
        _run_init(args=["--template", "default"])
        content = (tmp_path / "retrain.toml").read_text()
        assert "[model]" in content
        assert "[training]" in content
        assert "max_steps = 100" in content

    def test_campaign_refuses_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "campaign.toml").write_text("existing")
        with pytest.raises(SystemExit) as exc_info:
            _run_init(args=["--template", "campaign"])
        assert exc_info.value.code == 1


class TestExplainCommand:
    def _write_single_toml(self, tmp_path: Path) -> Path:
        p = tmp_path / "test.toml"
        p.write_text(
            "[model]\n"
            'model = "Qwen/Qwen3-4B-Instruct-2507"\n'
            "lora_rank = 32\n\n"
            "[algorithm]\n"
            'advantage_mode = "grpo"\n'
            'transform_mode = "none"\n\n'
            "[training]\n"
            "max_steps = 50\n"
            "batch_size = 4\n"
            "group_size = 8\n\n"
            "[backend]\n"
            'adapter_path = "adapters/test"\n'
        )
        return p

    def _write_campaign_toml(self, tmp_path: Path) -> Path:
        p = tmp_path / "campaign.toml"
        p.write_text(
            "[campaign]\n"
            "seeds = [42, 101]\n"
            "max_steps = 100\n\n"
            "[[campaign.conditions]]\n"
            'advantage_mode = "grpo"\n'
            'transform_mode = "none"\n\n'
            "[[campaign.conditions]]\n"
            'advantage_mode = "maxrl"\n'
            'transform_mode = "gtpo_sepa"\n\n'
            "[model]\n"
            'model = "Qwen/Qwen3-4B-Instruct-2507"\n'
            "lora_rank = 32\n"
        )
        return p

    def test_explain_single_text(self, tmp_path, capsys):
        p = self._write_single_toml(tmp_path)
        _run_explain([str(p)])
        out = capsys.readouterr().out
        assert "grpo+none" in out
        assert "50" in out
        assert "batch_size" in out

    def test_explain_single_json(self, tmp_path, capsys):
        p = self._write_single_toml(tmp_path)
        _run_explain(["--json", str(p)])
        payload = json.loads(capsys.readouterr().out)
        assert payload["mode"] == "single"
        assert payload["condition"] == "grpo+none"
        assert payload["max_steps"] == 50
        assert payload["datums_per_step"] == 32  # 4 * 8

    def test_explain_campaign_text(self, tmp_path, capsys):
        p = self._write_campaign_toml(tmp_path)
        _run_explain([str(p)])
        out = capsys.readouterr().out
        assert "campaign" in out
        assert "grpo+none" in out
        assert "maxrl+gtpo_sepa" in out
        assert "4" in out  # total_runs = 2 conditions x 2 seeds

    def test_explain_campaign_json(self, tmp_path, capsys):
        p = self._write_campaign_toml(tmp_path)
        _run_explain(["--json", str(p)])
        payload = json.loads(capsys.readouterr().out)
        assert payload["mode"] == "campaign"
        assert payload["total_runs"] == 4
        assert len(payload["conditions"]) == 2

    def test_explain_no_config_exits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            _run_explain([])
        assert exc_info.value.code == 1

    def test_explain_missing_file_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            _run_explain(["/nonexistent/path.toml"])
        assert exc_info.value.code == 1


class TestCustomizeToml:
    def test_replace_max_steps(self):
        content = "max_steps = 100\nseed = -1\n"
        result = _customize_toml(content, max_steps=50)
        assert "max_steps = 50" in result
        assert "seed = -1" in result

    def test_replace_seed(self):
        content = "max_steps = 100\nseed = -1\n"
        result = _customize_toml(content, seed=99)
        assert "seed = 99" in result
        assert "max_steps = 100" in result

    def test_replace_negative_seed(self):
        content = "seed = -1\n"
        result = _customize_toml(content, seed=42)
        assert "seed = 42" in result

    def test_uncomment_wandb(self):
        content = '# wandb_project = ""         # uncomment to enable wandb\n'
        result = _customize_toml(content, wandb_project="my_proj")
        assert 'wandb_project = "my_proj"' in result

    def test_replace_existing_wandb(self):
        content = 'wandb_project = ""           # set your project name\n'
        result = _customize_toml(content, wandb_project="my_proj")
        assert 'wandb_project = "my_proj"' in result

    def test_no_changes_when_none(self):
        content = "max_steps = 100\nseed = -1\n"
        result = _customize_toml(content)
        assert result == content


class TestInitInteractive:
    def test_non_tty_exits(self, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with pytest.raises(SystemExit) as exc_info:
            _run_init_interactive("retrain")
        assert exc_info.value.code == 1

    def test_quickstart_defaults(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        inputs = iter(["1", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        _run_init_interactive("retrain")
        content = (tmp_path / "retrain.toml").read_text()
        assert "max_steps = 20" in content
        assert "seed = 42" in content

    def test_experiment_custom_steps(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        inputs = iter(["2", "300", "7", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        _run_init_interactive("retrain")
        content = (tmp_path / "retrain.toml").read_text()
        assert "max_steps = 300" in content
        assert "seed = 7" in content

    def test_campaign_with_wandb(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        inputs = iter(["3", "", "", "my_project"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        _run_init_interactive("retrain")
        content = (tmp_path / "campaign.toml").read_text()
        assert "[campaign]" in content
        assert "max_steps = 200" in content
        assert "seed = 42" not in content  # campaign template has no seed line

    def test_invalid_choice_falls_back(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        inputs = iter(["9", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        _run_init_interactive("retrain")
        captured = capsys.readouterr()
        assert "Invalid choice" in captured.err
        # Falls back to quickstart
        assert (tmp_path / "retrain.toml").is_file()

    def test_bad_steps_uses_default(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        inputs = iter(["1", "abc", "", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        _run_init_interactive("retrain")
        captured = capsys.readouterr()
        assert "Not a number" in captured.err
        content = (tmp_path / "retrain.toml").read_text()
        assert "max_steps = 20" in content

    def test_bad_seed_uses_default(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        inputs = iter(["1", "", "xyz", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        _run_init_interactive("retrain")
        captured = capsys.readouterr()
        assert "Not a number" in captured.err
        content = (tmp_path / "retrain.toml").read_text()
        assert "seed = 42" in content

    def test_refuses_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "retrain.toml").write_text("existing")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        inputs = iter(["1", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        with pytest.raises(SystemExit) as exc_info:
            _run_init_interactive("retrain")
        assert exc_info.value.code == 1
        assert (tmp_path / "retrain.toml").read_text() == "existing"

    def test_interactive_flag_in_run_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        inputs = iter(["1", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        _run_init(args=["--interactive"], cli_name="retrain")
        assert (tmp_path / "retrain.toml").is_file()


class TestAutoInit:
    def test_auto_init_tty_yes(self, tmp_path, monkeypatch, capsys):
        """TTY user accepts auto-init → creates retrain.toml."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _: "y")
        monkeypatch.setattr("sys.argv", ["retrain"])

        from retrain.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        assert (tmp_path / "retrain.toml").is_file()

    def test_auto_init_tty_enter(self, tmp_path, monkeypatch, capsys):
        """TTY user presses Enter (default yes) → creates retrain.toml."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _: "")
        monkeypatch.setattr("sys.argv", ["retrain"])

        from retrain.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        assert (tmp_path / "retrain.toml").is_file()

    def test_auto_init_tty_no(self, tmp_path, monkeypatch, capsys):
        """TTY user declines → exit 1, no file."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _: "n")
        monkeypatch.setattr("sys.argv", ["retrain"])

        from retrain.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        assert not (tmp_path / "retrain.toml").is_file()

    def test_auto_init_non_tty(self, tmp_path, monkeypatch, capsys):
        """Non-TTY → no prompt, exit 1."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("sys.argv", ["retrain"])

        from retrain.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        assert not (tmp_path / "retrain.toml").is_file()
        captured = capsys.readouterr()
        assert "retrain init" in captured.out
