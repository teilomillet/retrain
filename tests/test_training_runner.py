"""Tests for the pluggable training runner system."""

import json
from dataclasses import MISSING, fields
from unittest.mock import patch

import pytest

from retrain.config import TrainConfig
from retrain.training_runner import (
    CommandRunner,
    RetainRunner,
    TrainingRunner,
)


def _bare_config(**overrides: object) -> TrainConfig:
    """Build a TrainConfig with defaults, skipping __post_init__ validation."""
    config = TrainConfig.__new__(TrainConfig)
    for f in fields(TrainConfig):
        if f.default is not MISSING:
            setattr(config, f.name, f.default)
        elif f.default_factory is not MISSING:
            setattr(config, f.name, f.default_factory())
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_retain_runner_is_training_runner(self):
        assert isinstance(RetainRunner(), TrainingRunner)

    def test_command_runner_is_training_runner(self):
        assert isinstance(CommandRunner("echo ok"), TrainingRunner)


# ---------------------------------------------------------------------------
# RetainRunner
# ---------------------------------------------------------------------------


class TestRetainRunner:
    @patch("retrain.trainer.train")
    def test_delegates_to_train(self, mock_train):
        mock_train.return_value = "/tmp/adapter"
        runner = RetainRunner()
        config = TrainConfig()
        result = runner.run(config)
        mock_train.assert_called_once_with(config)
        assert result == "/tmp/adapter"

    @patch("retrain.trainer.train")
    def test_returns_none_when_train_returns_none(self, mock_train):
        mock_train.return_value = None
        runner = RetainRunner()
        result = runner.run(TrainConfig())
        assert result is None


# ---------------------------------------------------------------------------
# CommandRunner
# ---------------------------------------------------------------------------


class TestCommandRunner:
    def test_runs_command_and_returns_adapter_path(self, tmp_path):
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = _bare_config(
            log_dir=str(tmp_path / "logs"),
            adapter_path=str(adapter_dir),
            model="test-model",
        )

        runner = CommandRunner("echo training with {model}")
        result = runner.run(config)
        assert result == str(adapter_dir)

    def test_returns_none_on_failure(self, tmp_path):
        config = _bare_config(
            log_dir=str(tmp_path / "logs"),
            adapter_path=str(tmp_path / "nonexistent_adapter"),
            model="test-model",
        )

        runner = CommandRunner("exit 1")
        result = runner.run(config)
        assert result is None

    def test_returns_none_when_adapter_missing(self, tmp_path):
        config = _bare_config(
            log_dir=str(tmp_path / "logs"),
            adapter_path=str(tmp_path / "no_adapter"),
            model="test-model",
        )

        runner = CommandRunner("echo ok")
        result = runner.run(config)
        assert result is None


class TestCommandRunnerConfigExport:
    def test_writes_json_config(self, tmp_path):
        log_dir = tmp_path / "logs"
        config = _bare_config(
            log_dir=str(log_dir),
            adapter_path=str(tmp_path / "adapter"),
            model="test-model",
            lr=1e-4,
        )

        runner = CommandRunner("echo ok")
        runner.run(config)

        config_json = log_dir / "retrain_config.json"
        assert config_json.exists()
        data = json.loads(config_json.read_text())
        assert data["model"] == "test-model"
        assert data["lr"] == pytest.approx(1e-4)
        assert "trainer" in data

    def test_substitutes_placeholders(self, tmp_path):
        marker = tmp_path / "marker.txt"
        config = _bare_config(
            log_dir=str(tmp_path / "logs"),
            adapter_path=str(tmp_path / "adapter"),
            model="test-model",
        )

        cmd = f'echo "{{model}}" > {marker}'
        runner = CommandRunner(cmd)
        runner.run(config)

        assert marker.exists()
        assert marker.read_text().strip() == "test-model"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestTrainerRegistry:
    def test_retrain_resolves(self):
        from retrain.registry import get_registry
        reg = get_registry("trainer")
        assert "retrain" in reg.builtin_names

    def test_command_resolves(self):
        from retrain.registry import get_registry
        reg = get_registry("trainer")
        assert "command" in reg.builtin_names

    def test_create_retrain_runner(self):
        from retrain.registry import get_registry
        config = TrainConfig()
        runner = get_registry("trainer").create("retrain", config)
        assert isinstance(runner, RetainRunner)

    def test_create_command_runner(self):
        from retrain.registry import get_registry
        config = _bare_config(trainer_command="echo ok")
        runner = get_registry("trainer").create("command", config)
        assert isinstance(runner, CommandRunner)

    def test_command_without_trainer_command_raises(self):
        from retrain.registry import get_registry
        config = _bare_config(trainer_command="")
        with pytest.raises(ValueError, match="trainer_command"):
            get_registry("trainer").create("command", config)

    def test_unknown_trainer_raises(self):
        from retrain.registry import get_registry
        config = TrainConfig()
        with pytest.raises(ValueError, match="Unknown trainer"):
            get_registry("trainer").create("nonexistent", config)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestTrainerValidation:
    def test_command_without_trainer_command_raises(self):
        with pytest.raises(ValueError, match="trainer_command"):
            TrainConfig(trainer="command", trainer_command="")

    def test_command_with_trainer_command_ok(self):
        config = TrainConfig(trainer="command", trainer_command="echo ok")
        assert config.trainer == "command"
        assert config.trainer_command == "echo ok"

    def test_default_trainer_is_retrain(self):
        config = TrainConfig()
        assert config.trainer == "retrain"
        assert config.trainer_command == ""


# ---------------------------------------------------------------------------
# Explain output
# ---------------------------------------------------------------------------


class TestExplainShowsTrainer:
    def test_explain_json_includes_trainer(self, tmp_path, capsys):
        toml_content = """\
[model]
model = "Qwen/Qwen3-4B-Instruct-2507"

[training]
trainer = "retrain"
"""
        config_file = tmp_path / "test.toml"
        config_file.write_text(toml_content)

        from retrain.cli import _explain_single
        _explain_single(str(config_file), "json")
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "trainer" in data
        assert data["trainer"] == "retrain"

    def test_explain_text_includes_trainer(self, tmp_path, capsys):
        toml_content = """\
[model]
model = "Qwen/Qwen3-4B-Instruct-2507"
"""
        config_file = tmp_path / "test.toml"
        config_file.write_text(toml_content)

        from retrain.cli import _explain_single
        _explain_single(str(config_file), "text")
        output = capsys.readouterr().out
        assert "trainer" in output
