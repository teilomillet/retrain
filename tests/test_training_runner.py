"""Tests for the pluggable training runner system."""

import json
from dataclasses import MISSING, fields
from typing import cast
from unittest.mock import patch

import pytest

from retrain.config import TrainConfig
from retrain.ttt_discover import TTTDiscoverRunner
from retrain.training_runner import (
    CommandRunner,
    RetainRunner,
    SftRunner,
    TrainingRunResult,
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

    def test_ttt_discover_runner_is_training_runner(self):
        assert isinstance(TTTDiscoverRunner(), TrainingRunner)

    def test_sft_runner_is_training_runner(self):
        assert isinstance(SftRunner(), TrainingRunner)


class TestTrainingRunResult:
    def test_to_dict_exports_all_dataclass_fields(self):
        result = TrainingRunResult(policy_ref="/tmp/adapter", run_id="run-1")

        assert set(result.to_dict()) == {f.name for f in fields(TrainingRunResult)}

    def test_to_dict_snapshots_mutable_payloads(self):
        result = TrainingRunResult(
            metrics={"loss": 0.5, "nested": {"values": [1, 2]}},
            artifacts={"metrics.jsonl": "/tmp/metrics.jsonl"},
        )

        payload = result.to_dict()
        metrics = cast(dict[str, object], payload["metrics"])
        artifacts = cast(dict[str, str], payload["artifacts"])
        nested = cast(dict[str, object], metrics["nested"])
        values = cast(list[int], nested["values"])
        assert isinstance(metrics, dict)
        assert isinstance(artifacts, dict)

        metrics["loss"] = 0.0
        values.append(3)
        artifacts["metrics.jsonl"] = "changed"

        assert result.metrics == {"loss": 0.5, "nested": {"values": [1, 2]}}
        assert result.artifacts == {"metrics.jsonl": "/tmp/metrics.jsonl"}


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
        assert isinstance(result, TrainingRunResult)
        assert result.ok
        assert result.policy_ref == "/tmp/adapter"

    @patch("retrain.trainer.train")
    def test_returns_failed_result_when_train_returns_none(self, mock_train):
        mock_train.return_value = None
        runner = RetainRunner()
        result = runner.run(TrainConfig())
        assert not result.ok
        assert result.failure_status == "missing_policy_ref"

    @patch("retrain.trainer.train")
    def test_returns_failed_result_when_train_raises(self, mock_train):
        mock_train.side_effect = RuntimeError("boom")
        runner = RetainRunner()
        result = runner.run(TrainConfig())
        assert not result.ok
        assert result.failure_status == "exception:RuntimeError"
        assert result.error_message == "boom"


# ---------------------------------------------------------------------------
# CommandRunner
# ---------------------------------------------------------------------------


class TestCommandRunner:
    def test_runs_command_and_returns_adapter_path(self, tmp_path):
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "metrics.jsonl").write_text('{"step": 1, "loss": 0.5}\n')
        config = _bare_config(
            log_dir=str(log_dir),
            adapter_path=str(adapter_dir),
            model="test-model",
        )

        runner = CommandRunner("echo training with {model}")
        result = runner.run(config)
        assert result.ok
        assert result.policy_ref == str(adapter_dir)
        assert result.run_id == "logs"
        assert result.metrics["loss"] == pytest.approx(0.5)
        assert "metrics.jsonl" in result.artifacts

    def test_returns_failed_result_on_failure(self, tmp_path):
        config = _bare_config(
            log_dir=str(tmp_path / "logs"),
            adapter_path=str(tmp_path / "nonexistent_adapter"),
            model="test-model",
        )

        runner = CommandRunner("exit 1")
        result = runner.run(config)
        assert not result.ok
        assert result.failure_status == "exit_code:1"

    def test_returns_failed_result_when_adapter_missing(self, tmp_path):
        config = _bare_config(
            log_dir=str(tmp_path / "logs"),
            adapter_path=str(tmp_path / "no_adapter"),
            model="test-model",
        )

        runner = CommandRunner("echo ok")
        result = runner.run(config)
        assert not result.ok
        assert result.failure_status == "missing_policy_ref"


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
# SftRunner
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "".join(f"{m['role']}:{m['content']}\n" for m in messages)
        if add_generation_prompt:
            rendered += "assistant:"
        return rendered

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


class _FakeSftHelper:
    def __init__(self):
        self.sft_loss_fn = ""
        self.calls = []
        self.loaded = []
        self.saved = []
        self.shutdown_called = False

    def sft_train_step(self, all_tokens, all_advantages, lr, weight_decay):
        self.calls.append(
            {
                "tokens": all_tokens,
                "advantages": all_advantages,
                "lr": lr,
                "weight_decay": weight_decay,
                "loss_fn": self.sft_loss_fn,
            }
        )
        return 0.25

    def save_adapter(self, path, name):
        from pathlib import Path

        save_dir = Path(path) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "adapter_model.safetensors").write_text("fake")
        self.saved.append((path, name))
        return str(save_dir)

    def load_state(self, ref):
        self.loaded.append(ref)

    def runtime_metrics(self):
        return {
            "fake_metric": 1,
            "none_metric": None,
            "object_metric": object(),
            123: "bad_key",
        }

    def shutdown(self):
        self.shutdown_called = True


class _NonCallableLoadStateSftHelper(_FakeSftHelper):
    load_state = 1


class _FakeBackendRegistry:
    def __init__(self, helper):
        self.helper = helper

    def create(self, name, config):
        assert name == "unsloth"
        return self.helper


class TestSftRunner:
    def test_runs_sft_without_rl_dataset_or_environment(self, tmp_path, monkeypatch):
        data_path = tmp_path / "sft.jsonl"
        data_path.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "prompt"},
                        {"role": "assistant", "content": "answer"},
                    ]
                }
            )
            + "\n"
        )
        adapter_path = tmp_path / "adapter"
        log_dir = tmp_path / "logs"
        helper = _FakeSftHelper()

        def fake_get_registry(name):
            assert name == "backend"
            return _FakeBackendRegistry(helper)

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda *args, **kwargs: _FakeTokenizer(),
        )
        monkeypatch.setattr("retrain.registry.get_registry", fake_get_registry)

        config = TrainConfig(
            trainer="sft",
            backend="unsloth",
            sft_data_path=str(data_path),
            sft_batch_size=1,
            sft_max_tokens=0,
            max_steps=2,
            save_every=1,
            batch_size=8,
            lr=1e-4,
            sft_lr=2e-4,
            model="fake-model",
            adapter_path=str(adapter_path),
            log_dir=str(log_dir),
        )

        result = SftRunner().run(config)

        assert result.ok
        assert result.policy_ref == str(adapter_path / "final")
        assert len(helper.calls) == 2
        assert helper.calls[0]["loss_fn"] == "cross_entropy"
        assert helper.calls[0]["lr"] == pytest.approx(2e-4)
        assert helper.shutdown_called is True

        metrics = [
            json.loads(line)
            for line in (log_dir / "metrics.jsonl").read_text().splitlines()
        ]
        assert metrics[-1]["phase"] == "sft"
        assert metrics[-1]["backend"] == "unsloth"
        assert metrics[-1]["sft_loss_fn"] == "cross_entropy"
        assert metrics[-1]["backend/fake_metric"] == 1
        assert "backend/none_metric" not in metrics[-1]
        assert "backend/object_metric" not in metrics[-1]
        assert "backend/123" not in metrics[-1]

        state = json.loads((log_dir / "trainer_state.json").read_text())
        assert state["checkpoint_name"] == "final"
        assert state["checkpoint_path"] == str(adapter_path / "final")
        assert (log_dir / "latest_sampler_path.txt").read_text().strip() == str(
            adapter_path / "final"
        )
        manifest = json.loads((log_dir / "sft_manifest.json").read_text())
        assert manifest["kind"] == "retrain_sft_adapter"
        assert manifest["base_model"] == "fake-model"
        assert manifest["adapter_path"] == str(adapter_path / "final")
        assert manifest["huggingface"]["format"] == "peft_lora_adapter"
        assert manifest["ergonomics"]["no_rl_rollouts"] is True
        adapter_manifest = json.loads(
            (adapter_path / "final" / "retrain_sft_manifest.json").read_text()
        )
        assert adapter_manifest["resume"]["from"] == str(log_dir)
        assert result.artifacts["sft_manifest.json"] == str(log_dir / "sft_manifest.json")

    def test_sft_resume_from_loads_initial_adapter(self, tmp_path, monkeypatch):
        data_path = tmp_path / "sft.jsonl"
        data_path.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "prompt"},
                        {"role": "assistant", "content": "answer"},
                    ]
                }
            )
            + "\n"
        )
        adapter_path = tmp_path / "adapter"
        resume_from = tmp_path / "previous" / "final"
        log_dir = tmp_path / "logs"
        helper = _FakeSftHelper()

        def fake_get_registry(name):
            assert name == "backend"
            return _FakeBackendRegistry(helper)

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda *args, **kwargs: _FakeTokenizer(),
        )
        monkeypatch.setattr("retrain.registry.get_registry", fake_get_registry)

        config = TrainConfig(
            trainer="sft",
            backend="unsloth",
            sft_data_path=str(data_path),
            sft_batch_size=1,
            max_steps=1,
            batch_size=1,
            lr=1e-4,
            model="fake-model",
            adapter_path=str(adapter_path),
            resume_from=str(resume_from),
            log_dir=str(log_dir),
        )

        result = SftRunner().run(config)

        assert result.ok
        assert helper.loaded == [str(resume_from)]
        assert len(helper.calls) == 1

    def test_sft_resume_requires_callable_load_state(self, tmp_path, monkeypatch):
        data_path = tmp_path / "sft.jsonl"
        data_path.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "prompt"},
                        {"role": "assistant", "content": "answer"},
                    ]
                }
            )
            + "\n"
        )
        helper = _NonCallableLoadStateSftHelper()

        def fake_get_registry(name):
            assert name == "backend"
            return _FakeBackendRegistry(helper)

        monkeypatch.setattr("retrain.registry.get_registry", fake_get_registry)

        result = SftRunner().run(
            TrainConfig(
                trainer="sft",
                backend="unsloth",
                sft_data_path=str(data_path),
                max_steps=1,
                batch_size=1,
                model="fake-model",
                adapter_path=str(tmp_path / "adapter"),
                resume_from=str(tmp_path / "previous" / "final"),
                log_dir=str(tmp_path / "logs"),
            )
        )

        assert not result.ok
        assert result.failure_status == "exception:RuntimeError"
        assert "requires a backend with load_state()" in result.error_message
        assert helper.calls == []

    def test_missing_dataset_returns_failed_result(self, tmp_path):
        config = _bare_config(
            trainer="sft",
            sft_data_path=str(tmp_path / "missing.jsonl"),
            log_dir=str(tmp_path / "logs"),
        )
        result = SftRunner().run(config)
        assert not result.ok
        assert result.failure_status == "exception:FileNotFoundError"


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

    def test_ttt_discover_resolves(self):
        from retrain.registry import get_registry
        reg = get_registry("trainer")
        assert "ttt_discover" in reg.builtin_names

    def test_sft_resolves(self):
        from retrain.registry import get_registry
        reg = get_registry("trainer")
        assert "sft" in reg.builtin_names

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

    def test_create_ttt_discover_runner(self):
        from retrain.registry import get_registry
        config = _bare_config()
        runner = get_registry("trainer").create("ttt_discover", config)
        assert isinstance(runner, TTTDiscoverRunner)

    def test_create_sft_runner(self):
        from retrain.registry import get_registry
        config = _bare_config()
        runner = get_registry("trainer").create("sft", config)
        assert isinstance(runner, SftRunner)

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

    def test_ttt_discover_trainer_is_valid(self):
        config = TrainConfig(trainer="ttt_discover")
        assert config.trainer == "ttt_discover"

    def test_sft_requires_dataset_path(self):
        with pytest.raises(ValueError, match="sft_data_path"):
            TrainConfig(trainer="sft")

    def test_sft_with_dataset_path_ok(self):
        config = TrainConfig(trainer="sft", sft_data_path="/tmp/data.jsonl")
        assert config.trainer == "sft"


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

    def test_explain_sft_json_uses_sft_datums(self, tmp_path, capsys):
        toml_content = f"""\
[backend]
backend = "local"

[model]
model = "Qwen/Qwen3-4B-Instruct-2507"

[training]
trainer = "sft"
max_steps = 3
batch_size = 8
group_size = 16
sft_batch_size = 2
sft_data_path = "{tmp_path / 'sft.jsonl'}"
"""
        config_file = tmp_path / "sft.toml"
        config_file.write_text(toml_content)

        from retrain.cli import _explain_single
        _explain_single(str(config_file), "json")
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["trainer"] == "sft"
        assert data["condition"] == "sft"
        assert data["datums_per_step"] == 2
        assert data["total_datums"] == 6
