"""Tests for resume preflight checks."""

from __future__ import annotations

import hashlib
import json

import pytest

from retrain.commands.resume_check import run as run_resume_check
from retrain.config import TrainConfig
from retrain.training.resume_check import check_resume_dir
from retrain.training.state import save_trainer_state


def _write_adapter(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_model.safetensors").write_text("fake")


def _write_state(log_dir, checkpoint_path, *, step=1):
    save_trainer_state(
        log_dir,
        step=step,
        example_idx=4,
        total_correct=2,
        total_completions=8,
        current_batch_size=2,
        current_group_size=4,
        checkpoint_name=f"checkpoint_step_{step + 1}",
        checkpoint_path=str(checkpoint_path),
        resume_mode="adapter_only",
        resume_warning="optimizer/scaler/RNG state is not restored",
        sepa_state={},
    )


def test_resume_check_accepts_local_adapter_checkpoint(tmp_path):
    checkpoint = tmp_path / "adapters" / "checkpoint_step_2"
    _write_adapter(checkpoint)
    log_dir = tmp_path / "logs"
    _write_state(log_dir, checkpoint)

    result = check_resume_dir(log_dir)

    assert result.ok
    assert result.trainer_state_valid
    assert result.checkpoint_payload == "local_dir"
    assert result.adapter_payload_ok is True
    assert result.adapter_payload_files == ["adapter_model.safetensors"]
    assert result.step == 1
    assert result.next_step == 2
    assert result.resume_mode == "adapter_only"


def test_resume_check_uses_artifact_local_adapter_fallback(tmp_path):
    log_dir = tmp_path / "downloaded"
    missing_original = tmp_path / "dead-machine" / "checkpoint_step_2"
    artifact_adapter = log_dir / "adapter"
    _write_adapter(artifact_adapter)
    _write_state(log_dir, missing_original)

    result = check_resume_dir(log_dir)

    assert result.ok
    assert result.checkpoint_path == str(missing_original)
    assert result.resolved_checkpoint_path == str(artifact_adapter)
    assert result.adapter_payload_ok is True


def test_resume_check_fails_missing_adapter_payload(tmp_path):
    checkpoint = tmp_path / "adapters" / "checkpoint_step_2"
    checkpoint.mkdir(parents=True)
    log_dir = tmp_path / "logs"
    _write_state(log_dir, checkpoint)

    result = check_resume_dir(log_dir)

    assert not result.ok
    assert any(issue.code == "adapter_payload_missing" for issue in result.issues)


def test_resume_check_rejects_resume_beyond_config_max_steps(tmp_path):
    checkpoint = tmp_path / "adapters" / "checkpoint_step_3"
    _write_adapter(checkpoint)
    log_dir = tmp_path / "logs"
    _write_state(log_dir, checkpoint, step=2)
    config = TrainConfig(max_steps=2)

    result = check_resume_dir(log_dir, config=config, config_path="run.toml")

    assert not result.ok
    assert result.next_step == 3
    assert result.max_steps == 2
    assert result.max_steps_ok is False
    assert any(issue.code == "resume_beyond_max_steps" for issue in result.issues)


def test_resume_check_verifies_sft_config_data_contract(tmp_path):
    checkpoint = tmp_path / "adapters" / "checkpoint_step_2"
    _write_adapter(checkpoint)
    log_dir = tmp_path / "logs"
    _write_state(log_dir, checkpoint)
    data_path = tmp_path / "datasets" / "sft.jsonl"
    data_content = json.dumps({"text": "plain next-token text"}) + "\n"
    data_path.parent.mkdir()
    data_path.write_text(data_content)
    data_sha256 = hashlib.sha256(data_content.encode("utf-8")).hexdigest()
    config = TrainConfig(
        trainer="sft",
        sft_data_path=str(data_path),
        sft_data_sha256=data_sha256,
        sft_data_rows=1,
    )

    result = check_resume_dir(log_dir, config=config, config_path="sft.toml")

    assert result.ok
    assert result.sft_data["config_data_contract_ok"] is True
    assert result.sft_data["recoverable"] is True
    assert result.sft_data["config_data_sha256"] == data_sha256


def test_resume_check_reports_missing_sft_config_data(tmp_path):
    checkpoint = tmp_path / "adapters" / "checkpoint_step_2"
    _write_adapter(checkpoint)
    log_dir = tmp_path / "logs"
    _write_state(log_dir, checkpoint)
    config = TrainConfig(
        trainer="sft",
        sft_data_path=str(tmp_path / "missing.jsonl"),
    )

    result = check_resume_dir(log_dir, config=config, config_path="sft.toml")

    assert not result.ok
    assert any(issue.code == "sft_config_data_missing" for issue in result.issues)


def test_resume_check_rejects_invalid_sft_recoverability_without_data(tmp_path):
    checkpoint = tmp_path / "adapters" / "checkpoint_step_2"
    _write_adapter(checkpoint)
    log_dir = tmp_path / "logs"
    _write_state(log_dir, checkpoint)
    (log_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "kind": "retrain_resolved_config",
                "config": {
                    "trainer": "sft",
                    "max_steps": 4,
                    "sft_data_path": str(tmp_path / "missing.jsonl"),
                },
            }
        )
    )
    (log_dir / "sft_data_recoverability.json").write_text("{bad json")

    result = check_resume_dir(log_dir)

    assert not result.ok
    assert any(issue.code == "sft_recoverability_invalid" for issue in result.issues)
    assert any(issue.code == "sft_data_not_recoverable" for issue in result.issues)


def test_resume_check_uses_resolved_config_sft_data_without_config_arg(tmp_path):
    checkpoint = tmp_path / "adapters" / "checkpoint_step_2"
    _write_adapter(checkpoint)
    log_dir = tmp_path / "logs"
    _write_state(log_dir, checkpoint)
    data_path = tmp_path / "datasets" / "sft.jsonl"
    data_path.parent.mkdir()
    data_path.write_text(json.dumps({"text": "plain next-token text"}) + "\n")
    (log_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "kind": "retrain_resolved_config",
                "config": {
                    "trainer": "sft",
                    "max_steps": 4,
                    "sft_data_path": str(data_path),
                },
            }
        )
    )

    result = check_resume_dir(log_dir)

    assert result.ok
    assert result.config_source == "resolved_config.json"
    assert result.max_steps == 4
    assert result.sft_data["resolved_config_data_exists"] is True
    assert result.sft_data["recoverable"] is True


def test_resume_check_command_json_exits_zero_for_good_checkpoint(
    tmp_path,
    capsys,
):
    checkpoint = tmp_path / "adapters" / "checkpoint_step_2"
    _write_adapter(checkpoint)
    log_dir = tmp_path / "logs"
    _write_state(log_dir, checkpoint)

    run_resume_check(["--json", str(log_dir)])
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["checkpoint_payload"] == "local_dir"


def test_resume_check_command_exits_nonzero_for_bad_checkpoint(
    tmp_path,
    capsys,
):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        run_resume_check([str(log_dir)])

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "ok            : no" in out
    assert "trainer_state_missing" in out
