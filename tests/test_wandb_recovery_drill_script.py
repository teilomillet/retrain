from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_drill_module():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "wandb_recovery_drill.py"
    spec = importlib.util.spec_from_file_location("wandb_recovery_drill", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wandb_recovery_drill_help_is_available_without_wandb_import():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "wandb_recovery_drill.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert "--project" in result.stdout
    assert "--poll-attempts" in result.stdout
    assert "--cleanup" in result.stdout


def test_wandb_recovery_drill_artifact_ref_is_checkpoint_alias():
    module = _load_drill_module()

    ref = module._artifact_ref(
        entity="team",
        project="proj",
        run_id="abc123",
        alias="checkpoint_step_1",
    )

    assert ref == "team/proj/retrain-abc123-checkpoints:checkpoint_step_1"


def test_wandb_recovery_drill_detects_incomplete_download(tmp_path):
    module = _load_drill_module()
    downloaded = tmp_path / "artifact"
    downloaded.mkdir()
    (downloaded / "trainer_state.json").write_text("{}\n")

    try:
        module._assert_downloaded_artifact(downloaded)
    except RuntimeError as exc:
        assert "adapter_model.safetensors" in str(exc)
    else:
        raise AssertionError("expected incomplete artifact to fail")


def test_wandb_recovery_drill_rejects_cleanup_output_inside_root(tmp_path):
    module = _load_drill_module()

    args = argparse.Namespace(
        steps=2,
        resume_max_steps=2,
        root=str(tmp_path),
        cleanup=True,
        output=str(tmp_path / "evidence.json"),
    )

    try:
        module.run_drill(args)
    except ValueError as exc:
        assert "--output must be outside --root" in str(exc)
    else:
        raise AssertionError("expected cleanup/output conflict to fail")
