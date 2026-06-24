from __future__ import annotations

import subprocess
import sys
import importlib.util
from pathlib import Path


def test_unsloth_smoke_script_help_is_available_without_training_stack_imports():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "smoke_unsloth_backend.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--max-seq-length" in result.stdout
    assert "--require-cuda" in result.stdout


def test_unsloth_sft_smoke_script_help_is_available_without_training_stack_imports():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "smoke_unsloth_sft.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--max-seq-length" in result.stdout
    assert "--compare-to" in result.stdout


def _load_sft_smoke_module():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "smoke_unsloth_sft.py"
    spec = importlib.util.spec_from_file_location("smoke_unsloth_sft", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unsloth_sft_smoke_empty_policy_ref_is_not_current_directory():
    module = _load_sft_smoke_module()

    checks = module._adapter_checks("")

    assert checks == {
        "adapter_dir": "",
        "exists": False,
        "files": [],
        "has_peft_config": False,
        "has_adapter_weights": False,
        "has_retrain_manifest": False,
    }


def test_unsloth_sft_smoke_comparison_requires_measured_sft_peak():
    module = _load_sft_smoke_module()

    assert module._comparison_payload(3000.0, 0.0) == {
        "baseline_peak_reserved_mb": 3000.0,
        "comparison_available": False,
    }


def test_unsloth_sft_smoke_comparison_reports_equal_peak_as_not_above():
    module = _load_sft_smoke_module()

    payload = module._comparison_payload(2926.0, 2926.0)

    assert payload["comparison_available"] is True
    assert payload["sft_reserved_below_baseline"] is False
    assert payload["sft_reserved_not_above_baseline"] is True
    assert payload["sft_peak_reserved_delta_mb"] == 0.0
