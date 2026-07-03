from __future__ import annotations

import subprocess
import sys
import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest


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


def test_unsloth_sft_usl_sweep_help_is_available_without_training_stack_imports():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "usl_unsloth_sft_sweep.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert "--batch-sizes" in result.stdout
    assert "--microbatch-sizes" in result.stdout
    assert "--dry-run" in result.stdout


def _load_sft_smoke_module():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "smoke_unsloth_sft.py"
    spec = importlib.util.spec_from_file_location("smoke_unsloth_sft", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_sft_usl_sweep_module():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "usl_unsloth_sft_sweep.py"
    spec = importlib.util.spec_from_file_location("usl_unsloth_sft_sweep", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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


def test_unsloth_sft_usl_sweep_fit_recovers_synthetic_curve():
    module = _load_sft_usl_sweep_module()
    from retrain.training.backpressure import usl_throughput

    rows = []
    for batch_size in [1, 2, 4, 8, 16, 32]:
        throughput = 10.0 * usl_throughput(float(batch_size), 0.08, 0.003)
        rows.append(
            {
                "status": "succeeded",
                "batch_size": batch_size,
                "throughput_datums_per_s": throughput,
            }
        )

    fit = module._fit_usl(rows)

    assert fit is not None
    assert fit.to_dict() == asdict(fit)
    assert fit.sigma == pytest.approx(0.08, abs=0.02)
    assert fit.kappa == pytest.approx(0.003, abs=0.001)
    assert fit.r2 > 0.99


def test_unsloth_sft_usl_sweep_detects_15_percent_microbatch_gain():
    module = _load_sft_usl_sweep_module()
    rows = [
        {
            "status": "succeeded",
            "batch_size": 4,
            "train_microbatch_size": 1,
            "throughput_datums_per_s": 1.0,
        },
        {
            "status": "succeeded",
            "batch_size": 4,
            "train_microbatch_size": 0,
            "throughput_datums_per_s": 1.2,
        },
    ]

    summary = module._summarize(rows)

    best = summary["best_improvement"]
    assert best["batch_size"] == 4
    assert best["gain_percent"] == pytest.approx(20.0)
    assert best["meets_15_percent"] is True


def test_unsloth_sft_usl_sweep_summary_serializes_fit_payload():
    module = _load_sft_usl_sweep_module()
    from retrain.training.backpressure import usl_throughput

    rows = [
        {
            "status": "succeeded",
            "batch_size": batch_size,
            "train_microbatch_size": 1,
            "throughput_datums_per_s": 10.0
            * usl_throughput(float(batch_size), 0.08, 0.003),
        }
        for batch_size in [1, 2, 4, 8, 16, 32]
    ]

    summary = module._summarize(rows)

    fit_payload = summary["fits_by_microbatch_size"]["1"]
    assert isinstance(fit_payload, dict)
    assert fit_payload["sigma"] == pytest.approx(0.08, abs=0.02)
    assert "classification" in fit_payload
