"""Tests for the energy retrain pipeline approved-export handoff."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_ENERGY_PIPELINE_PATH = (
    Path(__file__).resolve().parents[1] / "campaigns" / "energy_pipeline.py"
)
_ENERGY_PIPELINE_SPEC = importlib.util.spec_from_file_location(
    "energy_pipeline_test_module",
    _ENERGY_PIPELINE_PATH,
)
assert (
    _ENERGY_PIPELINE_SPEC is not None
    and _ENERGY_PIPELINE_SPEC.loader is not None
)
_ENERGY_PIPELINE = importlib.util.module_from_spec(_ENERGY_PIPELINE_SPEC)
sys.modules[_ENERGY_PIPELINE_SPEC.name] = _ENERGY_PIPELINE
_ENERGY_PIPELINE_SPEC.loader.exec_module(_ENERGY_PIPELINE)

PipelineConfig = _ENERGY_PIPELINE.PipelineConfig
_apply_approved_export_handoff = _ENERGY_PIPELINE._apply_approved_export_handoff


def test_approved_export_handoff_resolves_relative_paths(tmp_path):
    export_root = tmp_path / "approved-export"
    sft_dir = export_root / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = sft_dir / "energy_candidate_sft.jsonl"
    dataset_path.write_text('{"messages": []}\n', encoding="utf-8")

    export_bundle = {
        "schema_version": "soma_approved_candidate_export_v1",
        "export_id": "energy.job123.abc123",
        "domain": "energy",
        "candidate": {
            "job_id": "job123",
            "model": "candidate/model",
        },
        "retrain_input": {
            "domain": "energy",
            "sft_data_path": "sft/energy_candidate_sft.jsonl",
        },
    }
    export_index_path = export_root / "export.json"
    export_index_path.write_text(
        json.dumps(export_bundle, indent=2),
        encoding="utf-8",
    )

    cfg = PipelineConfig()
    loaded = _apply_approved_export_handoff(cfg, export_index_path)

    assert loaded["export_id"] == "energy.job123.abc123"
    assert cfg.approved_export_path == str(export_index_path.resolve())
    assert cfg.model == "candidate/model"
    assert cfg.sft_data_path == str(dataset_path.resolve())
    assert cfg.artifact_dir == str(export_root / "retrain_runs" / "energy.job123.abc123")


def test_pipeline_config_defaults_match_budgeted_energy_paths() -> None:
    cfg = PipelineConfig()

    assert cfg.sft_data_path.endswith("python/data/sft/energy_search_sft.train.jsonl")
    assert cfg.num_examples == 800
