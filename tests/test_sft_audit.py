"""Fail-closed coverage for the generic SFT audit binding."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from retrain.config import TrainConfig, load_config
from retrain.training.resume_check import check_resume_dir
from retrain.training.runner.sft import SftRunner
from retrain.training.sft import (
    SftDataProvenance,
    load_sft_dataset,
    verify_sft_data_contract,
)
from retrain.training.state import save_trainer_state
from retrain.training.warmup import load_sft_warmup_data


def _write_dataset(tmp_path: Path) -> tuple[Path, SftDataProvenance]:
    data_path = tmp_path / "datasets" / "sft.jsonl"
    data_path.parent.mkdir(parents=True)
    data_path.write_text(json.dumps({"text": "supervised target"}) + "\n")
    return data_path, load_sft_dataset(data_path).provenance


def _write_audit(
    tmp_path: Path,
    provenance: SftDataProvenance,
    *,
    status: str = "pass",
    corpus_mode: str = "replacement",
    lineage: Mapping[str, object] | None = None,
    data_sha256: str | None = None,
    data_rows: int | None = None,
) -> tuple[Path, str]:
    audit_path = tmp_path / "sft.audit.json"
    payload = {
        "schema": "retrain.sft_audit.v1",
        "status": status,
        "failed_checks": [] if status == "pass" else ["domain_guard"],
        "checks": {"domain_guard": status == "pass"},
        "audited_dataset": {
            "sha256": data_sha256 or provenance.data_sha256,
            "rows": (data_rows if data_rows is not None else provenance.data_rows),
        },
        "corpus_mode": corpus_mode,
        "lineage": dict(lineage or {}),
    }
    audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return audit_path, hashlib.sha256(audit_path.read_bytes()).hexdigest()


def _config(data_path: Path, audit_path: Path, audit_sha256: str, **kwargs):
    if kwargs.get("sft_token_audit_path"):
        kwargs.setdefault("model_revision", "audited-revision")
        kwargs.setdefault("model_local_files_only", True)
    return TrainConfig(
        sft_data_path=str(data_path),
        sft_audit_path=str(audit_path),
        sft_audit_sha256=audit_sha256,
        **kwargs,
    )


def test_audit_config_parses_and_requires_both_pins(tmp_path: Path):
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[training]
sft_data_path = "data/sft.jsonl"
sft_audit_path = "data/sft.audit.json"
sft_audit_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
"""
    )

    config = load_config(str(config_path))

    assert config.sft_audit_path == "data/sft.audit.json"
    assert config.sft_audit_sha256 == "a" * 64
    with pytest.raises(ValueError, match="must be configured together"):
        TrainConfig(sft_data_path="data.jsonl", sft_audit_path="audit.json")
    with pytest.raises(ValueError, match="sft_audit_path requires sft_data_path"):
        TrainConfig(sft_audit_path="audit.json", sft_audit_sha256="a" * 64)


def test_replacement_audit_binds_exact_loaded_dataset(tmp_path: Path):
    data_path, provenance = _write_dataset(tmp_path)
    audit_path, audit_sha = _write_audit(tmp_path, provenance)

    verified = verify_sft_data_contract(
        _config(data_path, audit_path, audit_sha),
        provenance,
    )

    assert verified is not None
    assert verified["audit_sha256"] == audit_sha
    assert verified["audited_dataset"] == {
        "sha256": provenance.data_sha256,
        "rows": 1,
    }
    assert verified["corpus_mode"] == "replacement"


def test_patch_audit_requires_explicit_base_and_patch_lineage(tmp_path: Path):
    data_path, provenance = _write_dataset(tmp_path)
    data_path.write_text(
        "".join(
            json.dumps({"text": f"supervised target {index}"}) + "\n"
            for index in range(3)
        )
    )
    provenance = load_sft_dataset(data_path).provenance
    missing_path, missing_sha = _write_audit(
        tmp_path,
        provenance,
        corpus_mode="patch",
    )

    with pytest.raises(ValueError, match="lineage.base"):
        verify_sft_data_contract(
            _config(data_path, missing_path, missing_sha),
            provenance,
        )

    lineage: dict[str, object] = {
        "base": {"sha256": "1" * 64, "rows": 2},
        "patch": {"sha256": "2" * 64, "rows": 1},
    }
    audit_path, audit_sha = _write_audit(
        tmp_path,
        provenance,
        corpus_mode="patch",
        lineage=lineage,
    )
    verified = verify_sft_data_contract(
        _config(data_path, audit_path, audit_sha),
        provenance,
    )

    assert verified is not None
    assert verified["corpus_mode"] == "patch"
    assert verified["lineage"] == lineage


def test_mode_lineage_semantics_match_emitter_contract(tmp_path: Path):
    data_path, provenance = _write_dataset(tmp_path)
    replacement_lineage: dict[str, object] = {"base": {"sha256": "1" * 64, "rows": 1}}
    audit_path, audit_sha = _write_audit(
        tmp_path,
        provenance,
        lineage=replacement_lineage,
    )
    with pytest.raises(ValueError, match="lineage must be empty"):
        verify_sft_data_contract(
            _config(data_path, audit_path, audit_sha),
            provenance,
        )

    bad_patch_lineage: dict[str, object] = {
        "base": {"sha256": "1" * 64, "rows": 1},
        "patch": {"sha256": "2" * 64, "rows": 1},
    }
    audit_path, audit_sha = _write_audit(
        tmp_path,
        provenance,
        corpus_mode="patch",
        lineage=bad_patch_lineage,
    )
    with pytest.raises(ValueError, match="must sum to audited_dataset.rows"):
        verify_sft_data_contract(
            _config(data_path, audit_path, audit_sha),
            provenance,
        )


def test_patch_with_token_audit_rechecks_source_bytes_and_rows(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.jsonl"
    patch_path = tmp_path / "patch.jsonl"
    combined_path = tmp_path / "datasets" / "combined.jsonl"
    combined_path.parent.mkdir()
    base_path.write_text(json.dumps({"text": "base"}) + "\n")
    patch_path.write_text(json.dumps({"text": "patch"}) + "\n")
    combined_path.write_bytes(base_path.read_bytes() + patch_path.read_bytes())
    provenance = replace(
        load_sft_dataset(combined_path).provenance,
        git_root="",
        data_root=str(combined_path.parent),
    )
    lineage: dict[str, object] = {
        "base": {
            "path": str(base_path.relative_to(tmp_path)),
            "sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
            "rows": 1,
        },
        "patch": {
            "path": str(patch_path.relative_to(tmp_path)),
            "sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
            "rows": 1,
        },
    }
    audit_path, audit_sha = _write_audit(
        tmp_path,
        provenance,
        corpus_mode="patch",
        lineage=lineage,
    )
    config = _config(
        combined_path,
        audit_path,
        audit_sha,
        sft_token_audit_path="configured-token-audit.json",
        sft_token_audit_sha256="a" * 64,
    )

    from retrain.training.sft_audit import verify_sft_audit_contract

    verified = verify_sft_audit_contract(config, provenance)
    assert verified is not None

    # Fresh training checkouts need only the tracked combined corpus and patch.
    base_path.unlink()
    assert verify_sft_audit_contract(config, provenance) is not None

    patch_path.write_text(
        json.dumps({"text": "changed"}) + "\n" + json.dumps({"text": "extra"}) + "\n"
    )
    with pytest.raises(ValueError, match="lineage.patch.sha256 mismatch") as exc_info:
        verify_sft_audit_contract(config, provenance)
    assert "lineage.patch.rows mismatch" in str(exc_info.value)
    assert "does not end byte-for-byte with lineage.patch" in str(exc_info.value)


def test_patch_source_proof_is_required_only_with_token_audit(
    tmp_path: Path,
) -> None:
    data_path, provenance = _write_dataset(tmp_path)
    lineage: dict[str, object] = {
        "base": {"sha256": "1" * 64, "rows": 1},
        "patch": {"sha256": "2" * 64, "rows": 0},
    }
    audit_path, audit_sha = _write_audit(
        tmp_path,
        provenance,
        corpus_mode="patch",
        lineage=lineage,
    )

    # Legacy v1 behavior remains path-free when no token audit is configured.
    with pytest.raises(ValueError, match="lineage.patch.rows must be a positive"):
        verify_sft_data_contract(
            _config(data_path, audit_path, audit_sha),
            provenance,
        )

    lineage["base"] = {"sha256": "1" * 64, "rows": 0}
    lineage["patch"] = {"sha256": "2" * 64, "rows": 1}
    audit_path, audit_sha = _write_audit(
        tmp_path,
        provenance,
        corpus_mode="patch",
        lineage=lineage,
    )
    legacy = _config(data_path, audit_path, audit_sha)
    from retrain.training.sft_audit import verify_sft_audit_contract

    # Existing semantics still reject zero-row lineage independently of paths.
    with pytest.raises(ValueError, match="lineage.base.rows must be a positive"):
        verify_sft_audit_contract(legacy, provenance)

    valid_path_free = {
        "base": {"sha256": "1" * 64, "rows": 1},
        "patch": {"sha256": "2" * 64, "rows": 1},
    }
    data_path.write_text(
        json.dumps({"text": "one"}) + "\n" + json.dumps({"text": "two"}) + "\n"
    )
    provenance = load_sft_dataset(data_path).provenance
    audit_path, audit_sha = _write_audit(
        tmp_path,
        provenance,
        corpus_mode="patch",
        lineage=valid_path_free,
    )
    assert (
        verify_sft_audit_contract(_config(data_path, audit_path, audit_sha), provenance)
        is not None
    )
    strict = _config(
        data_path,
        audit_path,
        audit_sha,
        sft_token_audit_path="configured-token-audit.json",
        sft_token_audit_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="lineage.patch.path must be"):
        verify_sft_audit_contract(strict, provenance)


def test_audit_rejects_hash_status_and_dataset_binding_mismatches(tmp_path: Path):
    data_path, provenance = _write_dataset(tmp_path)
    audit_path, audit_sha = _write_audit(tmp_path, provenance)
    with pytest.raises(ValueError, match="sft_audit_sha256 mismatch"):
        verify_sft_data_contract(
            _config(data_path, audit_path, "0" * 64),
            provenance,
        )

    audit_path, audit_sha = _write_audit(tmp_path, provenance, status="fail")
    with pytest.raises(ValueError, match="status must be 'pass'"):
        verify_sft_data_contract(
            _config(data_path, audit_path, audit_sha),
            provenance,
        )

    audit_path, audit_sha = _write_audit(
        tmp_path,
        provenance,
        data_sha256="3" * 64,
        data_rows=2,
    )
    with pytest.raises(ValueError, match="audited_dataset.sha256 mismatch") as exc_info:
        verify_sft_data_contract(
            _config(data_path, audit_path, audit_sha),
            provenance,
        )
    assert "audited_dataset.rows mismatch" in str(exc_info.value)


def test_explain_contract_path_rejects_failed_audit(tmp_path: Path):
    data_path, provenance = _write_dataset(tmp_path)
    audit_path, audit_sha = _write_audit(tmp_path, provenance, status="fail")
    config = _config(data_path, audit_path, audit_sha, trainer="sft")

    from retrain.commands.explain.single import _sft_provenance_info

    with pytest.raises(ValueError, match="status must be 'pass'"):
        _sft_provenance_info(config)


def test_warmup_contract_path_rejects_failed_audit(tmp_path: Path):
    data_path, provenance = _write_dataset(tmp_path)
    audit_path, audit_sha = _write_audit(tmp_path, provenance, status="fail")
    config = _config(
        data_path,
        audit_path,
        audit_sha,
        sft_warmup_steps=1,
    )

    with pytest.raises(ValueError, match="status must be 'pass'"):
        load_sft_warmup_data(config, object())


def test_standalone_run_rejects_failed_audit_before_backend_load(tmp_path: Path):
    data_path, provenance = _write_dataset(tmp_path)
    audit_path, audit_sha = _write_audit(tmp_path, provenance, status="fail")
    config = _config(
        data_path,
        audit_path,
        audit_sha,
        trainer="sft",
        model="unused-model",
        log_dir=str(tmp_path / "logs"),
    )

    result = SftRunner().run(config)

    assert not result.ok
    assert result.failure_status == "exception:ValueError"
    assert "status must be 'pass'" in result.error_message


def test_resume_check_contract_path_rejects_failed_audit(tmp_path: Path):
    data_path, provenance = _write_dataset(tmp_path)
    audit_path, audit_sha = _write_audit(tmp_path, provenance, status="fail")
    checkpoint = tmp_path / "adapter" / "checkpoint_step_2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_text("fake")
    log_dir = tmp_path / "logs"
    save_trainer_state(
        log_dir,
        step=1,
        example_idx=1,
        total_correct=0,
        total_completions=0,
        current_batch_size=1,
        current_group_size=1,
        checkpoint_name="checkpoint_step_2",
        checkpoint_path=str(checkpoint),
        sepa_state={},
    )
    config = _config(data_path, audit_path, audit_sha, trainer="sft")

    result = check_resume_dir(log_dir, config=config, config_path="run.toml")

    assert not result.ok
    issues = [
        issue for issue in result.issues if issue.code == "sft_config_data_invalid"
    ]
    assert len(issues) == 1
    assert "status must be 'pass'" in issues[0].message
