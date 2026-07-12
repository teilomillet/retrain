"""Fail-closed coverage for the runtime-sensitive SFT token audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from retrain.config import TrainConfig, load_config
from retrain.training.resume_check import check_resume_dir
from retrain.training.runner.sft import SftRunner
from retrain.training.sft import (
    load_sft_dataset,
    sft_tokenizer_load_kwargs,
    verify_sft_data_contract,
)
from retrain.training.state import save_trainer_state
import retrain.training.sft_token_audit as token_audit_module


MODEL = "example/model"
REVISION = "revision-123"
TRANSFORMERS_VERSION = "5.3.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TrainConfig, object, Path, list[tuple[str, str, str, bool]]]:
    dataset_path = tmp_path / "datasets" / "combined.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(json.dumps({"text": "supervised target"}) + "\n")
    provenance = replace(
        load_sft_dataset(dataset_path).provenance,
        git_root=str(tmp_path),
    )

    implementation_path = tmp_path / "eval" / "token_audit_impl.py"
    implementation_path.parent.mkdir()
    implementation_path.write_text("# exact audited implementation\n")

    snapshot_root = tmp_path / "cache" / "snapshots" / REVISION
    snapshot_root.mkdir(parents=True)
    tokenizer_path = snapshot_root / "tokenizer.json"
    tokenizer_path.write_text('{"version":"test"}\n')
    cached_calls: list[tuple[str, str, str, bool]] = []

    def cached_file(
        model: str,
        name: str,
        *,
        revision: str,
        local_files_only: bool,
    ) -> str | None:
        cached_calls.append((model, name, revision, local_files_only))
        return str(snapshot_root / name)

    monkeypatch.setattr(
        token_audit_module,
        "_transformers_runtime",
        lambda: (TRANSFORMERS_VERSION, cached_file),
    )
    payload = {
        "schema": "project.sft_token_audit.v1",
        "status": "pass",
        "dataset": {
            "path": "datasets/combined.jsonl",
            "sha256": provenance.data_sha256,
            "rows": provenance.data_rows,
        },
        "training_contract": {
            "trainer": "sft",
            "model": MODEL,
            "sft_max_tokens": 8,
            "transformers_version": TRANSFORMERS_VERSION,
            "tokenizer_revision": REVISION,
        },
        "rows_over_training_cap": 0,
        "tokens": {"max": 7},
        "implementation": [
            {
                "path": "eval/token_audit_impl.py",
                "sha256": _sha256(implementation_path),
            }
        ],
        "tokenizer_snapshot": [
            {"path": "tokenizer.json", "sha256": _sha256(tokenizer_path)}
        ],
    }
    audit_path = tmp_path / "datasets" / "combined.token-audit.json"
    audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    config = TrainConfig(
        trainer="sft",
        model=MODEL,
        model_revision=REVISION,
        model_local_files_only=True,
        max_tokens=8,
        sft_data_path=str(dataset_path),
        sft_token_audit_path=str(audit_path),
        sft_token_audit_sha256=_sha256(audit_path),
    )
    return config, provenance, audit_path, cached_calls


def _rewrite_audit(config: TrainConfig, audit_path: Path, payload: object) -> None:
    audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    config.sft_token_audit_sha256 = _sha256(audit_path)


def test_config_parses_and_validates_token_audit_pair(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[training]
trainer = "sft"
sft_data_path = "data/sft.jsonl"
sft_token_audit_path = "data/sft.token-audit.json"
sft_token_audit_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

[model]
revision = "revision-123"
local_files_only = true
"""
    )

    config = load_config(str(config_path))

    assert config.sft_token_audit_path == "data/sft.token-audit.json"
    assert config.sft_token_audit_sha256 == "a" * 64
    assert config.model_revision == REVISION
    assert config.model_local_files_only is True
    with pytest.raises(ValueError, match="must be configured together"):
        TrainConfig(
            sft_data_path="data.jsonl",
            sft_token_audit_path="audit.json",
            model_revision=REVISION,
            model_local_files_only=True,
        )
    with pytest.raises(
        ValueError,
        match="sft_token_audit_path requires sft_data_path",
    ):
        TrainConfig(
            sft_token_audit_path="audit.json",
            sft_token_audit_sha256="a" * 64,
            model_revision=REVISION,
            model_local_files_only=True,
        )
    with pytest.raises(ValueError, match="sft_token_audit_sha256"):
        TrainConfig(
            sft_data_path="data.jsonl",
            sft_token_audit_path="audit.json",
            sft_token_audit_sha256="not-a-sha",
        )
    with pytest.raises(ValueError, match="positive effective SFT token cap"):
        TrainConfig(
            max_tokens=-1024,
            sft_data_path="data.jsonl",
            sft_token_audit_path="audit.json",
            sft_token_audit_sha256="a" * 64,
        )


def test_valid_token_audit_binds_runtime_and_preserves_return_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, provenance, _, cached_calls = _fixture(tmp_path, monkeypatch)

    result = verify_sft_data_contract(config, provenance)

    assert result is None
    assert cached_calls
    assert set(cached_calls) == {(MODEL, "tokenizer.json", REVISION, True)}

    generic_path = tmp_path / "datasets" / "combined.sft-audit.json"
    generic_path.write_text(
        json.dumps(
            {
                "schema": "retrain.sft_audit.v1",
                "status": "pass",
                "audited_dataset": {
                    "sha256": provenance.data_sha256,
                    "rows": provenance.data_rows,
                },
                "corpus_mode": "replacement",
                "lineage": {},
            },
            sort_keys=True,
        )
        + "\n"
    )
    config.sft_audit_path = str(generic_path)
    config.sft_audit_sha256 = _sha256(generic_path)

    generic_result = verify_sft_data_contract(config, provenance)

    assert generic_result is not None
    assert generic_result["audit_sha256"] == config.sft_audit_sha256
    assert generic_result["corpus_mode"] == "replacement"


def test_audited_tokenizer_load_is_revision_pinned_and_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, provenance, _, cached_calls = _fixture(tmp_path, monkeypatch)

    kwargs = sft_tokenizer_load_kwargs(config, provenance)

    assert kwargs == {
        "trust_remote_code": True,
        "revision": REVISION,
        "local_files_only": True,
    }
    assert set(cached_calls) == {(MODEL, "tokenizer.json", REVISION, True)}


def test_token_audit_rejects_runtime_revision_or_network_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, provenance, _, _ = _fixture(tmp_path, monkeypatch)
    config.model_revision = "other-revision"

    with pytest.raises(ValueError, match=r"mismatch with \[model\] revision"):
        verify_sft_data_contract(config, provenance)

    config.model_revision = REVISION
    config.model_local_files_only = False
    with pytest.raises(ValueError, match="local_files_only must be true"):
        verify_sft_data_contract(config, provenance)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema=""), "schema must be a non-empty"),
        (lambda payload: payload.update(status="fail"), "status must be 'pass'"),
        (
            lambda payload: payload["dataset"].update(sha256="0" * 64),
            "dataset.sha256 mismatch",
        ),
        (
            lambda payload: payload["dataset"].update(rows=2),
            "dataset.rows mismatch",
        ),
        (
            lambda payload: payload["training_contract"].update(trainer="retrain"),
            "training_contract.trainer mismatch",
        ),
        (
            lambda payload: payload["training_contract"].update(model="other/model"),
            "training_contract.model mismatch",
        ),
        (
            lambda payload: payload["training_contract"].update(sft_max_tokens=9),
            "training_contract.sft_max_tokens mismatch",
        ),
        (
            lambda payload: payload.update(rows_over_training_cap=1),
            "rows_over_training_cap",
        ),
        (lambda payload: payload["tokens"].update(max=9), "tokens.max must be <="),
        (
            lambda payload: payload["training_contract"].update(
                transformers_version="0.0.0"
            ),
            "transformers_version mismatch",
        ),
        (
            lambda payload: payload["implementation"][0].update(sha256="0" * 64),
            "implementation[0].sha256 mismatch",
        ),
        (
            lambda payload: payload["training_contract"].update(
                tokenizer_revision="other-revision"
            ),
            "tokenizer_revision mismatch",
        ),
        (
            lambda payload: payload["tokenizer_snapshot"][0].update(sha256="0" * 64),
            "tokenizer_snapshot[0].sha256 mismatch",
        ),
    ],
)
def test_token_audit_rejects_stale_or_mismatched_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    config, provenance, audit_path, _ = _fixture(tmp_path, monkeypatch)
    payload = json.loads(audit_path.read_text())
    mutation(payload)
    _rewrite_audit(config, audit_path, payload)

    with pytest.raises(ValueError) as exc_info:
        verify_sft_data_contract(config, provenance)
    assert message in str(exc_info.value)


def test_token_audit_rejects_unpinned_byte_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, provenance, audit_path, _ = _fixture(tmp_path, monkeypatch)
    audit_path.write_text(audit_path.read_text() + "\n")

    with pytest.raises(ValueError, match="sft_token_audit_sha256 mismatch"):
        verify_sft_data_contract(config, provenance)


def test_explain_training_and_resume_all_enforce_token_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, audit_path, _ = _fixture(tmp_path, monkeypatch)
    payload = json.loads(audit_path.read_text())
    payload["status"] = "fail"
    _rewrite_audit(config, audit_path, payload)

    from retrain.commands.explain.single import _sft_provenance_info

    with pytest.raises(ValueError, match="status must be 'pass'"):
        _sft_provenance_info(config)

    config.log_dir = str(tmp_path / "training-logs")
    training = SftRunner().run(config)
    assert not training.ok
    assert "status must be 'pass'" in training.error_message

    checkpoint = tmp_path / "adapter" / "checkpoint_step_2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_text("fake")
    resume_dir = tmp_path / "resume-logs"
    save_trainer_state(
        resume_dir,
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
    resume = check_resume_dir(
        resume_dir,
        config=config,
        config_path="run.toml",
    )
    assert not resume.ok
    issues = [
        issue for issue in resume.issues if issue.code == "sft_config_data_invalid"
    ]
    assert len(issues) == 1
    assert "status must be 'pass'" in issues[0].message


def test_resume_schedule_binds_token_audit_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from retrain.training.sft import build_sft_resume_schedule_contract

    config, provenance, _, _ = _fixture(tmp_path, monkeypatch)
    contract = build_sft_resume_schedule_contract(
        config,
        provenance,
        batch_size=1,
        max_tokens=8,
        example_order=[0],
    )

    assert contract["version"] == 3
    assert contract["token_audit_sha256"] == config.sft_token_audit_sha256
