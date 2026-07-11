"""Generic, hash-pinned audit contract for supervised datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from retrain.config import TrainConfig
    from retrain.training.sft import SftDataProvenance


SFT_AUDIT_SCHEMA = "retrain.sft_audit.v1"
_CORPUS_MODES = frozenset({"patch", "replacement"})


def verify_sft_audit_contract(
    config: "TrainConfig",
    provenance: "SftDataProvenance",
) -> dict[str, object] | None:
    """Verify an optional audit artifact against the loaded SFT dataset.

    The audit implementation remains project-specific.  Retrain only trusts a
    configured audit after its exact bytes are externally pinned and its
    generic dataset/mode contract agrees with the data Retrain actually read.
    """

    configured_path = str(getattr(config, "sft_audit_path", "")).strip()
    expected_audit_sha = str(
        getattr(config, "sft_audit_sha256", "")
    ).strip().lower()
    if not configured_path and not expected_audit_sha:
        return None
    if not configured_path or not expected_audit_sha:
        raise ValueError(
            "SFT audit contract requires both sft_audit_path and "
            "sft_audit_sha256."
        )

    audit_path = Path(configured_path).expanduser().resolve(strict=True)
    audit_bytes = audit_path.read_bytes()
    actual_audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    if actual_audit_sha != expected_audit_sha:
        raise ValueError(
            "SFT audit contract mismatch: sft_audit_sha256 mismatch: "
            f"expected {expected_audit_sha}, got {actual_audit_sha} "
            f"for {audit_path}"
        )

    audit = _load_audit_object(audit_path, audit_bytes)
    errors: list[str] = []

    schema = audit.get("schema")
    if schema != SFT_AUDIT_SCHEMA:
        errors.append(
            f"schema must be {SFT_AUDIT_SCHEMA!r}, got {schema!r}"
        )
    status = audit.get("status")
    if status != "pass":
        errors.append(f"status must be 'pass', got {status!r}")
    _validate_reported_checks(audit, errors)

    audited_dataset = _mapping_field(audit, "audited_dataset", errors)
    audited_sha = _sha256_field(audited_dataset, "sha256", errors)
    audited_rows = _rows_field(audited_dataset, "rows", errors)
    if audited_sha is not None and audited_sha != provenance.data_sha256:
        errors.append(
            "audited_dataset.sha256 mismatch: "
            f"audit has {audited_sha}, loaded data has {provenance.data_sha256}"
        )
    if audited_rows is not None and audited_rows != provenance.data_rows:
        errors.append(
            "audited_dataset.rows mismatch: "
            f"audit has {audited_rows}, loaded data has {provenance.data_rows}"
        )

    corpus_mode = audit.get("corpus_mode")
    if corpus_mode not in _CORPUS_MODES:
        errors.append(
            "corpus_mode must be 'patch' or 'replacement', "
            f"got {corpus_mode!r}"
        )
    lineage_value = audit.get("lineage")
    lineage: Mapping[str, object] = {}
    if lineage_value is not None:
        if isinstance(lineage_value, Mapping):
            lineage = cast(Mapping[str, object], lineage_value)
        else:
            errors.append("lineage must be a JSON object when present")
    if corpus_mode == "patch":
        _validate_patch_lineage(lineage, audited_rows, errors)
    elif corpus_mode == "replacement" and lineage:
        errors.append("lineage must be empty for corpus_mode='replacement'")

    if errors:
        raise ValueError("SFT audit contract mismatch:\n- " + "\n- ".join(errors))

    return {
        "audit_path": str(audit_path),
        "audit_sha256": actual_audit_sha,
        "schema": SFT_AUDIT_SCHEMA,
        "status": "pass",
        "audited_dataset": {
            "sha256": provenance.data_sha256,
            "rows": provenance.data_rows,
        },
        "corpus_mode": cast(str, corpus_mode),
        "lineage": dict(lineage),
    }


def _load_audit_object(path: Path, raw_bytes: bytes) -> Mapping[str, object]:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Invalid SFT audit file {path}: expected UTF-8 JSON."
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid SFT audit file {path}: invalid JSON: {exc.msg}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid SFT audit file {path}: expected a JSON object.")
    return cast(Mapping[str, object], payload)


def _validate_reported_checks(
    audit: Mapping[str, object],
    errors: list[str],
) -> None:
    failed_checks = audit.get("failed_checks")
    if failed_checks is not None:
        if not isinstance(failed_checks, list):
            errors.append("failed_checks must be a JSON array when present")
        elif failed_checks:
            errors.append(f"failed_checks must be empty, got {failed_checks!r}")

    checks = audit.get("checks")
    if checks is None:
        return
    if not isinstance(checks, Mapping):
        errors.append("checks must be a JSON object when present")
        return
    non_passing = [
        str(name)
        for name, passed in checks.items()
        if type(passed) is not bool or passed is not True
    ]
    if non_passing:
        errors.append(
            "checks must contain only true booleans; non-passing keys: "
            + ", ".join(sorted(non_passing))
        )


def _mapping_field(
    payload: Mapping[str, object],
    key: str,
    errors: list[str],
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be a JSON object")
        return {}
    return cast(Mapping[str, object], value)


def _sha256_field(
    payload: Mapping[str, object],
    key: str,
    errors: list[str],
    *,
    prefix: str = "audited_dataset",
) -> str | None:
    value = payload.get(key)
    if not isinstance(value, str):
        errors.append(f"{prefix}.{key} must be a 64-character SHA256 digest")
        return None
    digest = value.strip()
    if (
        digest != value
        or len(digest) != 64
        or any(ch not in "0123456789abcdef" for ch in digest)
    ):
        errors.append(f"{prefix}.{key} must be a 64-character SHA256 digest")
        return None
    return digest


def _rows_field(
    payload: Mapping[str, object],
    key: str,
    errors: list[str],
    *,
    prefix: str = "audited_dataset",
) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{prefix}.{key} must be a non-negative integer")
        return None
    return value


def _validate_patch_lineage(
    lineage: Mapping[str, object],
    audited_rows: int | None,
    errors: list[str],
) -> None:
    lineage_rows: list[int] = []
    for role in ("base", "patch"):
        value = lineage.get(role)
        prefix = f"lineage.{role}"
        if not isinstance(value, Mapping):
            errors.append(f"{prefix} must be a JSON object for corpus_mode='patch'")
            continue
        item = cast(Mapping[str, object], value)
        _sha256_field(item, "sha256", errors, prefix=prefix)
        rows = _rows_field(item, "rows", errors, prefix=prefix)
        if rows is not None:
            if rows <= 0:
                errors.append(f"{prefix}.rows must be a positive integer")
            else:
                lineage_rows.append(rows)
    if (
        audited_rows is not None
        and len(lineage_rows) == 2
        and sum(lineage_rows) != audited_rows
    ):
        errors.append(
            "lineage base/patch rows must sum to audited_dataset.rows: "
            f"got {sum(lineage_rows)} versus {audited_rows}"
        )
