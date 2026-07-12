"""Generic, hash-pinned audit contract for supervised datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

from retrain.training.sft_audit_common import (
    load_audit_object as _load_audit_object,
    mapping_field as _mapping_field,
    sha256_field as _sha256_field,
    validate_reported_checks as _validate_reported_checks,
)

if TYPE_CHECKING:
    from retrain.config import TrainConfig
    from retrain.training.sft_data import SftDataProvenance


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
    expected_audit_sha = str(getattr(config, "sft_audit_sha256", "")).strip().lower()
    if not configured_path and not expected_audit_sha:
        return None
    if not configured_path or not expected_audit_sha:
        raise ValueError(
            "SFT audit contract requires both sft_audit_path and sft_audit_sha256."
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

    audit = _load_audit_object(
        audit_path,
        audit_bytes,
        audit_name="SFT audit",
    )
    errors: list[str] = []

    schema = audit.get("schema")
    if schema != SFT_AUDIT_SCHEMA:
        errors.append(f"schema must be {SFT_AUDIT_SCHEMA!r}, got {schema!r}")
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
            f"corpus_mode must be 'patch' or 'replacement', got {corpus_mode!r}"
        )
    lineage_value = audit.get("lineage")
    lineage: Mapping[str, object] = {}
    if lineage_value is not None:
        if isinstance(lineage_value, Mapping):
            lineage = cast(Mapping[str, object], lineage_value)
        else:
            errors.append("lineage must be a JSON object when present")
    if corpus_mode == "patch":
        _validate_patch_lineage(
            lineage,
            audited_rows,
            errors,
            require_source_proof=bool(
                str(getattr(config, "sft_token_audit_path", "")).strip()
            ),
            audit_path=audit_path,
            provenance=provenance,
        )
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
    *,
    require_source_proof: bool,
    audit_path: Path,
    provenance: "SftDataProvenance",
) -> None:
    lineage_rows: list[int] = []
    lineage_items: dict[str, Mapping[str, object]] = {}
    for role in ("base", "patch"):
        value = lineage.get(role)
        prefix = f"lineage.{role}"
        if not isinstance(value, Mapping):
            errors.append(f"{prefix} must be a JSON object for corpus_mode='patch'")
            continue
        item = cast(Mapping[str, object], value)
        lineage_items[role] = item
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
    if require_source_proof:
        _validate_patch_source_proof(
            lineage_items,
            audit_path=audit_path,
            provenance=provenance,
            errors=errors,
        )


def _validate_patch_source_proof(
    lineage: Mapping[str, Mapping[str, object]],
    *,
    audit_path: Path,
    provenance: "SftDataProvenance",
    errors: list[str],
) -> None:
    """Prove ``combined == declared-base-prefix || tracked-patch``.

    The standalone base file is useful generation evidence but is not required
    at training time: a fresh checkout can derive and verify the base prefix
    from the exact combined corpus plus the tracked patch suffix.
    """

    base_item = lineage.get("base")
    patch_item = lineage.get("patch")
    if base_item is None or patch_item is None:
        return
    try:
        live_bytes = Path(provenance.data_path).read_bytes()
    except OSError as exc:
        errors.append(f"loaded SFT data cannot be re-read for lineage proof: {exc}")
        return

    patch_path = _resolve_lineage_path(
        patch_item.get("path"),
        audit_path=audit_path,
        provenance=provenance,
        prefix="lineage.patch",
        required=True,
        errors=errors,
    )
    if patch_path is None:
        return
    try:
        patch_bytes = patch_path.read_bytes()
    except OSError as exc:
        errors.append(f"lineage.patch.path cannot be read: {exc}")
        return
    _validate_lineage_bytes(
        patch_item,
        patch_bytes,
        label=patch_path,
        prefix="lineage.patch",
        errors=errors,
    )
    if not patch_bytes:
        errors.append("lineage.patch.path must not be empty")
        return
    if not live_bytes.endswith(patch_bytes):
        errors.append(
            "loaded SFT data file does not end byte-for-byte with lineage.patch"
        )
        return

    base_bytes = live_bytes[: -len(patch_bytes)]
    _validate_lineage_bytes(
        base_item,
        base_bytes,
        label=Path(provenance.data_path),
        prefix="derived lineage.base prefix",
        errors=errors,
    )

    base_path = _resolve_lineage_path(
        base_item.get("path"),
        audit_path=audit_path,
        provenance=provenance,
        prefix="lineage.base",
        required=False,
        errors=errors,
    )
    if base_path is None:
        return
    try:
        explicit_base = base_path.read_bytes()
    except OSError as exc:
        errors.append(f"lineage.base.path cannot be read: {exc}")
        return
    _validate_lineage_bytes(
        base_item,
        explicit_base,
        label=base_path,
        prefix="lineage.base",
        errors=errors,
    )
    if explicit_base != base_bytes:
        errors.append(
            "lineage.base file does not match the verified combined-corpus prefix"
        )


def _validate_lineage_bytes(
    item: Mapping[str, object],
    raw_bytes: bytes,
    *,
    label: Path,
    prefix: str,
    errors: list[str],
) -> None:
    expected_sha = _sha256_field(item, "sha256", errors, prefix=prefix)
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha is not None and actual_sha != expected_sha:
        errors.append(
            f"{prefix}.sha256 mismatch for {label}: expected "
            f"{expected_sha}, got {actual_sha}"
        )
    expected_rows = _rows_field(item, "rows", errors, prefix=prefix)
    try:
        actual_rows = _jsonl_rows(label, raw_bytes)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if expected_rows is not None and actual_rows != expected_rows:
            errors.append(
                f"{prefix}.rows mismatch for {label}: expected "
                f"{expected_rows}, got {actual_rows}"
            )


def _resolve_lineage_path(
    raw_path: object,
    *,
    audit_path: Path,
    provenance: "SftDataProvenance",
    prefix: str,
    required: bool,
    errors: list[str],
) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        if required:
            errors.append(
                f"{prefix}.path must be a non-empty string when an SFT token "
                "audit is configured"
            )
        return None
    configured = Path(raw_path).expanduser()
    if not configured.is_absolute():
        if provenance.git_root:
            configured = Path(provenance.git_root) / configured
        elif provenance.data_root:
            data_root = Path(provenance.data_root).expanduser()
            configured = data_root.parent / configured
        else:
            configured = audit_path.parent / configured
    try:
        path = configured.resolve(strict=True)
    except OSError as exc:
        if required:
            errors.append(f"{prefix}.path cannot be resolved: {exc}")
        return None
    if not path.is_file():
        if required:
            errors.append(f"{prefix}.path is not a file: {path}")
        return None
    return path


def _jsonl_rows(path: Path, raw_bytes: bytes) -> int:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not UTF-8 JSONL") from exc
    rows = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{line_number} is invalid JSONL: {exc.msg}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows += 1
    return rows
