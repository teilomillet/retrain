"""Fail-closed, hash-pinned token-cap audit contract for SFT data."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

from retrain.io.digest import sha256_file as _file_sha256
from retrain.training.sft_audit_common import (
    load_audit_object as _load_audit_object,
    mapping_field as _mapping_field,
    sha256_field as _sha256_field,
    validate_reported_checks as _validate_reported_checks,
)

if TYPE_CHECKING:
    from retrain.config import TrainConfig
    from retrain.training.sft_data import SftDataProvenance


def effective_sft_token_cap(config: "TrainConfig") -> int:
    """Return the cap used by the matching standalone or warmup SFT path."""

    if config.sft_max_tokens > 0:
        return int(config.sft_max_tokens)
    if config.trainer == "sft":
        return int(config.max_tokens)
    return int(config.max_tokens) + 512


def verify_sft_token_audit_contract(
    config: "TrainConfig",
    provenance: "SftDataProvenance",
) -> dict[str, object] | None:
    """Verify an optional token audit against the exact loaded SFT corpus.

    The audit generator remains project-owned. Retrain trusts it only when the
    exact JSON bytes are pinned and all runtime-sensitive evidence still
    matches the launch: dataset bytes/rows, model, cap, Transformers version,
    implementation files, and the configured model's cached tokenizer files.
    No network access is permitted while resolving that tokenizer evidence.
    """

    configured_path = str(getattr(config, "sft_token_audit_path", "")).strip()
    expected_audit_sha = (
        str(getattr(config, "sft_token_audit_sha256", "")).strip().lower()
    )
    if not configured_path and not expected_audit_sha:
        return None
    if not configured_path or not expected_audit_sha:
        raise ValueError(
            "SFT token audit contract requires both sft_token_audit_path and "
            "sft_token_audit_sha256."
        )

    audit_path = Path(configured_path).expanduser().resolve(strict=True)
    audit_bytes = audit_path.read_bytes()
    actual_audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    if actual_audit_sha != expected_audit_sha:
        raise ValueError(
            "SFT token audit contract mismatch: sft_token_audit_sha256 "
            f"mismatch: expected {expected_audit_sha}, got "
            f"{actual_audit_sha} for {audit_path}"
        )

    audit = _load_audit_object(
        audit_path,
        audit_bytes,
        audit_name="SFT token audit",
    )
    errors: list[str] = []
    _validate_status(audit, errors)

    dataset = _mapping_field(audit, "dataset", errors)
    audited_sha = _sha256_field(dataset, "sha256", errors, prefix="dataset")
    audited_rows = _integer_field(
        dataset,
        "rows",
        errors,
        prefix="dataset",
        minimum=0,
    )
    if audited_sha is not None and audited_sha != provenance.data_sha256:
        errors.append(
            "dataset.sha256 mismatch: "
            f"audit has {audited_sha}, loaded data has {provenance.data_sha256}"
        )
    if audited_rows is not None and audited_rows != provenance.data_rows:
        errors.append(
            "dataset.rows mismatch: "
            f"audit has {audited_rows}, loaded data has {provenance.data_rows}"
        )

    effective_cap = effective_sft_token_cap(config)
    contract = _mapping_field(audit, "training_contract", errors)
    _validate_training_contract(
        contract,
        config=config,
        effective_cap=effective_cap,
        errors=errors,
    )
    audited_revision = contract.get("tokenizer_revision")
    configured_revision = str(config.model_revision).strip()
    if audited_revision != configured_revision:
        errors.append(
            "training_contract.tokenizer_revision mismatch with [model] "
            f"revision: expected {configured_revision!r}, got "
            f"{audited_revision!r}"
        )
    if not config.model_local_files_only:
        errors.append(
            "[model] local_files_only must be true for a token-audited SFT run"
        )
    rows_over = audit.get("rows_over_training_cap")
    if type(rows_over) is not int or rows_over != 0:
        errors.append(
            f"rows_over_training_cap must be the integer 0, got {rows_over!r}"
        )
    tokens = _mapping_field(audit, "tokens", errors)
    maximum = _integer_field(
        tokens,
        "max",
        errors,
        prefix="tokens",
        minimum=1,
    )
    if maximum is not None and maximum > effective_cap:
        errors.append(
            f"tokens.max must be <= effective SFT cap {effective_cap}, got {maximum}"
        )

    project_root = _resolve_project_root(audit_path, dataset, provenance)
    _validate_implementation_files(
        audit.get("implementation"),
        project_root=project_root,
        errors=errors,
    )
    _validate_tokenizer_snapshot(
        audit.get("tokenizer_snapshot"),
        contract=contract,
        model=str(config.model),
        errors=errors,
    )

    if errors:
        raise ValueError("SFT token audit contract mismatch:\n- " + "\n- ".join(errors))

    return {
        "audit_path": str(audit_path),
        "audit_sha256": actual_audit_sha,
        "schema": cast(str, audit["schema"]),
        "status": "pass",
        "dataset": {
            "sha256": provenance.data_sha256,
            "rows": provenance.data_rows,
        },
        "trainer": str(config.trainer),
        "model": str(config.model),
        "sft_max_tokens": effective_cap,
        "tokenizer_revision": cast(str, contract["tokenizer_revision"]),
    }


def _validate_status(
    audit: Mapping[str, object],
    errors: list[str],
) -> None:
    schema = audit.get("schema")
    if not isinstance(schema, str) or not schema.strip() or schema != schema.strip():
        errors.append("schema must be a non-empty, trimmed string")
    status = audit.get("status")
    if status != "pass":
        errors.append(f"status must be 'pass', got {status!r}")
    _validate_reported_checks(audit, errors)


def _validate_training_contract(
    contract: Mapping[str, object],
    *,
    config: "TrainConfig",
    effective_cap: int,
    errors: list[str],
) -> None:
    expected_fields: tuple[tuple[str, object], ...] = (
        ("trainer", str(config.trainer)),
        ("model", str(config.model)),
        ("sft_max_tokens", effective_cap),
    )
    for name, expected in expected_fields:
        actual = contract.get(name)
        if type(actual) is not type(expected) or actual != expected:
            errors.append(
                f"training_contract.{name} mismatch: expected "
                f"{expected!r}, got {actual!r}"
            )

    expected_transformers = contract.get("transformers_version")
    if not isinstance(expected_transformers, str) or not expected_transformers:
        errors.append(
            "training_contract.transformers_version must be a non-empty string"
        )
        return
    try:
        actual_transformers, _ = _transformers_runtime()
    except (ImportError, ModuleNotFoundError) as exc:
        errors.append(f"Transformers runtime is unavailable: {exc}")
        return
    if actual_transformers != expected_transformers:
        errors.append(
            "training_contract.transformers_version mismatch: expected "
            f"{expected_transformers!r}, got {actual_transformers!r}"
        )


def _validate_implementation_files(
    raw_records: object,
    *,
    project_root: Path | None,
    errors: list[str],
) -> None:
    if not isinstance(raw_records, list) or not raw_records:
        errors.append("implementation must be a non-empty JSON array")
        return
    for index, raw_record in enumerate(raw_records):
        prefix = f"implementation[{index}]"
        if not isinstance(raw_record, Mapping):
            errors.append(f"{prefix} must be a JSON object")
            continue
        record = cast(Mapping[str, object], raw_record)
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            errors.append(f"{prefix}.path must be a non-empty string")
            continue
        configured = Path(raw_path).expanduser()
        if not configured.is_absolute():
            if project_root is None:
                errors.append(
                    f"{prefix}.path is relative but the audit project root "
                    "cannot be resolved"
                )
                continue
            configured = project_root / configured
        try:
            path = configured.resolve(strict=True)
        except OSError as exc:
            errors.append(f"{prefix}.path cannot be resolved: {exc}")
            continue
        if not path.is_file():
            errors.append(f"{prefix}.path is not a file: {path}")
            continue
        expected_sha = _sha256_field(record, "sha256", errors, prefix=prefix)
        if expected_sha is None:
            continue
        actual_sha = _file_sha256(path)
        if actual_sha != expected_sha:
            errors.append(
                f"{prefix}.sha256 mismatch for {path}: expected "
                f"{expected_sha}, got {actual_sha}"
            )


def _validate_tokenizer_snapshot(
    raw_records: object,
    *,
    contract: Mapping[str, object],
    model: str,
    errors: list[str],
) -> None:
    if not isinstance(raw_records, list) or not raw_records:
        errors.append("tokenizer_snapshot must be a non-empty JSON array")
        return
    revision = contract.get("tokenizer_revision")
    if not isinstance(revision, str) or not revision.strip():
        errors.append("training_contract.tokenizer_revision must be a non-empty string")
        return
    try:
        _, cached_file = _transformers_runtime()
    except (ImportError, ModuleNotFoundError) as exc:
        errors.append(f"Transformers runtime is unavailable: {exc}")
        return

    snapshot_roots: set[Path] = set()
    for index, raw_record in enumerate(raw_records):
        prefix = f"tokenizer_snapshot[{index}]"
        if not isinstance(raw_record, Mapping):
            errors.append(f"{prefix} must be a JSON object")
            continue
        record = cast(Mapping[str, object], raw_record)
        raw_name = record.get("path")
        if not isinstance(raw_name, str) or not raw_name.strip():
            errors.append(f"{prefix}.path must be a non-empty string")
            continue
        name = Path(raw_name)
        if name.is_absolute() or ".." in name.parts:
            errors.append(f"{prefix}.path must be a safe relative path")
            continue
        expected_sha = _sha256_field(record, "sha256", errors, prefix=prefix)
        try:
            resolved = cached_file(
                model,
                raw_name,
                revision=revision,
                local_files_only=True,
            )
        except Exception as exc:  # noqa: BLE001 - dependency errors fail closed.
            errors.append(
                f"{prefix}.path is not in the configured model's local "
                f"tokenizer cache: {exc}"
            )
            continue
        if not isinstance(resolved, str):
            errors.append(
                f"{prefix}.path is not in the configured model's local tokenizer cache"
            )
            continue
        path = Path(resolved)
        if not path.is_file():
            errors.append(f"{prefix}.path is not a cached file: {path}")
            continue
        root = path
        for _ in name.parts:
            root = root.parent
        snapshot_roots.add(root)
        if expected_sha is not None:
            actual_sha = _file_sha256(path)
            if actual_sha != expected_sha:
                errors.append(
                    f"{prefix}.sha256 mismatch for {path}: expected "
                    f"{expected_sha}, got {actual_sha}"
                )

    if len(snapshot_roots) != 1:
        errors.append(
            "tokenizer_snapshot files must resolve to one cached snapshot, "
            f"got {len(snapshot_roots)}"
        )
    elif next(iter(snapshot_roots)).name != revision:
        errors.append(
            "training_contract.tokenizer_revision mismatch: expected cached "
            f"revision {revision!r}, got {next(iter(snapshot_roots)).name!r}"
        )


def _transformers_runtime() -> tuple[str, Callable[..., str | None]]:
    import transformers
    from transformers.utils.hub import cached_file

    return str(transformers.__version__), cached_file


def _resolve_project_root(
    audit_path: Path,
    dataset: Mapping[str, object],
    provenance: "SftDataProvenance",
) -> Path | None:
    if provenance.git_root:
        root = Path(provenance.git_root).expanduser()
        if root.is_dir():
            return root

    raw_dataset_path = dataset.get("path")
    if not isinstance(raw_dataset_path, str) or not raw_dataset_path.strip():
        return None
    relative = Path(raw_dataset_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None

    candidates: tuple[tuple[Path, tuple[str, ...]], ...] = (
        (Path(provenance.data_path), relative.parts),
        (audit_path.parent, relative.parent.parts),
    )
    for path, suffix in candidates:
        if suffix and tuple(path.parts[-len(suffix) :]) == suffix:
            root = path
            for _ in suffix:
                root = root.parent
            return root
    return None


def _integer_field(
    payload: Mapping[str, object],
    key: str,
    errors: list[str],
    *,
    prefix: str,
    minimum: int,
) -> int | None:
    value = payload.get(key)
    if type(value) is not int or value < minimum:
        errors.append(f"{prefix}.{key} must be an integer >= {minimum}")
        return None
    return value
