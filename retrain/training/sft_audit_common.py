"""Shared parsing fields for SFT audit contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast


def load_audit_object(
    path: Path,
    raw_bytes: bytes,
    *,
    audit_name: str,
) -> Mapping[str, object]:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Invalid {audit_name} file {path}: expected UTF-8 JSON."
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid {audit_name} file {path}: invalid JSON: {exc.msg}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid {audit_name} file {path}: expected a JSON object.")
    return cast(Mapping[str, object], payload)


def validate_reported_checks(
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
    non_passing = sorted(
        str(name)
        for name, passed in checks.items()
        if type(passed) is not bool or passed is not True
    )
    if non_passing:
        errors.append(
            "checks must contain only true booleans; non-passing keys: "
            + ", ".join(non_passing)
        )


def mapping_field(
    payload: Mapping[str, object],
    key: str,
    errors: list[str],
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be a JSON object")
        return {}
    return cast(Mapping[str, object], value)


def sha256_field(
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
    if (
        value != value.strip().lower()
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        errors.append(f"{prefix}.{key} must be a 64-character SHA256 digest")
        return None
    return value
