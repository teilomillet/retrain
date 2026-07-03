"""Uncertainty selector normalization for train configs."""

from __future__ import annotations

from retrain.advantages import get_uncertainty_kind_param


def _canonicalize_uncertainty_override(
    params: dict[str, object],
    *,
    label: str,
    errors: list[str],
) -> tuple[str, str] | None:
    """Canonicalize uncertainty selector params and return ``(label.key, value)``."""
    try:
        override = get_uncertainty_kind_param(params)
    except ValueError as exc:
        errors.append(f"{label}.{exc}")
        return None
    if override is None:
        return None
    key, value = override
    params[key] = value
    return f"{label}.{key}", value


def _first_uncertainty_override(
    overrides: list[tuple[str, str]],
    *,
    errors: list[str],
) -> tuple[str, str] | None:
    if not overrides:
        return None
    first_label, first_value = overrides[0]
    for label, value in overrides[1:]:
        if value != first_value:
            errors.append(
                "Conflicting uncertainty_kind overrides: "
                f"{first_label}={first_value!r} but {label}={value!r}."
            )
    return first_label, first_value
