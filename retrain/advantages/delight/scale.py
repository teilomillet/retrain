"""Delight surprisal scaling and mode parsing."""

from __future__ import annotations

import math
from collections.abc import Mapping

def _normalize_surprisals(surprisals: list[float]) -> list[float]:
    """Scale surprisals to unit-std without centering.

    Returns ℓ / std for each token. This keeps surprisals non-negative
    (preserving DG's sign semantics) while scaling them so the sigmoid
    argument covers its sensitive range.

    Centering (z-scoring) is wrong here because surprisal distributions
    are right-skewed: most tokens have z < 0 after centering, which
    inverts the gate for the majority of tokens. Scaling without
    centering preserves: high surprisal → large positive → gate opens
    for correct rollouts, closes for incorrect.
    """
    n = len(surprisals)
    if n < 2:
        return list(surprisals)
    mean_s = sum(surprisals) / n
    var_s = sum((s - mean_s) ** 2 for s in surprisals) / n
    std_s = var_s ** 0.5
    if std_s < 1e-8:
        return list(surprisals)
    return [s / std_s for s in surprisals]


def _median(values: list[float]) -> float:
    """Median of a non-empty list."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mad_scale_surprisals(surprisals: list[float]) -> list[float]:
    """Scale surprisals by robust MAD estimate without centering."""
    n = len(surprisals)
    if n < 2:
        return list(surprisals)
    med = _median(surprisals)
    abs_dev = [abs(s - med) for s in surprisals]
    mad = _median(abs_dev)
    robust_std = 1.4826 * mad
    if robust_std < 1e-8:
        return _normalize_surprisals(surprisals)
    return [s / robust_std for s in surprisals]


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile for q in [0, 1]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if q <= 0.0:
        return ordered[0]
    if q >= 1.0:
        return ordered[-1]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _coerce_delight_norm_mode(
    *,
    normalize: bool | None = None,
    norm_mode: str | None = None,
    default: str = "none",
) -> str:
    """Resolve Delight normalization mode from API parameters."""
    if norm_mode is not None:
        mode = norm_mode.strip().lower()
        if mode in {"none", "scale", "mad_scale"}:
            return mode
        raise ValueError(
            f"Invalid delight norm_mode '{norm_mode}'. "
            "Expected one of: none, scale, mad_scale."
        )

    if normalize is None:
        return default
    return "scale" if normalize else "none"


def _resolve_delight_norm_mode(
    params: Mapping[str, object], *, default: str
) -> str:
    """Resolve Delight normalization mode from transform params.

    `delight_norm_mode` is the primary API. `delight_normalize` remains as a
    backward-compatible boolean alias: true -> "scale", false -> "none".
    """
    raw_mode = params.get("delight_norm_mode")
    if raw_mode is not None:
        if not isinstance(raw_mode, str):
            raise ValueError("delight_norm_mode must be a string")
        return _coerce_delight_norm_mode(norm_mode=raw_mode, default=default)

    raw_normalize = params.get("delight_normalize")
    if raw_normalize is not None:
        if not isinstance(raw_normalize, bool):
            raise ValueError("delight_normalize must be a boolean")
        return _coerce_delight_norm_mode(
            normalize=raw_normalize,
            default=default,
        )

    return default


def _apply_delight_norm_mode(
    surprisals: list[float], norm_mode: str
) -> list[float]:
    """Apply the configured Delight normalization mode."""
    if norm_mode == "none":
        return list(surprisals)
    if norm_mode == "scale":
        return _normalize_surprisals(surprisals)
    if norm_mode == "mad_scale":
        return _mad_scale_surprisals(surprisals)
    raise ValueError(
        f"Invalid delight norm_mode '{norm_mode}'. Expected one of: none, scale, mad_scale."
    )


def _resolve_delight_eta_mode(
    params: Mapping[str, object], *, default: str = "fixed"
) -> str:
    """Resolve Delight eta mode from transform params."""
    raw_mode = params.get("delight_eta_mode")
    if raw_mode is None:
        raw_adaptive = params.get("delight_eta_adaptive")
        if raw_adaptive is None:
            return default
        if not isinstance(raw_adaptive, bool):
            raise ValueError("delight_eta_adaptive must be a boolean")
        return "adaptive" if raw_adaptive else "fixed"

    if not isinstance(raw_mode, str):
        raise ValueError("delight_eta_mode must be a string")
    mode = raw_mode.strip().lower()
    if mode in {"fixed", "adaptive"}:
        return mode
    raise ValueError(
        f"Invalid delight eta_mode '{raw_mode}'. Expected one of: fixed, adaptive."
    )
