"""Delight token gates."""

from __future__ import annotations

import math

from retrain.advantages.constants import MAX_SURPRISAL
from retrain.advantages.delight.scale import (
    _apply_delight_norm_mode,
    _coerce_delight_norm_mode,
)

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x > 20:
        return 1.0
    if x < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def apply_hard_delight_gating(
    advantage: float,
    surprisals: list[float],
    k_frac: float = 0.2,
) -> list[float]:
    """Hard top-K delight gating: sign-aware binary token selection.

    Ranks tokens by surprisal. For correct rollouts (A > 0), keeps the
    top k_frac% highest-surprisal tokens (fork-points) at full advantage
    and zeros the rest. For incorrect rollouts (A < 0), keeps the bottom
    k_frac% (routine tokens) and zeros the high-surprisal "blunders".

    This produces a ~60% gradient directional change vs uniform PG
    (compared to ~5% for sigmoid DG), making it a strong enough
    intervention for the optimizer to actually respond to.

    Args:
        advantage: Episode-level advantage.
        surprisals: Per-token surprisal values (-logprob).
        k_frac: Fraction of tokens to keep (0.2 = top/bottom 20%).
    """
    n = len(surprisals)
    if n == 0 or advantage == 0.0:
        return [0.0] * n

    k = max(1, int(n * k_frac))
    # For correct rollouts: keep highest surprisal (fork-points)
    # For incorrect rollouts: keep lowest surprisal (routine tokens)
    indexed = sorted(range(n), key=lambda i: surprisals[i],
                     reverse=(advantage > 0))
    keep = set(indexed[:k])
    return [advantage if i in keep else 0.0 for i in range(n)]


def apply_hard_delight_sepa_gating(
    advantage: float,
    surprisals: list[float],
    lambda_t: float,
    k_frac: float = 0.2,
) -> list[float]:
    """SEPA-annealed hard delight gating: PG → hard-DG transition.

    Interpolates between uniform PG (λ=0) and hard top-K gating (λ=1):
        token_adv = A × [(1-λ) + λ × mask_k(t)]
    where mask_k(t) is 1 for kept tokens and 0 for zeroed tokens.
    """
    n = len(surprisals)
    if n == 0 or advantage == 0.0:
        return [0.0] * n

    lam = max(0.0, min(1.0, lambda_t))
    if lam == 0.0:
        return [advantage] * n

    k = max(1, int(n * k_frac))
    indexed = sorted(range(n), key=lambda i: surprisals[i],
                     reverse=(advantage > 0))
    keep = set(indexed[:k])
    pg_w = 1.0 - lam
    return [advantage * (pg_w + lam * (1.0 if i in keep else 0.0))
            for i in range(n)]


def apply_delight_gating(
    advantage: float,
    surprisals: list[float],
    eta: float = 1.0,
    normalize: bool | None = None,
    norm_mode: str | None = None,
) -> list[float]:
    """Delight-gated token-level advantages."""
    n = len(surprisals)
    if n == 0:
        return []
    if advantage == 0.0:
        return [0.0] * n

    resolved_norm = _coerce_delight_norm_mode(
        normalize=normalize,
        norm_mode=norm_mode,
        default="none",
    )
    s_values = _apply_delight_norm_mode(
        [min(s, MAX_SURPRISAL) for s in surprisals],
        resolved_norm,
    )

    inv_eta = 1.0 / max(eta, 1e-8)
    result = []
    for s in s_values:
        gate = _sigmoid(advantage * s * inv_eta)
        result.append(advantage * gate)
    return result


def apply_delight_sepa_gating(
    advantage: float,
    surprisals: list[float],
    lambda_t: float,
    eta: float = 1.0,
    normalize: bool | None = None,
    norm_mode: str | None = None,
) -> list[float]:
    """SEPA-annealed delight gating: smooth PG to DG transition."""
    n = len(surprisals)
    if n == 0:
        return []
    if advantage == 0.0:
        return [0.0] * n

    lam = max(0.0, min(1.0, lambda_t))
    if lam == 0.0:
        return [advantage] * n

    resolved_norm = _coerce_delight_norm_mode(
        normalize=normalize,
        norm_mode=norm_mode,
        default="none",
    )
    s_values = _apply_delight_norm_mode(
        [min(s, MAX_SURPRISAL) for s in surprisals],
        resolved_norm,
    )

    inv_eta = 1.0 / max(eta, 1e-8)
    pg_weight = 1.0 - lam
    result = []
    for s in s_values:
        gate = _sigmoid(advantage * s * inv_eta)
        blended = pg_weight + lam * gate
        result.append(advantage * blended)
    return result
