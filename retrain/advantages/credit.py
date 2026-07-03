"""Token-credit transforms used by advantage pipelines."""

from __future__ import annotations

from collections.abc import Mapping

from retrain.advantages.constants import MAX_SURPRISAL

# 2. GTPO entropy-weighted credit assignment
# ---------------------------------------------------------------------------


def apply_gtpo_weighting(
    advantage: float, surprisals: list[float], beta: float = 0.1
) -> list[float]:
    """Surprisal-weighted token-level advantages."""
    n = len(surprisals)
    if n == 0:
        return []
    if beta == 0.0:
        return [advantage] * n

    surprisals = [min(e, MAX_SURPRISAL) for e in surprisals]
    mean_h = sum(surprisals) / n
    if mean_h < 1e-7:
        return [advantage] * n

    result = []
    for h in surprisals:
        h_norm = h / (mean_h + 1e-8)
        weight = max(0.0, 1.0 + beta * (h_norm - 1.0))
        result.append(advantage * weight)
    return result


# ---------------------------------------------------------------------------
# 3. HICRA planning token amplification
# ---------------------------------------------------------------------------


def apply_hicra(
    token_advs: list[float], planning_mask: list[int], alpha: float = 0.2
) -> list[float]:
    """A_HICRA(t) = A(t) + alpha * |A(t)| * mask(t)."""
    if len(token_advs) != len(planning_mask):
        raise ValueError(
            f"Length mismatch: token_advs ({len(token_advs)}) "
            f"vs planning_mask ({len(planning_mask)})"
        )
    if alpha == 0.0:
        return list(token_advs)
    return [
        a + alpha * abs(a) if m else a
        for a, m in zip(token_advs, planning_mask)
    ]


# ---------------------------------------------------------------------------
# 3b. Entropy masking (Yue et al. proxy replication)
# ---------------------------------------------------------------------------


def compute_entropy_mask_threshold(
    all_entropies: list[float], rho: float
) -> float:
    """Compute the threshold for top-ρ entropy masking.

    Returns the entropy value at the ρ-percentile boundary (descending).
    Tokens with entropy >= threshold are kept; the rest are zeroed.
    """
    if rho >= 1.0:
        return float("-inf")
    if rho <= 0.0:
        return float("inf")
    n = len(all_entropies)
    if n == 0:
        return 0.0
    sorted_desc = sorted(all_entropies, reverse=True)
    idx = max(1, int(n * rho)) - 1
    return sorted_desc[idx]


def apply_entropy_mask(
    token_advs: list[float], entropies: list[float], threshold: float
) -> list[float]:
    """Zero out advantages for tokens below the entropy threshold."""
    return [
        a if e >= threshold else 0.0
        for a, e in zip(token_advs, entropies)
    ]


def surprisal_mask_post_process(
    all_token_advs: list[list[float]],
    all_raw_surprisals: list[list[float]],
    params: Mapping[str, object],
) -> tuple[list[list[float]], dict[str, float]]:
    """Post-process hook: Yue et al. surprisal masking."""
    raw_rho = params.get("surprisal_mask_rho", params.get("entropy_mask_rho", 0.0))
    rho = float(raw_rho) if isinstance(raw_rho, int | float) else 0.0
    if rho <= 0.0:
        return all_token_advs, {}

    flat_surprisals = [e for seq in all_raw_surprisals for e in seq]
    threshold = compute_entropy_mask_threshold(flat_surprisals, rho)

    total_tokens = 0
    masked_tokens = 0
    for idx in range(len(all_token_advs)):
        all_token_advs[idx] = apply_entropy_mask(
            all_token_advs[idx], all_raw_surprisals[idx], threshold
        )
        for e in all_raw_surprisals[idx]:
            total_tokens += 1
            if e < threshold:
                masked_tokens += 1

    fraction = masked_tokens / total_tokens if total_tokens > 0 else 0.0
    return all_token_advs, {
        "entropy_mask_threshold": threshold,
        "entropy_mask_fraction": fraction,
    }


entropy_mask_post_process = surprisal_mask_post_process  # backward-compat alias


# ---------------------------------------------------------------------------
# 4. SEPA selective entropy pooling
# ---------------------------------------------------------------------------


def apply_sepa_pooling(
    surprisals: list[float], planning_mask: list[int], lambda_t: float
) -> list[float]:
    """Pull execution token surprisals toward their mean."""
    if len(surprisals) != len(planning_mask):
        raise ValueError(
            f"Length mismatch: surprisals ({len(surprisals)}) "
            f"vs planning_mask ({len(planning_mask)})"
        )
    lam = max(0.0, min(1.0, lambda_t))
    if lam == 0.0:
        return list(surprisals)

    surprisals = [min(e, MAX_SURPRISAL) for e in surprisals]
    exec_vals = [e for e, m in zip(surprisals, planning_mask) if m == 0]
    if not exec_vals:
        return list(surprisals)
    mean_h_exec = sum(exec_vals) / len(exec_vals)

    return [
        e if m else lam * mean_h_exec + (1.0 - lam) * e
        for e, m in zip(surprisals, planning_mask)
    ]


def apply_sepa_amplification(
    surprisals: list[float], planning_mask: list[int], lambda_t: float
) -> list[float]:
    """Push execution token surprisals away from their mean.

    h'_t = h_t + λ·(h_t - μ_exec) = (1+λ)·h_t - λ·μ_exec

    High-surprisal execution tokens get pushed higher (more GTPO gradient
    weight), low-surprisal ones get pushed lower. Planning tokens are
    left untouched.
    """
    if len(surprisals) != len(planning_mask):
        raise ValueError(
            f"Length mismatch: surprisals ({len(surprisals)}) "
            f"vs planning_mask ({len(planning_mask)})"
        )
    lam = max(0.0, min(1.0, lambda_t))
    if lam == 0.0:
        return list(surprisals)

    surprisals = [min(e, MAX_SURPRISAL) for e in surprisals]
    exec_vals = [e for e, m in zip(surprisals, planning_mask) if m == 0]
    if not exec_vals:
        return list(surprisals)
    mean_h_exec = sum(exec_vals) / len(exec_vals)

    return [
        e if m else (1.0 + lam) * e - lam * mean_h_exec
        for e, m in zip(surprisals, planning_mask)
    ]


def apply_sepa_amplification_clamped(
    surprisals: list[float], planning_mask: list[int], lambda_t: float
) -> list[float]:
    """Push execution token surprisals away from their mean, clamped to >= 0.

    Same as apply_sepa_amplification but floors results at zero so no token
    gets a negative surprisal value.  Keeps amplification purely soft —
    low-surprisal tokens shrink toward zero but never flip sign.
    """
    if len(surprisals) != len(planning_mask):
        raise ValueError(
            f"Length mismatch: surprisals ({len(surprisals)}) "
            f"vs planning_mask ({len(planning_mask)})"
        )
    lam = max(0.0, min(1.0, lambda_t))
    if lam == 0.0:
        return list(surprisals)

    surprisals = [min(e, MAX_SURPRISAL) for e in surprisals]
    exec_vals = [e for e, m in zip(surprisals, planning_mask) if m == 0]
    if not exec_vals:
        return list(surprisals)
    mean_h_exec = sum(exec_vals) / len(exec_vals)

    return [
        e if m else max(0.0, (1.0 + lam) * e - lam * mean_h_exec)
        for e, m in zip(surprisals, planning_mask)
    ]


# ---------------------------------------------------------------------------
