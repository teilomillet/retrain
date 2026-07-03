"""Advantage surprisal statistics."""

from __future__ import annotations

from retrain.advantages.constants import MAX_SURPRISAL
from retrain.advantages.types import EntropyStats

# 5. Entropy statistics
# ---------------------------------------------------------------------------


def compute_surprisal_stats(
    exec_surprisals: list[float], plan_surprisals: list[float]
) -> EntropyStats:
    """Compute summary stats for execution vs planning token surprisal."""
    stats = EntropyStats()

    exec_surprisals = [min(e, MAX_SURPRISAL) for e in exec_surprisals]
    plan_surprisals = [min(e, MAX_SURPRISAL) for e in plan_surprisals]

    if exec_surprisals:
        n = len(exec_surprisals)
        mean_e = sum(exec_surprisals) / n
        var_e = sum((e - mean_e) ** 2 for e in exec_surprisals) / n
        stats.exec_mean = mean_e
        stats.exec_var = var_e
        stats.exec_count = float(n)

    if plan_surprisals:
        n = len(plan_surprisals)
        mean_p = sum(plan_surprisals) / n
        var_p = sum((p - mean_p) ** 2 for p in plan_surprisals) / n
        stats.plan_mean = mean_p
        stats.plan_var = var_p
        stats.plan_count = float(n)

    return stats


compute_entropy_stats = compute_surprisal_stats  # backward-compat alias


# ---------------------------------------------------------------------------
