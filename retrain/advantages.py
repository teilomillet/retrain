"""Advantage computation: GRPO, MaxRL, GTPO, HICRA, SEPA + planning tokens.

Ports the core functions from src/advantages.mojo into pure Python.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Cap entropy values to prevent inf from poisoning downstream math.
# Real per-token entropy (-logprob) rarely exceeds ~15; 50 is a safe upper bound.
MAX_ENTROPY = 50.0


# ---------------------------------------------------------------------------
# EntropyStats
# ---------------------------------------------------------------------------

@dataclass
class EntropyStats:
    """Summary statistics for execution vs planning entropy distributions."""
    exec_mean: float = 0.0
    exec_var: float = 0.0
    exec_count: float = 0.0
    plan_mean: float = 0.0
    plan_var: float = 0.0
    plan_count: float = 0.0


@dataclass
class AdvantageResult:
    """Result of composable advantage computation."""
    token_advs: list[list[float]] = field(default_factory=list)
    has_stats: bool = False
    stats: EntropyStats = field(default_factory=EntropyStats)


# ---------------------------------------------------------------------------
# 0. GRPO advantages (simple reward centering)
# ---------------------------------------------------------------------------


def compute_grpo_advantages(rewards: list[float]) -> list[float]:
    """A_i = r_i - mean(r)."""
    n = len(rewards)
    if n == 0:
        return []
    mean_r = sum(rewards) / n
    return [r - mean_r for r in rewards]


# ---------------------------------------------------------------------------
# 1. MaxRL advantages (inverse success-rate reweighting)
# ---------------------------------------------------------------------------


def compute_maxrl_advantages(
    rewards: list[float], eps: float = 1e-6
) -> list[float]:
    """A_i = (r_i - mean(r)) / (mean(r) + eps). Zero if mean(r) ~ 0."""
    n = len(rewards)
    if n == 0:
        return []
    mean_r = sum(rewards) / n
    if mean_r <= eps:
        return [0.0] * n
    denom = mean_r + eps
    return [(r - mean_r) / denom for r in rewards]


# ---------------------------------------------------------------------------
# 2. GTPO entropy-weighted credit assignment
# ---------------------------------------------------------------------------


def apply_gtpo_weighting(
    advantage: float, entropies: list[float], beta: float = 0.1
) -> list[float]:
    """Entropy-weighted token-level advantages."""
    n = len(entropies)
    if n == 0:
        return []
    if beta == 0.0:
        return [advantage] * n

    entropies = [min(e, MAX_ENTROPY) for e in entropies]
    mean_h = sum(entropies) / n
    if mean_h < 1e-7:
        return [advantage] * n

    result = []
    for h in entropies:
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
# 4. SEPA selective entropy pooling
# ---------------------------------------------------------------------------


def apply_sepa_pooling(
    entropies: list[float], planning_mask: list[int], lambda_t: float
) -> list[float]:
    """Pull execution token entropies toward their mean."""
    if len(entropies) != len(planning_mask):
        raise ValueError(
            f"Length mismatch: entropies ({len(entropies)}) "
            f"vs planning_mask ({len(planning_mask)})"
        )
    lam = max(0.0, min(1.0, lambda_t))
    if lam == 0.0:
        return list(entropies)

    entropies = [min(e, MAX_ENTROPY) for e in entropies]
    exec_vals = [e for e, m in zip(entropies, planning_mask) if m == 0]
    if not exec_vals:
        return list(entropies)
    mean_h_exec = sum(exec_vals) / len(exec_vals)

    return [
        e if m else lam * mean_h_exec + (1.0 - lam) * e
        for e, m in zip(entropies, planning_mask)
    ]


def apply_sepa_amplification(
    entropies: list[float], planning_mask: list[int], lambda_t: float
) -> list[float]:
    """Push execution token entropies away from their mean.

    h'_t = h_t + λ·(h_t - μ_exec) = (1+λ)·h_t - λ·μ_exec

    High-entropy execution tokens get pushed higher (more GTPO gradient
    weight), low-entropy ones get pushed lower. Planning tokens are
    left untouched.
    """
    if len(entropies) != len(planning_mask):
        raise ValueError(
            f"Length mismatch: entropies ({len(entropies)}) "
            f"vs planning_mask ({len(planning_mask)})"
        )
    lam = max(0.0, min(1.0, lambda_t))
    if lam == 0.0:
        return list(entropies)

    entropies = [min(e, MAX_ENTROPY) for e in entropies]
    exec_vals = [e for e, m in zip(entropies, planning_mask) if m == 0]
    if not exec_vals:
        return list(entropies)
    mean_h_exec = sum(exec_vals) / len(exec_vals)

    return [
        e if m else (1.0 + lam) * e - lam * mean_h_exec
        for e, m in zip(entropies, planning_mask)
    ]


def apply_sepa_amplification_clamped(
    entropies: list[float], planning_mask: list[int], lambda_t: float
) -> list[float]:
    """Push execution token entropies away from their mean, clamped to >= 0.

    Same as apply_sepa_amplification but floors results at zero so no token
    gets a negative entropy value.  Keeps amplification purely soft —
    low-entropy tokens shrink toward zero but never flip sign.
    """
    if len(entropies) != len(planning_mask):
        raise ValueError(
            f"Length mismatch: entropies ({len(entropies)}) "
            f"vs planning_mask ({len(planning_mask)})"
        )
    lam = max(0.0, min(1.0, lambda_t))
    if lam == 0.0:
        return list(entropies)

    entropies = [min(e, MAX_ENTROPY) for e in entropies]
    exec_vals = [e for e, m in zip(entropies, planning_mask) if m == 0]
    if not exec_vals:
        return list(entropies)
    mean_h_exec = sum(exec_vals) / len(exec_vals)

    return [
        e if m else max(0.0, (1.0 + lam) * e - lam * mean_h_exec)
        for e, m in zip(entropies, planning_mask)
    ]


# ---------------------------------------------------------------------------
# 5. Entropy statistics
# ---------------------------------------------------------------------------


def compute_entropy_stats(
    exec_entropies: list[float], plan_entropies: list[float]
) -> EntropyStats:
    """Compute summary stats for execution vs planning entropy."""
    stats = EntropyStats()

    exec_entropies = [min(e, MAX_ENTROPY) for e in exec_entropies]
    plan_entropies = [min(e, MAX_ENTROPY) for e in plan_entropies]

    if exec_entropies:
        n = len(exec_entropies)
        mean_e = sum(exec_entropies) / n
        var_e = sum((e - mean_e) ** 2 for e in exec_entropies) / n
        stats.exec_mean = mean_e
        stats.exec_var = var_e
        stats.exec_count = float(n)

    if plan_entropies:
        n = len(plan_entropies)
        mean_p = sum(plan_entropies) / n
        var_p = sum((p - mean_p) ** 2 for p in plan_entropies) / n
        stats.plan_mean = mean_p
        stats.plan_var = var_p
        stats.plan_count = float(n)

    return stats


# ---------------------------------------------------------------------------
# 6. Planning token identification (regex-based)
# ---------------------------------------------------------------------------

DEFAULT_STRATEGIC_GRAMS = [
    "wait let me",
    "let me think",
    "on second thought",
    "let me check",
    "let me verify",
    "is this right",
    "double check",
    "try another approach",
    "go back and",
    "start over",
    "that's not right",
    "that doesn't work",
    "another way to",
    "or we could",
    "what if we",
    "notice that",
    "the key is",
    "the key insight",
]


def _clean_token_fragment(fragment: str) -> str:
    """Clean a tokenizer fragment: replace subword markers with space."""
    # sentencepiece: \u2581, GPT-2/BPE: \u0120
    return fragment.replace("\u2581", " ").replace("\u0120", " ").strip()


# Cache compiled regex patterns keyed by the tuple of grams
_pattern_cache: dict[tuple[str, ...], list[re.Pattern[str]]] = {}


def _get_gram_patterns(strategic_grams: list[str]) -> list[re.Pattern[str]]:
    """Return compiled regex patterns for strategic grams (cached)."""
    key = tuple(strategic_grams)
    if key not in _pattern_cache:
        _pattern_cache[key] = [
            re.compile(r"\b" + re.escape(gram) + r"\b", re.IGNORECASE)
            for gram in strategic_grams
        ]
    return _pattern_cache[key]


def identify_planning_tokens(
    token_strs: list[str],
    strategic_grams: list[str],
    max_window: int = 5,
) -> list[int]:
    """Identify planning tokens via strategic gram matching.

    Sliding window over token fragments, checking for word-boundary matches.
    """
    n_tokens = len(token_strs)
    if n_tokens == 0 or not strategic_grams:
        return [0] * n_tokens

    # Effective window covers longest gram by word count
    effective_window = max(max_window, max(len(g.split()) for g in strategic_grams))

    # Pre-clean all fragments
    cleaned = [_clean_token_fragment(t) for t in token_strs]

    patterns = _get_gram_patterns(strategic_grams)

    mask = [0] * n_tokens

    for start in range(n_tokens):
        window_text = ""
        window_end = min(start + effective_window, n_tokens)

        for end in range(start, window_end):
            if cleaned[end]:
                if window_text:
                    window_text += " " + cleaned[end]
                else:
                    window_text = cleaned[end]

            matched = False
            for pat in patterns:
                if pat.search(window_text):
                    for idx in range(start, end + 1):
                        mask[idx] = 1
                    matched = True
                    break
            if matched:
                break

    return mask


# ---------------------------------------------------------------------------
# Composable advantage pipeline
# ---------------------------------------------------------------------------


def compute_composable_advantages(
    rewards_G: list[float],
    logprobs_G: list[list[float]],
    planning_masks_G: list[list[int]],
    *,
    advantage_mode: str = "grpo",
    transform_mode: str = "none",
    gtpo_beta: float = 0.1,
    hicra_alpha: float = 0.2,
    sepa_lambda: float = 0.0,
) -> AdvantageResult:
    """Compute token-level advantages with composable transforms."""
    # Step 1: Episode-level advantages
    if advantage_mode == "maxrl":
        advantages_G = compute_maxrl_advantages(rewards_G)
    else:
        advantages_G = compute_grpo_advantages(rewards_G)

    # Step 2: Token-level expansion
    if transform_mode == "none":
        all_token_advs = [
            [advantages_G[i]] * len(logprobs_G[i])
            for i in range(len(logprobs_G))
        ]
        return AdvantageResult(all_token_advs, False, EntropyStats())

    # GTPO-based transforms need entropies
    all_token_advs = []
    all_exec_entropies: list[float] = []
    all_plan_entropies: list[float] = []

    for idx in range(len(logprobs_G)):
        logprobs = logprobs_G[idx]
        advantage = advantages_G[idx]
        planning_mask = planning_masks_G[idx]

        # Entropy proxy: -logprob (clamped to avoid inf)
        entropies = [min(-lp, MAX_ENTROPY) for lp in logprobs]

        # Collect entropy stats
        for j, e in enumerate(entropies):
            if planning_mask[j]:
                all_plan_entropies.append(e)
            else:
                all_exec_entropies.append(e)

        # SEPA pooling / amplification
        if transform_mode == "gtpo_sepa" and sepa_lambda > 0.0:
            entropies = apply_sepa_pooling(entropies, planning_mask, sepa_lambda)
        elif transform_mode == "gtpo_sepa_amp" and sepa_lambda > 0.0:
            entropies = apply_sepa_amplification(entropies, planning_mask, sepa_lambda)
        elif transform_mode == "gtpo_sepa_amp_c" and sepa_lambda > 0.0:
            entropies = apply_sepa_amplification_clamped(entropies, planning_mask, sepa_lambda)

        # GTPO weighting
        token_advs = apply_gtpo_weighting(advantage, entropies, beta=gtpo_beta)

        # HICRA amplification
        if transform_mode == "gtpo_hicra":
            token_advs = apply_hicra(token_advs, planning_mask, alpha=hicra_alpha)

        all_token_advs.append(token_advs)

    stats = compute_entropy_stats(all_exec_entropies, all_plan_entropies)
    return AdvantageResult(all_token_advs, True, stats)
