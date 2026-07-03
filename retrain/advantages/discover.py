"""TTT-Discover-style algorithm objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping

from retrain.advantages.types import AdvantageResult, AlgorithmContext, EntropyStats


def _discover_softmax_probs(rewards: list[float], beta: float) -> list[float]:
    """Boltzmann probabilities over sampled returns."""
    if not rewards:
        return []
    max_r = max(rewards)
    exps = [math.exp(max(-700.0, beta * (r - max_r))) for r in rewards]
    denom = sum(exps)
    if denom <= 0.0:
        n = len(rewards)
        return [1.0 / n] * n
    return [value / denom for value in exps]


def _discover_kl_to_uniform(probs: list[float]) -> float:
    """KL(probs || uniform)."""
    n = len(probs)
    if n <= 1:
        return 0.0
    log_n = math.log(float(n))
    kl = 0.0
    for prob in probs:
        if prob > 0.0:
            kl += prob * (math.log(prob) + log_n)
    return kl


def _resolve_discover_beta(
    rewards: list[float],
    params: Mapping[str, object],
) -> tuple[float, float]:
    """Choose beta from a fixed value or a target-KL constraint."""
    n = len(rewards)
    if n <= 1:
        return 0.0, 0.0
    if max(rewards) - min(rewards) <= 1e-12:
        return 0.0, 0.0

    fixed_beta = params.get("beta")
    if isinstance(fixed_beta, (int, float)) and not isinstance(fixed_beta, bool):
        beta = max(0.0, float(fixed_beta))
        probs = _discover_softmax_probs(rewards, beta)
        return beta, _discover_kl_to_uniform(probs)

    target_kl = params.get("target_kl", params.get("discover_target_kl", 0.2))
    beta_max = params.get("beta_max", params.get("discover_beta_max", 50.0))
    search_steps = params.get("beta_search_steps", 24)

    if not isinstance(target_kl, (int, float)) or isinstance(target_kl, bool):
        target_kl = 0.2
    if not isinstance(beta_max, (int, float)) or isinstance(beta_max, bool):
        beta_max = 50.0
    if not isinstance(search_steps, int) or isinstance(search_steps, bool):
        search_steps = 24

    target = max(0.0, float(target_kl))
    hi = max(0.0, float(beta_max))
    if target == 0.0 or hi == 0.0:
        return 0.0, 0.0

    probs_hi = _discover_softmax_probs(rewards, hi)
    kl_hi = _discover_kl_to_uniform(probs_hi)
    if kl_hi <= target:
        return hi, kl_hi

    lo = 0.0
    best_beta = 0.0
    best_kl = 0.0
    for _ in range(max(search_steps, 1)):
        mid = (lo + hi) / 2.0
        probs_mid = _discover_softmax_probs(rewards, mid)
        kl_mid = _discover_kl_to_uniform(probs_mid)
        if kl_mid <= target:
            best_beta = mid
            best_kl = kl_mid
            lo = mid
        else:
            hi = mid
    return best_beta, best_kl


def _compute_discover_entropic(ctx: AlgorithmContext) -> AdvantageResult:
    """TTT-Discover-style entropic objective over sampled returns."""
    beta, beta_kl = _resolve_discover_beta(ctx.rewards_G, ctx.params)
    probs = _discover_softmax_probs(ctx.rewards_G, beta)
    n = len(probs)
    weights = [prob * n for prob in probs] if probs else [0.0] * len(ctx.rewards_G)
    episode_advantages = [weight - 1.0 for weight in weights]
    token_advs = [
        [episode_advantages[idx]] * len(logprobs)
        for idx, logprobs in enumerate(ctx.logprobs_G)
    ]
    return AdvantageResult(
        token_advs=token_advs,
        has_stats=False,
        stats=EntropyStats(),
        extra_metrics={
            "discover_beta": beta,
            "discover_kl": beta_kl,
            "discover_reward_max": max(ctx.rewards_G) if ctx.rewards_G else 0.0,
            "discover_reward_mean": (
                sum(ctx.rewards_G) / len(ctx.rewards_G) if ctx.rewards_G else 0.0
            ),
        },
    )
