"""Delight transform implementations."""

from __future__ import annotations

from typing import cast

from retrain.advantages.constants import MAX_SURPRISAL
from retrain.advantages.delight.eta import _resolve_delight_eta
from retrain.advantages.delight.gate import (
    apply_delight_gating,
    apply_delight_sepa_gating,
    apply_hard_delight_sepa_gating,
)
from retrain.advantages.delight.metric import _compute_delight_gate_metrics
from retrain.advantages.delight.scale import _resolve_delight_norm_mode
from retrain.advantages.stats import compute_surprisal_stats
from retrain.advantages.types import AdvantageResult, TransformContext

def _compute_delight_transform(ctx: TransformContext) -> AdvantageResult:
    """Delightful Policy Gradient transform (Osband 2026, arxiv:2603.14608).

    Gates each token's advantage by σ(advantage × s / η) where s is raw
    surprisal or normalized surprisal according to delight_norm_mode.

    Scaling by rollout std (without centering) is critical for instruct-tuned
    models where mean surprisal is ~0.06: without it, the sigmoid argument is
    < 0.03 and 99% of tokens land in the neutral zone (gate ≈ 0.5). A robust
    `mad_scale` mode is also available for heavy-tailed batches.
    """
    norm_mode = _resolve_delight_norm_mode(ctx.params, default="none")
    eta, eta_metrics = _resolve_delight_eta(ctx, norm_mode=norm_mode)
    all_token_advs: list[list[float]] = []
    all_exec_surprisals: list[float] = []
    all_plan_surprisals: list[float] = []

    for idx in range(len(ctx.logprobs_G)):
        advantage = ctx.episode_advantages[idx]
        logprobs = ctx.logprobs_G[idx]
        planning_mask = ctx.planning_masks_G[idx]
        surprisals = [min(-lp, MAX_SURPRISAL) for lp in logprobs]

        token_advs = apply_delight_gating(
            advantage, surprisals, eta=eta, norm_mode=norm_mode
        )
        all_token_advs.append(token_advs)

        for j, s in enumerate(surprisals):
            if planning_mask[j]:
                all_plan_surprisals.append(s)
            else:
                all_exec_surprisals.append(s)

    stats = compute_surprisal_stats(all_exec_surprisals, all_plan_surprisals)
    extra = _compute_delight_gate_metrics(
        ctx, all_token_advs, eta, norm_mode=norm_mode
    )
    extra.update(eta_metrics)
    return AdvantageResult(all_token_advs, True, stats, extra_metrics=extra)


def _compute_delight_sepa_transform(ctx: TransformContext) -> AdvantageResult:
    """SEPA-annealed Delight gating: PG → DG transition over training.

    Uses the SEPA controller's lambda to interpolate between uniform PG
    (lambda=0, early training) and full DG gating (lambda=1, late training).

    delight_norm_mode (default "scale") controls how surprisals are
    normalized before gating. "scale" divides by rollout std without
    centering, while "mad_scale" uses a robust MAD estimate for
    outlier-heavy batches. delight_eta_mode can be "fixed" or "adaptive",
    and adaptive eta can be smoothed across steps with delight_eta_ema_decay.
    """
    norm_mode = _resolve_delight_norm_mode(ctx.params, default="scale")
    eta, eta_metrics = _resolve_delight_eta(ctx, norm_mode=norm_mode)
    # Allow fixed lambda override from transform_params (bypasses SEPA ramp)
    lam_override = ctx.params.get("delight_lambda")
    lam = float(cast(float, lam_override)) if lam_override is not None else ctx.sepa_lambda
    all_token_advs: list[list[float]] = []
    all_exec_surprisals: list[float] = []
    all_plan_surprisals: list[float] = []

    for idx in range(len(ctx.logprobs_G)):
        advantage = ctx.episode_advantages[idx]
        logprobs = ctx.logprobs_G[idx]
        planning_mask = ctx.planning_masks_G[idx]
        surprisals = [min(-lp, MAX_SURPRISAL) for lp in logprobs]

        token_advs = apply_delight_sepa_gating(
            advantage, surprisals, lambda_t=lam, eta=eta, norm_mode=norm_mode
        )
        all_token_advs.append(token_advs)

        for j, s in enumerate(surprisals):
            if planning_mask[j]:
                all_plan_surprisals.append(s)
            else:
                all_exec_surprisals.append(s)

    stats = compute_surprisal_stats(all_exec_surprisals, all_plan_surprisals)
    extra = _compute_delight_gate_metrics(
        ctx, all_token_advs, eta, lambda_t=lam, norm_mode=norm_mode
    )
    extra.update(eta_metrics)
    return AdvantageResult(all_token_advs, True, stats, extra_metrics=extra)


def _compute_hard_delight_transform(ctx: TransformContext) -> AdvantageResult:
    """Hard top-K delight gating: binary token selection.

    Keeps top k_frac% tokens by surprisal for correct rollouts (fork-points),
    bottom k_frac% for incorrect rollouts (routine tokens), zeros the rest.
    Produces ~60% gradient directional change vs PG (13x stronger than sigmoid DG).
    """
    k_frac = float(cast(float, ctx.params.get("delight_k_frac", 0.2)))
    lam_override = ctx.params.get("delight_lambda")
    lam = float(cast(float, lam_override)) if lam_override is not None else ctx.sepa_lambda
    all_token_advs: list[list[float]] = []
    all_exec_surprisals: list[float] = []
    all_plan_surprisals: list[float] = []

    for idx in range(len(ctx.logprobs_G)):
        advantage = ctx.episode_advantages[idx]
        logprobs = ctx.logprobs_G[idx]
        planning_mask = ctx.planning_masks_G[idx]
        surprisals = [min(-lp, MAX_SURPRISAL) for lp in logprobs]

        token_advs = apply_hard_delight_sepa_gating(
            advantage, surprisals, lambda_t=lam, k_frac=k_frac
        )
        all_token_advs.append(token_advs)

        for j, s in enumerate(surprisals):
            if planning_mask[j]:
                all_plan_surprisals.append(s)
            else:
                all_exec_surprisals.append(s)

    stats = compute_surprisal_stats(all_exec_surprisals, all_plan_surprisals)

    # Compute metrics: what fraction of tokens are active (non-zero)?
    all_flat = [a for seq in all_token_advs for a in seq]
    n_total = len(all_flat)
    n_active = sum(1 for a in all_flat if a != 0.0)
    extra: dict[str, float] = {
        "hard_dg_active_frac": n_active / max(n_total, 1),
        "hard_dg_k_frac": k_frac,
        "hard_dg_lambda": lam,
    }
    if n_total > 0:
        adv_sum = sum(all_flat)
        pg_sum = sum(
            ctx.episode_advantages[i] * len(ctx.logprobs_G[i])
            for i in range(len(ctx.logprobs_G))
        )
        extra["hard_dg_net_bias"] = adv_sum - pg_sum

    return AdvantageResult(all_token_advs, True, stats, extra_metrics=extra)
