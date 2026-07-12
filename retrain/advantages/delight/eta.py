"""Delight eta resolution."""

from __future__ import annotations

import math
from typing import cast

from retrain.advantages.constants import MAX_SURPRISAL
from retrain.advantages.delight.gate import _sigmoid
from retrain.advantages.delight.scale import (
    _apply_delight_norm_mode,
    _quantile,
    _resolve_delight_eta_mode,
)
from retrain.advantages.types import TransformContext


def _iter_delight_rollouts(
    ctx: TransformContext,
    norm_mode: str,
) -> list[tuple[float, list[float]]]:
    """Materialize normalized Delight surprisal sequences for the batch."""
    rollouts: list[tuple[float, list[float]]] = []
    for idx in range(len(ctx.logprobs_G)):
        advantage = ctx.episode_advantages[idx]
        if advantage == 0.0:
            continue
        raw_surprisals = [min(-lp, MAX_SURPRISAL) for lp in ctx.logprobs_G[idx]]
        s_values = _apply_delight_norm_mode(raw_surprisals, norm_mode)
        rollouts.append((advantage, s_values))
    return rollouts


def _compute_delight_corrected_ordering_gaps(
    rollouts: list[tuple[float, list[float]]],
    eta: float,
) -> tuple[list[float], list[float], list[float]]:
    """Return sign-corrected high-vs-low gate gaps for Delight rollouts."""
    inv_eta = 1.0 / max(eta, 1e-8)
    all_gaps: list[float] = []
    pos_gaps: list[float] = []
    neg_gaps: list[float] = []

    for advantage, s_values in rollouts:
        if len(s_values) < 2:
            continue
        rollout_pairs = [(s, _sigmoid(advantage * s * inv_eta)) for s in s_values]
        ordered = sorted(rollout_pairs, key=lambda p: p[0])
        half = len(ordered) // 2
        if half == 0:
            continue
        low_mean = sum(g for _, g in ordered[:half]) / half
        high_mean = sum(g for _, g in ordered[-half:]) / half
        gap = high_mean - low_mean
        corrected_gap = gap if advantage > 0 else -gap
        all_gaps.append(corrected_gap)
        if advantage > 0:
            pos_gaps.append(corrected_gap)
        else:
            neg_gaps.append(corrected_gap)

    return all_gaps, pos_gaps, neg_gaps


def _resolve_delight_eta(
    ctx: TransformContext,
    *,
    norm_mode: str,
) -> tuple[float, dict[str, float]]:
    """Resolve effective Delight eta for this batch.

    `delight_eta_mode = "adaptive"` chooses eta from the batch's current
    |A * s| logit magnitudes so the gate lands near a target neutral
    fraction, then optionally sharpens further to hit a minimum ordering gap.
    """
    eta_mode = _resolve_delight_eta_mode(ctx.params, default="fixed")
    base_eta = max(float(cast(float, ctx.params.get("delight_eta", 1.0))), 1e-8)
    ema_decay = float(cast(float, ctx.params.get("delight_eta_ema_decay", 0.0)))
    if ema_decay < 0.0 or ema_decay >= 1.0:
        raise ValueError(
            "delight_eta_ema_decay must be in [0, 1). Try: delight_eta_ema_decay = 0.8"
        )
    raw_prev_eta = ctx.params.get("delight_eta_prev")
    prev_eta: float | None = None
    if raw_prev_eta is not None:
        prev_eta = max(float(cast(float, raw_prev_eta)), 1e-8)

    if eta_mode == "fixed":
        return base_eta, {
            "dg_eta": base_eta,
            "dg_eta_adaptive": 0.0,
        }

    rollouts = _iter_delight_rollouts(ctx, norm_mode)
    if not rollouts:
        eta = base_eta
        metrics = {
            "dg_eta": eta,
            "dg_eta_adaptive": 1.0,
        }
        if prev_eta is not None and ema_decay > 0.0:
            eta = max(ema_decay * prev_eta + (1.0 - ema_decay) * eta, 1e-8)
            metrics.update(
                {
                    "dg_eta": eta,
                    "dg_eta_raw": base_eta,
                    "dg_eta_prev": prev_eta,
                    "dg_eta_ema_decay": ema_decay,
                }
            )
        return eta, metrics

    target_neutral = float(
        cast(float, ctx.params.get("delight_eta_target_neutral_frac", 0.5))
    )
    target_neutral = min(max(target_neutral, 0.0), 1.0)
    target_gap = max(
        0.0, float(cast(float, ctx.params.get("delight_eta_target_ordering_gap", 0.0)))
    )
    eta_min = max(1e-8, float(cast(float, ctx.params.get("delight_eta_min", 0.05))))
    eta_max = max(eta_min, float(cast(float, ctx.params.get("delight_eta_max", 5.0))))
    neutral_logit = math.log(0.6 / 0.4)

    magnitudes = [
        abs(advantage * s) for advantage, s_values in rollouts for s in s_values
    ]
    if magnitudes:
        eta_raw = _quantile(magnitudes, target_neutral) / max(neutral_logit, 1e-8)
    else:
        eta_raw = base_eta
    eta_raw = min(max(eta_raw, eta_min), eta_max)
    eta = eta_raw

    if target_gap > 0.0:
        ordering_gaps, _, _ = _compute_delight_corrected_ordering_gaps(rollouts, eta)
        if ordering_gaps:
            gap_now = sum(ordering_gaps) / len(ordering_gaps)
            if gap_now < target_gap:
                ratio = max(gap_now / max(target_gap, 1e-8), 0.25)
                eta = min(max(eta * ratio, eta_min), eta_max)
    eta_raw = eta

    if prev_eta is not None and ema_decay > 0.0:
        eta = min(
            max(ema_decay * prev_eta + (1.0 - ema_decay) * eta, eta_min),
            eta_max,
        )

    metrics = {
        "dg_eta": eta,
        "dg_eta_adaptive": 1.0,
        "dg_eta_target_neutral_frac": target_neutral,
        "dg_eta_target_ordering_gap": target_gap,
        "dg_eta_min": eta_min,
        "dg_eta_max": eta_max,
    }
    if prev_eta is not None and ema_decay > 0.0:
        metrics.update(
            {
                "dg_eta_raw": eta_raw,
                "dg_eta_prev": prev_eta,
                "dg_eta_ema_decay": ema_decay,
            }
        )
    return eta, metrics
