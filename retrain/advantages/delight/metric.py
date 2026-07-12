"""Delight diagnostic metrics."""

from __future__ import annotations

from retrain.advantages.delight.eta import (
    _compute_delight_corrected_ordering_gaps,
    _iter_delight_rollouts,
)
from retrain.advantages.delight.gate import _sigmoid
from retrain.advantages.types import TransformContext


def _compute_delight_gate_metrics(
    ctx: TransformContext,
    token_advs: list[list[float]],
    eta: float,
    lambda_t: float | None = None,
    norm_mode: str = "none",
) -> dict[str, float]:
    """Compute gate statistics for delight transforms.

    Collects per-token gate values σ(A·s/η) across all rollouts and
    returns summary metrics for logging/blog post diagnostics.

    When norm_mode uses scaling ("scale" or "mad_scale"), surprisals are
    rescaled without centering before computing gate values, matching what
    the actual gating does.

    Besides raw gate moments, this also logs a sign-corrected
    high-vs-low surprisal gate gap. That metric is positive when DG
    preserves its intended within-rollout ordering: high-surprisal
    tokens get larger gates for positive-advantage rollouts and smaller
    gates for negative-advantage rollouts.
    """
    all_gates: list[float] = []
    pos_gates: list[float] = []  # gates from correct rollouts (A > 0)
    neg_gates: list[float] = []  # gates from incorrect rollouts (A < 0)
    inv_eta = 1.0 / max(eta, 1e-8)
    rollouts = _iter_delight_rollouts(ctx, norm_mode)

    for advantage, s_values in rollouts:
        for s in s_values:
            gate = _sigmoid(advantage * s * inv_eta)
            all_gates.append(gate)
            if advantage > 0:
                pos_gates.append(gate)
            else:
                neg_gates.append(gate)

    if not all_gates:
        return {}

    ordering_gaps, ordering_gaps_pos, ordering_gaps_neg = (
        _compute_delight_corrected_ordering_gaps(rollouts, eta)
    )

    n = len(all_gates)
    mean_g = sum(all_gates) / n
    var_g = sum((g - mean_g) ** 2 for g in all_gates) / n
    breakthrough_frac = sum(1 for g in all_gates if g > 0.8) / n
    suppressed_frac = sum(1 for g in all_gates if g < 0.2) / n
    neutral_frac = sum(1 for g in all_gates if 0.4 <= g <= 0.6) / n

    metrics: dict[str, float] = {
        "dg_gate_mean": mean_g,
        "dg_gate_std": var_g**0.5,
        "dg_gate_min": min(all_gates),
        "dg_gate_max": max(all_gates),
        "dg_breakthrough_frac": breakthrough_frac,
        "dg_suppressed_frac": suppressed_frac,
        "dg_neutral_frac": neutral_frac,
    }

    if pos_gates:
        metrics["dg_gate_mean_pos"] = sum(pos_gates) / len(pos_gates)
    if neg_gates:
        metrics["dg_gate_mean_neg"] = sum(neg_gates) / len(neg_gates)
    if ordering_gaps:
        metrics["dg_gate_ordering_gap"] = sum(ordering_gaps) / len(ordering_gaps)
    if ordering_gaps_pos:
        metrics["dg_gate_ordering_gap_pos"] = sum(ordering_gaps_pos) / len(
            ordering_gaps_pos
        )
    if ordering_gaps_neg:
        metrics["dg_gate_ordering_gap_neg"] = sum(ordering_gaps_neg) / len(
            ordering_gaps_neg
        )
    if lambda_t is not None:
        metrics["dg_lambda"] = lambda_t

    # --- Zero-sum break: net advantage bias ---
    # With uniform PG, token_adv = A_i for each token, so the sum across
    # all tokens equals Σ(A_i × n_tokens_i). This is nonzero when the
    # batch has unequal correct/incorrect counts (natural GRPO imbalance).
    # DG's asymmetric gating ADDS to this imbalance.
    #
    # To isolate the DG-specific bias, we compute:
    #   dg_bias = Σ(actual_token_advs) - Σ(uniform_PG_token_advs)
    # This is zero when λ=0 (pure PG) and positive when DG is active.
    all_flat = [a for rollout in token_advs for a in rollout]
    if all_flat:
        n_tok = len(all_flat)
        adv_sum = sum(all_flat)
        adv_mean = adv_sum / n_tok

        # Compute what uniform PG would give (same advantage, every token)
        pg_sum = 0.0
        for idx in range(len(ctx.logprobs_G)):
            advantage = ctx.episode_advantages[idx]
            n_tokens = len(ctx.logprobs_G[idx])
            pg_sum += advantage * n_tokens

        dg_bias = adv_sum - pg_sum
        metrics["dg_net_advantage_bias"] = dg_bias
        metrics["dg_net_advantage_bias_per_token"] = dg_bias / n_tok
        adv_var = sum((a - adv_mean) ** 2 for a in all_flat) / n_tok
        metrics["dg_token_adv_std"] = adv_var**0.5

        # Per-rollout advantage variance (mean across rollouts):
        # how much DG differentiates tokens within a single rollout.
        # High = DG is actively selecting tokens. Low = uniform like PG.
        rollout_vars = []
        for rollout in token_advs:
            if len(rollout) < 2:
                continue
            r_mean = sum(rollout) / len(rollout)
            r_var = sum((a - r_mean) ** 2 for a in rollout) / len(rollout)
            rollout_vars.append(r_var)
        if rollout_vars:
            metrics["dg_within_rollout_adv_var"] = sum(rollout_vars) / len(rollout_vars)

    return metrics
