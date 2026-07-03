"""Composable advantage pipeline."""

from __future__ import annotations

from collections.abc import Mapping

from retrain.advantages.constants import MAX_SURPRISAL
from retrain.advantages.credit import apply_gtpo_weighting, apply_hicra
from retrain.advantages.episode import get_advantage_spec
from retrain.advantages.plugin import (
    _coerce_advantages_output,
    _coerce_transform_output,
    _validate_token_advs,
)
from retrain.advantages.stats import compute_surprisal_stats
from retrain.advantages.transform import get_transform_spec
from retrain.advantages.types import AdvantageResult, TransformContext, UncertaintyContext
from retrain.advantages.uncertainty import _resolve_uncertainty_kind, get_uncertainty_spec

def _raise_missing_data(
    uncertainty_kind: str,
    *,
    has_logprobs: bool,
    has_distributions: bool,
    n_episodes: int = 0,
) -> None:
    """Raise a diagnostic ValueError when required data is absent."""
    logprobs_status = f"provided ({n_episodes} episodes)" if has_logprobs else "absent"
    distributions_status = (
        f"provided ({n_episodes} episodes)" if has_distributions else "absent"
    )
    raise ValueError(
        f"uncertainty_kind='{uncertainty_kind}' requires per-position token distributions.\n"
        f"Data received: logprobs_G={logprobs_status}, "
        f"token_distributions_G={distributions_status}.\n"
        f"Use uncertainty_kind='surprisal' (requires only logprobs) or use a backend "
        f"that returns full token distributions."
    )


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
    advantage_params: Mapping[str, object] | None = None,
    transform_params: Mapping[str, object] | None = None,
    step: int = 0,
    post_process_params: Mapping[str, object] | None = None,
    token_distributions_G: list[list[list[float]]] | None = None,
    precomputed_entropies_G: list[list[float]] | None = None,
) -> AdvantageResult:
    """Compute token-level advantages with composable transforms."""
    advantage_spec = get_advantage_spec(advantage_mode)
    transform_spec = get_transform_spec(transform_mode)

    # Step 1: Episode-level advantages
    raw_advantages = advantage_spec.compute(rewards_G, advantage_params or {})
    advantages_G = _coerce_advantages_output(
        raw_advantages,
        expected_len=len(rewards_G),
        mode_name=advantage_spec.name,
    )
    expected_lens = [len(seq) for seq in logprobs_G]

    merged_transform_params: dict[str, object] = {}
    if post_process_params:
        merged_transform_params.update(post_process_params)
    if transform_params:
        merged_transform_params.update(transform_params)

    if transform_spec.compute_context is not None:
        ctx = TransformContext(
            episode_advantages=advantages_G,
            logprobs_G=logprobs_G,
            planning_masks_G=planning_masks_G,
            sepa_lambda=sepa_lambda,
            params=merged_transform_params,
            step=step,
            gtpo_beta=gtpo_beta,
            hicra_alpha=hicra_alpha,
        )
        raw_output = transform_spec.compute_context(ctx)
        return _coerce_transform_output(
            raw_output,
            expected_lens=expected_lens,
            mode_name=f"transform_mode '{transform_spec.name}'",
        )

    # Step 2: Token-level expansion
    if not transform_spec.use_gtpo:
        all_token_advs = [
            [advantages_G[i]] * len(logprobs_G[i])
            for i in range(len(logprobs_G))
        ]
        _validate_token_advs(
            all_token_advs,
            expected_lens=expected_lens,
            mode_name=f"transform_mode '{transform_spec.name}'",
        )
        # Observational stats — advantages are unchanged but we still
        # compute surprisal distributions for cross-condition comparison.
        obs_exec: list[float] = []
        obs_plan: list[float] = []
        for idx in range(len(logprobs_G)):
            for j, lp in enumerate(logprobs_G[idx]):
                s = min(-lp, MAX_SURPRISAL)
                if planning_masks_G[idx][j]:
                    obs_plan.append(s)
                else:
                    obs_exec.append(s)
        stats = compute_surprisal_stats(obs_exec, obs_plan)
        return AdvantageResult(all_token_advs, True, stats)

    uncertainty_kind = _resolve_uncertainty_kind(merged_transform_params)
    uncertainty_spec = get_uncertainty_spec(uncertainty_kind)

    if (
        uncertainty_spec.needs_distributions
        and token_distributions_G is None
        and precomputed_entropies_G is None
    ):
        _raise_missing_data(
            uncertainty_kind,
            has_logprobs=True,
            has_distributions=False,
            n_episodes=len(logprobs_G),
        )

    # GTPO-based transforms need per-token uncertainty values.
    # The pipeline is agnostic to the kind — the backend determines what's
    # available, and the user picks via uncertainty_kind.
    all_token_advs = []
    all_exec_surprisals: list[float] = []
    all_plan_surprisals: list[float] = []
    all_post_exec_surprisals: list[float] = []
    all_post_plan_surprisals: list[float] = []
    all_raw_surprisals: list[list[float]] = []  # for surprisal_mask compatibility

    for idx in range(len(logprobs_G)):
        logprobs = logprobs_G[idx]
        advantage = advantages_G[idx]
        planning_mask = planning_masks_G[idx]

        ctx = UncertaintyContext(
            logprobs=logprobs,
            token_distributions=token_distributions_G[idx] if token_distributions_G else None,
            precomputed_entropy=precomputed_entropies_G[idx] if precomputed_entropies_G else None,
            planning_mask=planning_mask,
            params=merged_transform_params,
        )
        surprisals = uncertainty_spec.compute(ctx)

        # Store raw surprisals before any transform (for surprisal masking)
        all_raw_surprisals.append(list(surprisals))

        # Collect pre-transform surprisal stats
        for j, e in enumerate(surprisals):
            if planning_mask[j]:
                all_plan_surprisals.append(e)
            else:
                all_exec_surprisals.append(e)

        # Optional surprisal transform (SEPA variants or custom plugin)
        if transform_spec.entropy_transform is not None:
            surprisals = transform_spec.entropy_transform(
                surprisals, planning_mask, sepa_lambda
            )

        # Collect post-transform surprisal stats (before GTPO weighting)
        for j, e in enumerate(surprisals):
            if planning_mask[j]:
                all_post_plan_surprisals.append(e)
            else:
                all_post_exec_surprisals.append(e)

        # GTPO weighting
        token_advs = apply_gtpo_weighting(advantage, surprisals, beta=gtpo_beta)

        # HICRA amplification
        if transform_spec.apply_hicra:
            token_advs = apply_hicra(token_advs, planning_mask, alpha=hicra_alpha)

        all_token_advs.append(token_advs)

    # Post-process hook (e.g. surprisal masking)
    extra_metrics: dict[str, float] = {}
    n_seqs = len(all_token_advs)
    seq_lens = [len(seq) for seq in all_token_advs]
    if transform_spec.post_process is not None:
        all_token_advs, extra_metrics = transform_spec.post_process(
            all_token_advs, all_raw_surprisals, merged_transform_params
        )
        # Validate hook output shape
        if len(all_token_advs) != n_seqs:
            raise ValueError(
                f"post_process hook '{transform_spec.name}' returned "
                f"{len(all_token_advs)} sequences, expected {n_seqs}"
            )
        for i, seq in enumerate(all_token_advs):
            if len(seq) != seq_lens[i]:
                raise ValueError(
                    f"post_process hook '{transform_spec.name}' returned "
                    f"{len(seq)} tokens for sequence {i}, expected {seq_lens[i]}"
                )

    _validate_token_advs(
        all_token_advs,
        expected_lens=expected_lens,
        mode_name=f"transform_mode '{transform_spec.name}'",
    )
    stats = compute_surprisal_stats(all_exec_surprisals, all_plan_surprisals)
    post_stats = compute_surprisal_stats(all_post_exec_surprisals, all_post_plan_surprisals)
    stats.post_exec_mean = post_stats.exec_mean
    stats.post_exec_var = post_stats.exec_var
    stats.post_exec_count = post_stats.exec_count
    stats.post_plan_mean = post_stats.plan_mean
    stats.post_plan_var = post_stats.plan_var
    stats.post_plan_count = post_stats.plan_count
    return AdvantageResult(all_token_advs, True, stats, extra_metrics=extra_metrics)
