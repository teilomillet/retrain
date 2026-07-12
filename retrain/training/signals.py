"""Shared training-signal helpers for RL-style runners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

from retrain.advantages import (
    AdvantageResult,
    compute_algorithm_advantages,
    compute_composable_advantages,
)
from retrain.config import TrainConfig
from retrain.training.flow import _UNIFORMITY_EPS


CORRECT_THRESHOLD = 0.5
_PROMPT_PAD_EPS = 1e-9


class RewardTieStats(TypedDict):
    """Approximate within-group reward tie summary."""

    eligible: bool
    has_tie: bool
    is_uniform: bool
    unique_count: int
    tied_pairs: int
    total_pairs: int


def apply_advantage_cap(
    all_advantages: list[list[float]],
    cap: float,
) -> tuple[list[list[float]], float, float]:
    """Cap per-token advantages to [-cap, +cap].

    This is not ratio clipping. It bounds the advantage magnitude sent to the
    backend, limiting how hard any single token can push the gradient.
    """
    total = 0
    capped_count = 0
    capped_magnitude_sum = 0.0
    result: list[list[float]] = []
    for seq in all_advantages:
        capped_seq: list[float] = []
        for advantage in seq:
            if advantage == 0.0:
                capped_seq.append(advantage)
                continue
            total += 1
            if advantage > cap:
                capped_magnitude_sum += abs(advantage)
                capped_count += 1
                capped_seq.append(cap)
            elif advantage < -cap:
                capped_magnitude_sum += abs(advantage)
                capped_count += 1
                capped_seq.append(-cap)
            else:
                capped_seq.append(advantage)
        result.append(capped_seq)
    cap_fraction = capped_count / max(total, 1)
    mean_cap_magnitude = capped_magnitude_sum / max(capped_count, 1)
    return result, cap_fraction, mean_cap_magnitude


def prepare_transform_params_for_step(
    params: Mapping[str, object] | None,
    *,
    delight_eta_prev: float | None,
) -> dict[str, object]:
    prepared = dict(params) if params is not None else {}
    if delight_eta_prev is not None and _uses_adaptive_delight(prepared):
        prepared["delight_eta_prev"] = delight_eta_prev
    return prepared


def prepare_algorithm_params_for_step(
    params: Mapping[str, object],
    *,
    delight_eta_prev: float | None,
) -> dict[str, object]:
    prepared = dict(params)
    raw_transform_params = prepared.get("transform_params")
    transform_params = (
        dict(raw_transform_params) if isinstance(raw_transform_params, Mapping) else {}
    )
    prepared["transform_params"] = prepare_transform_params_for_step(
        transform_params,
        delight_eta_prev=delight_eta_prev,
    )
    return prepared


def compute_group_advantages(
    config: TrainConfig,
    rewards_G: list[float],
    logprobs_G: list[list[float]],
    planning_masks_G: list[list[int]],
    *,
    step: int,
    sepa_lambda: float,
    algorithm_params: Mapping[str, object],
    transform_params: Mapping[str, object],
    precomputed_entropies_G: list[list[float]] | None = None,
) -> AdvantageResult:
    """Dispatch one group to the full-algorithm or composable advantage path."""
    if config.algorithm_mode:
        return compute_algorithm_advantages(
            rewards_G,
            logprobs_G,
            planning_masks_G,
            algorithm_mode=config.algorithm_mode,
            params=algorithm_params,
            gtpo_beta=config.gtpo_beta,
            hicra_alpha=config.hicra_alpha,
            sepa_lambda=sepa_lambda,
            step=step,
            token_distributions_G=None,
            precomputed_entropies_G=precomputed_entropies_G,
        )
    return compute_composable_advantages(
        rewards_G,
        logprobs_G,
        planning_masks_G,
        advantage_mode=config.advantage_mode,
        transform_mode=config.transform_mode,
        gtpo_beta=config.gtpo_beta,
        hicra_alpha=config.hicra_alpha,
        sepa_lambda=sepa_lambda,
        advantage_params=config.effective_advantage_params,
        transform_params=transform_params,
        step=step,
        post_process_params=config.post_process_params,
        token_distributions_G=None,
        precomputed_entropies_G=precomputed_entropies_G,
    )


def assert_uniform_completion_advantages_for_non_preserving_backend(
    all_logprobs: list[list[float]],
    all_advantages: list[list[float]],
    *,
    backend_name: str,
    eps: float = _UNIFORMITY_EPS,
) -> None:
    """Non-preserving backends must receive scalar completion advantages."""
    for sample_idx, (logprobs, advantages) in enumerate(
        zip(all_logprobs, all_advantages)
    ):
        n_tokens = min(len(logprobs), len(advantages))
        if n_tokens <= 1:
            continue
        prompt_len = 0
        for logprob, advantage in zip(logprobs[:n_tokens], advantages[:n_tokens]):
            if abs(logprob) <= _PROMPT_PAD_EPS and abs(advantage) <= _PROMPT_PAD_EPS:
                prompt_len += 1
            else:
                break
        completion = advantages[prompt_len:n_tokens]
        if len(completion) <= 1:
            continue
        if (max(completion) - min(completion)) > eps:
            raise RuntimeError(
                f"backend='{backend_name}' does not preserve token-level advantages, "
                f"but sample {sample_idx} contains non-uniform completion advantages."
            )


def summarize_reward_ties(
    rewards: list[float],
    *,
    eps: float = _UNIFORMITY_EPS,
) -> RewardTieStats:
    """Summarize approximate reward ties inside one prompt group."""
    n_rewards = len(rewards)
    if n_rewards < 2:
        return {
            "eligible": False,
            "has_tie": False,
            "is_uniform": False,
            "unique_count": n_rewards,
            "tied_pairs": 0,
            "total_pairs": 0,
        }

    bucket_sizes: list[int] = []
    bucket_anchor = 0.0
    for reward in sorted(rewards):
        if not bucket_sizes:
            bucket_sizes.append(1)
            bucket_anchor = reward
            continue
        if abs(reward - bucket_anchor) <= eps:
            bucket_sizes[-1] += 1
        else:
            bucket_sizes.append(1)
            bucket_anchor = reward

    unique_count = len(bucket_sizes)
    total_pairs = n_rewards * (n_rewards - 1) // 2
    tied_pairs = sum(size * (size - 1) // 2 for size in bucket_sizes)
    return {
        "eligible": True,
        "has_tie": unique_count < n_rewards,
        "is_uniform": unique_count == 1,
        "unique_count": unique_count,
        "tied_pairs": tied_pairs,
        "total_pairs": total_pairs,
    }


@dataclass
class RewardTieAccumulator:
    """Per-step aggregate of group-level reward-tie diagnostics."""

    eligible_groups: int = 0
    tie_groups: int = 0
    uniform_groups: int = 0
    tied_pairs: int = 0
    total_pairs: int = 0
    unique_fraction_sum: float = 0.0

    def observe(self, rewards: list[float]) -> RewardTieStats:
        stats = summarize_reward_ties(rewards)
        if stats["eligible"]:
            self.eligible_groups += 1
            self.tie_groups += int(stats["has_tie"])
            self.uniform_groups += int(stats["is_uniform"])
            self.tied_pairs += stats["tied_pairs"]
            self.total_pairs += stats["total_pairs"]
            self.unique_fraction_sum += stats["unique_count"] / len(rewards)
        return stats

    @property
    def tie_group_rate(self) -> float:
        return self.tie_groups / self.eligible_groups if self.eligible_groups else 0.0

    @property
    def uniform_group_rate(self) -> float:
        return (
            self.uniform_groups / self.eligible_groups if self.eligible_groups else 0.0
        )

    @property
    def tie_pair_rate(self) -> float:
        return self.tied_pairs / self.total_pairs if self.total_pairs else 0.0

    @property
    def unique_fraction_mean(self) -> float:
        return (
            self.unique_fraction_sum / self.eligible_groups
            if self.eligible_groups
            else 0.0
        )


def _uses_adaptive_delight(params: Mapping[str, object] | None) -> bool:
    if params is None:
        return False
    raw_mode = params.get("delight_eta_mode")
    if isinstance(raw_mode, str):
        return raw_mode.strip().lower() == "adaptive"
    return params.get("delight_eta_adaptive") is True
