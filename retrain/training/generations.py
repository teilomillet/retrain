"""Generation-log selection and token diagnostics."""

from __future__ import annotations

from heapq import nlargest

from retrain.training.rollouts import TokenTextLookup, top_surprisal_entries


def generation_log_indices(
    sample_count: int,
    *,
    samples_per_prompt: int,
    rewards: list[float] | None = None,
) -> list[int]:
    """Select generation indices to log for one prompt.

    Reward-ranked caps log the completions that best represent the learning
    signal; earlier sample indices break ties for deterministic JSONL output.
    """
    if sample_count <= 0:
        return []
    if samples_per_prompt <= 0 or samples_per_prompt >= sample_count:
        return list(range(sample_count))
    if rewards is None or len(rewards) != sample_count:
        return list(range(samples_per_prompt))
    if samples_per_prompt == 1:
        return [
            max(
                range(sample_count),
                key=lambda idx: (rewards[idx], -idx),
            )
        ]

    return nlargest(
        samples_per_prompt,
        range(sample_count),
        key=lambda idx: (rewards[idx], -idx),
    )


def top_surprisal_payload(
    logprobs: list[float],
    token_ids: list[int],
    token_lookup: TokenTextLookup,
    *,
    limit: int,
) -> list[dict[str, int | float | str]]:
    """Build optional top-surprisal diagnostics for one sampled completion."""
    if limit <= 0 or not logprobs or not token_ids:
        return []
    return top_surprisal_entries(
        logprobs,
        token_ids,
        token_lookup,
        limit=limit,
    )
