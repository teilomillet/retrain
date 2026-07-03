"""ECHO observation-token helpers.

ECHO trains on environment/tool tokens that appear in later multi-turn prompts.
The bridge records exact prompt-aligned observation masks when the prompt is a
role-structured chat transcript; suffix extraction remains a compatibility
fallback for older message renderers.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Protocol


class EchoTurnLike(Protocol):
    """Minimal turn shape produced by the verifiers bridge."""

    prompt_ids: list[int]
    completion_ids: list[int]
    completion_logprobs: list[float]


@dataclass(frozen=True)
class EchoRolloutDatum:
    """One faithful full-rollout training datum.

    The token row contains both assistant action tokens and later observation
    tokens. ``advantages`` is the RL/action-token mask, while
    ``echo_advantages`` is the environment-prediction mask scaled by lambda.
    """

    tokens: list[int]
    logprobs: list[float]
    advantages: list[float]
    echo_advantages: list[float]
    full_observation_count: int
    positive_tokens: int


@dataclass(frozen=True)
class EchoBuildStats:
    """Accounting for observation-token extraction before step-level caps."""

    candidate_datums: int = 0
    candidate_tokens: int = 0
    observation_mask_datums: int = 0
    skipped_first_turns: int = 0
    skipped_no_suffix: int = 0
    skipped_low_overlap: int = 0
    skipped_bad_observation_mask: int = 0


@dataclass(frozen=True)
class EchoLimitStats:
    """Accounting after step-level ECHO token caps."""

    kept_datums: int = 0
    kept_tokens: int = 0
    truncated_tokens: int = 0


def merge_echo_build_stats(
    left: EchoBuildStats,
    right: EchoBuildStats,
) -> EchoBuildStats:
    """Combine ECHO extraction counters across rollout groups."""

    return EchoBuildStats(
        candidate_datums=left.candidate_datums + right.candidate_datums,
        candidate_tokens=left.candidate_tokens + right.candidate_tokens,
        observation_mask_datums=(
            left.observation_mask_datums + right.observation_mask_datums
        ),
        skipped_first_turns=left.skipped_first_turns + right.skipped_first_turns,
        skipped_no_suffix=left.skipped_no_suffix + right.skipped_no_suffix,
        skipped_low_overlap=left.skipped_low_overlap + right.skipped_low_overlap,
        skipped_bad_observation_mask=(
            left.skipped_bad_observation_mask + right.skipped_bad_observation_mask
        ),
    )


def common_prefix_len(left: list[int], right: list[int]) -> int:
    """Return the length of the exact token prefix shared by two sequences."""

    n = min(len(left), len(right))
    for idx in range(n):
        if left[idx] != right[idx]:
            return idx
    return n


def build_rollout_echo_datum(
    turns: Sequence[EchoTurnLike],
    *,
    completion_advantages: Sequence[Sequence[float]],
    weight: float,
    min_prompt_overlap: float,
) -> tuple[EchoRolloutDatum | None, EchoBuildStats]:
    """Build one same-rollout RL+ECHO datum from a multi-turn transcript.

    This mirrors the reference ECHO shape: action tokens and observation tokens
    live in one sequence, with separate masks. Observation tokens are recovered
    from the suffix added to the next rendered prompt. Exact prompt-aligned
    observation masks are preferred; the suffix heuristic is only a compatibility
    fallback when a renderer cannot provide stable role masks.
    """

    if not turns:
        return None, EchoBuildStats()

    tokens = list(turns[0].prompt_ids)
    logprobs = [0.0] * len(tokens)
    advantages = [0.0] * len(tokens)
    echo_advantages = [0.0] * len(tokens)

    observation_mask_datums = 0
    skipped_first_turns = 1
    skipped_no_suffix = 0
    skipped_low_overlap = 0
    skipped_bad_observation_mask = 0
    positive_tokens = 0
    full_observation_count = 0

    for turn_idx, turn in enumerate(turns):
        completion_ids = list(turn.completion_ids)
        completion_logprobs = list(turn.completion_logprobs)
        turn_advantages = (
            list(completion_advantages[turn_idx])
            if turn_idx < len(completion_advantages)
            else []
        )
        if len(completion_logprobs) < len(completion_ids):
            completion_logprobs = completion_logprobs + [0.0] * (
                len(completion_ids) - len(completion_logprobs)
            )
        if len(turn_advantages) < len(completion_ids):
            turn_advantages = turn_advantages + [0.0] * (
                len(completion_ids) - len(turn_advantages)
            )

        tokens.extend(completion_ids)
        logprobs.extend(completion_logprobs[: len(completion_ids)])
        advantages.extend(turn_advantages[: len(completion_ids)])
        echo_advantages.extend([0.0] * len(completion_ids))

        if turn_idx + 1 >= len(turns):
            continue

        next_turn = turns[turn_idx + 1]
        next_prompt = list(next_turn.prompt_ids)
        common = common_prefix_len(tokens, next_prompt)
        overlap_base = max(min(len(tokens), len(next_prompt)), 1)
        overlap = common / overlap_base
        if overlap < min_prompt_overlap:
            return None, EchoBuildStats(
                skipped_first_turns=skipped_first_turns,
                skipped_low_overlap=1,
            )
        if common >= len(next_prompt):
            skipped_no_suffix += 1
            continue

        suffix = next_prompt[common:]
        suffix_echo = [0.0] * len(suffix)
        observation_mask = getattr(next_turn, "observation_mask", None)
        if observation_mask is not None:
            mask = [_coerce_mask_value(value) for value in observation_mask]
            if len(mask) != len(next_prompt):
                skipped_bad_observation_mask += 1
            else:
                suffix_mask = mask[common:]
                selected = sum(suffix_mask)
                if selected:
                    suffix_echo = [
                        weight if include else 0.0 for include in suffix_mask
                    ]
                    observation_mask_datums += 1
                    positive_tokens += selected
                    full_observation_count += selected
                else:
                    skipped_no_suffix += 1
        else:
            suffix_len = len(suffix)
            if suffix_len:
                suffix_echo = [weight] * suffix_len
                positive_tokens += suffix_len
                full_observation_count += suffix_len
            else:
                skipped_no_suffix += 1

        tokens.extend(suffix)
        logprobs.extend([0.0] * len(suffix))
        advantages.extend([0.0] * len(suffix))
        echo_advantages.extend(suffix_echo)

    stats = EchoBuildStats(
        candidate_datums=1 if positive_tokens else 0,
        candidate_tokens=positive_tokens,
        observation_mask_datums=observation_mask_datums,
        skipped_first_turns=skipped_first_turns,
        skipped_no_suffix=skipped_no_suffix,
        skipped_low_overlap=skipped_low_overlap,
        skipped_bad_observation_mask=skipped_bad_observation_mask,
    )
    return (
        EchoRolloutDatum(
            tokens=tokens,
            logprobs=logprobs,
            advantages=advantages,
            echo_advantages=echo_advantages,
            full_observation_count=full_observation_count,
            positive_tokens=positive_tokens,
        ),
        stats,
    )


def _coerce_mask_value(raw: object) -> int:
    return 1 if bool(raw) else 0


def zero_echo_mask(advantages: list[float]) -> list[float]:
    return [0.0] * len(advantages)


def limit_echo_masks(
    echo_advantages: list[list[float]],
    *,
    max_positive_tokens: int,
) -> tuple[list[list[float]], EchoLimitStats]:
    """Cap ECHO target tokens while preserving full rollout rows."""

    total_positive = sum(1 for row in echo_advantages for adv in row if adv > 0.0)
    if max_positive_tokens <= 0:
        return [zero_echo_mask(row) for row in echo_advantages], EchoLimitStats(
            kept_datums=0,
            kept_tokens=0,
            truncated_tokens=total_positive,
        )

    kept_rows: list[list[float]] = []
    remaining = max_positive_tokens
    kept_tokens = 0
    kept_datums = 0
    truncated = 0
    for row in echo_advantages:
        out: list[float] = []
        row_kept = 0
        for adv in row:
            if adv > 0.0:
                if remaining > 0:
                    out.append(adv)
                    remaining -= 1
                    kept_tokens += 1
                    row_kept += 1
                else:
                    out.append(0.0)
                    truncated += 1
            else:
                out.append(adv)
        if row_kept:
            kept_datums += 1
        kept_rows.append(out)

    return kept_rows, EchoLimitStats(
        kept_datums=kept_datums,
        kept_tokens=kept_tokens,
        truncated_tokens=truncated,
    )


def run_rl_echo_train_step(
    helper: object,
    all_tokens: list[list[int]],
    all_logprobs: list[list[float]],
    all_advantages: list[list[float]],
    echo_advantages: list[list[float]],
    echo_full_observation_counts: list[int],
    *,
    echo_loss_fn: str,
    lr: float,
    weight_decay: float,
) -> tuple[float, float, bool]:
    """Run one RL update, optionally with ECHO in the same optimizer step.

    ECHO is independent of the chosen RL algorithm: algorithms produce the
    sampled-token advantages above, while ECHO adds a same-rollout
    environment-token mask. Paper-faithful RL+ECHO requires a backend
    ``train_step_with_echo_masks`` implementation that computes both losses
    from the same actor forward/backward pass over those rollout rows.
    """

    if not all_tokens:
        return 0.0, 0.0, False

    if echo_advantages:
        train_step_with_echo_masks = getattr(helper, "train_step_with_echo_masks", None)
        if callable(train_step_with_echo_masks):
            rl_loss, echo_loss = train_step_with_echo_masks(
                all_tokens,
                all_logprobs,
                all_advantages,
                echo_advantages,
                echo_full_observation_counts,
                echo_loss_fn,
                lr,
                weight_decay,
            )
            return float(rl_loss), float(echo_loss), True
        raise RuntimeError(
            "ECHO requires a backend train_step_with_echo_masks implementation "
            "so RL and environment-token losses are computed from the same "
            "rollout rows in one actor forward/backward pass."
        )

    train_step = getattr(helper, "train_step")
    rl_loss = float(
        train_step(
            all_tokens,
            all_logprobs,
            all_advantages,
            lr,
            weight_decay,
        )
    )
    return rl_loss, 0.0, False
