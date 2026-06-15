"""ECHO prompt-suffix helpers.

ECHO trains on environment/tool tokens that appear in later multi-turn
prompts. The trainer owns this because it decides which tokens enter the loss;
environments should stay message-level and reward-focused.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Protocol


class EchoTurnLike(Protocol):
    """Minimal turn shape produced by the verifiers bridge."""

    prompt_ids: list[int]
    completion_ids: list[int]


@dataclass(frozen=True)
class EchoDatum:
    """One supervised prompt-suffix training datum."""

    tokens: list[int]
    advantages: list[float]
    positive_tokens: int


@dataclass(frozen=True)
class EchoBuildStats:
    """Accounting for prompt-suffix extraction before step-level caps."""

    candidate_datums: int = 0
    candidate_tokens: int = 0
    skipped_first_turns: int = 0
    skipped_no_suffix: int = 0
    skipped_low_overlap: int = 0


@dataclass(frozen=True)
class EchoLimitStats:
    """Accounting after step-level ECHO token caps."""

    kept_datums: int = 0
    kept_tokens: int = 0
    truncated_tokens: int = 0


def common_prefix_len(left: list[int], right: list[int]) -> int:
    """Return the length of the exact token prefix shared by two sequences."""

    n = min(len(left), len(right))
    for idx in range(n):
        if left[idx] != right[idx]:
            return idx
    return n


def build_prompt_suffix_echo_datums(
    rollout_turns: Sequence[Sequence[EchoTurnLike]],
    *,
    weight: float,
    min_prompt_overlap: float,
) -> tuple[list[EchoDatum], EchoBuildStats]:
    """Extract new prompt suffixes introduced after each previous turn.

    The suffix is computed against ``previous prompt + previous completion``.
    We require the next prompt to share at least the previous prompt prefix; this
    avoids training on the original user prompt when a tokenizer/chat template
    re-render is not prefix-stable enough to identify tool/environment tokens.
    """

    datums: list[EchoDatum] = []
    skipped_first_turns = 0
    skipped_no_suffix = 0
    skipped_low_overlap = 0

    for turns in rollout_turns:
        previous_full: list[int] | None = None
        previous_prompt_len = 0
        for turn in turns:
            prompt_ids = list(turn.prompt_ids)
            if previous_full is None:
                skipped_first_turns += 1
            else:
                common = common_prefix_len(previous_full, prompt_ids)
                overlap_base = max(min(len(previous_full), len(prompt_ids)), 1)
                overlap = common / overlap_base
                if common < previous_prompt_len or overlap < min_prompt_overlap:
                    skipped_low_overlap += 1
                elif common >= len(prompt_ids):
                    skipped_no_suffix += 1
                else:
                    suffix_len = len(prompt_ids) - common
                    advantages = [0.0] * common + [weight] * suffix_len
                    datums.append(
                        EchoDatum(
                            tokens=prompt_ids,
                            advantages=advantages,
                            positive_tokens=suffix_len,
                        )
                    )

            previous_prompt_len = len(prompt_ids)
            previous_full = prompt_ids + list(turn.completion_ids)

    stats = EchoBuildStats(
        candidate_datums=len(datums),
        candidate_tokens=sum(d.positive_tokens for d in datums),
        skipped_first_turns=skipped_first_turns,
        skipped_no_suffix=skipped_no_suffix,
        skipped_low_overlap=skipped_low_overlap,
    )
    return datums, stats


def limit_echo_datums(
    datums: list[EchoDatum],
    *,
    max_positive_tokens: int,
) -> tuple[list[EchoDatum], EchoLimitStats]:
    """Apply a step-level cap to positive ECHO tokens.

    The zero-weight prefix is kept only up to the last supervised token, so the
    backend sees the required context without wasting sequence length after the
    cap is reached.
    """

    if max_positive_tokens <= 0:
        return [], EchoLimitStats(
            kept_datums=0,
            kept_tokens=0,
            truncated_tokens=sum(d.positive_tokens for d in datums),
        )

    kept: list[EchoDatum] = []
    remaining = max_positive_tokens
    truncated = 0
    for datum in datums:
        if remaining <= 0:
            truncated += datum.positive_tokens
            continue
        keep_positive = min(datum.positive_tokens, remaining)
        if keep_positive <= 0:
            truncated += datum.positive_tokens
            continue

        seen_positive = 0
        cut_idx = 0
        for idx, adv in enumerate(datum.advantages):
            if adv > 0.0:
                seen_positive += 1
            if seen_positive == keep_positive:
                cut_idx = idx + 1
                break
        if cut_idx == 0:
            truncated += datum.positive_tokens
            continue

        kept.append(
            EchoDatum(
                tokens=datum.tokens[:cut_idx],
                advantages=datum.advantages[:cut_idx],
                positive_tokens=keep_positive,
            )
        )
        remaining -= keep_positive
        truncated += datum.positive_tokens - keep_positive

    kept_tokens = sum(d.positive_tokens for d in kept)
    return kept, EchoLimitStats(
        kept_datums=len(kept),
        kept_tokens=kept_tokens,
        truncated_tokens=truncated,
    )
