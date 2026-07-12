"""ECHO observation-token helpers.

ECHO trains on environment/tool tokens that appear in later multi-turn prompts.
The bridge records exact prompt-aligned observation masks when the prompt is a
role-structured chat transcript; suffix extraction remains a compatibility
fallback for older message renderers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Sequence
from typing import Protocol, cast


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
    terminal_observation_mask: list[int]
    full_observation_count: int
    positive_tokens: int
    action_token_count: int
    action_surprisal_sum: float


@dataclass(frozen=True)
class EchoBuildStats:
    """Accounting for observation-token extraction before step-level caps."""

    candidate_datums: int = 0
    candidate_tokens: int = 0
    observation_mask_datums: int = 0
    skipped_first_turns: int = 0
    skipped_no_suffix: int = 0
    skipped_low_overlap: int = 0
    split_non_prefix: int = 0
    skipped_bad_observation_mask: int = 0
    observation_responses: int = 0
    bridged_transition_datums: int = 0
    bridge_failures: int = 0
    renderer_parity_failures: int = 0
    terminal_candidate_tokens: int = 0
    explicit_transition_rollouts: int = 0


@dataclass(frozen=True)
class EchoLimitStats:
    """Accounting after step-level ECHO token caps."""

    kept_datums: int = 0
    kept_tokens: int = 0
    kept_terminal_tokens: int = 0
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
        split_non_prefix=left.split_non_prefix + right.split_non_prefix,
        skipped_bad_observation_mask=(
            left.skipped_bad_observation_mask + right.skipped_bad_observation_mask
        ),
        observation_responses=(
            left.observation_responses + right.observation_responses
        ),
        bridged_transition_datums=(
            left.bridged_transition_datums + right.bridged_transition_datums
        ),
        bridge_failures=left.bridge_failures + right.bridge_failures,
        renderer_parity_failures=(
            left.renderer_parity_failures + right.renderer_parity_failures
        ),
        terminal_candidate_tokens=(
            left.terminal_candidate_tokens + right.terminal_candidate_tokens
        ),
        explicit_transition_rollouts=(
            left.explicit_transition_rollouts + right.explicit_transition_rollouts
        ),
    )


def assert_echo_live_observation_contract(
    *,
    required: bool,
    build: EchoBuildStats,
    limit: EchoLimitStats,
    final_masks: Sequence[Sequence[float]],
    eligible_rollouts: int,
    skipped_entropy_floor: bool,
) -> None:
    """Abort strict ECHO before optimization if tool-token signal is absent."""

    if not required:
        return
    positive_final_tokens = sum(
        value > 0.0 for row in final_masks for value in row
    )
    checks = {
        "explicit_transition_rollouts": build.explicit_transition_rollouts > 0,
        "all_eligible_rollouts_use_explicit_bridge": (
            eligible_rollouts > 0
            and build.explicit_transition_rollouts == eligible_rollouts
        ),
        "observation_responses": build.observation_responses > 0,
        "bridged_transition_datums": build.bridged_transition_datums > 0,
        "zero_bridge_failures": build.bridge_failures == 0,
        "zero_renderer_parity_failures": build.renderer_parity_failures == 0,
        "candidate_tokens": build.candidate_tokens > 0,
        "entropy_floor_not_skipped": not skipped_entropy_floor,
        "kept_tokens": limit.kept_tokens > 0,
        "final_mask_matches_kept_tokens": (
            positive_final_tokens == limit.kept_tokens
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "ECHO live-observation contract failed before optimizer: "
            + ", ".join(failed)
            + f"; build={build!r}, limit={limit!r}, "
            + f"eligible_rollouts={eligible_rollouts}, "
            + f"positive_final_tokens={positive_final_tokens}"
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
    """Compatibility wrapper for rollouts that fit one exact-prefix segment.

    Multi-turn training must use :func:`build_rollout_echo_datums`; returning
    ``None`` when segmentation is required prevents callers from silently
    dropping later action tokens.
    """

    datums, stats = build_rollout_echo_datums(
        turns,
        completion_advantages=completion_advantages,
        weight=weight,
        min_prompt_overlap=min_prompt_overlap,
    )
    return (datums[0] if len(datums) == 1 else None), stats


def build_rollout_echo_datums(
    turns: Sequence[EchoTurnLike],
    *,
    completion_advantages: Sequence[Sequence[float]],
    weight: float,
    min_prompt_overlap: float,
) -> tuple[list[EchoRolloutDatum], EchoBuildStats]:
    """Build exact-prefix RL+ECHO segments covering every rollout action.

    This mirrors the reference ECHO shape: action tokens and observation tokens
    live in one sequence, with separate masks. Observation tokens are recovered
    from the suffix added to the next rendered prompt. Exact prompt-aligned
    observation masks are preferred; the suffix heuristic is only a compatibility
    fallback when a renderer cannot provide stable role masks. A prompt is
    stitched only when the whole current segment is its exact prefix. Any
    divergence finalizes the segment and restarts from the next exact prompt.
    """

    if not turns:
        return [], EchoBuildStats()

    if any(
        bool(getattr(turn, "echo_observation_capture_supported", False))
        for turn in turns
    ):
        return _build_explicit_transition_datums(
            turns,
            completion_advantages=completion_advantages,
            weight=weight,
        )

    observation_mask_datums = 0
    skipped_no_suffix = 0
    skipped_low_overlap = 0
    split_non_prefix = 0
    skipped_bad_observation_mask = 0
    full_observation_count = 0
    segments: list[EchoRolloutDatum] = []

    tokens: list[int] | None = None
    logprobs: list[float] = []
    advantages: list[float] = []
    echo_advantages: list[float] = []
    positive_tokens = 0
    action_token_count = 0
    action_surprisal_sum = 0.0

    def start_segment(prompt_ids: Sequence[int]) -> None:
        nonlocal tokens, logprobs, advantages, echo_advantages
        nonlocal positive_tokens, action_token_count, action_surprisal_sum
        tokens = list(prompt_ids)
        logprobs = [0.0] * len(tokens)
        advantages = [0.0] * len(tokens)
        echo_advantages = [0.0] * len(tokens)
        positive_tokens = 0
        action_token_count = 0
        action_surprisal_sum = 0.0

    def finish_segment() -> None:
        nonlocal tokens
        if tokens is None:
            return
        segments.append(
            EchoRolloutDatum(
                tokens=tokens,
                logprobs=logprobs,
                advantages=advantages,
                echo_advantages=echo_advantages,
                terminal_observation_mask=[0] * len(tokens),
                full_observation_count=0,
                positive_tokens=positive_tokens,
                action_token_count=action_token_count,
                action_surprisal_sum=action_surprisal_sum,
            )
        )
        tokens = None

    for turn_idx, turn in enumerate(turns):
        turn_prompt = list(turn.prompt_ids)
        if tokens is None:
            start_segment(turn_prompt)
        elif tokens != turn_prompt:
            split_non_prefix += 1
            finish_segment()
            start_segment(turn_prompt)
        assert tokens is not None
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
        action_token_count += len(completion_ids)
        action_surprisal_sum += sum(
            -value for value in completion_logprobs[: len(completion_ids)]
        )

        if turn_idx + 1 >= len(turns):
            continue

        next_turn = turns[turn_idx + 1]
        next_prompt = list(next_turn.prompt_ids)
        common = common_prefix_len(tokens, next_prompt)
        overlap_base = max(min(len(tokens), len(next_prompt)), 1)
        overlap = common / overlap_base
        if common != len(tokens):
            split_non_prefix += 1
            if overlap < min_prompt_overlap:
                skipped_low_overlap += 1
            finish_segment()
            continue
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

    finish_segment()
    segments = [
        replace(datum, full_observation_count=full_observation_count)
        for datum in segments
    ]
    stats = EchoBuildStats(
        candidate_datums=sum(datum.positive_tokens > 0 for datum in segments),
        candidate_tokens=full_observation_count,
        observation_mask_datums=observation_mask_datums,
        skipped_first_turns=len(segments),
        skipped_no_suffix=skipped_no_suffix,
        skipped_low_overlap=skipped_low_overlap,
        split_non_prefix=split_non_prefix,
        skipped_bad_observation_mask=skipped_bad_observation_mask,
    )
    return segments, stats


def _build_explicit_transition_datums(
    turns: Sequence[EchoTurnLike],
    *,
    completion_advantages: Sequence[Sequence[float]],
    weight: float,
) -> tuple[list[EchoRolloutDatum], EchoBuildStats]:
    """Build one exact sampled-action/current-observation row per turn."""

    datums: list[EchoRolloutDatum] = []
    observation_responses = 0
    bridged_transition_datums = 0
    bridge_failures = 0
    renderer_parity_failures = 0
    bad_masks = 0
    terminal_candidate_tokens = 0

    for turn_idx, turn in enumerate(turns):
        prompt_ids = list(turn.prompt_ids)
        completion_ids = list(turn.completion_ids)
        completion_logprobs = list(turn.completion_logprobs)
        turn_advantages = (
            list(completion_advantages[turn_idx])
            if turn_idx < len(completion_advantages)
            else []
        )
        completion_logprobs = _pad_float_row(
            completion_logprobs,
            len(completion_ids),
        )
        turn_advantages = _pad_float_row(turn_advantages, len(completion_ids))

        action_tokens = prompt_ids + completion_ids
        tokens = action_tokens
        observation_mask = [0] * len(tokens)
        observation_seen = bool(getattr(turn, "post_observation_seen", False))
        if observation_seen:
            observation_responses += 1
            if bool(getattr(turn, "echo_renderer_parity_failed", False)):
                renderer_parity_failures += 1
            post_ids = getattr(turn, "post_observation_ids", None)
            post_mask = getattr(turn, "post_observation_mask", None)
            if _valid_post_observation(
                post_ids,
                post_mask,
                action_tokens=action_tokens,
            ):
                assert isinstance(post_ids, list)
                assert isinstance(post_mask, list)
                tokens = [cast(int, token) for token in post_ids]
                observation_mask = [_coerce_mask_value(value) for value in post_mask]
                bridged_transition_datums += 1
            else:
                bridge_failures += 1
                if post_ids is not None or post_mask is not None:
                    bad_masks += 1

        action_end = len(action_tokens)
        suffix_len = len(tokens) - action_end
        selected = sum(observation_mask)
        is_terminal = bool(getattr(turn, "post_observation_terminal", False))
        terminal_observation_mask = (
            list(observation_mask) if is_terminal else [0] * len(tokens)
        )
        if is_terminal:
            terminal_candidate_tokens += selected
        datums.append(
            EchoRolloutDatum(
                tokens=tokens,
                logprobs=(
                    [0.0] * len(prompt_ids) + completion_logprobs + [0.0] * suffix_len
                ),
                advantages=(
                    [0.0] * len(prompt_ids) + turn_advantages + [0.0] * suffix_len
                ),
                echo_advantages=[
                    weight if include else 0.0 for include in observation_mask
                ],
                terminal_observation_mask=terminal_observation_mask,
                full_observation_count=0,
                positive_tokens=selected,
                action_token_count=len(completion_ids),
                action_surprisal_sum=sum(-value for value in completion_logprobs),
            )
        )

    full_observation_count = sum(datum.positive_tokens for datum in datums)
    datums = [
        replace(datum, full_observation_count=full_observation_count)
        for datum in datums
    ]
    return datums, EchoBuildStats(
        candidate_datums=sum(datum.positive_tokens > 0 for datum in datums),
        candidate_tokens=full_observation_count,
        observation_mask_datums=sum(datum.positive_tokens > 0 for datum in datums),
        skipped_bad_observation_mask=bad_masks,
        observation_responses=observation_responses,
        bridged_transition_datums=bridged_transition_datums,
        bridge_failures=bridge_failures,
        renderer_parity_failures=renderer_parity_failures,
        terminal_candidate_tokens=terminal_candidate_tokens,
        explicit_transition_rollouts=1,
    )


def _pad_float_row(values: list[float], length: int) -> list[float]:
    return (values + [0.0] * max(0, length - len(values)))[:length]


def _valid_post_observation(
    post_ids: object,
    post_mask: object,
    *,
    action_tokens: list[int],
) -> bool:
    if not isinstance(post_ids, list) or not isinstance(post_mask, list):
        return False
    if not all(isinstance(token, int) for token in post_ids):
        return False
    if len(post_ids) != len(post_mask) or len(post_ids) < len(action_tokens):
        return False
    if post_ids[: len(action_tokens)] != action_tokens:
        return False
    return all(not include for include in post_mask[: len(action_tokens)])


def _coerce_mask_value(raw: object) -> int:
    return 1 if bool(raw) else 0


def zero_echo_mask(advantages: list[float]) -> list[float]:
    return [0.0] * len(advantages)


def limit_echo_masks(
    echo_advantages: list[list[float]],
    *,
    max_positive_tokens: int,
    terminal_observation_masks: Sequence[Sequence[int]] | None = None,
) -> tuple[list[list[float]], EchoLimitStats]:
    """Cap ECHO targets and retain exact accounting for terminal targets."""

    if terminal_observation_masks is None:
        terminal_observation_masks = [([0] * len(row)) for row in echo_advantages]
    if len(terminal_observation_masks) != len(echo_advantages):
        raise ValueError("terminal observation masks must match ECHO rows")
    if any(
        len(mask) != len(row)
        for row, mask in zip(
            echo_advantages,
            terminal_observation_masks,
            strict=True,
        )
    ):
        raise ValueError("terminal observation masks must match ECHO row lengths")

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
    kept_terminal_tokens = 0
    kept_datums = 0
    truncated = 0
    for row, terminal_mask in zip(
        echo_advantages,
        terminal_observation_masks,
        strict=True,
    ):
        out: list[float] = []
        row_kept = 0
        for token_idx, adv in enumerate(row):
            if adv > 0.0:
                if remaining > 0:
                    out.append(adv)
                    remaining -= 1
                    kept_tokens += 1
                    if terminal_mask[token_idx]:
                        kept_terminal_tokens += 1
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
        kept_terminal_tokens=kept_terminal_tokens,
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
    echo_rollout_denominator: int = 0,
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
                echo_rollout_denominator=echo_rollout_denominator,
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
