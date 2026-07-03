"""TL-GRPO branching and turn-level advantages for verifiers rollouts."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from retrain.backends import TrainHelper


StateDict = dict[str, object]
ForkExecute = Callable[[list[object]], Mapping[str, object]]


class _Tokenizer(Protocol):
    def batch_decode(
        self,
        token_ids: list[list[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]: ...


class _TurnSample(Protocol):
    prompt_ids: list[int]


def compute_advantages(
    states: list[StateDict],
    branch_rewards: list[list[list[float]]],
    turn_weight: float = 0.5,
    outcome_baseline: float | None = None,
) -> None:
    """Write TL-GRPO per-turn advantages into completed rollout states."""
    eps = 1e-8
    if outcome_baseline is not None:
        outcome_advantages: list[float] = [
            _coerce_reward(state.get("reward")) - outcome_baseline
            for state in states
        ]
    else:
        outcome_advantages = [
            _coerce_reward(state.get("advantage")) for state in states
        ]

    for i, state in enumerate(states):
        if i >= len(branch_rewards):
            state["turn_advantages"] = []
            continue

        outcome_adv = outcome_advantages[i]
        turn_advs: list[float] = []

        for group_rewards in branch_rewards[i]:
            if len(group_rewards) < 2:
                turn_advs.append(turn_weight * outcome_adv)
                continue

            primary_reward = group_rewards[0]
            mean_reward = sum(group_rewards) / len(group_rewards)
            variance = sum(
                (reward - mean_reward) ** 2 for reward in group_rewards
            ) / len(group_rewards)
            local_adv = (primary_reward - mean_reward) / (variance**0.5 + eps)
            turn_advs.append(local_adv + turn_weight * outcome_adv)

        state["turn_advantages"] = turn_advs


def run(
    state: StateDict,
    turns: Sequence[_TurnSample],
    env: object,
    helper: "TrainHelper",
    tokenizer: object,
    *,
    branch_mode: str = "action_space",
    branch_size: int = 4,
    lookahead_steps: int = 0,
    max_tokens: int = 768,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> list[list[float]]:
    """Run TL-GRPO branch probes from a completed primary rollout."""
    turn_log = cast(list[dict[str, object]], state.get("turn_log") or [])
    fork_execute = _resolve_fork_execute(state.get("env"))
    if fork_execute is None:
        return []

    all_valid_ops: list[object] = []
    valid_op_indices: list[int] = []
    for entry in turn_log:
        if entry.get("valid") and entry.get("operation") is not None:
            valid_op_indices.append(len(all_valid_ops))
            all_valid_ops.append(entry["operation"])
        else:
            valid_op_indices.append(-1)

    branch_rewards: list[list[float]] = []
    valid_ops_so_far: list[object] = []

    for turn_idx, entry in enumerate(turn_log):
        raw_primary_delta = float(cast(int | float, entry.get("reward_delta", 0.0)))

        if not entry.get("valid") or entry.get("operation") is None:
            branch_rewards.append([raw_primary_delta])
            continue

        operation = entry["operation"]
        ops_before = list(valid_ops_so_far)
        continuation = _primary_continuation(
            all_valid_ops,
            valid_op_indices[turn_idx],
            lookahead_steps=lookahead_steps,
        )
        cumulative_reward = float(
            cast(int | float, entry.get("cumulative_reward", 0.0))
        )
        pre_cumulative = cumulative_reward - raw_primary_delta

        primary_delta = raw_primary_delta
        if continuation:
            try:
                primary_delta = _fork_and_measure(
                    fork_execute,
                    ops_before,
                    operation,
                    pre_cumulative,
                    continuation=continuation,
                )
            except (ValueError, RuntimeError):
                primary_delta = raw_primary_delta
        group_rewards = [primary_delta]

        if branch_mode == "action_space":
            primary_op = cast(dict[str, object], operation)
            legal_actions = _get_legal_actions_at_turn(fork_execute, ops_before)
            alt_actions = [action for action in legal_actions if action != primary_op]
            if len(alt_actions) > branch_size - 1:
                alt_actions = random.sample(alt_actions, branch_size - 1)
            for alt_op in alt_actions:
                group_rewards.append(
                    _measure_or_zero(
                        fork_execute,
                        ops_before,
                        alt_op,
                        pre_cumulative,
                        continuation=continuation,
                    )
                )

        elif branch_mode == "llm":
            extract_operation = getattr(
                getattr(env, "domain", None),
                "extract_operation",
                None,
            )
            if not callable(extract_operation) or turn_idx >= len(turns):
                branch_rewards.append([raw_primary_delta])
                valid_ops_so_far.append(operation)
                continue

            tokenizer_typed = cast(_Tokenizer, tokenizer)
            num_alts = branch_size - 1
            sampled = helper.sample(
                [turns[turn_idx].prompt_ids],
                num_alts,
                max_tokens,
                temperature,
                top_p,
            )
            alt_ids_list = [list(sampled[0][j][0]) for j in range(num_alts)]
            alt_texts = tokenizer_typed.batch_decode(
                alt_ids_list,
                skip_special_tokens=True,
            )
            for alt_text in alt_texts:
                try:
                    alt_op = extract_operation(alt_text)
                    delta = _fork_and_measure(
                        fork_execute,
                        ops_before,
                        alt_op,
                        pre_cumulative,
                        continuation=continuation,
                    )
                except (ValueError, RuntimeError):
                    delta = 0.0
                group_rewards.append(delta)

        branch_rewards.append(group_rewards)
        valid_ops_so_far.append(operation)

    return branch_rewards


def _coerce_reward(raw: object) -> float:
    if raw is None:
        return 0.0
    try:
        return float(cast(int | float | str, raw))
    except (TypeError, ValueError):
        return 0.0


def _primary_continuation(
    all_valid_ops: Sequence[object],
    valid_op_index: int,
    *,
    lookahead_steps: int,
) -> list[object] | None:
    if lookahead_steps <= 0:
        return None
    continuation = list(
        all_valid_ops[valid_op_index + 1 : valid_op_index + 1 + lookahead_steps]
    )
    return continuation or None


def _measure_or_zero(
    fork_execute: ForkExecute,
    ops_before: list[object],
    alt_op: object,
    pre_cumulative: float,
    *,
    continuation: list[object] | None,
) -> float:
    try:
        return _fork_and_measure(
            fork_execute,
            ops_before,
            alt_op,
            pre_cumulative,
            continuation=continuation,
        )
    except (ValueError, RuntimeError):
        return 0.0


def _fork_and_measure(
    fork_execute: ForkExecute,
    ops_before: list[object],
    alt_op: object,
    pre_cumulative: float,
    continuation: list[object] | None = None,
) -> float:
    """Execute an alternative action in a forked kernel and return reward delta."""
    ops = ops_before + [alt_op]
    if continuation:
        ops = ops + list(continuation)
    alt_snapshot = fork_execute(ops)
    run_data = alt_snapshot.get("run")
    run_map: Mapping[str, object] = (
        cast(Mapping[str, object], run_data)
        if isinstance(run_data, Mapping)
        else {}
    )
    return _coerce_reward(run_map.get("cumulative_reward", 0.0)) - pre_cumulative


_FALLBACK_ACTION_SPACE: list[dict[str, object]] = [
    {"kind": "act", "action": {"type": "accept_customer"}},
    {"kind": "act", "action": {"type": "reject_customer"}},
    {"kind": "act", "action": {"type": "schedule_restock"}},
    {"kind": "wait"},
]


def _get_legal_actions_at_turn(
    fork_execute: ForkExecute,
    ops_before: list[object],
) -> list[dict[str, object]]:
    try:
        snapshot = fork_execute(ops_before)
        model_view = snapshot.get("model_view")
        model_view_map: Mapping[str, object] = (
            cast(Mapping[str, object], model_view)
            if isinstance(model_view, Mapping)
            else {}
        )
        legal = snapshot.get("legal_actions") or model_view_map.get("legal_actions")
        legal_actions = (
            list(legal)
            if isinstance(legal, Sequence) and not isinstance(legal, (str, bytes))
            else []
        )
        if legal_actions:
            actions: list[dict[str, object]] = [
                {"kind": "act", "action": action} for action in legal_actions
            ]
            actions.append({"kind": "wait"})
            return actions
    except (ValueError, RuntimeError):
        pass
    return list(_FALLBACK_ACTION_SPACE)


def _resolve_fork_execute(env_obj: object) -> ForkExecute | None:
    fork_execute = getattr(env_obj, "fork_execute", None)
    if callable(fork_execute):
        return cast(ForkExecute, fork_execute)

    client = getattr(env_obj, "client", None)
    client_execute = getattr(client, "execute", None)
    if callable(client_execute):
        return cast(ForkExecute, client_execute)

    return None
