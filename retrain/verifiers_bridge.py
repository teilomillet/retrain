"""Bridge utilities to run verifiers environments from retrain.

This keeps the user-facing workflow TOML-first:
- select env in [environment]
- keep training through retrain backends (local/tinker)

Supports:
- dataset loading from verifiers environments
- rubric scoring for single-turn and multi-turn rollouts
- multi-turn rollouts driven by retrain sampling backends
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import types
from typing import TYPE_CHECKING, Protocol, cast

from retrain.data import Example
from retrain.type_defs import ExampleInfoLike, JSONObject, PromptLike

if TYPE_CHECKING:
    from retrain.backends import TrainHelper
    from retrain.config import TrainConfig


_FALLBACK_TRAINING_ENVS = (
    "primeintellect/gsm8k",
    "primeintellect/wordle",
    "primeintellect/hendrycks-math",
)


StateDict = dict[str, object]


class _Rubric(Protocol):
    async def score_group(self, states: list[StateDict]) -> object: ...


class _DatasetEnvironment(Protocol):
    env_id: str

    def get_dataset(self, *, n: int, seed: int | None) -> Iterable[Mapping[str, object]]: ...


class _SingleTurnEnvironment(Protocol):
    message_type: str
    rubric: _Rubric


class _MultiTurnEnvironment(Protocol):
    message_type: str
    rubric: _Rubric

    async def init_state(
        self,
        *,
        input: dict[str, object],
        client: object,
        model: str,
        sampling_args: object,
    ) -> StateDict: ...

    async def setup_state(self, state: StateDict) -> StateDict: ...
    async def is_completed(self, state: StateDict) -> bool: ...
    async def get_prompt_messages(self, state: StateDict) -> PromptLike: ...
    async def add_trajectory_step(self, state: StateDict, step: object) -> object: ...
    async def render_completion(self, state: StateDict) -> object: ...


class _Tokenizer(Protocol):
    def encode(self, text: str) -> object: ...
    def batch_decode(
        self, token_ids: list[list[int]], *, skip_special_tokens: bool = True
    ) -> list[str]: ...


def _require_verifiers() -> types.ModuleType:
    try:
        import verifiers as vf
    except ModuleNotFoundError:
        raise ImportError(
            "Verifiers environment bridge requires the verifiers package.\n"
            "Install it with: pip install 'retrain[verifiers]'"
        ) from None
    return vf


def _coerce_prompt(raw: object) -> PromptLike:
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return str(raw)
    messages: list[dict[str, object]] = []
    for msg in raw:
        if isinstance(msg, Mapping):
            messages.append(dict(cast(Mapping[str, object], msg)))
        else:
            messages.append({"role": "", "content": str(msg)})
    return messages


def _coerce_example_info(raw: object) -> ExampleInfoLike:
    if raw is None or isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        return cast(ExampleInfoLike, dict(cast(Mapping[str, object], raw)))
    return str(raw)


def _coerce_int_ids(raw: object) -> list[int]:
    if not isinstance(raw, list):
        raise TypeError(f"Expected list of token ids, got {type(raw).__name__}")
    return [_coerce_int(tok) for tok in raw]


def _coerce_int(raw: object) -> int:
    try:
        return int(cast(int | str | float, raw))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected int-like value, got {raw!r}.") from exc


def _coerce_float_list(raw: object) -> list[float]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    result: list[float] = []
    for item in raw:
        try:
            result.append(float(cast(int | float | str, item)))
        except (TypeError, ValueError):
            result.append(0.0)
    return result


def _coerce_reward(raw: object) -> float:
    if raw is None:
        return 0.0
    try:
        return float(cast(int | float | str, raw))
    except (TypeError, ValueError):
        return 0.0


def _coerce_example_id(raw: object) -> int | str:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int | str):
        return raw
    return str(raw)


def parse_environment_args(raw: str | JSONObject | None) -> JSONObject:
    """Parse [environment].args from TOML/CLI into a dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return cast(JSONObject, raw)
    if not isinstance(raw, str):
        raise ValueError(
            f"[environment].args must be a JSON string/object, got {type(raw).__name__}"
        )
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"[environment].args must be valid JSON, got: {raw}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("[environment].args must decode to a JSON object.")
    return cast(JSONObject, parsed)


def _hub_env_suggestions(env_id: str, limit: int = 5) -> list[str]:
    """Best-effort suggestions for similar Hub environment IDs."""
    if "/" not in env_id:
        return []

    query = env_id.rsplit("/", 1)[-1].split("@", 1)[0].strip()
    if not query:
        return []

    try:
        import requests
        from verifiers.utils.install_utils import ENVIRONMENTS_HUB_URL
    except Exception:
        return []

    try:
        response = requests.get(
            ENVIRONMENTS_HUB_URL,
            params={"search": query, "limit": max(1, limit)},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    suggestions: list[str] = []
    for row in payload.get("data", []):
        owner = (row.get("owner") or {}).get("name")
        name = row.get("name")
        if owner and name:
            env_key = f"{owner}/{name}"
            if env_key not in suggestions:
                suggestions.append(env_key)
        if len(suggestions) >= limit:
            break
    return suggestions


def _format_hub_suggestions(env_id: str) -> str:
    suggestions = _hub_env_suggestions(env_id)
    if not suggestions:
        return ""
    return " Similar IDs: " + ", ".join(suggestions) + "."


def load_verifiers_environment(config: "TrainConfig") -> object:
    """Load a verifiers environment from local install or Prime Hub package."""
    vf = _require_verifiers()
    env_id = config.environment_id
    env_args = parse_environment_args(config.environment_args)

    if config.environment_auto_install:
        try:
            from verifiers.utils.install_utils import (
                check_hub_env_installed,
                install_from_hub,
                is_hub_env,
            )
        except Exception:
            # Keep loading path robust even if helper APIs change in verifiers.
            check_hub_env_installed = None  # type: ignore[assignment]
            install_from_hub = None  # type: ignore[assignment]
            is_hub_env = None  # type: ignore[assignment]

        if (
            check_hub_env_installed is not None
            and install_from_hub is not None
            and is_hub_env is not None
            and is_hub_env(env_id)
            and not check_hub_env_installed(env_id)
        ):
            ok = install_from_hub(env_id)
            if not ok:
                suggestion_hint = _format_hub_suggestions(env_id)
                raise RuntimeError(
                    f"Failed to auto-install verifiers environment '{env_id}'. "
                    "The environment ID may be invalid, private, or inaccessible."
                    f"{suggestion_hint} "
                    "Try manually: uv run python -m verifiers.cli.commands.install "
                    f"{env_id}"
                )

    try:
        return vf.load_environment(env_id, **env_args)
    except Exception as exc:
        suggestion_hint = _format_hub_suggestions(env_id)
        raise RuntimeError(
            f"Failed to load verifiers environment '{env_id}': {exc}."
            f"{suggestion_hint}"
        ) from exc


def load_examples_from_environment(env: object, config: "TrainConfig") -> list[Example]:
    """Convert verifiers dataset rows into retrain Example objects."""
    n = config.max_examples if config.max_examples > 0 else -1
    seed = config.seed if config.seed >= 0 else None
    env_id = str(getattr(env, "env_id", config.environment_id or "unknown"))
    dataset_env = cast(_DatasetEnvironment, env)
    try:
        dataset = dataset_env.get_dataset(n=n, seed=seed)
    except Exception as exc:
        msg = str(exc).lower()
        if "dataset is not set" in msg:
            fallback = ", ".join(_FALLBACK_TRAINING_ENVS)
            raise RuntimeError(
                f"Environment '{env_id}' does not expose a training dataset "
                "(likely eval-only). Use a trainable environment such as "
                f"{fallback}. If you intended evaluation, use verifiers eval flow."
            ) from None
        raise RuntimeError(
            f"Failed to load dataset from verifiers environment '{env_id}': {exc}"
        ) from exc

    examples: list[Example] = []
    for row in dataset:
        row_data = row
        prompt = row_data.get("prompt")
        if prompt is None:
            # Fallback for raw pre-format datasets.
            question = row_data.get("question", "")
            prompt = str(question)
        answer = row_data.get("answer", "")
        task = row_data.get("task", getattr(env, "env_id", "") or "default")
        info = row_data.get("info", None)
        example_id = _coerce_example_id(row_data.get("example_id", -1))
        examples.append(
            Example(
                prompt=_coerce_prompt(prompt),
                reference=str(answer),
                task=str(task),
                info=_coerce_example_info(info),
                example_id=example_id,
            )
        )
    return examples


def prompt_preview(prompt: PromptLike, max_chars: int = 200) -> str:
    """Render a compact text preview for logs."""
    if isinstance(prompt, str):
        text = prompt
    else:
        parts: list[str] = []
        for msg in prompt:
            role = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            if content:
                parts.append(f"{role}: {content}")
        text = "\n".join(parts)
    return text[:max_chars]


def encode_prompt_for_sampling(tokenizer: object, prompt: PromptLike) -> list[int]:
    """Encode a prompt object (string or chat messages) for model sampling."""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    encode = getattr(tokenizer, "encode", None)
    if isinstance(prompt, str):
        if callable(apply_chat_template):
            messages = [{"role": "user", "content": prompt}]
            ids = apply_chat_template(messages, add_generation_prompt=True)
        else:
            if not callable(encode):
                raise TypeError("Tokenizer must expose encode() for string prompts.")
            ids = encode(prompt)
    else:
        if callable(apply_chat_template):
            ids = apply_chat_template(prompt, add_generation_prompt=True)
        else:
            text = "\n".join(str(msg.get("content", "")) for msg in prompt)
            if not callable(encode):
                raise TypeError("Tokenizer must expose encode() for chat prompts.")
            ids = encode(text)

    if isinstance(ids, Mapping):
        input_ids = ids.get("input_ids")
        if input_ids is not None:
            return _coerce_int_ids(input_ids)
    if hasattr(ids, "input_ids"):
        return _coerce_int_ids(getattr(ids, "input_ids"))
    return _coerce_int_ids(ids)


def _completion_messages_for_env(
    env: _SingleTurnEnvironment, completion_text: str
) -> list[dict[str, str]] | str:
    if getattr(env, "message_type", "chat") == "chat":
        return [{"role": "assistant", "content": completion_text}]
    return completion_text


def _messages_to_text(messages: object) -> str:
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return str(messages)
    chunks: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            chunks.append(str(msg))
            continue
        msg_data = cast(Mapping[str, object], msg)
        content = msg_data.get("content")
        if content:
            chunks.append(str(content))
    return "\n".join(chunks)


def score_singleturn_group(
    env: object,
    *,
    prompt: PromptLike,
    answer: str,
    task: str,
    info: ExampleInfoLike,
    completion_texts: list[str],
) -> list[float]:
    """Score a group of single-turn completions with env rubric."""
    vf = _require_verifiers()
    env_typed = cast(_SingleTurnEnvironment, env)

    states: list[StateDict] = []
    for i, text in enumerate(completion_texts):
        input_payload: dict[str, object] = {
            "prompt": prompt,
            "answer": answer,
            "task": task,
            "example_id": i,
        }
        if info is not None:
            input_payload["info"] = info

        state = cast(StateDict, vf.State(input=input_payload))
        state["completion"] = _completion_messages_for_env(env_typed, text)
        state["trajectory"] = []
        state["reward"] = None
        state["advantage"] = None
        state["metrics"] = {}
        state["error"] = None
        state["is_completed"] = True
        state["is_truncated"] = False
        state["timing"] = {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}
        states.append(state)

    asyncio.run(env_typed.rubric.score_group(states))
    return [_coerce_reward(s.get("reward")) for s in states]


@dataclass
class VerifiersTurnSample:
    """One assistant turn sampled by retrain for a rollout."""

    prompt_ids: list[int]
    completion_ids: list[int]
    completion_logprobs: list[float]
    completion_text: str


def is_multiturn_environment(env: object) -> bool:
    """Whether env is a verifiers MultiTurnEnv."""
    vf = _require_verifiers()
    return isinstance(env, vf.MultiTurnEnv)


def _compute_tl_grpo_advantages(
    states: list[StateDict],
    branch_rewards: list[list[list[float]]],
    turn_weight: float = 0.5,
    outcome_baseline: float | None = None,
) -> None:
    """TL-GRPO per-turn advantage estimation from branching rewards.

    Compares alternative actions branched from the *same* kernel state at each
    turn.  ``branch_rewards[i][t]`` is a list of ``G`` reward-deltas for
    rollout ``i``, turn ``t`` (index 0 = primary action).

    When ``outcome_baseline`` is provided, the outcome advantage is computed as
    ``R_episode - baseline`` instead of using the group-normalized advantage
    from ``score_group``.  This is necessary for ``group_size=1`` where the
    group-normalized advantage is always 0.

    Overwrites ``state["turn_advantages"]`` (previously set by MT-GRPO in
    ``score_group``).
    """
    eps = 1e-8
    if outcome_baseline is not None:
        outcome_advantages: list[float] = [
            _coerce_reward(s.get("reward")) - outcome_baseline for s in states
        ]
    else:
        outcome_advantages = [
            _coerce_reward(s.get("advantage")) for s in states
        ]

    for i, state in enumerate(states):
        if i >= len(branch_rewards):
            state["turn_advantages"] = []
            continue

        rollout_branches = branch_rewards[i]
        outcome_adv = outcome_advantages[i]
        turn_advs: list[float] = []

        for group_rewards in rollout_branches:
            if len(group_rewards) < 2:
                turn_advs.append(turn_weight * outcome_adv)
                continue

            primary_reward = group_rewards[0]
            mean_r = sum(group_rewards) / len(group_rewards)
            var_r = sum((r - mean_r) ** 2 for r in group_rewards) / len(
                group_rewards
            )
            std_r = var_r**0.5
            turn_local = (primary_reward - mean_r) / (std_r + eps)
            turn_advs.append(turn_local + turn_weight * outcome_adv)

        state["turn_advantages"] = turn_advs


def _fork_and_measure(
    fork_execute: object,
    ops_before: list[object],
    alt_op: object,
    pre_cumulative: float,
    continuation: list[object] | None = None,
) -> float:
    """Execute an alternative action in a forked kernel, return reward delta.

    When *continuation* is provided, the primary trajectory's next K actions
    are appended after the alternative.  This captures delayed effects (e.g.
    a price change that only affects the next customer interaction).
    """
    ops = ops_before + [alt_op]
    if continuation:
        ops = ops + list(continuation)
    alt_snapshot = cast(object, fork_execute)(ops)
    alt_cum = float(
        cast(dict[str, object], alt_snapshot.get("run", {})).get(
            "cumulative_reward", 0.0
        )
    )
    return alt_cum - pre_cumulative


# -- Fallback action-space alternatives (vending domain). ------------------
# Used only when the kernel response doesn't include legal_actions.
_FALLBACK_ACTION_SPACE: list[dict[str, object]] = [
    {"kind": "act", "action": {"type": "accept_customer"}},
    {"kind": "act", "action": {"type": "reject_customer"}},
    {"kind": "act", "action": {"type": "schedule_restock"}},
    {"kind": "wait"},
]


def _get_legal_actions_at_turn(
    fork_execute: object,
    ops_before: list[object],
) -> list[dict[str, object]]:
    """Get the kernel's legal actions at the state before a turn.

    Falls back to the hardcoded vending action list if the kernel response
    doesn't include legal_actions.
    """
    try:
        snapshot = cast(object, fork_execute)(ops_before)
        # legal_actions can be at top level or nested in model_view
        legal = snapshot.get("legal_actions") or snapshot.get(
            "model_view", {}
        ).get("legal_actions", [])
        if legal:
            return [{"kind": "act", "action": a} for a in legal] + [{"kind": "wait"}]
    except (ValueError, RuntimeError):
        pass
    return list(_FALLBACK_ACTION_SPACE)


def _run_tl_grpo_branching(
    state: StateDict,
    turns: list["VerifiersTurnSample"],
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
    """Run TL-GRPO branching from a completed primary rollout.

    Two branch modes:

    ``action_space`` (default)
        Enumerates legal kernel actions directly — guaranteed diversity,
        zero LLM cost.  Best for domains where the LLM is too confident
        to produce diverse samples.

    ``llm``
        Samples ``branch_size - 1`` alternative completions from the LLM.
        Useful when the action space is too large to enumerate or when
        the LLM's uncertainty is the signal of interest.

    When ``lookahead_steps > 0``, appends the next K valid operations from
    the primary trajectory after each alternative.  This captures delayed
    effects — e.g. a price change whose reward only materialises when a
    customer arrives 2 turns later.  The primary action at turn t also
    gets the same continuation so the comparison is fair.

    Returns ``branch_rewards[turn_idx]`` — a list of reward-deltas
    where index 0 is the primary action's delta.
    """
    turn_log = cast(
        list[dict[str, object]], state.get("turn_log") or []
    )
    env_obj = state.get("env")

    # fork_execute: replay an operation sequence from scratch without
    # mutating the primary rollout.  Works with both subprocess and HTTP envs.
    fork_execute = getattr(env_obj, "fork_execute", None)
    if fork_execute is None:
        client = getattr(env_obj, "client", None)
        if client is not None:
            fork_execute = client.execute

    if fork_execute is None:
        return []

    # Collect ALL valid operations upfront so we can build continuations.
    all_valid_ops: list[object] = []
    valid_op_indices: list[int] = []  # turn index → position in all_valid_ops
    for t, entry in enumerate(turn_log):
        if entry.get("valid") and entry.get("operation") is not None:
            valid_op_indices.append(len(all_valid_ops))
            all_valid_ops.append(entry["operation"])
        else:
            valid_op_indices.append(-1)

    branch_rewards: list[list[float]] = []
    valid_ops_so_far: list[object] = []

    for t, entry in enumerate(turn_log):
        primary_delta = float(
            cast(int | float, entry.get("reward_delta", 0.0))
        )

        if not entry.get("valid") or entry.get("operation") is None:
            branch_rewards.append([primary_delta])
            continue

        ops_before = list(valid_ops_so_far)
        vop_idx = valid_op_indices[t]

        # Continuation: next K valid operations from the primary trajectory.
        continuation: list[object] | None = None
        if lookahead_steps > 0:
            continuation = list(
                all_valid_ops[vop_idx + 1 : vop_idx + 1 + lookahead_steps]
            )
            if not continuation:
                continuation = None

        # Pre-action cumulative reward (deterministic replay invariant).
        cum_reward = float(
            cast(int | float, entry.get("cumulative_reward", 0.0))
        )
        pre_cumulative = cum_reward - primary_delta

        # When using lookahead, re-measure the primary action WITH
        # continuation so that primary and alternatives are compared
        # on equal footing.
        if continuation:
            try:
                primary_with_lookahead = _fork_and_measure(
                    fork_execute,
                    ops_before,
                    entry["operation"],
                    pre_cumulative,
                    continuation=continuation,
                )
            except (ValueError, RuntimeError):
                primary_with_lookahead = primary_delta
            group_rewards: list[float] = [primary_with_lookahead]
        else:
            group_rewards = [primary_delta]

        if branch_mode == "action_space":
            # Enumerate kernel actions — skip the primary action itself.
            primary_op = cast(dict[str, object], entry["operation"])
            legal_actions = _get_legal_actions_at_turn(fork_execute, ops_before)
            for alt_op in legal_actions:
                if alt_op == primary_op:
                    continue
                try:
                    delta = _fork_and_measure(
                        fork_execute, ops_before, alt_op, pre_cumulative,
                        continuation=continuation,
                    )
                    group_rewards.append(delta)
                except (ValueError, RuntimeError):
                    group_rewards.append(0.0)

        elif branch_mode == "llm":
            # Sample from the LLM (original approach).
            extract_operation = getattr(
                getattr(env, "domain", None), "extract_operation", None,
            )
            if extract_operation is None or t >= len(turns):
                branch_rewards.append([primary_delta])
                valid_ops_so_far.append(entry["operation"])
                continue

            tokenizer_typed = cast(_Tokenizer, tokenizer)
            num_alts = branch_size - 1
            sampled = helper.sample(
                [turns[t].prompt_ids], num_alts, max_tokens, temperature, top_p,
            )
            alt_ids_list = [list(sampled[0][j][0]) for j in range(num_alts)]
            alt_texts = tokenizer_typed.batch_decode(
                alt_ids_list, skip_special_tokens=True,
            )
            for alt_text in alt_texts:
                try:
                    alt_op = extract_operation(alt_text)
                    delta = _fork_and_measure(
                        fork_execute, ops_before, alt_op, pre_cumulative,
                        continuation=continuation,
                    )
                    group_rewards.append(delta)
                except (ValueError, RuntimeError):
                    group_rewards.append(0.0)

        branch_rewards.append(group_rewards)
        valid_ops_so_far.append(entry["operation"])

    return branch_rewards


def run_multiturn_group(
    env: object,
    *,
    helper: "TrainHelper",
    tokenizer: object,
    model_name: str,
    prompt: PromptLike,
    answer: str,
    task: str,
    info: ExampleInfoLike,
    num_rollouts: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_turns_override: int = -1,
    tl_grpo: bool = False,
    tl_grpo_branch_mode: str = "action_space",
    tl_grpo_branch_size: int = 4,
    tl_grpo_lookahead_steps: int = 0,
    tl_grpo_outcome_baseline: float | None = None,
) -> tuple[list[float], list[list[VerifiersTurnSample]], list[str], list[list[float]], list[list[float]], list[list[dict[str, object]]], list[list[list[float]]]]:
    """Run group rollouts for verifiers MultiTurnEnv using retrain sampling.

    Returns:
        (rewards, per_rollout_turns, completions_text, turn_rewards, turn_advantages, turn_logs, branch_rewards)
        turn_rewards: per-turn reward deltas for each rollout (from env state)
        turn_advantages: MT-GRPO per-turn advantages for each rollout (from env rubric)
        turn_logs: per-turn action log for each rollout (observation, action, result)
        branch_rewards: raw per-turn branch reward vectors (TL-GRPO only, else empty)
    """

    async def _run() -> tuple[list[float], list[list[VerifiersTurnSample]], list[str], list[list[float]], list[list[float]], list[list[dict[str, object]]], list[list[list[float]]]]:
        vf = _require_verifiers()
        env_typed = cast(_MultiTurnEnvironment, env)
        tokenizer_typed = cast(_Tokenizer, tokenizer)
        states: list[StateDict] = []
        per_rollout_turns: list[list[VerifiersTurnSample]] = [[] for _ in range(num_rollouts)]

        for i in range(num_rollouts):
            input_payload: dict[str, object] = {
                "prompt": prompt,
                "answer": answer,
                "task": task,
                "example_id": i,
            }
            if info is not None:
                input_payload["info"] = info
            state = await env_typed.init_state(
                input=input_payload,
                client=None,  # retrain handles model calls through helper.sample()
                model=model_name,
                sampling_args=None,
            )
            state = await env_typed.setup_state(state)
            states.append(state)

        turn_count = 0
        while True:
            active: list[tuple[int, PromptLike, list[int]]] = []
            for idx, state in enumerate(states):
                if await env_typed.is_completed(state):
                    continue
                prompt_messages = await env_typed.get_prompt_messages(state)
                if state.get("final_env_response") is not None:
                    continue
                prompt_ids = encode_prompt_for_sampling(tokenizer, prompt_messages)
                active.append((idx, prompt_messages, prompt_ids))

            if not active:
                break

            if max_turns_override > 0 and turn_count >= max_turns_override:
                for idx, _messages, _prompt_ids in active:
                    states[idx]["is_completed"] = True
                    states[idx]["is_truncated"] = True
                    states[idx]["stop_condition"] = "retrain_max_turns"
                break

            prompt_ids_batch = [item[2] for item in active]
            sampled_groups = helper.sample(
                prompt_ids_batch,
                1,
                max_tokens,
                temperature,
                top_p,
            )
            completion_ids_batch = [list(group[0][0]) for group in sampled_groups]
            completion_logprobs_batch = [
                [float(lp) for lp in group[0][1]] for group in sampled_groups
            ]
            completion_texts = tokenizer_typed.batch_decode(
                completion_ids_batch, skip_special_tokens=True
            )

            for pos, (idx, prompt_messages, prompt_ids) in enumerate(active):
                completion_ids = completion_ids_batch[pos]
                completion_logprobs = completion_logprobs_batch[pos]
                completion_text = completion_texts[pos]
                completion_messages = _completion_messages_for_env(env_typed, completion_text)

                tokens_payload = {
                    "prompt_ids": list(prompt_ids),
                    "prompt_mask": [0] * len(prompt_ids),
                    "completion_ids": list(completion_ids),
                    "completion_mask": [1] * len(completion_ids),
                    "completion_logprobs": list(completion_logprobs),
                    "overlong_prompt": False,
                    "is_truncated": False,
                }
                _TrajectoryStep = getattr(vf, "TrajectoryStep", None)
                if _TrajectoryStep is not None:
                    trajectory_step = _TrajectoryStep(
                        prompt=prompt_messages,
                        completion=completion_messages,
                        response=None,
                        tokens=tokens_payload,
                        reward=None,
                        advantage=None,
                        is_truncated=False,
                        trajectory_id=states[idx]["trajectory_id"],
                        extras={},
                    )
                else:
                    # Fallback for verifiers versions without TrajectoryStep.
                    trajectory_step = types.SimpleNamespace(
                        prompt=prompt_messages,
                        completion=completion_messages,
                        tokens=tokens_payload,
                        reward=None,
                        advantage=None,
                        is_truncated=False,
                        trajectory_id=states[idx].get("trajectory_id", idx),
                    )
                await env_typed.add_trajectory_step(states[idx], trajectory_step)
                per_rollout_turns[idx].append(
                    VerifiersTurnSample(
                        prompt_ids=list(prompt_ids),
                        completion_ids=list(completion_ids),
                        completion_logprobs=list(completion_logprobs),
                        completion_text=completion_text,
                    )
                )

            turn_count += 1

        for state in states:
            await env_typed.render_completion(state)
        await env_typed.rubric.score_group(states)

        # TL-GRPO: branch from each turn to get epistemically sound
        # per-turn advantages (alternatives compared against same state).
        all_branch_rewards: list[list[list[float]]] = []
        if tl_grpo:
            for i, state in enumerate(states):
                br = _run_tl_grpo_branching(
                    state,
                    per_rollout_turns[i],
                    env,
                    helper,
                    tokenizer,
                    branch_mode=tl_grpo_branch_mode,
                    branch_size=tl_grpo_branch_size,
                    lookahead_steps=tl_grpo_lookahead_steps,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                all_branch_rewards.append(br)
            _compute_tl_grpo_advantages(
                states,
                all_branch_rewards,
                outcome_baseline=tl_grpo_outcome_baseline,
            )

        rewards = [_coerce_reward(s.get("reward")) for s in states]
        completions_text = [_messages_to_text(s.get("completion")) for s in states]
        turn_rewards = [_coerce_float_list(s.get("turn_rewards")) for s in states]
        turn_advantages = [_coerce_float_list(s.get("turn_advantages")) for s in states]
        turn_logs = [
            cast(list[dict[str, object]], s.get("turn_log") or [])
            for s in states
        ]
        return rewards, per_rollout_turns, completions_text, turn_rewards, turn_advantages, turn_logs, all_branch_rewards

    return asyncio.run(_run())
