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
import importlib
import json
import math
import sys
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
import time
from dataclasses import dataclass
import types
from typing import TYPE_CHECKING, Protocol, cast

from retrain.data.source import Example
from retrain.environments import prompt as prompt_utils
from retrain.types import ExampleInfoLike, JSONObject, PromptLike

if TYPE_CHECKING:
    from retrain.backends import TrainHelper
    from retrain.config import TrainConfig


_FALLBACK_TRAINING_ENVS = (
    "primeintellect/gsm8k",
    "primeintellect/wordle",
    "primeintellect/hendrycks-math",
)

StateDict = dict[str, object]
ForkExecute = Callable[[list[object]], Mapping[str, object]]


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
    async def cleanup(self, state: StateDict) -> object: ...


class _Tokenizer(Protocol):
    def encode(self, text: str) -> object: ...
    def batch_decode(
        self, token_ids: list[list[int]], *, skip_special_tokens: bool = True
    ) -> list[str]: ...


def _require_verifiers() -> types.ModuleType:
    try:
        return importlib.import_module("verifiers")
    except ModuleNotFoundError:
        raise ImportError(
            "Verifiers environment bridge requires the verifiers package.\n"
            "Install it with: pip install 'retrain[verifiers]'"
        ) from None


_NULL_CLIENT_MSG = (
    "retrain performs sampling via TrainHelper; the verifiers client must never be used"
)


def _make_env_client() -> object | None:
    """Inert client to satisfy Environment.init_state.

    retrain samples through helper.sample(), never through the verifiers
    client. Newer verifiers (>= 0.1.12) validate the client argument in
    init_state (resolve_client raises on None), so we hand it a Client whose
    sampling surface fails loudly if anything ever tries to use it. Older
    verifiers accepted None; fall back to that.
    """
    try:
        from verifiers.clients import Client  # type: ignore[unresolved-import]
    except ImportError:
        return None

    class _RetrainNullClient(Client):
        def setup_client(self, config: object) -> object:
            raise NotImplementedError(_NULL_CLIENT_MSG)

        async def to_native_tool(self, tool: object) -> object:
            raise NotImplementedError(_NULL_CLIENT_MSG)

        async def to_native_prompt(self, messages: object) -> object:
            raise NotImplementedError(_NULL_CLIENT_MSG)

        async def get_native_response(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError(_NULL_CLIENT_MSG)

        async def raise_from_native_response(self, response: object) -> None:
            raise NotImplementedError(_NULL_CLIENT_MSG)

        async def from_native_response(self, response: object) -> object:
            raise NotImplementedError(_NULL_CLIENT_MSG)

        async def close(self) -> None:  # cleanup paths may call this; no-op
            return None

    return _RetrainNullClient(None)


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


def _object_field(obj: object, key: str) -> object:
    if isinstance(obj, Mapping):
        return cast(Mapping[str, object], obj).get(key)
    return getattr(obj, key, None)


def _collect_observation_timing(
    state: StateDict,
    totals: dict[str, float] | None,
) -> None:
    if totals is None:
        return
    trajectory = state.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        return
    step = trajectory[-1]
    extras = _object_field(step, "extras")
    candidates: list[tuple[object, bool]] = []
    if isinstance(extras, Mapping):
        extras_map = cast(Mapping[str, object], extras)
        candidates.extend(
            [
                (extras_map.get("openenv_info"), False),
                (extras_map.get("info"), False),
                (extras_map, False),
            ]
        )
    candidates.append((_object_field(step, "timing"), True))

    for candidate, direct_timing in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_map = cast(Mapping[str, object], candidate)
        timing = candidate_map.get("timing")
        if isinstance(timing, Mapping):
            _accumulate_numeric_timing(cast(Mapping[object, object], timing), totals)
        elif direct_timing:
            _accumulate_numeric_timing(cast(Mapping[object, object], candidate_map), totals)


def _accumulate_numeric_timing(
    timing: Mapping[object, object],
    totals: dict[str, float],
) -> None:
    for raw_key, raw_value in timing.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            continue
        if not math.isfinite(raw_value):
            continue
        key = str(raw_key)
        totals[key] = totals.get(key, 0.0) + float(raw_value)


def _coerce_example_id(raw: object) -> int | str:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int | str):
        return raw
    return str(raw)


def _rollout_temperatures(
    *,
    temperature: float,
    temperature_spread: float,
    num_rollouts: int,
) -> list[float]:
    """Return one sampling temperature per rollout."""
    if temperature_spread <= 0.0 or num_rollouts <= 1:
        return [temperature] * num_rollouts
    return [
        max(
            0.1,
            temperature + temperature_spread * (2 * i / (num_rollouts - 1) - 1),
        )
        for i in range(num_rollouts)
    ]


def _sample_active_rollouts(
    *,
    helper: "TrainHelper",
    active: list[tuple[int, PromptLike, list[int], list[int] | None]],
    rollout_temps: list[float],
    max_tokens: int,
    top_p: float,
) -> list[list[tuple[list[int], list[float]]]]:
    """Sample active rollout prompts, batching only equal-temperature prompts."""
    sampled_groups: list[list[tuple[list[int], list[float]]]] = [
        [] for _ in active
    ]
    batch_positions: list[int] = []
    batch_prompt_ids: list[list[int]] = []
    batch_temperature: float | None = None

    def flush_batch() -> None:
        nonlocal batch_positions, batch_prompt_ids, batch_temperature
        if batch_temperature is None:
            return
        groups = helper.sample(
            batch_prompt_ids,
            1,
            max_tokens,
            batch_temperature,
            top_p,
        )
        for position, group in zip(batch_positions, groups):
            sampled_groups[position] = group
        batch_positions = []
        batch_prompt_ids = []
        batch_temperature = None

    for position, (
        rollout_idx,
        _messages,
        prompt_ids,
        _observation_mask,
    ) in enumerate(active):
        rollout_temperature = rollout_temps[rollout_idx]
        if batch_temperature is None:
            batch_temperature = rollout_temperature
        elif rollout_temperature != batch_temperature:
            flush_batch()
            batch_temperature = rollout_temperature
        batch_positions.append(position)
        batch_prompt_ids.append(prompt_ids)
    flush_batch()

    return sampled_groups


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
        install_utils = importlib.import_module("verifiers.utils.install_utils")
        environments_hub_url = str(getattr(install_utils, "ENVIRONMENTS_HUB_URL"))
    except Exception:
        return []

    try:
        response = requests.get(
            environments_hub_url,
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
        check_hub_env_installed_fn: Callable[[str], bool] | None
        install_from_hub_fn: Callable[[str], bool] | None
        is_hub_env_fn: Callable[[str], bool] | None
        try:
            install_utils = importlib.import_module("verifiers.utils.install_utils")
            check_hub_env_installed_obj = getattr(
                install_utils,
                "check_hub_env_installed",
                None,
            )
            install_from_hub_obj = getattr(install_utils, "install_from_hub", None)
            is_hub_env_obj = getattr(install_utils, "is_hub_env", None)
            check_hub_env_installed_fn = (
                cast(Callable[[str], bool], check_hub_env_installed_obj)
                if callable(check_hub_env_installed_obj)
                else None
            )
            install_from_hub_fn = (
                cast(Callable[[str], bool], install_from_hub_obj)
                if callable(install_from_hub_obj)
                else None
            )
            is_hub_env_fn = (
                cast(Callable[[str], bool], is_hub_env_obj)
                if callable(is_hub_env_obj)
                else None
            )
        except Exception:
            # Keep loading path robust even if helper APIs change in verifiers.
            check_hub_env_installed_fn = None
            install_from_hub_fn = None
            is_hub_env_fn = None

        if (
            check_hub_env_installed_fn is not None
            and install_from_hub_fn is not None
            and is_hub_env_fn is not None
            and is_hub_env_fn(env_id)
            and not check_hub_env_installed_fn(env_id)
        ):
            ok = install_from_hub_fn(env_id)
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
    return prompt_utils.preview(prompt, max_chars=max_chars)


def encode_prompt_for_sampling(tokenizer: object, prompt: PromptLike) -> list[int]:
    return prompt_utils.encode_for_sampling(tokenizer, prompt)


def observation_mask_for_prompt(
    tokenizer: object,
    prompt: PromptLike,
    prompt_ids: list[int],
) -> list[int] | None:
    return prompt_utils.observation_mask(tokenizer, prompt, prompt_ids)


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
    observation_mask: list[int] | None = None


@dataclass
class VerifiersRolloutTiming:
    """Stage timings for a verifiers multi-turn rollout group."""

    init_state_s: float = 0.0
    prompt_render_s: float = 0.0
    prompt_encode_s: float = 0.0
    generation_s: float = 0.0
    decode_s: float = 0.0
    trajectory_step_s: float = 0.0
    render_completion_s: float = 0.0
    score_s: float = 0.0
    branch_s: float = 0.0
    scheduler_wait_s: float = 0.0
    scheduler_worker_s: float = 0.0
    scheduler_buffer_wait_s: float = 0.0
    total_s: float = 0.0
    turns: int = 0
    model_tokens: int = 0
    env_workers: int = 1
    buffer_size: int = 1
    env_timing_s: dict[str, float] | None = None

    def as_metrics(self, prefix: str = "rollout/") -> dict[str, float]:
        metrics = {
            f"{prefix}init_state_s": self.init_state_s,
            f"{prefix}prompt_render_s": self.prompt_render_s,
            f"{prefix}prompt_encode_s": self.prompt_encode_s,
            f"{prefix}generation_s": self.generation_s,
            f"{prefix}decode_s": self.decode_s,
            f"{prefix}trajectory_step_s": self.trajectory_step_s,
            f"{prefix}render_completion_s": self.render_completion_s,
            f"{prefix}score_s": self.score_s,
            f"{prefix}branch_s": self.branch_s,
            f"{prefix}scheduler_wait_s": self.scheduler_wait_s,
            f"{prefix}scheduler_worker_s": self.scheduler_worker_s,
            f"{prefix}scheduler_buffer_wait_s": self.scheduler_buffer_wait_s,
            f"{prefix}total_s": self.total_s,
            f"{prefix}turns": float(self.turns),
            f"{prefix}model_tokens": float(self.model_tokens),
            f"{prefix}env_workers": float(self.env_workers),
            f"{prefix}buffer_size": float(self.buffer_size),
        }
        if self.env_timing_s:
            for key, value in self.env_timing_s.items():
                metrics[f"{prefix}env/{key}"] = value
        return metrics


class RolloutScheduler:
    """Bound async environment work for one multi-turn rollout group."""

    def __init__(
        self,
        *,
        max_env_workers: int,
        max_buffered_rollouts: int,
    ) -> None:
        self.max_env_workers = max(1, int(max_env_workers))
        self.max_buffered_rollouts = max(1, int(max_buffered_rollouts))

    async def map_ordered(
        self,
        items: Sequence[object],
        worker: Callable[[object], Awaitable[object]],
        timing: VerifiersRolloutTiming,
    ) -> list[object]:
        """Run async item work with bounded concurrency and stable ordering."""
        if not items:
            return []

        if self.max_env_workers <= 1:
            results = []
            for item in items:
                started = time.perf_counter()
                result = await worker(item)
                timing.scheduler_worker_s += time.perf_counter() - started
                results.append(result)
            return results

        env_sem = asyncio.Semaphore(self.max_env_workers)
        buffer_sem = asyncio.Semaphore(min(self.max_buffered_rollouts, len(items)))

        async def run_one(pos: int, item: object) -> tuple[int, object]:
            buffer_wait_started = time.perf_counter()
            await buffer_sem.acquire()
            timing.scheduler_buffer_wait_s += (
                time.perf_counter() - buffer_wait_started
            )
            worker_wait_started = time.perf_counter()
            await env_sem.acquire()
            timing.scheduler_wait_s += time.perf_counter() - worker_wait_started
            try:
                started = time.perf_counter()
                result = await worker(item)
                timing.scheduler_worker_s += time.perf_counter() - started
                return pos, result
            finally:
                env_sem.release()
                buffer_sem.release()

        slots: list[object | None] = [None] * len(items)
        for pos, result in await asyncio.gather(
            *(run_one(pos, item) for pos, item in enumerate(items))
        ):
            slots[pos] = result
        return [cast(object, result) for result in slots]


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
    fork_execute: ForkExecute,
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
    alt_snapshot = fork_execute(ops)
    run = alt_snapshot.get("run")
    run_map: Mapping[str, object] = (
        cast(Mapping[str, object], run) if isinstance(run, Mapping) else {}
    )
    alt_cum = _coerce_reward(run_map.get("cumulative_reward", 0.0))
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
    fork_execute: ForkExecute,
    ops_before: list[object],
) -> list[dict[str, object]]:
    """Get the kernel's legal actions at the state before a turn.

    Falls back to the hardcoded vending action list if the kernel response
    doesn't include legal_actions.
    """
    try:
        snapshot = fork_execute(ops_before)
        # legal_actions can be at top level or nested in model_view
        model_view = snapshot.get("model_view")
        model_view_map: Mapping[str, object] = (
            cast(Mapping[str, object], model_view)
            if isinstance(model_view, Mapping)
            else {}
        )
        legal = snapshot.get("legal_actions")
        if not legal:
            legal = model_view_map.get("legal_actions")
        if legal:
            legal_actions = (
                list(legal)
                if isinstance(legal, Sequence) and not isinstance(legal, (str, bytes))
                else []
            )
            if legal_actions:
                actions: list[dict[str, object]] = [
                    {"kind": "act", "action": a} for a in legal_actions
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
    fork_execute = _resolve_fork_execute(env_obj)
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
            # Cap at branch_size alternatives to control cost.
            import random as _rng
            primary_op = cast(dict[str, object], entry["operation"])
            legal_actions = _get_legal_actions_at_turn(fork_execute, ops_before)
            alt_actions = [a for a in legal_actions if a != primary_op]
            if len(alt_actions) > branch_size - 1:
                alt_actions = _rng.sample(alt_actions, branch_size - 1)
            for alt_op in alt_actions:
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
    temperature_spread: float = 0.0,
    rollout_env_workers: int = 1,
    rollout_buffer_size: int = 0,
) -> tuple[
    list[float],
    list[list[VerifiersTurnSample]],
    list[str],
    list[list[float]],
    list[list[float]],
    list[list[dict[str, object]]],
    list[list[list[float]]],
    VerifiersRolloutTiming,
]:
    """Run group rollouts for verifiers MultiTurnEnv using retrain sampling.

    Args:
        temperature_spread: When > 0, each rollout in the group uses a
            different temperature: ``temperature + linspace(-spread, +spread, num_rollouts)``.
            This ensures diverse actions even from deterministic models.
            Example: temperature=1.0, spread=0.3 → temps [0.7, 0.8, ..., 1.3]

    Returns:
        (rewards, per_rollout_turns, completions_text, turn_rewards,
        turn_advantages, turn_logs, branch_rewards, timing)
        turn_rewards: per-turn reward deltas for each rollout (from env state)
        turn_advantages: MT-GRPO per-turn advantages for each rollout (from env rubric)
        turn_logs: per-turn action log for each rollout (observation, action, result)
        branch_rewards: raw per-turn branch reward vectors (TL-GRPO only, else empty)
    """

    async def _run() -> tuple[
        list[float],
        list[list[VerifiersTurnSample]],
        list[str],
        list[list[float]],
        list[list[float]],
        list[list[dict[str, object]]],
        list[list[list[float]]],
        VerifiersRolloutTiming,
    ]:
        vf = _require_verifiers()
        env_typed = cast(_MultiTurnEnvironment, env)
        tokenizer_typed = cast(_Tokenizer, tokenizer)
        states: list[StateDict] = []
        per_rollout_turns: list[list[VerifiersTurnSample]] = [[] for _ in range(num_rollouts)]
        rollout_timing = VerifiersRolloutTiming(env_timing_s={})
        scheduler = RolloutScheduler(
            max_env_workers=rollout_env_workers,
            max_buffered_rollouts=rollout_buffer_size or max(1, num_rollouts),
        )
        rollout_timing.env_workers = scheduler.max_env_workers
        rollout_timing.buffer_size = scheduler.max_buffered_rollouts
        rollout_started = time.perf_counter()

        async def cleanup_states() -> None:
            cleanup_error: Exception | None = None
            active_exception = sys.exc_info()[0] is not None
            cleanup_openenv_state = getattr(env_typed, "_cleanup_openenv_state", None)
            for state in states:
                try:
                    if callable(cleanup_openenv_state):
                        await cleanup_openenv_state(state)
                    else:
                        await env_typed.cleanup(state)
                except Exception as exc:  # Preserve rollout failures.
                    cleanup_error = cleanup_error or exc
            if cleanup_error is not None and not active_exception:
                raise cleanup_error

        rollout_temps = _rollout_temperatures(
            temperature=temperature,
            temperature_spread=temperature_spread,
            num_rollouts=num_rollouts,
        )

        try:
            async def init_one(raw_idx: object) -> StateDict:
                i = int(cast(int, raw_idx))
                input_payload: dict[str, object] = {
                    "prompt": prompt,
                    "answer": answer,
                    "task": task,
                    "example_id": i,
                }
                if info is not None:
                    input_payload["info"] = info
                init_started = time.perf_counter()
                state = await env_typed.init_state(
                    input=input_payload,
                    client=_make_env_client(),
                    model=model_name,
                    sampling_args=None,
                )
                state = await env_typed.setup_state(state)
                rollout_timing.init_state_s += time.perf_counter() - init_started
                return state

            states = [
                cast(StateDict, state)
                for state in await scheduler.map_ordered(
                    list(range(num_rollouts)),
                    init_one,
                    rollout_timing,
                )
            ]

            turn_count = 0
            while True:
                active: list[tuple[int, PromptLike, list[int], list[int] | None]] = []
                indexed_states = list(enumerate(states))

                async def render_active(
                    raw_item: object,
                ) -> tuple[int, PromptLike, list[int], list[int] | None] | None:
                    idx, state = cast(tuple[int, StateDict], raw_item)
                    if await env_typed.is_completed(state):
                        return None
                    render_started = time.perf_counter()
                    prompt_messages = await env_typed.get_prompt_messages(state)
                    rollout_timing.prompt_render_s += (
                        time.perf_counter() - render_started
                    )
                    # OpenEnvEnv applies the prior action while rendering the next
                    # prompt, so collect observation timings before appending a step.
                    _collect_observation_timing(state, rollout_timing.env_timing_s)
                    if state.get("final_env_response") is not None:
                        return None
                    encode_started = time.perf_counter()
                    prompt_ids = encode_prompt_for_sampling(tokenizer, prompt_messages)
                    observation_mask = observation_mask_for_prompt(
                        tokenizer,
                        prompt_messages,
                        prompt_ids,
                    )
                    rollout_timing.prompt_encode_s += (
                        time.perf_counter() - encode_started
                    )
                    return (idx, prompt_messages, prompt_ids, observation_mask)

                active = [
                    cast(tuple[int, PromptLike, list[int], list[int] | None], item)
                    for item in await scheduler.map_ordered(
                        indexed_states,
                        render_active,
                        rollout_timing,
                    )
                    if item is not None
                ]

                if not active:
                    break

                if max_turns_override > 0 and turn_count >= max_turns_override:
                    for idx, _messages, _prompt_ids, _observation_mask in active:
                        states[idx]["is_completed"] = True
                        states[idx]["is_truncated"] = True
                        states[idx]["stop_condition"] = "retrain_max_turns"
                    break

                generation_started = time.perf_counter()
                sampled_groups = _sample_active_rollouts(
                    helper=helper,
                    active=active,
                    rollout_temps=rollout_temps,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                rollout_timing.generation_s += (
                    time.perf_counter() - generation_started
                )
                # Handle empty groups: if the sampler returns no completions
                # (e.g. prompt too long for max_tokens), use a fallback empty
                # completion so the turn loop can continue gracefully.
                completion_ids_batch = [
                    list(group[0][0]) if group else []
                    for group in sampled_groups
                ]
                completion_logprobs_batch = [
                    [float(lp) for lp in group[0][1]] if group else []
                    for group in sampled_groups
                ]
                decode_started = time.perf_counter()
                completion_texts = tokenizer_typed.batch_decode(
                    completion_ids_batch, skip_special_tokens=True
                )
                rollout_timing.decode_s += time.perf_counter() - decode_started

                step_jobs = list(enumerate(active))

                async def add_step(
                    raw_job: object,
                ) -> tuple[int, VerifiersTurnSample, int]:
                    pos, active_item = cast(
                        tuple[
                            int,
                            tuple[int, PromptLike, list[int], list[int] | None],
                        ],
                        raw_job,
                    )
                    idx, prompt_messages, prompt_ids, observation_mask = active_item
                    completion_ids = completion_ids_batch[pos]
                    completion_logprobs = completion_logprobs_batch[pos]
                    completion_text = completion_texts[pos]
                    completion_messages = _completion_messages_for_env(
                        env_typed, completion_text
                    )

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
                    step_started = time.perf_counter()
                    await env_typed.add_trajectory_step(states[idx], trajectory_step)
                    rollout_timing.trajectory_step_s += (
                        time.perf_counter() - step_started
                    )
                    _collect_observation_timing(states[idx], rollout_timing.env_timing_s)
                    turn_sample = VerifiersTurnSample(
                        prompt_ids=list(prompt_ids),
                        completion_ids=list(completion_ids),
                        completion_logprobs=list(completion_logprobs),
                        completion_text=completion_text,
                        observation_mask=(
                            list(observation_mask)
                            if observation_mask is not None
                            else None
                        ),
                    )
                    return idx, turn_sample, len(completion_ids)

                for idx, turn_sample, token_count in (
                    cast(tuple[int, VerifiersTurnSample, int], item)
                    for item in await scheduler.map_ordered(
                        step_jobs,
                        add_step,
                        rollout_timing,
                    )
                ):
                    per_rollout_turns[idx].append(turn_sample)
                    rollout_timing.model_tokens += token_count

                turn_count += 1
                rollout_timing.turns += len(active)

            async def render_completion_one(raw_state: object) -> None:
                state = cast(StateDict, raw_state)
                render_started = time.perf_counter()
                await env_typed.render_completion(state)
                rollout_timing.render_completion_s += (
                    time.perf_counter() - render_started
                )

            await scheduler.map_ordered(states, render_completion_one, rollout_timing)
            score_started = time.perf_counter()
            await env_typed.rubric.score_group(states)
            rollout_timing.score_s += time.perf_counter() - score_started

            # TL-GRPO: branch from each turn to get epistemically sound
            # per-turn advantages (alternatives compared against same state).
            all_branch_rewards: list[list[list[float]]] = []
            if tl_grpo:
                branch_started = time.perf_counter()
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
                rollout_timing.branch_s += time.perf_counter() - branch_started

            rewards = [_coerce_reward(s.get("reward")) for s in states]
            completions_text = [_messages_to_text(s.get("completion")) for s in states]
            turn_rewards = [_coerce_float_list(s.get("turn_rewards")) for s in states]
            turn_advantages = [
                _coerce_float_list(s.get("turn_advantages")) for s in states
            ]
            turn_logs = [
                cast(list[dict[str, object]], s.get("turn_log") or [])
                for s in states
            ]
            rollout_timing.total_s = time.perf_counter() - rollout_started
            return (
                rewards,
                per_rollout_turns,
                completions_text,
                turn_rewards,
                turn_advantages,
                turn_logs,
                all_branch_rewards,
                rollout_timing,
            )
        finally:
            await cleanup_states()

    return asyncio.run(_run())
