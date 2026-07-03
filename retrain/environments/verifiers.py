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
import math
import sys
from collections.abc import Mapping
import time
import types
from typing import TYPE_CHECKING, Protocol, cast

from retrain.environments import load as env_load
from retrain.environments import branch as tl_branch
from retrain.environments import prompt as prompt_utils
from retrain.environments.rollout import (
    RolloutScheduler,
    VerifiersRolloutTiming,
    VerifiersTurnSample,
    rollout_temperatures,
    sample_active_rollouts,
)
from retrain.types import ExampleInfoLike, JSONObject, PromptLike

if TYPE_CHECKING:
    from retrain.backends import TrainHelper
    from retrain.config import TrainConfig
    from retrain.data.source import Example


_FALLBACK_TRAINING_ENVS = env_load.FALLBACK_TRAINING_ENVS

StateDict = dict[str, object]


class _Rubric(Protocol):
    async def score_group(self, states: list[StateDict]) -> object: ...


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
    return env_load.require_verifiers()


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


def parse_environment_args(raw: str | JSONObject | None) -> JSONObject:
    return env_load.parse_args(raw)


def _hub_env_suggestions(env_id: str, limit: int = 5) -> list[str]:
    return env_load.hub_suggestions(env_id, limit=limit)


def _format_hub_suggestions(env_id: str) -> str:
    return env_load.format_hub_suggestions(env_id)


def load_verifiers_environment(config: "TrainConfig") -> object:
    return env_load.load_environment(config, require_fn=_require_verifiers)


def load_examples_from_environment(env: object, config: "TrainConfig") -> list[Example]:
    return env_load.examples_from_environment(env, config)


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


def is_multiturn_environment(env: object) -> bool:
    """Whether env is a verifiers MultiTurnEnv."""
    vf = _require_verifiers()
    return isinstance(env, vf.MultiTurnEnv)


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

        rollout_temps = rollout_temperatures(
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
                sampled_groups = sample_active_rollouts(
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
                    br = tl_branch.run(
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
                tl_branch.compute_advantages(
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
