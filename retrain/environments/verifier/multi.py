"""Multi-turn rollout bridge for verifiers-style environments."""

from __future__ import annotations

import asyncio
import sys
import time
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from retrain.environments import branch as tl_branch
from retrain.environments import load as env_load
from retrain.environments import prompt as prompt_utils
from retrain.environments.rollout import (
    RolloutScheduler,
    VerifiersRolloutTiming,
    VerifiersTurnSample,
    rollout_temperatures,
    sample_active_rollouts,
)
from retrain.environments.timing import collect_observation_timing
from retrain.environments.verifier import client, coerce
from retrain.environments.verifier.score import completion_messages, messages_to_text
from retrain.environments.verifier.types import (
    MultiTurnEnvironment,
    StateDict,
    Tokenizer,
)
from retrain.types import ExampleInfoLike, PromptLike

if TYPE_CHECKING:
    from retrain.backends import TrainHelper


def is_multiturn_environment(
    env: object,
    *,
    require_fn: Callable[[], types.ModuleType] = env_load.require_verifiers,
) -> bool:
    """Whether env drives retrain's multi-turn rollout protocol."""
    if getattr(env, "retrain_multiturn", False):
        return True
    vf = require_fn()
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
    verifiers_loader: Callable[[], types.ModuleType | None] = client.optional,
    env_client_factory: Callable[[], object | None] = client.make,
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
    """Run group rollouts with retrain sampling and environment-side scoring."""

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
        vf = verifiers_loader()
        env_typed = cast(MultiTurnEnvironment, env)
        tokenizer_typed = cast(Tokenizer, tokenizer)
        states: list[StateDict] = []
        per_rollout_turns: list[list[VerifiersTurnSample]] = [
            [] for _ in range(num_rollouts)
        ]
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
                except Exception as exc:
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
                    client=env_client_factory(),
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
                    # OpenEnv applies the prior action while rendering the next prompt.
                    collect_observation_timing(state, rollout_timing.env_timing_s)
                    if state.get("final_env_response") is not None:
                        return None
                    encode_started = time.perf_counter()
                    prompt_ids = prompt_utils.encode_for_sampling(
                        tokenizer,
                        prompt_messages,
                    )
                    observation_mask = prompt_utils.observation_mask(
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
                # Empty sampler groups can happen when prompts leave no generation budget.
                completion_ids_batch = [
                    list(group[0][0]) if group else [] for group in sampled_groups
                ]
                completion_logprobs_batch = [
                    [float(lp) for lp in group[0][1]] if group else []
                    for group in sampled_groups
                ]
                decode_started = time.perf_counter()
                completion_texts = tokenizer_typed.batch_decode(
                    completion_ids_batch,
                    skip_special_tokens=True,
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
                    completion = completion_messages(env_typed, completion_text)

                    tokens_payload = {
                        "prompt_ids": list(prompt_ids),
                        "prompt_mask": [0] * len(prompt_ids),
                        "completion_ids": list(completion_ids),
                        "completion_mask": [1] * len(completion_ids),
                        "completion_logprobs": list(completion_logprobs),
                        "overlong_prompt": False,
                        "is_truncated": False,
                    }
                    trajectory_step_cls = (
                        getattr(vf, "TrajectoryStep", None) if vf is not None else None
                    )
                    if trajectory_step_cls is not None:
                        trajectory_step = trajectory_step_cls(
                            prompt=prompt_messages,
                            completion=completion,
                            response=None,
                            tokens=tokens_payload,
                            reward=None,
                            advantage=None,
                            is_truncated=False,
                            trajectory_id=states[idx]["trajectory_id"],
                            extras={},
                        )
                    else:
                        # Older verifiers versions do not expose TrajectoryStep.
                        trajectory_step = types.SimpleNamespace(
                            prompt=prompt_messages,
                            completion=completion,
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
                    collect_observation_timing(
                        states[idx],
                        rollout_timing.env_timing_s,
                    )
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

            all_branch_rewards: list[list[list[float]]] = []
            if tl_grpo:
                # Compare each action against alternatives from the same state.
                branch_started = time.perf_counter()
                for i, state in enumerate(states):
                    branch_rewards = tl_branch.run(
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
                    all_branch_rewards.append(branch_rewards)
                tl_branch.compute_advantages(
                    states,
                    all_branch_rewards,
                    outcome_baseline=tl_grpo_outcome_baseline,
                )
                rollout_timing.branch_s += time.perf_counter() - branch_started

            rewards = [coerce.reward(s.get("reward")) for s in states]
            completions_text = [messages_to_text(s.get("completion")) for s in states]
            turn_rewards = [coerce.float_list(s.get("turn_rewards")) for s in states]
            turn_advantages = [
                coerce.float_list(s.get("turn_advantages")) for s in states
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
