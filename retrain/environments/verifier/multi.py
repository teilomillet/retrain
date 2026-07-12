"""Multi-turn rollout bridge for verifiers-style environments."""

from __future__ import annotations

import asyncio
import sys
import time
import types
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

from retrain.environments import branch as tl_branch
from retrain.environments import load as env_load
from retrain.environments import prompt as prompt_utils
from retrain.environments.echo_tokens import (
    EchoTokenBridgeError,
    EchoTokenRenderer,
    bridge_observation_tokens,
    create_echo_token_renderer,
)
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
    capture_echo_transitions: bool = False,
    echo_token_renderer: EchoTokenRenderer | None = None,
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
        raw_rollout_contract = getattr(env_typed, "retrain_rollout_contract", None)
        rollout_contract = (
            raw_rollout_contract
            if isinstance(raw_rollout_contract, Mapping)
            else {}
        )
        context_window = rollout_contract.get("context_window")
        completion_reserve = rollout_contract.get("completion_reserve_tokens")
        history_turns = rollout_contract.get("history_turns")
        if rollout_contract:
            if history_turns != 0:
                raise ValueError(
                    "retrain_rollout_contract.history_turns must be 0 (full history)"
                )
            if (
                not isinstance(context_window, int)
                or isinstance(context_window, bool)
                or not isinstance(completion_reserve, int)
                or isinstance(completion_reserve, bool)
                or context_window <= completion_reserve
            ):
                raise ValueError(
                    "retrain_rollout_contract requires a context_window larger "
                    "than completion_reserve_tokens"
                )
            if max_tokens != completion_reserve:
                raise ValueError(
                    f"training max_tokens={max_tokens} does not match the live "
                    f"completion reserve {completion_reserve}"
                )
        pop_echo_observation_messages = getattr(
            env_typed,
            "pop_echo_observation_messages",
            None,
        )
        echo_capture_supported = capture_echo_transitions and callable(
            pop_echo_observation_messages
        )
        active_echo_renderer = echo_token_renderer
        echo_renderer_parity_failed = False
        if echo_capture_supported and active_echo_renderer is None:
            try:
                active_echo_renderer = create_echo_token_renderer(tokenizer)
            except (AssertionError, RuntimeError, TypeError, ValueError):
                # Rollouts and GRPO remain valid; explicit bridge-failure metrics
                # prevent this from being mistaken for a working ECHO update.
                active_echo_renderer = None
        states: list[StateDict] = []
        initialized_states: list[StateDict] = []
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
            for state in initialized_states:
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
                # Keep cleanup ownership even if a later sibling fails setup
                # before map_ordered can return the complete state list.
                initialized_states.append(state)
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
                    nonlocal echo_renderer_parity_failed
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
                    if (
                        isinstance(context_window, int)
                        and isinstance(completion_reserve, int)
                        and len(prompt_ids) + completion_reserve > context_window
                    ):
                        raise RuntimeError(
                            f"Full-history prompt for rollout {idx} has "
                            f"{len(prompt_ids)} tokens; with the reserved "
                            f"{completion_reserve}-token action it exceeds the "
                            f"{context_window}-token context window. No action "
                            "was sampled, sent to the environment, or optimizer."
                        )
                    observation_mask = prompt_utils.observation_mask(
                        tokenizer,
                        prompt_messages,
                        prompt_ids,
                    )
                    if (
                        echo_capture_supported
                        and active_echo_renderer is not None
                        and not echo_renderer_parity_failed
                        and isinstance(prompt_messages, list)
                    ):
                        try:
                            renderer_prompt_ids = active_echo_renderer.render_ids(
                                [dict(message) for message in prompt_messages],
                                add_generation_prompt=True,
                            )
                        except (AssertionError, RuntimeError, TypeError, ValueError):
                            echo_renderer_parity_failed = True
                        else:
                            if list(renderer_prompt_ids) != prompt_ids:
                                echo_renderer_parity_failed = True
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
                rollout_timing.generation_s += time.perf_counter() - generation_started
                malformed_groups = [
                    active[pos][0]
                    for pos, group in enumerate(sampled_groups)
                    if len(group) != 1
                ]
                if malformed_groups:
                    raise RuntimeError(
                        "Sampler must return exactly one action for every active "
                        f"rollout; malformed rollout(s): {malformed_groups}. No "
                        "action from this sampled batch was sent to the environment "
                        "or optimizer."
                    )
                completion_ids_batch = [
                    list(group[0].token_ids) for group in sampled_groups
                ]
                completion_logprobs_batch = [
                    list(group[0].logprobs)
                    for group in sampled_groups
                ]
                finish_reasons = [
                    group[0].finish_reason for group in sampled_groups
                ]
                token_limit_hits = [
                    group[0].hit_token_limit
                    for group in sampled_groups
                ]
                if any(token_limit_hits):
                    rejected_rollouts = [
                        active[pos][0]
                        for pos, hit_token_limit in enumerate(token_limit_hits)
                        if hit_token_limit
                    ]
                    raise RuntimeError(
                        "Refusing incomplete sampled action(s) for rollout(s) "
                        f"{rejected_rollouts}: generation did not finish with an "
                        "explicit normal stop. No action from this sampled batch "
                        "was sent to the environment or optimizer."
                    )
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
                    finish_reason = finish_reasons[pos]
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
                    observation_messages: list[dict[str, object]] = []
                    post_observation_ids: list[int] | None = None
                    post_observation_mask: list[int] | None = None
                    post_observation_bridge_failed = False
                    if echo_capture_supported:
                        assert callable(pop_echo_observation_messages)
                        raw_messages = pop_echo_observation_messages(states[idx])
                        if not isinstance(raw_messages, list):
                            raise TypeError(
                                "ECHO observation capture must return a messages list."
                            )
                        for raw_message in raw_messages:
                            if not isinstance(raw_message, dict):
                                raise TypeError(
                                    "ECHO observation messages must be mappings."
                                )
                            observation_messages.append(dict(raw_message))
                        if observation_messages:
                            if (
                                active_echo_renderer is None
                                or echo_renderer_parity_failed
                            ):
                                post_observation_bridge_failed = True
                            else:
                                try:
                                    transition = bridge_observation_tokens(
                                        active_echo_renderer,
                                        prompt_ids=list(prompt_ids),
                                        completion_ids=list(completion_ids),
                                        observation_messages=observation_messages,
                                    )
                                except EchoTokenBridgeError:
                                    post_observation_bridge_failed = True
                                else:
                                    post_observation_ids = transition.token_ids
                                    post_observation_mask = transition.observation_mask
                    turn_sample = VerifiersTurnSample(
                        prompt_ids=list(prompt_ids),
                        completion_ids=list(completion_ids),
                        completion_logprobs=list(completion_logprobs),
                        completion_text=completion_text,
                        finish_reason=finish_reason,
                        is_truncated=False,
                        observation_mask=(
                            list(observation_mask)
                            if observation_mask is not None
                            else None
                        ),
                        echo_observation_capture_supported=echo_capture_supported,
                        post_observation_ids=post_observation_ids,
                        post_observation_mask=post_observation_mask,
                        post_observation_seen=bool(observation_messages),
                        post_observation_bridge_failed=(post_observation_bridge_failed),
                        echo_renderer_parity_failed=echo_renderer_parity_failed,
                        post_observation_terminal=bool(states[idx].get("openenv_done")),
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
                cast(list[dict[str, object]], s.get("turn_log") or []) for s in states
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
