"""A retrain-native multi-turn environment over an OpenEnv gym server.

Implements the same structural protocol the verifiers bridge drives
(``init_state`` / ``setup_state`` / ``is_completed`` / ``get_prompt_messages``
/ ``add_trajectory_step`` / ``render_completion`` / ``cleanup`` plus an async
``rubric.score_group``), so ``run_multiturn_group`` runs OpenEnv rollouts
with the exact machinery used for verifiers environments — scheduling,
timing, ECHO observation masks, and turn accounting included.

Scope: the OpenEnv *gym* contract (reset/step) against an already-running
server. MCP tool environments need tool-call message plumbing that retrain's
text-completion bridge does not model; use the verifiers integration for
those.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from retrain.environments.openenv.actions import ActionParseError, parse_action
from retrain.environments.openenv.client import OpenEnvClient
from retrain.environments.openenv.render import (
    Renderer,
    default_renderer,
    render_messages,
)
from retrain.types import JSONObject, PromptMessage

StateDict = dict[str, object]
_MISSING = object()

_CORRECTIVE_MESSAGE = (
    "Your last message was not a valid action. Respond with exactly one "
    "JSON object matching the action schema — no prose, no code fences, "
    "no trailing text."
)


class EpisodeRewardRubric:
    """Scores each rollout as the sum of its per-step env rewards.

    Matches the episodic-sum scoring of verifiers' OpenEnv integration, so
    reward semantics do not change when switching providers.
    """

    async def score_group(self, states: list[StateDict]) -> None:
        for state in states:
            turn_rewards = cast(list[float], state.get("turn_rewards") or [])
            state["reward"] = float(sum(turn_rewards))


class OpenEnvEnvironment:
    """Drive one OpenEnv gym server through retrain's multi-turn protocol."""

    # Structural marker read by is_multiturn_environment(); lets the trainer
    # route to multi-turn rollouts without importing verifiers.
    retrain_multiturn = True
    message_type = "chat"

    def __init__(
        self,
        base_url: str,
        *,
        action_schema: Mapping[str, object],
        renderer: Renderer | None = None,
        message_timeout_s: float = 300.0,
        client_factory: Callable[[str], OpenEnvClient] | None = None,
    ) -> None:
        self.base_url = base_url
        self.action_schema = action_schema
        self.renderer: Renderer = renderer or default_renderer
        self.client_factory = client_factory or (
            lambda url: OpenEnvClient(url, message_timeout_s=message_timeout_s)
        )
        self.rubric = EpisodeRewardRubric()

    # -- lifecycle ---------------------------------------------------------

    async def init_state(
        self,
        *,
        input: dict[str, object],
        client: object = None,
        model: str = "",
        sampling_args: object = None,
    ) -> StateDict:
        """Build the rollout state dict. Pure — no I/O until setup_state.

        ``client``/``model``/``sampling_args`` exist for protocol parity;
        retrain samples through TrainHelper, never through an API client.
        """
        del client, model, sampling_args
        info = input.get("info")
        seed = 0
        if isinstance(info, dict):
            raw_seed = cast(dict[str, object], info).get("seed", 0)
            if isinstance(raw_seed, int | str):
                seed = int(raw_seed)
        return {
            "input": input,
            "seed": seed,
            "trajectory_id": input.get("example_id", 0),
            "messages": [],
            "prompt_message_count": 0,
            "trajectory": [],
            "turn_rewards": [],
            "turn_log": [],
            "openenv_done": False,
            "completion": [],
        }

    async def setup_state(self, state: StateDict) -> StateDict:
        """Connect and reset the episode; render the first prompt."""
        env_client = self.client_factory(self.base_url)
        state["openenv_client"] = env_client
        try:
            await env_client.connect()
            result = await env_client.reset(seed=cast(int, state["seed"]))
        except Exception:
            await self.cleanup(state)
            raise
        state["openenv_done"] = bool(result.done)
        messages = render_messages(
            self.renderer,
            result.observation,
            context="reset",
            action_schema=self.action_schema,
            seed=cast(int, state["seed"]),
        )
        state["messages"] = messages
        state["prompt_message_count"] = len(messages)
        return state

    async def is_completed(self, state: StateDict) -> bool:
        # "is_completed" may be forced by the rollout engine (max-turns).
        return bool(state.get("is_completed")) or bool(state.get("openenv_done"))

    async def get_prompt_messages(self, state: StateDict) -> list[PromptMessage]:
        return cast(list[PromptMessage], state["messages"])

    # -- stepping ----------------------------------------------------------

    async def add_trajectory_step(self, state: StateDict, step: object) -> None:
        """Apply one model turn: parse the action, step the env, observe.

        A malformed completion gets a corrective observation instead of
        crashing the rollout group: model formatting errors are training
        signal, not infrastructure failures. Env/transport errors still
        propagate. (verifiers' integration raises on malformed actions;
        deviating here keeps long training runs alive through bad samples.)
        """
        completion_text = _completion_text(step)
        messages = cast(list[PromptMessage], state["messages"])
        messages.append({"role": "assistant", "content": completion_text})

        turn_rewards = cast(list[float], state["turn_rewards"])
        turn_log = cast(list[dict[str, object]], state["turn_log"])
        trajectory = cast(list[object], state["trajectory"])
        trajectory.append(step)

        try:
            action = parse_action(completion_text, self.action_schema)
        except ActionParseError:
            messages.append({"role": "user", "content": _CORRECTIVE_MESSAGE})
            turn_rewards.append(0.0)
            turn_log.append({"malformed": True, "reward": 0.0, "done": False})
            return

        env_client = cast(OpenEnvClient, state["openenv_client"])
        result = await env_client.step(action)
        reward = float(result.reward) if result.reward is not None else 0.0
        turn_rewards.append(reward)
        turn_log.append({"action": action, "reward": reward, "done": result.done})
        state["openenv_done"] = bool(result.done)
        messages.extend(
            render_messages(
                self.renderer,
                result.observation,
                context="step",
                action_schema=self.action_schema,
                seed=None,
            )
        )

    # -- teardown ----------------------------------------------------------

    async def render_completion(self, state: StateDict) -> None:
        messages = cast(list[PromptMessage], state["messages"])
        prompt_count = int(cast(int, state.get("prompt_message_count") or 0))
        state["completion"] = messages[prompt_count:]

    async def cleanup(self, state: StateDict) -> None:
        env_client = state.pop("openenv_client", None)
        if env_client is not None:
            await cast(OpenEnvClient, env_client).close()


def _completion_text(step: object) -> str:
    """Extract assistant text without stringifying trajectory bookkeeping."""
    completion = (
        cast(Mapping[str, object], step).get("completion", _MISSING)
        if isinstance(step, Mapping)
        else getattr(step, "completion", _MISSING)
    )
    if completion is _MISSING:
        raise ValueError("Trajectory step is missing required 'completion' data.")
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        raise TypeError(
            "Trajectory step 'completion' must be text or a list of messages, "
            f"got {type(completion).__name__}."
        )

    chunks: list[str] = []
    for message in cast(list[object], completion):
        content = (
            cast(Mapping[str, object], message).get("content")
            if isinstance(message, Mapping)
            else getattr(message, "content", None)
        )
        if content is None:
            continue
        if not isinstance(content, str):
            raise TypeError(
                "Trajectory completion message content must be text, "
                f"got {type(content).__name__}."
            )
        if content:
            chunks.append(content)
    if not chunks:
        raise ValueError("Trajectory step completion contains no textual content.")
    return "\n".join(chunks)
