"""Episode-lifecycle tests for OpenEnvEnvironment (fake client, no network)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from retrain.environments.openenv.client import OpenEnvServerError, StepResult
from retrain.environments.openenv.environment import OpenEnvEnvironment
from retrain.environments.openenv.provenance import (
    OpenEnvProvenanceError,
    ResetProvenanceGuard,
)

_SCHEMA = {
    "properties": {"tool": {"type": "string"}},
    "required": ["tool"],
}


class _FakeClient:
    """Scripted env: rewards each step from a queue, done when empty."""

    def __init__(self, step_rewards: list[float]) -> None:
        self.step_rewards = list(step_rewards)
        self.reset_seeds: list[object] = []
        self.reset_calls: list[dict[str, object]] = []
        self.actions: list[dict] = []
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def reset(self, **kwargs: object) -> StepResult:
        self.reset_calls.append(dict(kwargs))
        self.reset_seeds.append(kwargs.get("seed"))
        return StepResult(observation={"board": "start"}, reward=None, done=False)

    async def step(self, action: dict) -> StepResult:
        self.actions.append(action)
        reward = self.step_rewards.pop(0)
        return StepResult(
            observation={"board": f"after-{len(self.actions)}"},
            reward=reward,
            done=not self.step_rewards,
        )

    async def close(self) -> None:
        self.closed = True


def _env(client: _FakeClient) -> OpenEnvEnvironment:
    return OpenEnvEnvironment(
        "http://localhost:8765",
        action_schema=_SCHEMA,
        client_factory=lambda _url: client,
    )


def _assistant_step(text: str) -> SimpleNamespace:
    return SimpleNamespace(completion=[{"role": "assistant", "content": text}])


async def _run_episode(env: OpenEnvEnvironment, completions: list[str]) -> dict:
    state = await env.init_state(input={"info": {"seed": 3}, "example_id": 5})
    state = await env.setup_state(state)
    for text in completions:
        if await env.is_completed(state):
            break
        await env.add_trajectory_step(state, _assistant_step(text))
    await env.render_completion(state)
    await env.rubric.score_group([state])
    await env.cleanup(state)
    return state


class TestEpisode:
    def test_full_episode_sums_step_rewards(self):
        client = _FakeClient(step_rewards=[0.0, 1.0])
        state = asyncio.run(
            _run_episode(_env(client), ['{"tool": "look"}', '{"tool": "solve"}'])
        )
        assert client.reset_seeds == [3]
        assert client.actions == [{"tool": "look"}, {"tool": "solve"}]
        assert state["reward"] == 1.0
        assert state["turn_rewards"] == [0.0, 1.0]
        assert state["openenv_done"] is True
        assert client.closed

    def test_trajectory_id_comes_from_example_id(self):
        client = _FakeClient(step_rewards=[1.0])
        state = asyncio.run(_run_episode(_env(client), ['{"tool": "go"}']))
        assert state["trajectory_id"] == 5

    def test_positive_max_turns_is_forwarded_to_live_reset(self):
        client = _FakeClient(step_rewards=[1.0])
        env = OpenEnvEnvironment(
            "http://localhost:8765",
            action_schema=_SCHEMA,
            reset_max_turns=16,
            client_factory=lambda _url: client,
        )

        asyncio.run(_run_episode(env, ['{"tool": "go"}']))

        assert client.reset_calls == [{"seed": 3, "max_turns": 16}]

    def test_prompt_messages_accumulate_transcript(self):
        client = _FakeClient(step_rewards=[0.5, 0.5])
        env = _env(client)

        async def run() -> tuple[list, list]:
            state = await env.init_state(input={"info": {"seed": 0}})
            state = await env.setup_state(state)
            first_prompt = list(await env.get_prompt_messages(state))
            await env.add_trajectory_step(state, _assistant_step('{"tool": "a"}'))
            second_prompt = list(await env.get_prompt_messages(state))
            await env.cleanup(state)
            return first_prompt, second_prompt

        first_prompt, second_prompt = asyncio.run(run())
        assert [m["role"] for m in first_prompt] == ["user"]
        assert [m["role"] for m in second_prompt] == ["user", "assistant", "user"]

    def test_mapping_trajectory_step_extracts_only_completion(self):
        client = _FakeClient(step_rewards=[1.0])
        env = _env(client)

        async def run() -> None:
            state = await env.init_state(input={})
            state = await env.setup_state(state)
            await env.add_trajectory_step(
                state,
                {
                    "prompt": [{"role": "user", "content": "large prompt"}],
                    "completion": [{"role": "assistant", "content": '{"tool": "go"}'}],
                    "tokens": {"prompt_ids": list(range(32))},
                },
            )
            await env.cleanup(state)

        asyncio.run(run())
        assert client.actions == [{"tool": "go"}]

    def test_object_message_completion_is_supported(self):
        client = _FakeClient(step_rewards=[1.0])
        env = _env(client)

        async def run() -> None:
            state = await env.init_state(input={})
            state = await env.setup_state(state)
            await env.add_trajectory_step(
                state,
                SimpleNamespace(completion=[SimpleNamespace(content='{"tool": "go"}')]),
            )
            await env.cleanup(state)

        asyncio.run(run())
        assert client.actions == [{"tool": "go"}]

    def test_missing_completion_fails_without_stringifying_step(self):
        client = _FakeClient(step_rewards=[1.0])
        env = _env(client)

        async def run() -> None:
            state = await env.init_state(input={})
            state = await env.setup_state(state)
            initial_messages = list(state["messages"])
            with pytest.raises(ValueError, match="missing required 'completion'"):
                await env.add_trajectory_step(
                    state,
                    {
                        "prompt": [{"role": "user", "content": "large prompt"}],
                        "tokens": {"prompt_ids": list(range(32))},
                    },
                )
            assert state["messages"] == initial_messages
            assert client.actions == []
            await env.cleanup(state)

        asyncio.run(run())

    def test_render_completion_excludes_initial_prompt(self):
        client = _FakeClient(step_rewards=[1.0])
        state = asyncio.run(_run_episode(_env(client), ['{"tool": "go"}']))
        completion = state["completion"]
        assert [m["role"] for m in completion] == ["assistant", "user"]

    def test_echo_observations_are_one_shot_and_include_terminal_output(self):
        client = _FakeClient(step_rewards=[0.0, 1.0])
        env = _env(client)

        async def run() -> tuple[list, list, list, bool]:
            state = await env.init_state(input={})
            state = await env.setup_state(state)
            await env.add_trajectory_step(state, _assistant_step('{"tool": "a"}'))
            first = env.pop_echo_observation_messages(state)
            repeated = env.pop_echo_observation_messages(state)
            await env.add_trajectory_step(state, _assistant_step('{"tool": "b"}'))
            terminal = env.pop_echo_observation_messages(state)
            done = bool(state["openenv_done"])
            await env.cleanup(state)
            return first, repeated, terminal, done

        first, repeated, terminal, done = asyncio.run(run())
        assert first and first[0]["role"] == "user"
        assert "after-1" in str(first[0]["content"])
        assert repeated == []
        assert terminal and "after-2" in str(terminal[0]["content"])
        assert done is True

    def test_external_is_completed_flag_stops_episode(self):
        # run_multiturn_group forces is_completed on max-turns cutoffs.
        client = _FakeClient(step_rewards=[0.0, 0.0])
        env = _env(client)

        async def run() -> bool:
            state = await env.init_state(input={})
            state = await env.setup_state(state)
            state["is_completed"] = True
            done = await env.is_completed(state)
            await env.cleanup(state)
            return done

        assert asyncio.run(run()) is True


class TestMalformedCompletions:
    def test_bare_text_wraps_via_single_string_field(self):
        client = _FakeClient(step_rewards=[1.0])
        env = _env(client)

        async def run() -> None:
            state = await env.init_state(input={})
            state = await env.setup_state(state)
            await env.add_trajectory_step(state, _assistant_step("look around"))
            await env.cleanup(state)

        asyncio.run(run())
        assert client.actions == [{"tool": "look around"}]

    def test_unwrappable_completion_is_corrected(self):
        client = _FakeClient(step_rewards=[1.0])
        env = OpenEnvEnvironment(
            "http://localhost:8765",
            action_schema={
                "properties": {
                    "tool": {"type": "string"},
                    "args": {"type": "object"},
                },
                "required": ["tool", "args"],
            },
            client_factory=lambda _url: client,
        )

        async def run() -> dict:
            state = await env.init_state(input={})
            state = await env.setup_state(state)
            await env.add_trajectory_step(state, _assistant_step("not an action"))
            await env.cleanup(state)
            return state

        state = asyncio.run(run())
        assert client.actions == []  # env was never stepped
        assert state["turn_rewards"] == [0.0]
        messages = state["messages"]
        assert messages[-1]["role"] == "user"
        assert "JSON object" in messages[-1]["content"]
        assert state["turn_log"][-1]["malformed"] is True
        assert env.pop_echo_observation_messages(state) == []

    def test_server_schema_validation_is_corrected_without_crashing_group(self):
        class _SchemaRejectingClient(_FakeClient):
            async def step(self, action: dict) -> StepResult:
                self.actions.append(action)
                raise OpenEnvServerError("Invalid message", code="VALIDATION_ERROR")

        client = _SchemaRejectingClient(step_rewards=[])
        env = _env(client)

        async def run() -> dict:
            state = await env.init_state(input={})
            state = await env.setup_state(state)
            await env.add_trajectory_step(state, _assistant_step('{"tool": 123}'))
            await env.cleanup(state)
            return state

        state = asyncio.run(run())
        assert client.actions == [{"tool": 123}]
        assert state["turn_rewards"] == [0.0]
        assert state["messages"][-1]["role"] == "user"
        assert "Server detail: Invalid message" in state["messages"][-1]["content"]
        assert state["turn_log"][-1]["malformed"] is True
        assert state["turn_log"][-1]["server_validation_error"] is True
        assert env.pop_echo_observation_messages(state) == []

    def test_setup_failure_still_closes_client(self):
        class _FailingResetClient(_FakeClient):
            async def reset(self, **kwargs: object) -> StepResult:
                raise RuntimeError("server exploded")

        client = _FailingResetClient(step_rewards=[])
        env = _env(client)

        async def run() -> None:
            state = await env.init_state(input={})
            try:
                await env.setup_state(state)
            except RuntimeError:
                return
            raise AssertionError("expected setup_state to raise")

        asyncio.run(run())
        assert client.closed


class TestResetProvenanceGuard:
    def test_live_reset_must_match_the_identity_preloaded_for_its_seed(self):
        client = _ProvenanceClient(task_id="task-b")
        env = _provenance_env(client)

        async def run() -> None:
            state = await env.init_state(
                input={
                    "info": {
                        "seed": 0,
                        "openenv_task_id": "task-a",
                        "openenv_task_source": "factory",
                    }
                }
            )
            await env.setup_state(state)

        with pytest.raises(OpenEnvProvenanceError, match="changed task_id"):
            asyncio.run(run())
        assert client.closed

    def test_live_reset_accepts_preloaded_identity_and_forwards_horizon(self):
        client = _ProvenanceClient(task_id="task-a")
        env = _provenance_env(client, reset_max_turns=16)

        async def run() -> dict:
            state = await env.init_state(
                input={
                    "info": {
                        "seed": 7,
                        "openenv_task_id": "task-a",
                        "openenv_task_source": "factory",
                    }
                }
            )
            state = await env.setup_state(state)
            await env.cleanup(state)
            return state

        state = asyncio.run(run())
        assert state["openenv_done"] is False
        assert client.reset_calls == [{"seed": 7, "max_turns": 16}]
        assert client.closed

    def test_guarded_live_reset_requires_preload_identity_in_input(self):
        client = _ProvenanceClient(task_id="task-a")
        env = _provenance_env(client)

        async def run() -> None:
            await env.init_state(input={"info": {"seed": 0}})

        with pytest.raises(ValueError, match="openenv_task_id"):
            asyncio.run(run())
        assert client.reset_calls == []


class _ProvenanceClient(_FakeClient):
    def __init__(self, *, task_id: str, source: str = "factory") -> None:
        super().__init__(step_rewards=[])
        self.task_id = task_id
        self.source = source

    async def reset(self, **kwargs: object) -> StepResult:
        self.reset_calls.append(dict(kwargs))
        self.reset_seeds.append(kwargs.get("seed"))
        return StepResult(
            observation={
                "text": "start",
                "task_id": self.task_id,
                "metadata": {
                    "task_id": self.task_id,
                    "task_source": self.source,
                },
            },
            reward=None,
            done=False,
        )


def _provenance_env(
    client: _ProvenanceClient,
    *,
    reset_max_turns: int | None = None,
) -> OpenEnvEnvironment:
    return OpenEnvEnvironment(
        "http://localhost:8765",
        action_schema=_SCHEMA,
        provenance_guard=ResetProvenanceGuard(
            expected_task_source="factory",
            expected_task_ids=frozenset({"task-a", "task-b"}),
        ),
        reset_max_turns=reset_max_turns,
        client_factory=lambda _url: client,
    )
