"""OpenEnvEnvironment driven by run_multiturn_group, without verifiers.

This is the provider's load-bearing integration contract: the same rollout
engine that drives verifiers environments must complete OpenEnv episodes,
score them, and enforce turn limits — even when the verifiers package is
absent.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest

import retrain.environments.load as env_load
from retrain.backends import TrainHelper
from retrain.environments.openenv.client import StepResult
from retrain.environments.openenv.environment import OpenEnvEnvironment
from retrain.environments.openenv.provenance import (
    OpenEnvProvenanceError,
    ResetProvenanceGuard,
)
from retrain.environments.verifiers import (
    is_multiturn_environment,
    run_multiturn_group,
)
from retrain.training.echo import build_rollout_echo_datums

_SCHEMA = {"properties": {"tool": {"type": "string"}}, "required": ["tool"]}


class _ScriptedClient:
    """Env that rewards 1.0 and finishes on the second step."""

    instances: list["_ScriptedClient"] = []

    def __init__(self) -> None:
        self.actions: list[dict] = []
        self.closed = False
        _ScriptedClient.instances.append(self)

    async def connect(self) -> None:
        pass

    async def reset(self, **kwargs: object) -> StepResult:
        return StepResult(
            observation=f"start-{kwargs.get('seed')}", reward=None, done=False
        )

    async def step(self, action: dict) -> StepResult:
        self.actions.append(action)
        done = len(self.actions) >= 2
        return StepResult(
            observation=f"obs-{len(self.actions)}",
            reward=1.0 if done else 0.25,
            done=done,
        )

    async def close(self) -> None:
        self.closed = True


class _JsonActionTokenizer:
    """Decodes every completion to a fixed valid JSON action."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        del add_generation_prompt, tokenize
        return list(range(len(messages) + 1))

    def encode(self, text):
        return [len(text)]

    def batch_decode(self, token_ids, *, skip_special_tokens=True):
        del skip_special_tokens
        return ['{"tool": "advance"}' for _ in token_ids]


class _DummyHelper:
    def sample(self, prompt_ids_batch, num_samples, max_tokens, temperature, top_p):
        del num_samples, max_tokens, temperature, top_p
        return [[([101 + idx], [-0.1])] for idx, _ in enumerate(prompt_ids_batch)]


class _TransitionRenderer:
    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        del tools, add_generation_prompt
        return list(range(len(messages) + 1))

    def bridge_to_next_turn(
        self,
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        *,
        tools=None,
    ):
        del tools
        body = str(new_messages[0]["content"])
        return SimpleNamespace(
            token_ids=(
                list(previous_prompt_ids)
                + list(previous_completion_ids)
                + [900]
                + [ord(char) for char in body]
                + [901]
            )
        )


def _fresh_env() -> OpenEnvEnvironment:
    _ScriptedClient.instances.clear()
    return OpenEnvEnvironment(
        "http://localhost:8765",
        action_schema=_SCHEMA,
        client_factory=lambda _url: _ScriptedClient(),
    )


def _run(
    env: OpenEnvEnvironment,
    *,
    num_rollouts: int = 2,
    max_turns: int = -1,
    info: dict[str, object] | None = None,
    rollout_env_workers: int = 1,
    capture_echo_transitions: bool = False,
    echo_token_renderer=None,
):
    return run_multiturn_group(
        env,
        helper=cast(TrainHelper, _DummyHelper()),
        tokenizer=_JsonActionTokenizer(),
        model_name="test-model",
        prompt="ignored",
        answer="",
        task="openenv",
        info=info or {"seed": 4},
        num_rollouts=num_rollouts,
        max_tokens=16,
        temperature=1.0,
        top_p=1.0,
        max_turns_override=max_turns,
        rollout_env_workers=rollout_env_workers,
        capture_echo_transitions=capture_echo_transitions,
        echo_token_renderer=echo_token_renderer,
    )


class TestOpenEnvRollout:
    def test_is_multiturn_without_verifiers_import(self, monkeypatch):
        def _unavailable():
            raise ImportError("verifiers not installed")

        monkeypatch.setattr(env_load, "require_verifiers", _unavailable)
        assert is_multiturn_environment(_fresh_env()) is True

    def test_group_completes_and_scores_episodic_rewards(self, monkeypatch):
        def _unavailable():
            raise ImportError("verifiers not installed")

        monkeypatch.setattr(env_load, "require_verifiers", _unavailable)
        rewards, turns, completions, turn_rewards, _, _, _, timing = _run(_fresh_env())
        assert rewards == [1.25, 1.25]  # 0.25 + 1.0 per rollout
        assert turn_rewards == [[0.25, 1.0], [0.25, 1.0]]
        assert [len(rollout_turns) for rollout_turns in turns] == [2, 2]
        assert all("advance" in text for text in completions)
        assert timing.turns == 4
        # One isolated connection per rollout, all closed after cleanup.
        assert len(_ScriptedClient.instances) == 2
        assert all(client.closed for client in _ScriptedClient.instances)

    def test_max_turns_override_truncates_episode(self):
        rewards, turns, _, _, _, _, _, _ = _run(
            _fresh_env(), num_rollouts=1, max_turns=1
        )
        assert [len(rollout_turns) for rollout_turns in turns] == [1]
        assert rewards == [0.25]  # only the first step's reward accrued

    def test_captures_nonterminal_and_terminal_echo_transition_rows(self):
        rewards, turns, *_ = _run(
            _fresh_env(),
            num_rollouts=1,
            capture_echo_transitions=True,
            echo_token_renderer=_TransitionRenderer(),
        )

        assert rewards == [1.25]
        assert len(turns[0]) == 2
        assert all(turn.echo_observation_capture_supported for turn in turns[0])
        assert all(turn.post_observation_seen for turn in turns[0])
        assert all(turn.post_observation_ids is not None for turn in turns[0])
        assert all(sum(turn.post_observation_mask or []) > 0 for turn in turns[0])
        assert turns[0][0].post_observation_terminal is False
        assert turns[0][1].post_observation_terminal is True

        datums, stats = build_rollout_echo_datums(
            turns[0],
            completion_advantages=[[0.25], [-0.25]],
            weight=0.05,
            min_prompt_overlap=1.0,
        )
        assert sum(datum.action_token_count for datum in datums) == 2
        assert stats.observation_responses == 2
        assert stats.bridged_transition_datums == 2
        assert stats.bridge_failures == 0
        assert stats.terminal_candidate_tokens > 0

    def test_renderer_parity_failure_keeps_rollout_and_marks_bridge_failure(self):
        class _MismatchedRenderer(_TransitionRenderer):
            def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
                del messages, tools, add_generation_prompt
                return [999]

        rewards, turns, *_ = _run(
            _fresh_env(),
            num_rollouts=1,
            capture_echo_transitions=True,
            echo_token_renderer=_MismatchedRenderer(),
        )

        assert rewards == [1.25]
        assert all(turn.post_observation_seen for turn in turns[0])
        assert all(turn.post_observation_ids is None for turn in turns[0])
        assert all(turn.post_observation_bridge_failed for turn in turns[0])
        assert all(turn.echo_renderer_parity_failed for turn in turns[0])

    def test_renderer_bridge_exception_falls_back_to_action_only_grpo(self):
        class _FailingBridgeRenderer(_TransitionRenderer):
            def bridge_to_next_turn(self, *args, **kwargs):
                del args, kwargs
                raise ValueError("unsupported transition")

        rewards, turns, *_ = _run(
            _fresh_env(),
            num_rollouts=1,
            capture_echo_transitions=True,
            echo_token_renderer=_FailingBridgeRenderer(),
        )

        datums, stats = build_rollout_echo_datums(
            turns[0],
            completion_advantages=[[0.25], [-0.25]],
            weight=0.05,
            min_prompt_overlap=1.0,
        )

        assert rewards == [1.25]
        assert sum(datum.action_token_count for datum in datums) == 2
        assert sum(datum.positive_tokens for datum in datums) == 0
        assert stats.observation_responses == 2
        assert stats.bridge_failures == 2
        assert all(turn.post_observation_bridge_failed for turn in turns[0])

    def test_mapping_trajectory_step_preserves_only_assistant_completion(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "retrain.environments.verifiers._optional_verifiers",
            lambda: SimpleNamespace(TrajectoryStep=dict),
        )
        rewards, _, _, _, _, _, _, _ = _run(
            _fresh_env(),
            num_rollouts=1,
        )
        assert rewards == [1.25]
        assert len(_ScriptedClient.instances) == 1
        assert _ScriptedClient.instances[0].actions == [
            {"tool": "advance"},
            {"tool": "advance"},
        ]

    def test_infra_errors_propagate_and_still_clean_up(self):
        class _ExplodingClient(_ScriptedClient):
            async def step(self, action: dict) -> StepResult:
                raise RuntimeError("connection lost")

        _ScriptedClient.instances.clear()
        env = OpenEnvEnvironment(
            "http://localhost:8765",
            action_schema=_SCHEMA,
            client_factory=lambda _url: _ExplodingClient(),
        )
        with pytest.raises(RuntimeError, match="connection lost"):
            _run(env, num_rollouts=1)
        assert all(client.closed for client in _ScriptedClient.instances)

    def test_mixed_live_provenance_failure_cleans_successful_sibling(self):
        class _IdentityClient(_ScriptedClient):
            def __init__(self, task_id: str) -> None:
                super().__init__()
                self.task_id = task_id

            async def reset(self, **kwargs: object) -> StepResult:
                if self.task_id == "task-a":
                    await asyncio.sleep(0.01)
                return StepResult(
                    observation={
                        "text": f"start-{kwargs.get('seed')}",
                        "task_id": self.task_id,
                        "info": {
                            "task_id": self.task_id,
                            "task_source": "factory",
                        },
                    },
                    reward=None,
                    done=False,
                )

        task_ids = iter(("task-a", "task-b"))
        _ScriptedClient.instances.clear()
        env = OpenEnvEnvironment(
            "http://localhost:8765",
            action_schema=_SCHEMA,
            provenance_guard=ResetProvenanceGuard(
                expected_task_source="factory",
                expected_task_ids=frozenset({"task-a", "task-b"}),
            ),
            client_factory=lambda _url: _IdentityClient(next(task_ids)),
        )

        with pytest.raises(OpenEnvProvenanceError, match="changed task_id"):
            _run(
                env,
                num_rollouts=2,
                info={
                    "seed": 4,
                    "openenv_task_id": "task-a",
                    "openenv_task_source": "factory",
                },
                rollout_env_workers=2,
            )

        assert len(_ScriptedClient.instances) == 2
        assert all(client.closed for client in _ScriptedClient.instances)
