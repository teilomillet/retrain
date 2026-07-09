"""OpenEnvEnvironment driven by run_multiturn_group, without verifiers.

This is the provider's load-bearing integration contract: the same rollout
engine that drives verifiers environments must complete OpenEnv episodes,
score them, and enforce turn limits — even when the verifiers package is
absent.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

import retrain.environments.load as env_load
from retrain.backends import TrainHelper
from retrain.environments.openenv.client import StepResult
from retrain.environments.openenv.environment import OpenEnvEnvironment
from retrain.environments.verifiers import (
    is_multiturn_environment,
    run_multiturn_group,
)

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
):
    return run_multiturn_group(
        env,
        helper=cast(TrainHelper, _DummyHelper()),
        tokenizer=_JsonActionTokenizer(),
        model_name="test-model",
        prompt="ignored",
        answer="",
        task="openenv",
        info={"seed": 4},
        num_rollouts=num_rollouts,
        max_tokens=16,
        temperature=1.0,
        top_p=1.0,
        max_turns_override=max_turns,
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
        rewards, turns, completions, turn_rewards, _, _, _, timing = _run(
            _fresh_env()
        )
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
