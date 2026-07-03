"""Config loading and seed-dataset tests for the openenv provider."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import retrain.environments.openenv.load as openenv_load
from retrain.environments.openenv.client import StepResult
from retrain.environments.openenv.environment import OpenEnvEnvironment

_SCHEMA = {"properties": {"tool": {"type": "string"}}, "required": ["tool"]}


class _SeedEchoClient:
    def __init__(self) -> None:
        self.reset_seeds: list[object] = []
        self.closed = False

    async def connect(self) -> None:
        pass

    async def reset(self, **kwargs: object) -> StepResult:
        seed = kwargs.get("seed")
        self.reset_seeds.append(seed)
        return StepResult(observation=f"obs-{seed}", reward=None, done=False)

    async def close(self) -> None:
        self.closed = True


def _config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "environment_id": "http://localhost:8765",
        "environment_args": None,
        "max_examples": 0,
        "seed": -1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestLoadEnvironment:
    def test_unknown_args_fail_loudly(self, monkeypatch):
        monkeypatch.setattr(
            openenv_load, "fetch_action_schema", lambda _url: _SCHEMA
        )
        with pytest.raises(ValueError, match="renderr"):
            openenv_load.load_environment(
                _config(environment_args='{"renderr": "typo.path"}')
            )

    def test_builds_environment_with_fetched_schema(self, monkeypatch):
        monkeypatch.setattr(
            openenv_load, "fetch_action_schema", lambda _url: _SCHEMA
        )
        env = openenv_load.load_environment(_config())
        assert isinstance(env, OpenEnvEnvironment)
        assert env.action_schema == _SCHEMA
        assert env.retrain_multiturn is True


class TestExamplesFromEnvironment:
    def _env(self, client: _SeedEchoClient) -> OpenEnvEnvironment:
        return OpenEnvEnvironment(
            "http://localhost:8765",
            action_schema=_SCHEMA,
            client_factory=lambda _url: client,
        )

    def test_one_example_per_seed_with_rendered_reset_prompt(self):
        client = _SeedEchoClient()
        examples = openenv_load.examples_from_environment(
            self._env(client),
            _config(max_examples=3, seed=10),
        )
        assert client.reset_seeds == [10, 11, 12]
        assert client.closed
        assert [e.info for e in examples] == [
            {"seed": 10},
            {"seed": 11},
            {"seed": 12},
        ]
        assert [e.example_id for e in examples] == [0, 1, 2]
        # Prompt is the rendered reset observation for that exact seed.
        assert "obs-11" in str(examples[1].prompt)

    def test_defaults_apply_when_unset(self):
        client = _SeedEchoClient()
        examples = openenv_load.examples_from_environment(
            self._env(client),
            _config(),  # max_examples=0, seed=-1
        )
        assert len(examples) == openenv_load.DEFAULT_NUM_EXAMPLES
        assert client.reset_seeds[0] == 0
