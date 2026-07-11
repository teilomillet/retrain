"""Config loading and seed-dataset tests for the openenv provider."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import retrain.environments.openenv.load as openenv_load
from retrain.environments.openenv.client import StepResult
from retrain.environments.openenv.environment import OpenEnvEnvironment
from retrain.environments.openenv.provenance import (
    OpenEnvProvenanceError,
    ResetProvenanceGuard,
)

_SCHEMA = {"properties": {"tool": {"type": "string"}}, "required": ["tool"]}


class _SeedEchoClient:
    def __init__(self) -> None:
        self.reset_seeds: list[object] = []
        self.reset_calls: list[dict[str, object]] = []
        self.closed = False

    async def connect(self) -> None:
        pass

    async def reset(self, **kwargs: object) -> StepResult:
        self.reset_calls.append(dict(kwargs))
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
        "environment_max_turns": -1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestLoadEnvironment:
    def test_unknown_args_fail_loudly(self, monkeypatch):
        monkeypatch.setattr(openenv_load, "fetch_action_schema", lambda _url: _SCHEMA)
        with pytest.raises(ValueError, match="renderr"):
            openenv_load.load_environment(
                _config(environment_args='{"renderr": "typo.path"}')
            )

    def test_builds_environment_with_fetched_schema(self, monkeypatch):
        monkeypatch.setattr(openenv_load, "fetch_action_schema", lambda _url: _SCHEMA)
        env = openenv_load.load_environment(_config())
        assert isinstance(env, OpenEnvEnvironment)
        assert env.action_schema == _SCHEMA
        assert env.retrain_multiturn is True

    def test_builds_opt_in_provenance_and_horizon_guards(self, monkeypatch):
        monkeypatch.setattr(openenv_load, "fetch_action_schema", lambda _url: _SCHEMA)
        env = openenv_load.load_environment(
            _config(
                environment_max_turns=16,
                environment_args=(
                    '{"expected_task_source":"factory",'
                    '"expected_task_ids":["task-a","task-b"]}'
                ),
            )
        )

        assert env.provenance_guard == ResetProvenanceGuard(
            expected_task_source="factory",
            expected_task_ids=frozenset({"task-a", "task-b"}),
        )
        assert env.reset_max_turns == 16

    @pytest.mark.parametrize("max_turns", [-1, 0])
    def test_nonpositive_horizon_preserves_server_default(self, monkeypatch, max_turns):
        monkeypatch.setattr(openenv_load, "fetch_action_schema", lambda _url: _SCHEMA)
        env = openenv_load.load_environment(_config(environment_max_turns=max_turns))

        assert env.reset_max_turns is None

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            ('{"expected_task_source":""}', "non-empty string"),
            ('{"expected_task_ids":"task-a"}', "non-empty list"),
            ('{"expected_task_ids":[]}', "non-empty list"),
            (
                '{"expected_task_ids":["task-a","task-a"]}',
                "must not contain duplicates",
            ),
        ],
    )
    def test_invalid_provenance_args_fail_closed(self, monkeypatch, args, message):
        monkeypatch.setattr(openenv_load, "fetch_action_schema", lambda _url: _SCHEMA)
        with pytest.raises(ValueError, match=message):
            openenv_load.load_environment(_config(environment_args=args))


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

    def test_positive_max_turns_is_forwarded_to_every_preload_reset(self):
        client = _SeedEchoClient()
        env = OpenEnvEnvironment(
            "http://localhost:8765",
            action_schema=_SCHEMA,
            reset_max_turns=16,
            client_factory=lambda _url: client,
        )

        openenv_load.examples_from_environment(
            env,
            _config(max_examples=2, seed=10),
        )

        assert client.reset_calls == [
            {"seed": 10, "max_turns": 16},
            {"seed": 11, "max_turns": 16},
        ]

    def test_guard_validates_exact_set_and_records_per_seed_identity(self):
        client = _TaskClient({0: "task-a", 1: "task-b"})
        env = _guarded_env(client, task_ids={"task-a", "task-b"})

        examples = openenv_load.examples_from_environment(
            env,
            _config(max_examples=2, seed=0),
        )

        assert [example.info for example in examples] == [
            {
                "seed": 0,
                "openenv_task_id": "task-a",
                "openenv_task_source": "factory",
            },
            {
                "seed": 1,
                "openenv_task_id": "task-b",
                "openenv_task_source": "factory",
            },
        ]
        assert client.closed

    def test_guard_rejects_preload_source_mismatch_and_closes_client(self):
        client = _TaskClient({0: "task-a"}, source="eval")
        env = _guarded_env(client, task_ids={"task-a"})

        with pytest.raises(OpenEnvProvenanceError, match="task_source mismatch"):
            openenv_load.examples_from_environment(
                env,
                _config(max_examples=1, seed=0),
            )
        assert client.closed

    def test_guard_rejects_missing_preload_provenance(self):
        client = _TaskClient({0: "task-a"}, source="")
        env = _guarded_env(client, task_ids={"task-a"})

        with pytest.raises(OpenEnvProvenanceError, match="missing task_source"):
            openenv_load.examples_from_environment(
                env,
                _config(max_examples=1, seed=0),
            )

    def test_guard_rejects_task_outside_expected_set(self):
        client = _TaskClient({0: "task-c"})
        env = _guarded_env(client, task_ids={"task-a"})

        with pytest.raises(OpenEnvProvenanceError, match="unexpected task_id"):
            openenv_load.examples_from_environment(
                env,
                _config(max_examples=1, seed=0),
            )

    def test_guard_rejects_missing_expected_task_set_coverage(self):
        client = _TaskClient({0: "task-a", 1: "task-a"})
        env = _guarded_env(client, task_ids={"task-a", "task-b"})

        with pytest.raises(OpenEnvProvenanceError, match=r"missing=\['task-b'\]"):
            openenv_load.examples_from_environment(
                env,
                _config(max_examples=2, seed=0),
            )

    def test_guard_rejects_too_few_preload_examples_before_reset(self):
        client = _TaskClient({0: "task-a"})
        env = _guarded_env(client, task_ids={"task-a", "task-b"})

        with pytest.raises(ValueError, match="max_examples must be at least"):
            openenv_load.examples_from_environment(
                env,
                _config(max_examples=1, seed=0),
            )
        assert client.reset_calls == []


class _TaskClient(_SeedEchoClient):
    def __init__(self, tasks_by_seed: dict[int, str], *, source: str = "factory"):
        super().__init__()
        self.tasks_by_seed = tasks_by_seed
        self.source = source

    async def reset(self, **kwargs: object) -> StepResult:
        self.reset_calls.append(dict(kwargs))
        seed = int(kwargs["seed"])
        self.reset_seeds.append(seed)
        task_id = self.tasks_by_seed[seed]
        return StepResult(
            observation={
                "text": f"obs-{seed}",
                "task_id": task_id,
                "info": {"task_id": task_id, "task_source": self.source},
            },
            reward=None,
            done=False,
        )


def _guarded_env(client: _TaskClient, *, task_ids: set[str]) -> OpenEnvEnvironment:
    return OpenEnvEnvironment(
        "http://localhost:8765",
        action_schema=_SCHEMA,
        provenance_guard=ResetProvenanceGuard(
            expected_task_source="factory",
            expected_task_ids=frozenset(task_ids),
        ),
        client_factory=lambda _url: client,
    )
