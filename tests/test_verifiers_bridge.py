"""Unit tests for retrain.verifiers_bridge helper functions."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from retrain import verifiers_bridge as bridge_mod
from retrain.backends import TrainHelper
from retrain.config import TrainConfig
from retrain.verifiers_bridge import (
    StateDict,
    VerifiersRolloutTiming,
    _coerce_float_list,
    _collect_observation_timing,
    encode_prompt_for_sampling,
    load_examples_from_environment,
    observation_mask_for_prompt,
    parse_environment_args,
    prompt_preview,
    run_multiturn_group,
)


class _DummyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        # Stable synthetic IDs for tests.
        size = len(messages)
        return [10 + size, 20 + size]

    def encode(self, text):
        return [len(text), len(text) + 1]

    def batch_decode(self, token_ids, *, skip_special_tokens=True):
        return [f"completion-{idx}" for idx, _ids in enumerate(token_ids)]


class _RoleTemplateTokenizer:
    role_ids = {
        "system": 1,
        "user": 2,
        "assistant": 3,
        "tool": 4,
        "environment": 5,
        "observation": 6,
    }

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        ids = []
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            ids.append(self.role_ids.get(role, 0))
            ids.extend(ord(char) for char in content)
        if add_generation_prompt:
            ids.append(99)
        return ids

    def batch_decode(self, token_ids, *, skip_special_tokens=True):
        return [f"completion-{idx}" for idx, _ids in enumerate(token_ids)]


class _NonPrefixRoleTemplateTokenizer(_RoleTemplateTokenizer):
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        ids = super().apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
        )
        if not add_generation_prompt:
            return list(reversed(ids))
        return ids


class _DummyHelper:
    def sample(self, prompt_ids_batch, num_samples, max_tokens, temperature, top_p):  # noqa: ARG002
        return [[([101 + idx], [-0.1])] for idx, _prompt in enumerate(prompt_ids_batch)]


def _helper() -> TrainHelper:
    return cast(TrainHelper, _DummyHelper())


class _CleanupTrackingRubric:
    async def score_group(self, states):
        for state in states:
            state["reward"] = 1.0


class _FailingRubric:
    async def score_group(self, states):  # noqa: ARG002
        raise RuntimeError("score failed")


class _CleanupTrackingMultiTurnEnv:
    message_type = "chat"

    def __init__(self) -> None:
        self.rubric = _CleanupTrackingRubric()
        self.cleaned: list[int] = []

    async def init_state(self, *, input, client, model, sampling_args):  # noqa: A002, ARG002
        return {
            "prompt": input["prompt"],
            "trajectory": [],
            "trajectory_id": input["example_id"],
            "reward": None,
            "completion": None,
        }

    async def setup_state(self, state):
        state["setup"] = True
        return state

    async def is_completed(self, state):
        return bool(state["trajectory"])

    async def get_prompt_messages(self, state):
        return state["prompt"]

    async def add_trajectory_step(self, state, step):
        state["trajectory"].append(step)

    async def render_completion(self, state):
        step = state["trajectory"][0]
        state["completion"] = getattr(step, "completion", [])

    async def cleanup(self, state):
        self.cleaned.append(int(state["trajectory_id"]))


class _TwoTurnObservationEnv(_CleanupTrackingMultiTurnEnv):
    async def is_completed(self, state):
        return len(state["trajectory"]) >= 2

    async def get_prompt_messages(self, state):
        if not state["trajectory"]:
            return [{"role": "user", "content": "task"}]
        first_completion = state["trajectory"][0].completion[0]["content"]
        return [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": first_completion},
            {"role": "tool", "content": "result"},
        ]


class _OpenEnvCleanupTrackingMultiTurnEnv(_CleanupTrackingMultiTurnEnv):
    def __init__(self) -> None:
        super().__init__()
        self.openenv_cleaned: list[int] = []

    async def _cleanup_openenv_state(self, state):
        self.openenv_cleaned.append(int(state["trajectory_id"]))

    async def cleanup(self, state):  # pragma: no cover - must not be called.
        raise AssertionError("full cleanup should not run for OpenEnv resources")


class _FailingScoreCleanupTrackingMultiTurnEnv(_CleanupTrackingMultiTurnEnv):
    def __init__(self) -> None:
        super().__init__()
        self.rubric = _FailingRubric()


class _ConcurrentInitTrackingMultiTurnEnv(_CleanupTrackingMultiTurnEnv):
    def __init__(self) -> None:
        super().__init__()
        self.active_init = 0
        self.max_active_init = 0

    async def init_state(self, *, input, client, model, sampling_args):  # noqa: A002, ARG002
        self.active_init += 1
        self.max_active_init = max(self.max_active_init, self.active_init)
        await asyncio.sleep(0)
        self.active_init -= 1
        return await super().init_state(
            input=input,
            client=client,
            model=model,
            sampling_args=sampling_args,
        )


class _FakeTrajectoryStep:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _FakeVerifiersModule:
    TrajectoryStep = _FakeTrajectoryStep


class _EvalOnlyEnv:
    env_id = "primeintellect/aime2025"

    def get_dataset(self, n=-1, seed=None):
        raise ValueError("dataset is not set")


class _BrokenDatasetEnv:
    env_id = "primeintellect/broken"

    def get_dataset(self, n=-1, seed=None):
        raise RuntimeError("backend unavailable")


class TestParseEnvironmentArgs:
    def test_empty_args(self):
        assert parse_environment_args("") == {}
        assert parse_environment_args("   ") == {}
        assert parse_environment_args(None) == {}

    def test_dict_args_passthrough(self):
        assert parse_environment_args({"game": "Wordle-v0", "seed": 42}) == {
            "game": "Wordle-v0",
            "seed": 42,
        }

    def test_valid_json_object(self):
        parsed = parse_environment_args('{"game":"Wordle-v0","seed":42}')
        assert parsed == {"game": "Wordle-v0", "seed": 42}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="must be valid JSON"):
            parse_environment_args("{bad-json")

    def test_non_object_json_raises(self):
        with pytest.raises(ValueError, match="JSON object"):
            parse_environment_args('["not","object"]')

    def test_non_str_non_dict_raises(self):
        with pytest.raises(ValueError, match="JSON string/object"):
            parse_environment_args(123)  # type: ignore[arg-type]


class TestPromptHelpers:
    def test_prompt_preview_string(self):
        assert prompt_preview("hello world", max_chars=5) == "hello"

    def test_prompt_preview_messages(self):
        preview = prompt_preview(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Solve 2+2"},
            ]
        )
        assert "system: You are helpful." in preview
        assert "user: Solve 2+2" in preview

    def test_encode_prompt_string(self):
        tok = _DummyTokenizer()
        ids = encode_prompt_for_sampling(tok, "test prompt")
        assert ids == [11, 21]

    def test_encode_prompt_messages(self):
        tok = _DummyTokenizer()
        ids = encode_prompt_for_sampling(
            tok,
            [{"role": "user", "content": "test prompt"}],
        )
        assert ids == [11, 21]

    def test_observation_mask_for_prompt_marks_tool_content_span(self):
        tok = _RoleTemplateTokenizer()
        prompt: list[dict[str, object]] = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "act"},
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "continue"},
        ]
        prompt_ids = encode_prompt_for_sampling(tok, prompt)

        mask = observation_mask_for_prompt(tok, prompt, prompt_ids)

        assert mask is not None
        marked = [idx for idx, value in enumerate(mask) if value]
        assert len(marked) == len("result")
        assert prompt_ids[marked[0] : marked[-1] + 1] == [*map(ord, "result")]

    def test_observation_mask_for_prompt_returns_none_on_unstable_prefix(self):
        tok = _NonPrefixRoleTemplateTokenizer()
        prompt: list[dict[str, object]] = [
            {"role": "user", "content": "task"},
            {"role": "tool", "content": "result"},
        ]
        prompt_ids = encode_prompt_for_sampling(tok, prompt)

        mask = observation_mask_for_prompt(tok, prompt, prompt_ids)

        assert mask is None


class TestLoadExamplesFromEnvironment:
    def test_eval_only_env_has_actionable_error(self):
        cfg = TrainConfig(
            environment_provider="verifiers",
            environment_id="primeintellect/aime2025",
        )
        with pytest.raises(RuntimeError, match="does not expose a training dataset"):
            load_examples_from_environment(_EvalOnlyEnv(), cfg)

    def test_other_dataset_errors_preserve_context(self):
        cfg = TrainConfig(
            environment_provider="verifiers",
            environment_id="primeintellect/broken",
        )
        with pytest.raises(RuntimeError, match="Failed to load dataset"):
            load_examples_from_environment(_BrokenDatasetEnv(), cfg)


class TestCoerceFloatList:
    def test_none_returns_empty(self):
        assert _coerce_float_list(None) == []

    def test_non_list_returns_empty(self):
        assert _coerce_float_list(42) == []
        assert _coerce_float_list("not a list") == []

    def test_valid_floats(self):
        assert _coerce_float_list([1.5, 0.0, -0.3]) == [1.5, 0.0, -0.3]

    def test_ints_coerced_to_float(self):
        assert _coerce_float_list([1, 2, 3]) == [1.0, 2.0, 3.0]

    def test_invalid_items_become_zero(self):
        assert _coerce_float_list([1.0, "bad", None, 2.0]) == [1.0, 0.0, 0.0, 2.0]

    def test_empty_list(self):
        assert _coerce_float_list([]) == []


class TestRunMultiturnGroup:
    def test_cleans_up_rollout_states_after_scoring(self, monkeypatch):
        monkeypatch.setattr(
            bridge_mod,
            "_require_verifiers",
            lambda: _FakeVerifiersModule,
        )
        env = _CleanupTrackingMultiTurnEnv()

        rewards, turns, *_ = run_multiturn_group(
            env,
            helper=_helper(),
            tokenizer=_DummyTokenizer(),
            model_name="dummy-model",
            prompt=[{"role": "user", "content": "hello"}],
            answer="",
            task="task",
            info={},
            num_rollouts=3,
            max_tokens=4,
            temperature=1.0,
            top_p=1.0,
        )

        assert rewards == [1.0, 1.0, 1.0]
        assert [len(rollout_turns) for rollout_turns in turns] == [1, 1, 1]
        assert env.cleaned == [0, 1, 2]

    def test_turn_samples_include_prompt_observation_mask(self, monkeypatch):
        monkeypatch.setattr(
            bridge_mod,
            "_require_verifiers",
            lambda: _FakeVerifiersModule,
        )
        env = _TwoTurnObservationEnv()

        rewards, turns, *_ = run_multiturn_group(
            env,
            helper=_helper(),
            tokenizer=_RoleTemplateTokenizer(),
            model_name="dummy-model",
            prompt=[{"role": "user", "content": "task"}],
            answer="",
            task="task",
            info={},
            num_rollouts=1,
            max_tokens=4,
            temperature=1.0,
            top_p=1.0,
        )

        assert rewards == [1.0]
        assert len(turns[0]) == 2
        assert turns[0][0].observation_mask is None
        assert turns[0][1].observation_mask is not None
        assert sum(turns[0][1].observation_mask or []) == len("result")

    def test_prefers_openenv_resource_cleanup_when_available(self, monkeypatch):
        monkeypatch.setattr(
            bridge_mod,
            "_require_verifiers",
            lambda: _FakeVerifiersModule,
        )
        env = _OpenEnvCleanupTrackingMultiTurnEnv()

        rewards, turns, *_ = run_multiturn_group(
            env,
            helper=_helper(),
            tokenizer=_DummyTokenizer(),
            model_name="dummy-model",
            prompt=[{"role": "user", "content": "hello"}],
            answer="",
            task="task",
            info={},
            num_rollouts=2,
            max_tokens=4,
            temperature=1.0,
            top_p=1.0,
        )

        assert rewards == [1.0, 1.0]
        assert [len(rollout_turns) for rollout_turns in turns] == [1, 1]
        assert env.openenv_cleaned == [0, 1]

    def test_cleans_up_rollout_states_when_scoring_fails(self, monkeypatch):
        monkeypatch.setattr(
            bridge_mod,
            "_require_verifiers",
            lambda: _FakeVerifiersModule,
        )
        env = _FailingScoreCleanupTrackingMultiTurnEnv()

        with pytest.raises(RuntimeError, match="score failed"):
            run_multiturn_group(
                env,
                helper=_helper(),
                tokenizer=_DummyTokenizer(),
                model_name="dummy-model",
                prompt=[{"role": "user", "content": "hello"}],
                answer="",
                task="task",
                info={},
                num_rollouts=2,
                max_tokens=4,
                temperature=1.0,
                top_p=1.0,
            )

        assert env.cleaned == [0, 1]

    def test_rollout_scheduler_bounds_async_env_workers(self, monkeypatch):
        monkeypatch.setattr(
            bridge_mod,
            "_require_verifiers",
            lambda: _FakeVerifiersModule,
        )
        env = _ConcurrentInitTrackingMultiTurnEnv()

        rewards, turns, *_rest, timing = run_multiturn_group(
            env,
            helper=_helper(),
            tokenizer=_DummyTokenizer(),
            model_name="dummy-model",
            prompt=[{"role": "user", "content": "hello"}],
            answer="",
            task="task",
            info={},
            num_rollouts=4,
            max_tokens=4,
            temperature=1.0,
            top_p=1.0,
            rollout_env_workers=2,
            rollout_buffer_size=2,
        )

        assert rewards == [1.0, 1.0, 1.0, 1.0]
        assert [len(rollout_turns) for rollout_turns in turns] == [1, 1, 1, 1]
        assert env.max_active_init == 2
        assert env.cleaned == [0, 1, 2, 3]
        assert timing.env_workers == 2
        assert timing.buffer_size == 2
        assert timing.scheduler_worker_s >= 0.0


def test_collect_observation_timing_from_trajectory_extras():
    state = cast(
        StateDict,
        {
            "trajectory": [
                {
                    "extras": {
                        "openenv_info": {
                            "timing": {
                                "dbt_total_s": 1.25,
                                "step_total_s": 1.5,
                                "dbt_target_scoped": True,
                                "action": "Dbt",
                            }
                        }
                    }
                }
            ]
        },
    )
    totals: dict[str, float] = {}

    _collect_observation_timing(state, totals)

    assert totals == {"dbt_total_s": 1.25, "step_total_s": 1.5}


def test_collect_observation_timing_from_direct_step_timing():
    state = cast(
        StateDict,
        {
            "trajectory": [
                _FakeTrajectoryStep(
                    timing={
                        "env_step_s": 0.25,
                        "nan_step_s": float("nan"),
                        "inf_step_s": float("inf"),
                        "flag": True,
                        "action": "Dbt",
                    }
                )
            ]
        },
    )
    totals: dict[str, float] = {}

    _collect_observation_timing(state, totals)

    assert totals == {"env_step_s": 0.25}


def test_verifiers_rollout_timing_as_metrics_includes_env_timing():
    timing = VerifiersRolloutTiming(
        generation_s=2.0,
        trajectory_step_s=1.0,
        total_s=3.5,
        turns=4,
        model_tokens=12,
        env_timing_s={"dbt_total_s": 0.75},
    )

    metrics = timing.as_metrics()

    assert metrics["rollout/generation_s"] == 2.0
    assert metrics["rollout/trajectory_step_s"] == 1.0
    assert metrics["rollout/turns"] == 4.0
    assert metrics["rollout/model_tokens"] == 12.0
    assert metrics["rollout/env/dbt_total_s"] == 0.75
