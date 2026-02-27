"""Backend contract tests for built-in backends."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from retrain.backends import TrainHelper
from retrain.config import TrainConfig
from retrain.registry import backend


class _BaseFakeHelper:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def checkpoint(self, name: str) -> None:
        self.calls.append(f"checkpoint:{name}")

    def sample(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[tuple[list[int], list[float]]]]:
        self.calls.append("sample")
        _ = max_tokens, temperature, top_p
        return [
            [([101, 102], [-0.1, -0.2]) for _ in range(num_samples)]
            for _ in prompt_ids_list
        ]

    def sample_with_entropy(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[tuple[list[int], list[float], list[float] | None]]]:
        self.calls.append("sample_with_entropy")
        _ = max_tokens, temperature, top_p
        return [
            [([101, 102], [-0.1, -0.2], None) for _ in range(num_samples)]
            for _ in prompt_ids_list
        ]

    def train_step(
        self,
        all_tokens: list[list[int]],
        all_logprobs: list[list[float]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float:
        self.calls.append("train_step")
        _ = all_tokens, all_logprobs, all_advantages, lr, weight_decay
        return 0.0

    def save_adapter(self, path: str, name: str) -> str:
        self.calls.append(f"save_adapter:{name}")
        return f"{path.rstrip('/')}/{name}"

    def load_state(self, name: str) -> None:
        self.calls.append(f"load_state:{name}")


class _FakeLocalTrainHelper(_BaseFakeHelper):
    def __init__(self, *args, **kwargs):
        _ = args, kwargs
        super().__init__()


class _FakeTinkerTrainHelper(_BaseFakeHelper):
    def __init__(self, *args, **kwargs):
        _ = args, kwargs
        super().__init__()


class _FakePrimeRLTrainHelper(_BaseFakeHelper):
    def __init__(self, *args, **kwargs):
        _ = args, kwargs
        super().__init__()


def _exercise_lifecycle_step(helper: TrainHelper, step_name: str) -> None:
    helper.checkpoint(step_name)
    out = helper.sample(
        prompt_ids_list=[[1, 2, 3]],
        num_samples=1,
        max_tokens=32,
        temperature=0.7,
        top_p=0.95,
    )
    assert out == [[([101, 102], [-0.1, -0.2])]]

    loss = helper.train_step(
        all_tokens=[[1, 2, 3]],
        all_logprobs=[[0.0, -0.1, -0.2]],
        all_advantages=[[0.0, 1.0, 1.0]],
        lr=1e-4,
        weight_decay=0.0,
    )
    assert isinstance(loss, float)

    save_path = helper.save_adapter("/tmp/backend-contract", step_name)
    assert save_path.endswith(f"/{step_name}")

    helper.load_state(step_name)


def _assert_lifecycle_calls(helper: _BaseFakeHelper) -> None:
    assert helper.calls == [
        "checkpoint:step_0",
        "sample",
        "train_step",
        "save_adapter:step_0",
        "load_state:step_0",
        "checkpoint:step_1",
        "sample",
        "train_step",
        "save_adapter:step_1",
        "load_state:step_1",
    ]


def test_local_backend_contract(monkeypatch):
    fake_mod = SimpleNamespace(LocalTrainHelper=_FakeLocalTrainHelper)
    monkeypatch.setitem(sys.modules, "retrain.local_train_helper", fake_mod)

    cfg = TrainConfig(backend="local")
    helper = backend.create("local", cfg)
    assert isinstance(helper, TrainHelper)
    assert isinstance(helper, _BaseFakeHelper)
    _exercise_lifecycle_step(helper, "step_0")
    _exercise_lifecycle_step(helper, "step_1")
    _assert_lifecycle_calls(helper)


def test_tinker_backend_contract(monkeypatch):
    fake_mod = SimpleNamespace(TinkerTrainHelper=_FakeTinkerTrainHelper)
    monkeypatch.setitem(sys.modules, "retrain.tinker_backend", fake_mod)

    cfg = TrainConfig(backend="tinker")
    helper = backend.create("tinker", cfg)
    assert isinstance(helper, TrainHelper)
    assert isinstance(helper, _BaseFakeHelper)
    _exercise_lifecycle_step(helper, "step_0")
    _exercise_lifecycle_step(helper, "step_1")
    _assert_lifecycle_calls(helper)


def test_prime_rl_backend_contract(monkeypatch):
    fake_mod = SimpleNamespace(PrimeRLTrainHelper=_FakePrimeRLTrainHelper)
    monkeypatch.setitem(sys.modules, "retrain.prime_rl_backend", fake_mod)

    cfg = TrainConfig(backend="prime_rl")
    helper = backend.create("prime_rl", cfg)
    assert isinstance(helper, TrainHelper)
    assert isinstance(helper, _BaseFakeHelper)
    _exercise_lifecycle_step(helper, "step_0")
    _exercise_lifecycle_step(helper, "step_1")
    _assert_lifecycle_calls(helper)
