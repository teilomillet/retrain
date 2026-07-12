from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from retrain.backends.local import state as local_state
from retrain.backends.local.train import LocalTrainHelper


class _ControlledFuture:
    def __init__(self, *, done: bool = False, result: float = 0.0) -> None:
        self._done = done
        self._result = result
        self.result_calls = 0

    def done(self) -> bool:
        return self._done

    def result(self) -> float:
        self.result_calls += 1
        if not self._done:
            raise AssertionError("test attempted to wait for unfinished training")
        return self._result

    def complete(self, result: float) -> None:
        self._done = True
        self._result = result


class _RecordingExecutor:
    def __init__(self) -> None:
        self.submissions: list[tuple[object, tuple[object, ...]]] = []
        self.futures: list[_ControlledFuture] = []

    def submit(self, fn, *args) -> _ControlledFuture:
        future = _ControlledFuture()
        self.submissions.append((fn, args))
        self.futures.append(future)
        return future


class _BlockingFuture:
    def __init__(self, result: float) -> None:
        self._result = result
        self.result_started = threading.Event()
        self.release_result = threading.Event()

    def result(self) -> float:
        self.result_started.set()
        if not self.release_result.wait(timeout=5):
            raise AssertionError("timed out waiting to release prior training")
        return self._result


class _PrefixCacheEngine:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear_prefix_cache(self) -> None:
        self.clear_calls += 1


def _split_sequence_helper() -> LocalTrainHelper:
    helper = object.__new__(LocalTrainHelper)
    helper.optimizer = SimpleNamespace(param_groups=[{}])
    helper.train_device = "cpu"
    helper.train_microbatch_size = 1
    helper.train_supervised_context_tokens = 0
    helper.train_selective_suffix_logits = False
    helper.split_mode = True
    helper._external_engine = False
    helper._train_future = None
    helper._pending_loss = 0.0
    helper._train_executor = _RecordingExecutor()
    helper.engine = _PrefixCacheEngine()
    return helper


def test_split_train_step_returns_prior_loss_and_applies_backpressure() -> None:
    helper = _split_sequence_helper()
    tokens = [[1, 2], [3, 4]]
    logprobs = [[0.0, -0.1], [0.0, -0.2]]
    advantages = [[0.0, 1.0], [0.0, -1.0]]

    first_loss = helper.train_step(
        tokens,
        logprobs,
        advantages,
        lr=3e-5,
        weight_decay=0.1,
    )

    executor = helper._train_executor
    first_future = executor.futures[0]
    assert first_loss == 0.0
    assert first_future.result_calls == 0
    assert helper.optimizer.param_groups == [{"lr": 3e-5, "weight_decay": 0.1}]

    first_future.complete(1.25)
    second_loss = helper.train_step(
        tokens,
        logprobs,
        advantages,
        lr=2e-5,
        weight_decay=0.2,
    )

    assert second_loss == pytest.approx(1.25)
    assert first_future.result_calls == 1
    assert len(executor.submissions) == 2
    assert helper._train_future is executor.futures[1]
    assert helper.optimizer.param_groups == [{"lr": 2e-5, "weight_decay": 0.2}]
    assert helper.engine.clear_calls == 2


def test_split_sequence_submission_snapshots_caller_owned_rows() -> None:
    helper = _split_sequence_helper()
    tokens = [[1, 2], [3, 4]]
    logprobs = [[0.0, -0.1], [0.0, -0.2]]
    advantages = [[0.0, 1.0], [0.0, -1.0]]

    helper.train_step(
        tokens,
        logprobs,
        advantages,
        lr=3e-5,
        weight_decay=0.0,
    )
    _, submitted_args = helper._train_executor.submissions[0]

    tokens[0][1] = 99
    logprobs[0][1] = -9.0
    advantages[0][1] = 9.0

    assert submitted_args == (
        ((1, 2), (3, 4)),
        ((0.0, -0.1), (0.0, -0.2)),
        ((0.0, 1.0), (0.0, -1.0)),
    )


def test_split_train_step_waits_before_updating_optimizer_options() -> None:
    helper = _split_sequence_helper()
    tokens = [[1, 2], [3, 4]]
    logprobs = [[0.0, -0.1], [0.0, -0.2]]
    advantages = [[0.0, 1.0], [0.0, -1.0]]
    helper.train_step(
        tokens,
        logprobs,
        advantages,
        lr=3e-5,
        weight_decay=0.1,
    )
    prior = _BlockingFuture(result=1.25)
    helper._train_future = prior
    outcome: list[float] = []

    caller = threading.Thread(
        target=lambda: outcome.append(
            helper.train_step(
                tokens,
                logprobs,
                advantages,
                lr=2e-5,
                weight_decay=0.2,
            )
        )
    )
    caller.start()
    assert prior.result_started.wait(timeout=5)

    assert helper.optimizer.param_groups == [{"lr": 3e-5, "weight_decay": 0.1}]

    prior.release_result.set()
    caller.join(timeout=5)
    assert not caller.is_alive()
    assert outcome == [pytest.approx(1.25)]
    assert helper.optimizer.param_groups == [{"lr": 2e-5, "weight_decay": 0.2}]


def test_checkpoint_does_not_wait_for_unfinished_training() -> None:
    helper = object.__new__(LocalTrainHelper)
    future = _ControlledFuture()
    helper._train_future = future
    helper._pending_loss = 0.0
    calls: list[str] = []
    helper._sync_lora_weights = lambda: calls.append("sync")
    helper._clear_inference_prefix_cache = lambda: calls.append("clear")

    helper.checkpoint("step_1")

    assert helper._train_future is future
    assert helper._pending_loss == 0.0
    assert future.result_calls == 0
    assert calls == ["sync", "clear"]


def test_checkpoint_collects_completed_loss_before_sync() -> None:
    helper = object.__new__(LocalTrainHelper)
    future = _ControlledFuture(done=True, result=2.5)
    helper._train_future = future
    helper._pending_loss = 0.0
    observations: list[tuple[str, float]] = []
    helper._sync_lora_weights = lambda: observations.append(
        ("sync", helper._pending_loss)
    )
    helper._clear_inference_prefix_cache = lambda: observations.append(
        ("clear", helper._pending_loss)
    )

    helper.checkpoint("step_1")

    assert helper._train_future is None
    assert helper._pending_loss == pytest.approx(2.5)
    assert future.result_calls == 1
    assert observations == [("sync", 2.5), ("clear", 2.5)]


def test_save_adapter_flushes_pending_training_before_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = object.__new__(LocalTrainHelper)
    future = _ControlledFuture(done=True, result=3.5)
    helper._train_future = future
    helper._pending_loss = 0.0
    helper.train_model = object()
    observations: list[tuple[object, str, str, float]] = []

    def fake_save_model(model, *, path: str, name: str) -> str:
        observations.append((model, path, name, helper._pending_loss))
        return f"{path}/{name}"

    monkeypatch.setattr(local_state, "save_model", fake_save_model)

    save_dir = helper.save_adapter("/tmp/adapters", "step_1")

    assert save_dir == "/tmp/adapters/step_1"
    assert helper._train_future is None
    assert helper._pending_loss == pytest.approx(3.5)
    assert future.result_calls == 1
    assert observations == [
        (helper.train_model, "/tmp/adapters", "step_1", 3.5),
    ]
