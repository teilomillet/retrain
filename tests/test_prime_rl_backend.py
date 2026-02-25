"""Tests for retrain.prime_rl_backend."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from retrain.prime_rl_backend import PrimeRLTrainHelper


class _FakeTrainingSample:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeTrainingBatch:
    def __init__(self, examples, step, run_idx=None):
        self.examples = examples
        self.step = step
        self.run_idx = run_idx


class _FakeSender:
    def __init__(self):
        self.sent: list[_FakeTrainingBatch] = []

    def send(self, batch):
        self.sent.append(batch)

    def close(self):
        return None


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, sample_payload):
        self.sample_payload = sample_payload
        self.calls: list[tuple[str, dict]] = []

    def post(self, url, json, timeout=120):
        self.calls.append((url, dict(json)))
        if url.endswith("/v1/chat/completions/tokens"):
            return _FakeResponse(self.sample_payload)
        if url.endswith("/update_weights"):
            return _FakeResponse({"status": "ok"})
        raise AssertionError(f"Unexpected URL: {url}")


def _install_fake_prime_rl(monkeypatch):
    sender = _FakeSender()

    prime_rl = ModuleType("prime_rl")
    prime_rl_configs = ModuleType("prime_rl.configs")
    prime_rl_transport = ModuleType("prime_rl.transport")
    prime_rl_configs_shared = ModuleType("prime_rl.configs.shared")

    prime_rl_configs_shared.FileSystemTransportConfig = lambda: SimpleNamespace(
        type="filesystem"
    )
    prime_rl_configs_shared.ZMQTransportConfig = lambda **kwargs: SimpleNamespace(
        type="zmq", **kwargs
    )

    prime_rl_transport.TrainingSample = _FakeTrainingSample
    prime_rl_transport.TrainingBatch = _FakeTrainingBatch
    prime_rl_transport.setup_training_batch_sender = lambda output_dir, transport: sender

    monkeypatch.setitem(sys.modules, "prime_rl", prime_rl)
    monkeypatch.setitem(sys.modules, "prime_rl.configs", prime_rl_configs)
    monkeypatch.setitem(sys.modules, "prime_rl.configs.shared", prime_rl_configs_shared)
    monkeypatch.setitem(sys.modules, "prime_rl.transport", prime_rl_transport)
    return sender


def test_sample_parses_prime_tokens_payload(monkeypatch, tmp_path):
    _install_fake_prime_rl(monkeypatch)
    helper = PrimeRLTrainHelper(
        model_name="m",
        output_dir=str(tmp_path),
        inference_url="http://localhost:8000",
    )
    session = _FakeSession(
        {
            "choices": [
                {
                    "tokens": {
                        "completion_ids": [101, 102],
                        "completion_logprobs": [-0.1, -0.2],
                    }
                }
            ]
        }
    )
    helper._session = session

    out = helper.sample(
        prompt_ids_list=[[1, 2, 3]],
        num_samples=1,
        max_tokens=32,
        temperature=0.7,
        top_p=0.95,
    )
    assert out == [[([101, 102], [-0.1, -0.2])]]
    assert session.calls and session.calls[0][0].endswith("/v1/chat/completions/tokens")


def test_train_step_sends_training_batch(monkeypatch, tmp_path):
    sender = _install_fake_prime_rl(monkeypatch)
    helper = PrimeRLTrainHelper(
        model_name="m",
        output_dir=str(tmp_path),
        inference_url="http://localhost:8000",
    )
    helper.checkpoint("step_0")
    loss = helper.train_step(
        all_tokens=[[10, 11, 12]],
        all_logprobs=[[0.0, -0.3, -0.2]],
        all_advantages=[[0.0, 1.25, 1.25]],
        lr=1e-4,
        weight_decay=0.0,
    )
    assert loss == 0.0
    assert len(sender.sent) == 1
    batch = sender.sent[0]
    assert batch.step == 0
    assert len(batch.examples) == 1
    sample = batch.examples[0]
    assert sample.prompt_ids == [10]
    assert sample.completion_ids == [11, 12]
    assert sample.completion_logprobs == [-0.3, -0.2]
    assert sample.advantage == pytest.approx(1.25)


def test_train_step_rejects_non_uniform_advantages_by_default(monkeypatch, tmp_path):
    _install_fake_prime_rl(monkeypatch)
    helper = PrimeRLTrainHelper(
        model_name="m",
        output_dir=str(tmp_path),
        inference_url="http://localhost:8000",
    )
    with pytest.raises(RuntimeError, match="non-uniform token advantages"):
        helper.train_step(
            all_tokens=[[10, 11, 12]],
            all_logprobs=[[0.0, -0.3, -0.2]],
            all_advantages=[[0.0, 1.0, 2.0]],
            lr=1e-4,
            weight_decay=0.0,
        )


def test_train_step_can_aggregate_advantages_when_relaxed(monkeypatch, tmp_path):
    sender = _install_fake_prime_rl(monkeypatch)
    helper = PrimeRLTrainHelper(
        model_name="m",
        output_dir=str(tmp_path),
        inference_url="http://localhost:8000",
        strict_advantages=False,
    )
    helper.train_step(
        all_tokens=[[10, 11, 12]],
        all_logprobs=[[0.0, -0.3, -0.2]],
        all_advantages=[[0.0, 1.0, 3.0]],
        lr=1e-4,
        weight_decay=0.0,
    )
    sample = sender.sent[0].examples[0]
    assert sample.advantage == pytest.approx(2.0)


def test_checkpoint_syncs_latest_stable_broadcast_step(monkeypatch, tmp_path):
    _install_fake_prime_rl(monkeypatch)
    (tmp_path / "broadcasts" / "step_3").mkdir(parents=True)
    (tmp_path / "broadcasts" / "step_3" / "STABLE").touch()

    helper = PrimeRLTrainHelper(
        model_name="m",
        output_dir=str(tmp_path),
        inference_url="http://localhost:8000",
        sync_wait_s=0,
    )
    session = _FakeSession({"choices": []})
    helper._session = session

    helper.checkpoint("step_3")
    assert helper._last_loaded_step == 3
    urls = [u for u, _ in session.calls]
    assert any(u.endswith("/update_weights") for u in urls)
    call = [c for c in session.calls if c[0].endswith("/update_weights")][0]
    assert Path(call[1]["weight_dir"]) == (tmp_path / "broadcasts" / "step_3")

