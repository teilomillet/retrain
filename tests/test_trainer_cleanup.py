"""Tests for trainer cleanup on exceptional exits."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from retrain.config import TrainConfig
from retrain import trainer as trainer_mod
from retrain.flow import TraceIssue, TraceResult


class _FakeLogger:
    instances: list["_FakeLogger"] = []

    def __init__(self, path: str, *args, **kwargs) -> None:
        self.path = path
        self.closed = False
        _FakeLogger.instances.append(self)

    def log(self, payload) -> None:
        _ = payload

    def close(self) -> None:
        self.closed = True


class _FakeBackend:
    def __init__(self) -> None:
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeExample:
    def __init__(self) -> None:
        self.prompt = "prompt"
        self.reference = "reference"
        self.task = "task"
        self.info = {}


class _FakeRegistry:
    def create(self, *args, **kwargs):
        _ = args, kwargs
        return SimpleNamespace(load=lambda: [_FakeExample()])


def test_train_shuts_down_backend_when_flow_trace_fails(tmp_path, monkeypatch) -> None:
    backend = _FakeBackend()

    class _BadFlow:
        def __init__(self, backend: _FakeBackend) -> None:
            self.backend_capabilities = SimpleNamespace(
                reports_sync_loss=True,
                preserves_token_advantages=True,
                supports_checkpoint_resume=True,
                resume_runtime_dependent=False,
            )
            self.backend_capability_source = "builtin"
            self.backend = backend

        def trace(self) -> TraceResult:
            return TraceResult(
                issues=[
                    TraceIssue(
                        severity="error",
                        category="compat",
                        message="bad flow",
                    )
                ],
            )

    monkeypatch.setattr(
        trainer_mod,
        "build_flow",
        lambda *args, **kwargs: _BadFlow(backend),
    )
    monkeypatch.setattr(trainer_mod, "_print_config_summary", lambda config: None)
    config = TrainConfig(
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapter"),
        max_steps=1,
    )

    with pytest.raises(ValueError, match="bad flow"):
        trainer_mod.train(config)

    assert backend.shutdown_called is True


def test_train_closes_loggers_when_tokenizer_load_fails(tmp_path, monkeypatch) -> None:
    _FakeLogger.instances.clear()
    monkeypatch.setattr(trainer_mod, "JsonlLogger", _FakeLogger)
    monkeypatch.setattr(
        trainer_mod,
        "get_registry",
        lambda name: _FakeRegistry() if name == "data_source" else None,
    )
    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(trainer_mod, "_print_config_summary", lambda config: None)
    monkeypatch.setattr(
        trainer_mod,
        "_print_backend_capability_summary",
        lambda *args, **kwargs: None,
    )

    backend = _FakeBackend()
    flow = SimpleNamespace(
        backend=backend,
        backend_capabilities=SimpleNamespace(
            reports_sync_loss=True,
            preserves_token_advantages=True,
            supports_checkpoint_resume=True,
            resume_runtime_dependent=False,
        ),
        backend_capability_source="builtin",
        planning_detector=object(),
        sepa_controller=SimpleNamespace(state_dict=lambda: {}),
        backpressure=object(),
        needs_planning=False,
        uses_sepa_controller=False,
    )
    config = TrainConfig(
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapter"),
        max_steps=1,
    )

    with pytest.raises(RuntimeError, match="boom"):
        trainer_mod.train(config, flow=flow)

    assert len(_FakeLogger.instances) == 3
    assert all(logger.closed for logger in _FakeLogger.instances)
    assert backend.shutdown_called is True
