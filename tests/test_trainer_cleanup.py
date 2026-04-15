"""Tests for trainer cleanup on exceptional exits."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from retrain.config import TrainConfig
from retrain import trainer as trainer_mod


class _FakeLogger:
    instances: list["_FakeLogger"] = []

    def __init__(self, path: str, *args, **kwargs) -> None:  # noqa: ARG002
        self.path = path
        self.closed = False
        _FakeLogger.instances.append(self)

    def log(self, payload) -> None:  # noqa: ANN001, D401
        _ = payload

    def close(self) -> None:
        self.closed = True


class _FakeExample:
    def __init__(self) -> None:
        self.prompt = "prompt"
        self.reference = "reference"
        self.task = "task"
        self.info = {}


class _FakeRegistry:
    def create(self, *args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
        return SimpleNamespace(load=lambda: [_FakeExample()])


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
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),  # noqa: ARG005
    )
    monkeypatch.setattr(trainer_mod, "_print_config_summary", lambda config: None)
    monkeypatch.setattr(
        trainer_mod,
        "_print_backend_capability_summary",
        lambda *args, **kwargs: None,  # noqa: ARG005
    )

    flow = SimpleNamespace(
        backend=object(),
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
