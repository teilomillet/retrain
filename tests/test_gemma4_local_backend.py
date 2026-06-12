"""Focused tests for Gemma4 local-backend integration helpers."""

from __future__ import annotations

from retrain.inference_engine.pytorch_engine import PyTorchEngine


class _ExistingModel:
    def __init__(self) -> None:
        self.to_calls: list[str] = []

    def to(self, device: str):
        self.to_calls.append(device)
        return self


def test_pytorch_engine_moves_existing_model_to_requested_device():
    model = _ExistingModel()

    engine = PyTorchEngine(
        model_name="unused",
        device="cuda:7",
        peft_config=None,
        dtype=None,
        existing_model=model,
    )

    assert engine.model is model
    assert model.to_calls == ["cuda:7"]
