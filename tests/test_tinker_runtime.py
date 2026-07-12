"""Tests for the optional Tinker SDK runtime boundary."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from retrain.backends.tinker import runtime as tinker_runtime
from retrain.backends.tinker.runtime import (
    load_tensor_data,
    load_tinker,
    load_tinker_types,
)


class _FakeTensorData:
    @staticmethod
    def from_torch(tensor: object) -> object:
        return {"tensor": tensor}


def _module(name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_tinker_runtime_loads_optional_sdk_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_client = object()
    model_input = SimpleNamespace(from_ints=lambda tokens: tuple(tokens))
    modules = {
        "tinker": _module("tinker", ServiceClient=service_client),
        "tinker.types": _module("tinker.types", ModelInput=model_input),
        "tinker.types.tensor_data": _module(
            "tinker.types.tensor_data",
            TensorData=_FakeTensorData,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert load_tinker().ServiceClient is service_client
    assert load_tinker_types().ModelInput.from_ints([1, 2]) == (1, 2)
    assert load_tensor_data().from_torch("x") == {"tensor": "x"}


def test_tinker_runtime_preserves_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ModuleNotFoundError("No module named 'tinker'")

    def _missing_tinker(name: str) -> ModuleType:
        assert name == "tinker"
        raise expected

    monkeypatch.setattr(
        tinker_runtime,
        "importlib",
        SimpleNamespace(import_module=_missing_tinker),
    )

    with pytest.raises(ModuleNotFoundError) as exc_info:
        load_tinker()
    assert exc_info.value is expected
