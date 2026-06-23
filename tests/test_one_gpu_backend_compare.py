"""Tests for the one-GPU backend comparison helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_helper():
    path = Path(__file__).resolve().parents[1] / "scripts" / "one_gpu_backend_compare.py"
    spec = importlib.util.spec_from_file_location("one_gpu_backend_compare", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(*, steps: int, reload_calls: float | None, reload_failures: float | None):
    aggregates = {}
    if reload_calls is not None:
        aggregates["engine_adapter_reload_calls"] = SimpleNamespace(mean=reload_calls)
    if reload_failures is not None:
        aggregates["engine_adapter_reload_failures"] = SimpleNamespace(
            mean=reload_failures
        )
    return SimpleNamespace(
        runs=[SimpleNamespace(final_correct_rate=1.0, steps=steps)],
        aggregates=aggregates,
    )


def test_adapter_reload_gate_requires_reload_after_multi_step_run() -> None:
    helper = _load_helper()

    gates = helper._quality_gates(
        engine="vllm",
        summary=_summary(steps=2, reload_calls=0.0, reload_failures=0.0),
        min_correct_rate=0.0,
        require_token_native=False,
        require_adapter_reload=True,
    )

    assert gates["adapter_reload"]["passed"] is False
    assert gates["adapter_reload"]["min_calls"] == 1


def test_adapter_reload_gate_passes_when_reload_succeeds() -> None:
    helper = _load_helper()

    gates = helper._quality_gates(
        engine="sglang",
        summary=_summary(steps=2, reload_calls=1.0, reload_failures=0.0),
        min_correct_rate=0.0,
        require_token_native=False,
        require_adapter_reload=True,
    )

    assert gates["adapter_reload"]["passed"] is True


def test_adapter_reload_gate_applies_to_trtllm() -> None:
    helper = _load_helper()

    gates = helper._quality_gates(
        engine="trtllm",
        summary=_summary(steps=2, reload_calls=0.0, reload_failures=0.0),
        min_correct_rate=0.0,
        require_token_native=False,
        require_adapter_reload=True,
    )

    assert gates["adapter_reload"]["passed"] is False


def test_parse_engines_accepts_trtllm() -> None:
    helper = _load_helper()

    assert helper._parse_engines("pytorch,trtllm") == ["pytorch", "trtllm"]


def test_adapter_reload_gate_allows_single_step_smoke_without_reload() -> None:
    helper = _load_helper()

    gates = helper._quality_gates(
        engine="vllm",
        summary=_summary(steps=1, reload_calls=0.0, reload_failures=0.0),
        min_correct_rate=0.0,
        require_token_native=False,
        require_adapter_reload=True,
    )

    assert gates["adapter_reload"]["passed"] is True
    assert gates["adapter_reload"]["min_calls"] == 0
