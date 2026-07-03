"""Tests for the optional Unsloth runtime boundary."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from retrain.backends.unsloth.runtime import load_fast_language_model


def test_load_fast_language_model_reports_malformed_unsloth_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "unsloth", ModuleType("unsloth"))

    with pytest.raises(ImportError, match="FastLanguageModel") as exc_info:
        load_fast_language_model()

    assert isinstance(exc_info.value.__cause__, AttributeError)
