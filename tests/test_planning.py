"""Tests for retrain.planning detectors."""

import importlib

import pytest

from retrain.advantages import identify_planning_tokens
from retrain.planning import RegexPlanningDetector, SemanticPlanningDetector


def test_regex_planning_detector_matches_identify_helper() -> None:
    detector = RegexPlanningDetector(
        ["let me think", "that's not right", "notice that"]
    )
    token_groups = [
        ["let", "me", "think", "about", "this"],
        ["that's", "not", "right,", "let", "me", "check."],
        ["ordinary", "tokens"],
        ["\u2581notice", "\u2581that", "\u2581works"],
    ]

    for tokens in token_groups:
        assert detector.detect(tokens) == identify_planning_tokens(
            tokens,
            detector._grams,  # type: ignore[attr-defined]
        )


def test_semantic_planning_detector_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def import_module(name: str, package: str | None = None) -> object:
        if name == "sentence_transformers":
            raise ModuleNotFoundError("No module named 'sentence_transformers'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)

    with pytest.raises(ImportError, match=r"retrain\[semantic\]") as exc_info:
        SemanticPlanningDetector()

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
