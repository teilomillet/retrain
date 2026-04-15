"""Tests for retrain.planning detectors."""

from retrain.advantages import identify_planning_tokens
from retrain.planning import RegexPlanningDetector


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
