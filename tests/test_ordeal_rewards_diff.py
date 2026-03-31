"""Ordeal property tests for retrain.rewards and retrain.diff.

Mined invariants: extract_boxed is idempotent with len(output) <= len(input).
Fuzz tests for crash safety on diff formatting functions.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
from hypothesis import given, settings
from ordeal.auto import fuzz
from ordeal.quickcheck import quickcheck

from retrain.diff import DiffResult, format_diff, format_time
from retrain.rewards import extract_boxed


# ═══════════════════════════════════════════
# extract_boxed Properties (mined by ordeal)
# ═══════════════════════════════════════════


class TestExtractBoxedProperties:
    @quickcheck
    def test_returns_string(self, text: str) -> None:
        assert isinstance(extract_boxed(text), str)

    @quickcheck
    def test_deterministic(self, text: str) -> None:
        assert extract_boxed(text) == extract_boxed(text)

    @quickcheck
    def test_idempotent(self, text: str) -> None:
        """Extracting twice gives the same result as extracting once."""
        assert extract_boxed(extract_boxed(text)) == extract_boxed(text)

    @quickcheck
    def test_output_not_longer_than_input(self, text: str) -> None:
        """Output is always a substring or shorter than input."""
        assert len(extract_boxed(text)) <= len(text)

    def test_extracts_boxed_content(self) -> None:
        assert extract_boxed(r"The answer is \boxed{42}") == "42"

    def test_extracts_last_boxed(self) -> None:
        assert extract_boxed(r"\boxed{1} then \boxed{2}") == "2"

    def test_no_boxed_returns_empty(self) -> None:
        assert extract_boxed("no boxed content here") == ""

    def test_empty_input(self) -> None:
        assert extract_boxed("") == ""

    def test_nested_braces(self) -> None:
        result = extract_boxed(r"\boxed{x^{2}}")
        assert "x" in result

    @given(
        inner=st.text(
            alphabet=st.characters(
                categories=("L", "N", "P"),
                exclude_characters=("\\", "{", "}"),
            ),
            min_size=1,
            max_size=20,
        )
    )
    def test_roundtrip_simple_content(self, inner: str) -> None:
        """Simple content (no braces/backslash) roundtrips through boxed."""
        text = rf"\boxed{{{inner}}}"
        assert extract_boxed(text) == inner


# ═══════════════════════════════════════════
# format_diff Properties
# ═══════════════════════════════════════════


def _make_diff_result(
    cr_a: float = 0.5, cr_b: float = 0.6, steps: int = 100
) -> DiffResult:
    return DiffResult(
        label_a="run_a",
        label_b="run_b",
        final_a={"correct_rate": cr_a},
        final_b={"correct_rate": cr_b},
        wall_time_a=float(steps),
        wall_time_b=float(steps),
        steps_a=steps,
        steps_b=steps,
        curve_a=[cr_a] * 5,
        curve_b=[cr_b] * 5,
    )


class TestFormatDiffProperties:
    @given(
        cr_a=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        cr_b=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_returns_non_empty_string(self, cr_a: float, cr_b: float) -> None:
        result = format_diff(_make_diff_result(cr_a, cr_b))
        assert isinstance(result, str)
        assert len(result) > 0

    @given(
        cr_a=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        cr_b=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_deterministic(self, cr_a: float, cr_b: float) -> None:
        dr = _make_diff_result(cr_a, cr_b)
        assert format_diff(dr) == format_diff(dr)

    def test_contains_labels(self) -> None:
        result = format_diff(_make_diff_result())
        assert "run_a" in result
        assert "run_b" in result


# ═══════════════════════════════════════════
# format_time Properties
# ═══════════════════════════════════════════


class TestFormatTimeProperties:
    @given(
        seconds=st.floats(
            min_value=0.0,
            max_value=1e8,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    def test_returns_non_empty_string(self, seconds: float) -> None:
        result = format_time(seconds)
        assert isinstance(result, str)
        assert len(result) > 0

    @given(
        seconds=st.floats(
            min_value=0.0,
            max_value=1e8,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    def test_deterministic(self, seconds: float) -> None:
        assert format_time(seconds) == format_time(seconds)

    def test_seconds_format(self) -> None:
        result = format_time(30.0)
        assert "s" in result

    def test_minutes_format(self) -> None:
        result = format_time(120.0)
        assert "m" in result or "min" in result or ":" in result

    def test_zero(self) -> None:
        result = format_time(0.0)
        assert isinstance(result, str)


# ═══════════════════════════════════════════
# Fuzz: Crash Safety
# ═══════════════════════════════════════════


class TestFuzzCrashSafety:
    def test_extract_boxed_no_crash(self) -> None:
        result = fuzz(extract_boxed, max_examples=100)
        assert result.passed, result.summary()

    def test_format_time_no_crash(self) -> None:
        result = fuzz(format_time, max_examples=100)
        assert result.passed, result.summary()
