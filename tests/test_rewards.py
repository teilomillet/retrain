"""Tests for retrain.rewards — extract_boxed and BoxedMathReward."""

import pytest

from retrain.rewards import BoxedMathReward, extract_boxed


class TestExtractBoxed:
    def test_simple(self):
        assert extract_boxed(r"The answer is \boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_deeply_nested(self):
        assert extract_boxed(r"\boxed{a{b{c}d}e}") == "a{b{c}d}e"

    def test_multiple_takes_last(self):
        assert extract_boxed(r"\boxed{wrong} then \boxed{right}") == "right"

    def test_no_boxed(self):
        assert extract_boxed("no boxed here") == ""

    def test_empty_boxed(self):
        assert extract_boxed(r"\boxed{}") == ""

    def test_unclosed_brace(self):
        # Unclosed brace — should still extract what's inside
        result = extract_boxed(r"\boxed{abc")
        assert result == "abc"

    def test_whitespace_stripping(self):
        assert extract_boxed(r"\boxed{  hello  }") == "hello"

    def test_empty_string(self):
        assert extract_boxed("") == ""

    def test_boxed_at_end_no_content(self):
        assert extract_boxed(r"\boxed{") == ""


class TestBoxedMathReward:
    def setup_method(self):
        self.reward = BoxedMathReward()

    def test_correct(self):
        assert self.reward.score(r"Answer is \boxed{42}", "42") == 1.0

    def test_incorrect(self):
        assert self.reward.score(r"Answer is \boxed{41}", "42") == 0.0

    def test_no_boxed(self):
        assert self.reward.score("no answer", "42") == 0.0

    def test_whitespace_match(self):
        assert self.reward.score(r"\boxed{ 42 }", " 42 ") == 1.0

    def test_latex_expression(self):
        ref = r"\frac{1}{2}"
        assert self.reward.score(rf"\boxed{{{ref}}}", ref) == 1.0
