"""Tests for reward.mojo: RewardFn trait, BoxedMathReward, extract_boxed, grade_answer.

Covers:
    - extract_boxed: LaTeX \\boxed{...} extraction with nested braces
    - grade_answer: whitespace-tolerant string matching
    - get_reward: end-to-end binary reward
    - BoxedMathReward: trait conformance
    - AlwaysOneReward: custom RewardFn compiles and works
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal

from src.reward import (
    RewardFn,
    BoxedMathReward,
    extract_boxed,
    grade_answer,
    get_reward,
)


# ---------------------------------------------------------------------------
# extract_boxed
# ---------------------------------------------------------------------------


fn test_extract_boxed_basic() raises:
    """Simple \\boxed{42}."""
    var result = extract_boxed("The answer is \\boxed{42}")
    assert_equal(result, "42")


fn test_extract_boxed_nested() raises:
    """Nested braces: \\boxed{\\frac{1}{2}}."""
    var result = extract_boxed("\\boxed{\\frac{1}{2}}")
    assert_equal(result, "\\frac{1}{2}")


fn test_extract_boxed_deeply_nested() raises:
    """Deeply nested: \\boxed{a{b{c}d}e}."""
    var result = extract_boxed("\\boxed{a{b{c}d}e}")
    assert_equal(result, "a{b{c}d}e")


fn test_extract_boxed_no_boxed() raises:
    """No \\boxed -> empty string."""
    var result = extract_boxed("There is no boxed answer here.")
    assert_equal(result, "")


fn test_extract_boxed_empty() raises:
    """Empty input -> empty string."""
    var result = extract_boxed("")
    assert_equal(result, "")


fn test_extract_boxed_empty_content() raises:
    """\\boxed{} -> empty string."""
    var result = extract_boxed("\\boxed{}")
    assert_equal(result, "")


fn test_extract_boxed_multiple_takes_last() raises:
    """Multiple \\boxed -> takes the last one (rfind)."""
    var result = extract_boxed("\\boxed{wrong} then \\boxed{right}")
    assert_equal(result, "right")


fn test_extract_boxed_with_whitespace() raises:
    """Strips whitespace from extracted content."""
    var result = extract_boxed("\\boxed{  42  }")
    assert_equal(result, "42")


fn test_extract_boxed_text_after() raises:
    """Text after \\boxed{...} doesn't interfere."""
    var result = extract_boxed("\\boxed{42} is correct.")
    assert_equal(result, "42")


# ---------------------------------------------------------------------------
# grade_answer
# ---------------------------------------------------------------------------


fn test_grade_exact_match() raises:
    """Exact match -> True."""
    assert_true(grade_answer("42", "42"), "Exact match should return True")


fn test_grade_whitespace_stripping() raises:
    """Whitespace differences -> still matches."""
    assert_true(grade_answer("  42  ", "42"), "Should strip whitespace")
    assert_true(grade_answer("42", "  42  "), "Should strip whitespace")


fn test_grade_different() raises:
    """Different answers -> False."""
    assert_false(grade_answer("42", "43"), "Different answers should return False")


fn test_grade_empty_both() raises:
    """Both empty -> True."""
    assert_true(grade_answer("", ""), "Both empty should return True")


fn test_grade_one_empty() raises:
    """One empty -> False."""
    assert_false(grade_answer("", "42"), "Empty vs non-empty should return False")
    assert_false(grade_answer("42", ""), "Non-empty vs empty should return False")


fn test_grade_case_sensitive() raises:
    """Grading is case-sensitive."""
    assert_false(grade_answer("ABC", "abc"), "Should be case-sensitive")


# ---------------------------------------------------------------------------
# get_reward
# ---------------------------------------------------------------------------


fn test_get_reward_correct() raises:
    """Correct \\boxed answer -> 1.0."""
    var response = "The answer is \\boxed{42}"
    var reward = get_reward(response, "42")
    assert_almost_equal(reward, 1.0, atol=1e-6)


fn test_get_reward_incorrect() raises:
    """Incorrect \\boxed answer -> 0.0."""
    var response = "The answer is \\boxed{43}"
    var reward = get_reward(response, "42")
    assert_almost_equal(reward, 0.0, atol=1e-6)


fn test_get_reward_no_boxed() raises:
    """No \\boxed in response -> 0.0 (empty extracted != expected)."""
    var response = "I think the answer is 42"
    var reward = get_reward(response, "42")
    assert_almost_equal(reward, 0.0, atol=1e-6)


fn test_get_reward_nested_latex() raises:
    """Nested LaTeX in boxed -> correct extraction."""
    var response = "\\boxed{\\frac{1}{2}}"
    var reward = get_reward(response, "\\frac{1}{2}")
    assert_almost_equal(reward, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# BoxedMathReward trait conformance
# ---------------------------------------------------------------------------


fn test_boxed_math_reward_score() raises:
    """BoxedMathReward.score() matches get_reward()."""
    var r = BoxedMathReward()
    assert_almost_equal(r.score("\\boxed{42}", "42"), 1.0, atol=1e-6)
    assert_almost_equal(r.score("\\boxed{43}", "42"), 0.0, atol=1e-6)
    assert_almost_equal(r.score("no boxed", "42"), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Custom RewardFn: AlwaysOneReward
# ---------------------------------------------------------------------------


struct AlwaysOneReward(RewardFn):
    """Trivial reward that always returns 1.0 — verifies trait conformance."""

    fn __init__(out self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass

    fn score(self, response: String, reference: String) -> Float64:
        return 1.0


fn test_always_one_reward() raises:
    """Custom RewardFn compiles and returns expected value."""
    var r = AlwaysOneReward()
    assert_almost_equal(r.score("anything", "anything"), 1.0, atol=1e-6)
    assert_almost_equal(r.score("", ""), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


fn main() raises:
    # extract_boxed
    test_extract_boxed_basic()
    test_extract_boxed_nested()
    test_extract_boxed_deeply_nested()
    test_extract_boxed_no_boxed()
    test_extract_boxed_empty()
    test_extract_boxed_empty_content()
    test_extract_boxed_multiple_takes_last()
    test_extract_boxed_with_whitespace()
    test_extract_boxed_text_after()

    # grade_answer
    test_grade_exact_match()
    test_grade_whitespace_stripping()
    test_grade_different()
    test_grade_empty_both()
    test_grade_one_empty()
    test_grade_case_sensitive()

    # get_reward
    test_get_reward_correct()
    test_get_reward_incorrect()
    test_get_reward_no_boxed()
    test_get_reward_nested_latex()

    # BoxedMathReward
    test_boxed_math_reward_score()

    # Custom RewardFn
    test_always_one_reward()

    print("All 22 reward tests passed!")
