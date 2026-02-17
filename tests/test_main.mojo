"""Tests for main.mojo utility functions and the composable advantage pipeline.

Covers:
    - extract_boxed: LaTeX \\boxed{...} extraction with nested braces
    - grade_answer: whitespace-tolerant string matching
    - get_reward: end-to-end binary reward
    - compute_composable_advantages (H10): full pipeline integration
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from python import Python, PythonObject
from math import abs

from src.main import (
    extract_boxed,
    grade_answer,
    get_reward,
    compute_composable_advantages,
    AdvantageResult,
    default_strategic_grams,
)
from src.advantages import EntropyStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


fn assert_list_approx_eq(
    xs: List[Float64], ys: List[Float64], tol: Float64 = 1e-6
) raises:
    assert_equal(len(xs), len(ys))
    for i in range(len(xs)):
        assert_almost_equal(xs[i], ys[i], atol=tol)


fn all_finite(xs: List[Float64]) -> Bool:
    for i in range(len(xs)):
        var v = xs[i]
        # Check not NaN and not Inf
        if v != v:
            return False
        if v > 1e300 or v < -1e300:
            return False
    return True


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
# default_strategic_grams
# ---------------------------------------------------------------------------


fn test_default_strategic_grams_nonempty() raises:
    """Default strategic grams should be non-empty."""
    var grams = default_strategic_grams()
    assert_true(len(grams) > 0, "Should have at least one strategic gram")
    # Check a few known grams
    var found_think = False
    var found_check = False
    for i in range(len(grams)):
        if grams[i] == "let me think":
            found_think = True
        if grams[i] == "let me check":
            found_check = True
    assert_true(found_think, "'let me think' should be in default grams")
    assert_true(found_check, "'let me check' should be in default grams")


# ---------------------------------------------------------------------------
# H10: compute_composable_advantages pipeline
# ---------------------------------------------------------------------------


fn test_pipeline_none_mode_single_completion() raises:
    """Single completion -> advantage=0 (no group contrast). H10."""
    var rewards = List[Float64]()
    rewards.append(1.0)

    var tokens = List[List[Int]]()
    var t = List[Int]()
    t.append(1)
    t.append(2)
    t.append(3)
    tokens.append(t^)

    var logprobs = List[List[Float64]]()
    var lp = List[Float64]()
    lp.append(-0.5)
    lp.append(-1.0)
    lp.append(-0.3)
    logprobs.append(lp^)

    var result = compute_composable_advantages(
        rewards, tokens, logprobs, Python.none(),
        advantage_mode="maxrl",
        transform_mode="none",
    )

    assert_equal(len(result.token_advs), 1)
    assert_equal(len(result.token_advs[0]), 3)
    # Single completion: mean = reward -> advantage = 0 for all tokens
    for j in range(3):
        assert_almost_equal(result.token_advs[0][j], 0.0, atol=1e-4)


fn test_pipeline_none_mode_binary() raises:
    """K=1 correct of N=4, transform_mode=none -> uniform token advantages. H10."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)
    rewards.append(0.0)
    rewards.append(0.0)

    var tokens = List[List[Int]]()
    var logprobs = List[List[Float64]]()
    for i in range(4):
        var t = List[Int]()
        t.append(10)
        t.append(20)
        tokens.append(t^)
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.0)
        logprobs.append(lp^)

    var result = compute_composable_advantages(
        rewards, tokens, logprobs, Python.none(),
        advantage_mode="maxrl",
        transform_mode="none",
    )

    # 4 completions, each with 2 tokens
    assert_equal(len(result.token_advs), 4)
    for i in range(4):
        assert_equal(len(result.token_advs[i]), 2)
        assert_true(all_finite(result.token_advs[i]), "All advantages should be finite")

    # Correct completion (index 0): advantage ~ 3.0 (MaxRL formula) for all tokens
    assert_almost_equal(result.token_advs[0][0], 3.0, atol=1e-3)
    assert_almost_equal(result.token_advs[0][1], 3.0, atol=1e-3)

    # Incorrect completions: advantage ~ -1.0 for all tokens
    assert_almost_equal(result.token_advs[1][0], -1.0, atol=1e-3)
    assert_almost_equal(result.token_advs[1][1], -1.0, atol=1e-3)

    # has_stats should be False for "none" mode
    assert_false(result.has_stats, "none mode should not have entropy stats")


fn test_pipeline_grpo_none_mode() raises:
    """GRPO + none: advantages = r - mean(r) broadcast to all tokens."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)

    var tokens = List[List[Int]]()
    var logprobs = List[List[Float64]]()
    for i in range(2):
        var t = List[Int]()
        t.append(10)
        t.append(20)
        t.append(30)
        tokens.append(t^)
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.0)
        lp.append(-0.3)
        logprobs.append(lp^)

    var result = compute_composable_advantages(
        rewards, tokens, logprobs, Python.none(),
        advantage_mode="grpo",
        transform_mode="none",
    )

    assert_equal(len(result.token_advs), 2)
    # GRPO: mean = 0.5, correct = 0.5, incorrect = -0.5
    for j in range(3):
        assert_almost_equal(result.token_advs[0][j], 0.5, atol=1e-6)
        assert_almost_equal(result.token_advs[1][j], -0.5, atol=1e-6)


fn test_pipeline_gtpo_mode() raises:
    """GTPO mode (no hicra/sepa): entropy-weighted, no planning mask needed."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)
    rewards.append(0.0)
    rewards.append(0.0)

    var tokens = List[List[Int]]()
    var logprobs = List[List[Float64]]()
    for i in range(4):
        var t = List[Int]()
        t.append(10)
        t.append(20)
        t.append(30)
        tokens.append(t^)
        # Varying logprobs -> varying entropy proxy
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.5)
        lp.append(-3.0)
        logprobs.append(lp^)

    var result = compute_composable_advantages(
        rewards, tokens, logprobs, Python.none(),
        advantage_mode="maxrl",
        transform_mode="gtpo",
        gtpo_beta=0.5,
    )

    assert_equal(len(result.token_advs), 4)
    for i in range(4):
        assert_equal(len(result.token_advs[i]), 3)
        assert_true(all_finite(result.token_advs[i]), "All advantages should be finite")

    # has_stats should be True for GTPO mode
    assert_true(result.has_stats, "GTPO mode should have entropy stats")

    # Correct completion: high-entropy token (index 2, entropy=3.0) should have
    # higher advantage magnitude than low-entropy token (index 0, entropy=0.5)
    assert_true(
        result.token_advs[0][2] > result.token_advs[0][0],
        "High-entropy token should have larger positive advantage"
    )

    # Incorrect completion: all negative, but high-entropy more negative
    assert_true(
        result.token_advs[1][2] < result.token_advs[1][0],
        "High-entropy token should have more negative advantage"
    )


fn test_pipeline_empty_inputs() raises:
    """Empty input group -> empty output."""
    var result = compute_composable_advantages(
        List[Float64](),
        List[List[Int]](),
        List[List[Float64]](),
        Python.none(),
        advantage_mode="maxrl",
        transform_mode="none",
    )
    assert_equal(len(result.token_advs), 0)


fn test_pipeline_all_same_reward() raises:
    """All same reward -> zero advantages for all tokens."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(1.0)
    rewards.append(1.0)

    var tokens = List[List[Int]]()
    var logprobs = List[List[Float64]]()
    for i in range(3):
        var t = List[Int]()
        t.append(10)
        tokens.append(t^)
        var lp = List[Float64]()
        lp.append(-0.5)
        logprobs.append(lp^)

    var result = compute_composable_advantages(
        rewards, tokens, logprobs, Python.none(),
        advantage_mode="maxrl",
        transform_mode="none",
    )

    for i in range(3):
        assert_almost_equal(result.token_advs[i][0], 0.0, atol=1e-4)


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

    # default_strategic_grams
    test_default_strategic_grams_nonempty()

    # H10: pipeline
    test_pipeline_none_mode_single_completion()
    test_pipeline_none_mode_binary()
    test_pipeline_grpo_none_mode()
    test_pipeline_gtpo_mode()
    test_pipeline_empty_inputs()
    test_pipeline_all_same_reward()

    print("All 27 main tests passed!")
