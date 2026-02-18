"""Tests for main.mojo utility functions and the composable advantage pipeline.

Covers:
    - extract_boxed: LaTeX \\boxed{...} extraction with nested braces
    - grade_answer: whitespace-tolerant string matching
    - get_reward: end-to-end binary reward
    - compute_composable_advantages (H10): full pipeline integration
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from math import abs

from src.reward import extract_boxed, grade_answer, get_reward
from src.main import compute_composable_advantages, default_strategic_grams
from src.advantage_fns import AdvantageResult
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


fn _zero_masks(n_seqs: Int, n_tokens: Int) -> List[List[Int]]:
    """Helper: create all-zero planning masks."""
    var masks = List[List[Int]]()
    for _ in range(n_seqs):
        masks.append(List[Int](length=n_tokens, fill=0))
    return masks^


fn test_pipeline_none_mode_single_completion() raises:
    """Single completion -> advantage=0 (no group contrast). H10."""
    var rewards = List[Float64]()
    rewards.append(1.0)

    var logprobs = List[List[Float64]]()
    var lp = List[Float64]()
    lp.append(-0.5)
    lp.append(-1.0)
    lp.append(-0.3)
    logprobs.append(lp^)

    var masks = _zero_masks(1, 3)
    var result = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl",
        transform_mode="none",
    )

    assert_equal(len(result.token_advs), 1)
    assert_equal(len(result.token_advs[0]), 3)
    for j in range(3):
        assert_almost_equal(result.token_advs[0][j], 0.0, atol=1e-4)


fn test_pipeline_none_mode_binary() raises:
    """K=1 correct of N=4, transform_mode=none -> uniform token advantages. H10."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)
    rewards.append(0.0)
    rewards.append(0.0)

    var logprobs = List[List[Float64]]()
    for _ in range(4):
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.0)
        logprobs.append(lp^)

    var masks = _zero_masks(4, 2)
    var result = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl",
        transform_mode="none",
    )

    assert_equal(len(result.token_advs), 4)
    for i in range(4):
        assert_equal(len(result.token_advs[i]), 2)
        assert_true(all_finite(result.token_advs[i]), "All advantages should be finite")

    assert_almost_equal(result.token_advs[0][0], 3.0, atol=1e-3)
    assert_almost_equal(result.token_advs[0][1], 3.0, atol=1e-3)
    assert_almost_equal(result.token_advs[1][0], -1.0, atol=1e-3)
    assert_almost_equal(result.token_advs[1][1], -1.0, atol=1e-3)
    assert_false(result.has_stats, "none mode should not have entropy stats")


fn test_pipeline_grpo_none_mode() raises:
    """GRPO + none: advantages = r - mean(r) broadcast to all tokens."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)

    var logprobs = List[List[Float64]]()
    for _ in range(2):
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.0)
        lp.append(-0.3)
        logprobs.append(lp^)

    var masks = _zero_masks(2, 3)
    var result = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="grpo",
        transform_mode="none",
    )

    assert_equal(len(result.token_advs), 2)
    for j in range(3):
        assert_almost_equal(result.token_advs[0][j], 0.5, atol=1e-6)
        assert_almost_equal(result.token_advs[1][j], -0.5, atol=1e-6)


fn test_pipeline_gtpo_mode() raises:
    """GTPO mode (no hicra/sepa): entropy-weighted, all-zero masks."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)
    rewards.append(0.0)
    rewards.append(0.0)

    var logprobs = List[List[Float64]]()
    for _ in range(4):
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.5)
        lp.append(-3.0)
        logprobs.append(lp^)

    var masks = _zero_masks(4, 3)
    var result = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl",
        transform_mode="gtpo",
        gtpo_beta=0.5,
    )

    assert_equal(len(result.token_advs), 4)
    for i in range(4):
        assert_equal(len(result.token_advs[i]), 3)
        assert_true(all_finite(result.token_advs[i]), "All advantages should be finite")

    assert_true(result.has_stats, "GTPO mode should have entropy stats")
    assert_true(
        result.token_advs[0][2] > result.token_advs[0][0],
        "High-entropy token should have larger positive advantage"
    )
    assert_true(
        result.token_advs[1][2] < result.token_advs[1][0],
        "High-entropy token should have more negative advantage"
    )


fn test_pipeline_empty_inputs() raises:
    """Empty input group -> empty output."""
    var result = compute_composable_advantages(
        List[Float64](),
        List[List[Float64]](),
        List[List[Int]](),
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

    var logprobs = List[List[Float64]]()
    for _ in range(3):
        var lp = List[Float64]()
        lp.append(-0.5)
        logprobs.append(lp^)

    var masks = _zero_masks(3, 1)
    var result = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl",
        transform_mode="none",
    )

    for i in range(3):
        assert_almost_equal(result.token_advs[i][0], 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# H10: HICRA pipeline (gtpo_hicra)
# ---------------------------------------------------------------------------


fn test_pipeline_hicra_amplifies_planning() raises:
    """HICRA amplifies positive advantage, dampens negative at planning tokens."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)

    var logprobs = List[List[Float64]]()
    for _ in range(2):
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.0)
        logprobs.append(lp^)

    # Token 1 is planning in both sequences
    var masks = List[List[Int]]()
    for _ in range(2):
        var m = List[Int]()
        m.append(0)
        m.append(1)
        masks.append(m^)

    var result = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl",
        transform_mode="gtpo_hicra",
        gtpo_beta=0.0,  # disable entropy weighting for predictable values
        hicra_alpha=0.5,
    )

    assert_equal(len(result.token_advs), 2)
    # Seq 0 (reward=1): positive advantage — planning token amplified
    assert_true(
        result.token_advs[0][1] > result.token_advs[0][0],
        "Planning token should have larger positive advantage",
    )
    # Seq 1 (reward=0): negative advantage — planning token dampened
    assert_true(
        result.token_advs[1][1] > result.token_advs[1][0],
        "Planning token should have less negative advantage (dampened)",
    )
    assert_true(result.has_stats, "HICRA mode should have entropy stats")


fn test_pipeline_hicra_exact_values() raises:
    """HICRA with beta=0: exact computation check.

    MaxRL: mean=0.5, denom~0.5, adv[0]~1.0, adv[1]~-1.0
    beta=0 -> uniform token advs = episode advantage
    HICRA alpha=0.5: plan tokens get A + 0.5*|A|
      Seq 0: [1.0, 1.5]   Seq 1: [-1.0, -0.5]
    """
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)

    var logprobs = List[List[Float64]]()
    for _ in range(2):
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.0)
        logprobs.append(lp^)

    var masks = List[List[Int]]()
    for _ in range(2):
        var m = List[Int]()
        m.append(0)
        m.append(1)
        masks.append(m^)

    var result = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl",
        transform_mode="gtpo_hicra",
        gtpo_beta=0.0,
        hicra_alpha=0.5,
    )

    assert_almost_equal(result.token_advs[0][0], 1.0, atol=1e-3)
    assert_almost_equal(result.token_advs[0][1], 1.5, atol=1e-3)
    assert_almost_equal(result.token_advs[1][0], -1.0, atol=1e-3)
    assert_almost_equal(result.token_advs[1][1], -0.5, atol=1e-3)


fn test_pipeline_hicra_alpha_zero_equals_gtpo() raises:
    """HICRA with alpha=0 should equal plain GTPO (no amplification)."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)
    rewards.append(0.0)

    var logprobs = List[List[Float64]]()
    for _ in range(3):
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.5)
        logprobs.append(lp^)

    var masks = List[List[Int]]()
    for _ in range(3):
        var m = List[Int]()
        m.append(0)
        m.append(1)
        masks.append(m^)

    var result_gtpo = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl",
        transform_mode="gtpo",
        gtpo_beta=0.1,
    )
    var result_hicra = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl",
        transform_mode="gtpo_hicra",
        gtpo_beta=0.1,
        hicra_alpha=0.0,
    )

    for i in range(3):
        for j in range(2):
            assert_almost_equal(
                result_gtpo.token_advs[i][j],
                result_hicra.token_advs[i][j],
                atol=1e-10,
            )


# ---------------------------------------------------------------------------
# H10: SEPA pipeline (gtpo_sepa)
# ---------------------------------------------------------------------------


fn test_pipeline_sepa_modifies_advantages() raises:
    """SEPA with lambda>0 produces different advantages than plain GTPO."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)
    rewards.append(0.0)
    rewards.append(0.0)

    var logprobs = List[List[Float64]]()
    for _ in range(4):
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.5)
        lp.append(-3.0)
        logprobs.append(lp^)

    # Token 2 is planning
    var masks = List[List[Int]]()
    for _ in range(4):
        var m = List[Int]()
        m.append(0)
        m.append(0)
        m.append(1)
        masks.append(m^)

    var result_gtpo = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl",
        transform_mode="gtpo",
        gtpo_beta=0.5,
    )
    var result_sepa = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl",
        transform_mode="gtpo_sepa",
        gtpo_beta=0.5,
        sepa_lambda=0.5,
    )

    assert_true(result_sepa.has_stats, "SEPA mode should have entropy stats")
    assert_true(all_finite(result_sepa.token_advs[0]), "SEPA advantages should be finite")
    # SEPA pooling changes exec token entropies -> different advantages
    var differs = False
    for j in range(3):
        if result_sepa.token_advs[0][j] != result_gtpo.token_advs[0][j]:
            differs = True
            break
    assert_true(differs, "SEPA should produce different advantages than plain GTPO")


fn test_pipeline_sepa_lambda_zero_equals_gtpo() raises:
    """SEPA with lambda=0 should equal plain GTPO (no pooling)."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)

    var logprobs = List[List[Float64]]()
    for _ in range(2):
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.0)
        logprobs.append(lp^)

    var masks = List[List[Int]]()
    for _ in range(2):
        var m = List[Int]()
        m.append(0)
        m.append(1)
        masks.append(m^)

    var result_gtpo = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl",
        transform_mode="gtpo",
        gtpo_beta=0.1,
    )
    var result_sepa = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl",
        transform_mode="gtpo_sepa",
        gtpo_beta=0.1,
        sepa_lambda=0.0,
    )

    for i in range(2):
        for j in range(2):
            assert_almost_equal(
                result_gtpo.token_advs[i][j],
                result_sepa.token_advs[i][j],
                atol=1e-10,
            )


# ---------------------------------------------------------------------------
# H10: Edge cases
# ---------------------------------------------------------------------------


fn test_pipeline_none_ignores_masks() raises:
    """transform_mode=none ignores planning masks entirely."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)

    var logprobs = List[List[Float64]]()
    for _ in range(2):
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.0)
        logprobs.append(lp^)

    var masks_zero = _zero_masks(2, 2)
    var masks_one = List[List[Int]]()
    for _ in range(2):
        var m = List[Int]()
        m.append(1)
        m.append(1)
        masks_one.append(m^)

    var result_zero = compute_composable_advantages(
        rewards, logprobs, masks_zero,
        advantage_mode="grpo", transform_mode="none",
    )
    var result_one = compute_composable_advantages(
        rewards, logprobs, masks_one,
        advantage_mode="grpo", transform_mode="none",
    )

    for i in range(2):
        for j in range(2):
            assert_almost_equal(
                result_zero.token_advs[i][j],
                result_one.token_advs[i][j],
                atol=1e-10,
            )


fn test_pipeline_all_planning_tokens() raises:
    """All tokens are planning -> SEPA exec pool is empty, still works."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)

    var logprobs = List[List[Float64]]()
    for _ in range(2):
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.0)
        logprobs.append(lp^)

    # All tokens are planning
    var masks = List[List[Int]]()
    for _ in range(2):
        var m = List[Int]()
        m.append(1)
        m.append(1)
        masks.append(m^)

    var result = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl",
        transform_mode="gtpo_sepa",
        gtpo_beta=0.1,
        sepa_lambda=0.5,
    )

    assert_equal(len(result.token_advs), 2)
    for i in range(2):
        assert_true(all_finite(result.token_advs[i]), "Should handle all-planning gracefully")
    assert_true(result.has_stats, "Should still have entropy stats")


fn test_pipeline_mixed_sequence_lengths() raises:
    """Sequences with different token counts."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)

    var logprobs = List[List[Float64]]()
    var lp1 = List[Float64]()
    lp1.append(-0.5)
    lp1.append(-1.0)
    lp1.append(-0.3)
    logprobs.append(lp1^)
    var lp2 = List[Float64]()
    lp2.append(-0.5)
    logprobs.append(lp2^)

    var masks = List[List[Int]]()
    masks.append(List[Int](length=3, fill=0))
    masks.append(List[Int](length=1, fill=0))

    var result = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="grpo",
        transform_mode="none",
    )

    assert_equal(len(result.token_advs), 2)
    assert_equal(len(result.token_advs[0]), 3)
    assert_equal(len(result.token_advs[1]), 1)
    # GRPO: mean = 0.5, advantages = [0.5, -0.5]
    assert_almost_equal(result.token_advs[0][0], 0.5, atol=1e-6)
    assert_almost_equal(result.token_advs[1][0], -0.5, atol=1e-6)


fn test_pipeline_entropy_stats_counts() raises:
    """Entropy stats correctly count exec vs planning tokens."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)

    var logprobs = List[List[Float64]]()
    for _ in range(2):
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.0)
        lp.append(-2.0)
        logprobs.append(lp^)

    # Token 2 is planning in both sequences
    var masks = List[List[Int]]()
    for _ in range(2):
        var m = List[Int]()
        m.append(0)
        m.append(0)
        m.append(1)
        masks.append(m^)

    var result = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl",
        transform_mode="gtpo",
        gtpo_beta=0.1,
    )

    assert_true(result.has_stats, "GTPO should have stats")
    # 2 sequences x 2 exec tokens = 4 exec
    assert_almost_equal(result.stats.exec_count, 4.0, atol=1e-6)
    # 2 sequences x 1 plan token = 2 plan
    assert_almost_equal(result.stats.plan_count, 2.0, atol=1e-6)


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

    # H10: pipeline — basic
    test_pipeline_none_mode_single_completion()
    test_pipeline_none_mode_binary()
    test_pipeline_grpo_none_mode()
    test_pipeline_gtpo_mode()
    test_pipeline_empty_inputs()
    test_pipeline_all_same_reward()

    # H10: pipeline — HICRA
    test_pipeline_hicra_amplifies_planning()
    test_pipeline_hicra_exact_values()
    test_pipeline_hicra_alpha_zero_equals_gtpo()

    # H10: pipeline — SEPA
    test_pipeline_sepa_modifies_advantages()
    test_pipeline_sepa_lambda_zero_equals_gtpo()

    # H10: pipeline — edge cases
    test_pipeline_none_ignores_masks()
    test_pipeline_all_planning_tokens()
    test_pipeline_mixed_sequence_lengths()
    test_pipeline_entropy_stats_counts()

    print("All 37 main tests passed!")
