"""Tests for advantage math — port of textpolicy/tests/test_advantages.py.

Validates the Mojo advantage functions against the same formulas
from the MLX originals (grpo.py, hicra.py, sepa.py).

Hypotheses tested:
    H1: MaxRL advantages zero out when mean reward ~ 0
    H2: MaxRL gives correct A = (N-K)/K for binary rewards
    H3: GTPO with beta=0 or uniform entropy returns uniform advantages
    H4: GTPO amplifies high-entropy tokens and dampens low-entropy
    H5: HICRA with alpha=0 is identity; with alpha>0 amplifies masked tokens
    H6: HICRA sign behavior: positive amplified, negative dampened
    H7: SEPA with lambda=0 is identity; lambda=1 fully pools execution entropy
"""

from testing import assert_true, assert_almost_equal, assert_equal
from math import abs

from src.advantages import (
    compute_grpo_advantages,
    compute_maxrl_advantages,
    apply_gtpo_weighting,
    apply_hicra,
    apply_sepa_pooling,
    compute_entropy_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


fn approx_eq(a: Float64, b: Float64, tol: Float64 = 1e-6) -> Bool:
    return abs(a - b) < tol


fn assert_list_approx_eq(
    xs: List[Float64], ys: List[Float64], tol: Float64 = 1e-6
) raises:
    assert_equal(len(xs), len(ys))
    for i in range(len(xs)):
        assert_almost_equal(xs[i], ys[i], atol=tol)


# ---------------------------------------------------------------------------
# H0: GRPO advantages (baseline)
# ---------------------------------------------------------------------------


fn test_grpo_empty() raises:
    """Edge case: empty input returns empty output."""
    var result = compute_grpo_advantages(List[Float64]())
    assert_equal(len(result), 0)


fn test_grpo_basic() raises:
    """GRPO: A_i = r_i - mean(r)."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)
    rewards.append(0.0)
    rewards.append(0.0)
    var result = compute_grpo_advantages(rewards)
    # mean = 0.25, correct = 0.75, incorrect = -0.25
    assert_almost_equal(result[0], 0.75, atol=1e-6)
    assert_almost_equal(result[1], -0.25, atol=1e-6)
    assert_almost_equal(result[2], -0.25, atol=1e-6)
    assert_almost_equal(result[3], -0.25, atol=1e-6)


fn test_grpo_all_equal() raises:
    """GRPO: all equal rewards -> zero advantages."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(1.0)
    rewards.append(1.0)
    var result = compute_grpo_advantages(rewards)
    for i in range(len(result)):
        assert_almost_equal(result[i], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# H1, H2: MaxRL advantages
# ---------------------------------------------------------------------------


fn test_maxrl_empty() raises:
    """Edge case: empty input returns empty output."""
    var result = compute_maxrl_advantages(List[Float64]())
    assert_equal(len(result), 0)


fn test_maxrl_all_zero_returns_zeros() raises:
    """H1: When mean ~ 0, all advantages are zero."""
    var rewards = List[Float64]()
    rewards.append(0.0)
    rewards.append(0.0)
    rewards.append(0.0)
    rewards.append(0.0)
    var result = compute_maxrl_advantages(rewards)
    for i in range(len(result)):
        assert_almost_equal(result[i], 0.0, atol=1e-6)


fn test_maxrl_binary_formula() raises:
    """H2: For K=1 correct out of N=4, correct gets A~3.0."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)
    rewards.append(0.0)
    rewards.append(0.0)
    var result = compute_maxrl_advantages(rewards)
    # Correct: (1.0 - 0.25) / (0.25 + eps) ~ 3.0
    assert_almost_equal(result[0], 3.0, atol=1e-4)
    # Incorrect: (0.0 - 0.25) / (0.25 + eps) ~ -1.0
    assert_almost_equal(result[1], -1.0, atol=1e-4)
    assert_almost_equal(result[2], -1.0, atol=1e-4)
    assert_almost_equal(result[3], -1.0, atol=1e-4)


fn test_maxrl_all_correct_zero() raises:
    """When all rewards are equal, advantages are zero."""
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(1.0)
    rewards.append(1.0)
    var result = compute_maxrl_advantages(rewards)
    for i in range(len(result)):
        assert_almost_equal(result[i], 0.0, atol=1e-4)


fn test_maxrl_harder_higher_advantage() raises:
    """H2: Harder problems -> higher correct advantage."""
    # Easy: 3/4 correct
    var easy_rewards = List[Float64]()
    easy_rewards.append(1.0)
    easy_rewards.append(1.0)
    easy_rewards.append(1.0)
    easy_rewards.append(0.0)
    var easy = compute_maxrl_advantages(easy_rewards)

    # Hard: 1/4 correct
    var hard_rewards = List[Float64]()
    hard_rewards.append(1.0)
    hard_rewards.append(0.0)
    hard_rewards.append(0.0)
    hard_rewards.append(0.0)
    var hard = compute_maxrl_advantages(hard_rewards)

    assert_true(hard[0] > easy[0], "Hard problem should have higher correct advantage")


# ---------------------------------------------------------------------------
# H3, H4: GTPO entropy weighting
# ---------------------------------------------------------------------------


fn test_gtpo_empty() raises:
    """Edge case: empty input."""
    var result = apply_gtpo_weighting(1.0, List[Float64]())
    assert_equal(len(result), 0)


fn test_gtpo_beta_zero_uniform() raises:
    """H3: beta=0 -> all advantages equal to base."""
    var entropies = List[Float64]()
    entropies.append(1.0)
    entropies.append(2.0)
    entropies.append(3.0)
    var result = apply_gtpo_weighting(0.5, entropies, beta=0.0)
    for i in range(len(result)):
        assert_almost_equal(result[i], 0.5, atol=1e-6)


fn test_gtpo_uniform_entropy_uniform() raises:
    """H3: Uniform entropy -> H_norm(t)=1 -> weight=1 -> unchanged."""
    var entropies = List[Float64]()
    entropies.append(2.0)
    entropies.append(2.0)
    entropies.append(2.0)
    var result = apply_gtpo_weighting(0.5, entropies, beta=0.1)
    for i in range(len(result)):
        assert_almost_equal(result[i], 0.5, atol=1e-6)


fn test_gtpo_high_entropy_amplified() raises:
    """H4: High-entropy tokens get larger advantage magnitude."""
    var entropies = List[Float64]()
    entropies.append(1.0)
    entropies.append(1.0)
    entropies.append(3.0)
    var result = apply_gtpo_weighting(1.0, entropies, beta=0.5)

    assert_true(result[2] > result[0], "High-entropy token should have larger advantage")
    assert_true(result[2] > 1.0, "High-entropy token should be amplified beyond base")
    assert_true(result[0] < 1.0, "Low-entropy token should be dampened below base")


fn test_gtpo_all_zero_entropy_uniform() raises:
    """H3: All-zero entropy -> no signal -> uniform advantages."""
    var entropies = List[Float64]()
    entropies.append(0.0)
    entropies.append(0.0)
    entropies.append(0.0)
    var result = apply_gtpo_weighting(0.5, entropies, beta=0.1)
    for i in range(len(result)):
        assert_almost_equal(result[i], 0.5, atol=1e-6)


fn test_gtpo_negative_advantage_preserved() raises:
    """GTPO preserves sign: negative advantage stays negative."""
    var entropies = List[Float64]()
    entropies.append(1.0)
    entropies.append(2.0)
    entropies.append(3.0)
    var result = apply_gtpo_weighting(-1.0, entropies, beta=0.1)
    for i in range(len(result)):
        assert_true(result[i] <= 0.0, "Negative advantage should stay non-positive")


fn test_gtpo_weight_clamped_nonnegative() raises:
    """Large beta with low-entropy tokens -> weight clamped to 0."""
    var entropies = List[Float64]()
    entropies.append(0.1)
    entropies.append(10.0)
    entropies.append(10.0)
    var result = apply_gtpo_weighting(1.0, entropies, beta=5.0)
    # Token 0 should have weight clamped to 0
    assert_almost_equal(result[0], 0.0, atol=0.1)


# ---------------------------------------------------------------------------
# H5, H6: HICRA amplification
# ---------------------------------------------------------------------------


fn test_hicra_alpha_zero_identity() raises:
    """H5: alpha=0 -> no amplification."""
    var advs = List[Float64]()
    advs.append(0.5)
    advs.append(-0.3)
    advs.append(0.2)
    var mask = List[Int]()
    mask.append(1)
    mask.append(0)
    mask.append(1)
    var result = apply_hicra(advs, mask, alpha=0.0)
    assert_list_approx_eq(result, advs)


fn test_hicra_no_mask_identity() raises:
    """H5: All-zero mask -> no amplification."""
    var advs = List[Float64]()
    advs.append(0.5)
    advs.append(-0.3)
    advs.append(0.2)
    var mask = List[Int]()
    mask.append(0)
    mask.append(0)
    mask.append(0)
    var result = apply_hicra(advs, mask, alpha=0.5)
    assert_list_approx_eq(result, advs)


fn test_hicra_positive_amplified() raises:
    """H6: A>0, mask=1 -> A*(1+alpha)."""
    var advs = List[Float64]()
    advs.append(1.0)
    var mask = List[Int]()
    mask.append(1)
    var result = apply_hicra(advs, mask, alpha=0.2)
    assert_almost_equal(result[0], 1.2, atol=1e-6)


fn test_hicra_negative_dampened() raises:
    """H6: A<0, mask=1 -> A*(1-alpha) = blame dampened."""
    var advs = List[Float64]()
    advs.append(-1.0)
    var mask = List[Int]()
    mask.append(1)
    var result = apply_hicra(advs, mask, alpha=0.2)
    # -1.0 + 0.2 * 1.0 = -0.8
    assert_almost_equal(result[0], -0.8, atol=1e-6)


fn test_hicra_length_mismatch_raises() raises:
    """Length mismatch should raise."""
    var advs = List[Float64]()
    advs.append(0.5)
    advs.append(0.3)
    var mask = List[Int]()
    mask.append(1)
    var raised = False
    try:
        _ = apply_hicra(advs, mask, alpha=0.2)
    except:
        raised = True
    assert_true(raised, "Should raise on length mismatch")


fn test_hicra_mixed_mask() raises:
    """Mixed mask: only masked tokens affected."""
    var advs = List[Float64]()
    advs.append(0.5)
    advs.append(-0.3)
    advs.append(0.2)
    var mask = List[Int]()
    mask.append(1)
    mask.append(0)
    mask.append(1)
    var result = apply_hicra(advs, mask, alpha=0.5)
    # Token 0: 0.5 + 0.5*0.5 = 0.75
    assert_almost_equal(result[0], 0.75, atol=1e-6)
    # Token 1: unchanged
    assert_almost_equal(result[1], -0.3, atol=1e-6)
    # Token 2: 0.2 + 0.5*0.2 = 0.3
    assert_almost_equal(result[2], 0.3, atol=1e-6)


# ---------------------------------------------------------------------------
# H7: SEPA pooling
# ---------------------------------------------------------------------------


fn test_sepa_lambda_zero_identity() raises:
    """H7: lambda=0 -> no pooling."""
    var entropies = List[Float64]()
    entropies.append(1.0)
    entropies.append(2.0)
    entropies.append(3.0)
    entropies.append(4.0)
    var mask = List[Int]()
    mask.append(0)
    mask.append(1)
    mask.append(0)
    mask.append(0)
    var result = apply_sepa_pooling(entropies, mask, lambda_t=0.0)
    assert_list_approx_eq(result, entropies)


fn test_sepa_lambda_one_fully_pools() raises:
    """H7: lambda=1 -> execution tokens fully pooled to their mean."""
    var entropies = List[Float64]()
    entropies.append(1.0)
    entropies.append(5.0)
    entropies.append(3.0)
    entropies.append(7.0)
    var mask = List[Int]()
    mask.append(0)
    mask.append(1)
    mask.append(0)
    mask.append(0)
    var result = apply_sepa_pooling(entropies, mask, lambda_t=1.0)

    # Execution tokens: [1.0, 3.0, 7.0], mean = 11/3
    var exec_mean: Float64 = (1.0 + 3.0 + 7.0) / 3.0

    # Planning token unchanged
    assert_almost_equal(result[1], 5.0, atol=1e-6)
    # Execution tokens should all be exec_mean
    assert_almost_equal(result[0], exec_mean, atol=1e-4)
    assert_almost_equal(result[2], exec_mean, atol=1e-4)
    assert_almost_equal(result[3], exec_mean, atol=1e-4)


fn test_sepa_partial_pooling() raises:
    """Intermediate lambda -> interpolation between original and mean."""
    var entropies = List[Float64]()
    entropies.append(1.0)
    entropies.append(5.0)
    entropies.append(3.0)
    var mask = List[Int]()
    mask.append(0)
    mask.append(1)
    mask.append(0)
    var result = apply_sepa_pooling(entropies, mask, lambda_t=0.5)

    # exec_mean = (1.0 + 3.0) / 2 = 2.0
    # Token 0: 0.5 * 2.0 + 0.5 * 1.0 = 1.5
    assert_almost_equal(result[0], 1.5, atol=1e-6)
    # Token 1: planning, unchanged
    assert_almost_equal(result[1], 5.0, atol=1e-6)
    # Token 2: 0.5 * 2.0 + 0.5 * 3.0 = 2.5
    assert_almost_equal(result[2], 2.5, atol=1e-6)


fn test_sepa_all_planning_unchanged() raises:
    """If all tokens are planning, nothing to pool."""
    var entropies = List[Float64]()
    entropies.append(1.0)
    entropies.append(2.0)
    entropies.append(3.0)
    var mask = List[Int]()
    mask.append(1)
    mask.append(1)
    mask.append(1)
    var result = apply_sepa_pooling(entropies, mask, lambda_t=1.0)
    assert_list_approx_eq(result, entropies)


fn test_sepa_length_mismatch_raises() raises:
    """Length mismatch should raise."""
    var entropies = List[Float64]()
    entropies.append(1.0)
    entropies.append(2.0)
    var mask = List[Int]()
    mask.append(0)
    var raised = False
    try:
        _ = apply_sepa_pooling(entropies, mask, lambda_t=0.5)
    except:
        raised = True
    assert_true(raised, "Should raise on length mismatch")


# ---------------------------------------------------------------------------
# Entropy stats
# ---------------------------------------------------------------------------


fn test_entropy_stats_empty() raises:
    """Empty inputs produce zero stats."""
    var stats = compute_entropy_stats(List[Float64](), List[Float64]())
    assert_almost_equal(stats.exec_mean, 0.0, atol=1e-6)
    assert_almost_equal(stats.exec_var, 0.0, atol=1e-6)
    assert_almost_equal(stats.exec_count, 0.0, atol=1e-6)
    assert_almost_equal(stats.plan_mean, 0.0, atol=1e-6)
    assert_almost_equal(stats.plan_var, 0.0, atol=1e-6)
    assert_almost_equal(stats.plan_count, 0.0, atol=1e-6)


fn test_entropy_stats_basic() raises:
    """Basic entropy stats computation."""
    var exec_e = List[Float64]()
    exec_e.append(1.0)
    exec_e.append(2.0)
    exec_e.append(3.0)
    var plan_e = List[Float64]()
    plan_e.append(4.0)
    plan_e.append(6.0)
    var stats = compute_entropy_stats(exec_e, plan_e)

    # exec: mean=2, var = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3
    assert_almost_equal(stats.exec_mean, 2.0, atol=1e-6)
    assert_almost_equal(stats.exec_var, 2.0 / 3.0, atol=1e-6)
    assert_almost_equal(stats.exec_count, 3.0, atol=1e-6)

    # plan: mean=5, var = ((4-5)^2 + (6-5)^2)/2 = 1.0
    assert_almost_equal(stats.plan_mean, 5.0, atol=1e-6)
    assert_almost_equal(stats.plan_var, 1.0, atol=1e-6)
    assert_almost_equal(stats.plan_count, 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


fn main() raises:
    # GRPO
    test_grpo_empty()
    test_grpo_basic()
    test_grpo_all_equal()

    # MaxRL
    test_maxrl_empty()
    test_maxrl_all_zero_returns_zeros()
    test_maxrl_binary_formula()
    test_maxrl_all_correct_zero()
    test_maxrl_harder_higher_advantage()

    # GTPO
    test_gtpo_empty()
    test_gtpo_beta_zero_uniform()
    test_gtpo_uniform_entropy_uniform()
    test_gtpo_high_entropy_amplified()
    test_gtpo_all_zero_entropy_uniform()
    test_gtpo_negative_advantage_preserved()
    test_gtpo_weight_clamped_nonnegative()

    # HICRA
    test_hicra_alpha_zero_identity()
    test_hicra_no_mask_identity()
    test_hicra_positive_amplified()
    test_hicra_negative_dampened()
    test_hicra_length_mismatch_raises()
    test_hicra_mixed_mask()

    # SEPA pooling
    test_sepa_lambda_zero_identity()
    test_sepa_lambda_one_fully_pools()
    test_sepa_partial_pooling()
    test_sepa_all_planning_unchanged()
    test_sepa_length_mismatch_raises()

    # Entropy stats
    test_entropy_stats_empty()
    test_entropy_stats_basic()

    print("All 30 advantage tests passed!")
