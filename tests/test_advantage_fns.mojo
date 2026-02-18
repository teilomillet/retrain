"""Tests for advantage_fns.mojo: trait structs produce identical output to string-dispatch path.

Covers:
    - GRPOAdvantage: numerical equivalence with compute_grpo_advantages
    - MaxRLAdvantage: numerical equivalence with compute_maxrl_advantages
    - UniformExpand: broadcast episode advantages uniformly
    - GTPOTransform: entropy-weighted credit assignment
    - GTPOHicraTransform: GTPO + HICRA amplification
    - GTPOSepaTransform: GTPO + SEPA pooling
    - ConstantAdvantage: custom trait struct compiles and works
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal

from src.advantage_fns import (
    EpisodeAdvantageFn,
    TokenTransformFn,
    AdvantageResult,
    GRPOAdvantage,
    MaxRLAdvantage,
    UniformExpand,
    GTPOTransform,
    GTPOHicraTransform,
    GTPOSepaTransform,
)
from src.main import compute_composable_advantages


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
        if v != v:
            return False
        if v > 1e300 or v < -1e300:
            return False
    return True


fn _zero_masks(n_seqs: Int, n_tokens: Int) -> List[List[Int]]:
    var masks = List[List[Int]]()
    for _ in range(n_seqs):
        masks.append(List[Int](length=n_tokens, fill=0))
    return masks^


fn _make_rewards() -> List[Float64]:
    """Standard test rewards: [1.0, 0.0, 0.0, 0.0]."""
    var r = List[Float64]()
    r.append(1.0)
    r.append(0.0)
    r.append(0.0)
    r.append(0.0)
    return r^


fn _make_logprobs(n_seqs: Int, n_tokens: Int) -> List[List[Float64]]:
    """Standard test logprobs: varying per-token."""
    var logprobs = List[List[Float64]]()
    var vals = List[Float64]()
    vals.append(-0.5)
    vals.append(-1.5)
    vals.append(-3.0)
    for _ in range(n_seqs):
        var lp = List[Float64]()
        for j in range(n_tokens):
            lp.append(vals[j % len(vals)])
        logprobs.append(lp^)
    return logprobs^


# ---------------------------------------------------------------------------
# GRPOAdvantage
# ---------------------------------------------------------------------------


fn test_grpo_advantage_equivalence() raises:
    """GRPOAdvantage.compute() matches compute_grpo_advantages."""
    var rewards = _make_rewards()
    var ep = GRPOAdvantage()
    var result = ep.compute(rewards)
    # mean = 0.25, so [0.75, -0.25, -0.25, -0.25]
    assert_almost_equal(result[0], 0.75, atol=1e-6)
    assert_almost_equal(result[1], -0.25, atol=1e-6)
    assert_almost_equal(result[2], -0.25, atol=1e-6)
    assert_almost_equal(result[3], -0.25, atol=1e-6)


fn test_grpo_advantage_empty() raises:
    """Empty rewards -> empty advantages."""
    var ep = GRPOAdvantage()
    var result = ep.compute(List[Float64]())
    assert_equal(len(result), 0)


# ---------------------------------------------------------------------------
# MaxRLAdvantage
# ---------------------------------------------------------------------------


fn test_maxrl_advantage_equivalence() raises:
    """MaxRLAdvantage.compute() matches compute_maxrl_advantages."""
    var rewards = _make_rewards()
    var ep = MaxRLAdvantage()
    var result = ep.compute(rewards)
    # mean=0.25, denom=0.25+eps, adv[0]~3.0, adv[1]~-1.0
    assert_almost_equal(result[0], 3.0, atol=1e-3)
    assert_almost_equal(result[1], -1.0, atol=1e-3)


fn test_maxrl_advantage_all_zero() raises:
    """All zero rewards -> all zero advantages (no signal)."""
    var rewards = List[Float64]()
    rewards.append(0.0)
    rewards.append(0.0)
    var ep = MaxRLAdvantage()
    var result = ep.compute(rewards)
    assert_almost_equal(result[0], 0.0, atol=1e-6)
    assert_almost_equal(result[1], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# UniformExpand
# ---------------------------------------------------------------------------


fn test_uniform_expand_equivalence() raises:
    """UniformExpand matches compute_composable_advantages with mode=none."""
    var rewards = _make_rewards()
    var logprobs = _make_logprobs(4, 3)
    var masks = _zero_masks(4, 3)

    # Old path
    var old = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl", transform_mode="none",
    )

    # New path
    var ep = MaxRLAdvantage()
    var advs = ep.compute(rewards)
    var tx = UniformExpand()
    var new = tx.transform(advs, logprobs, masks)

    assert_equal(len(old.token_advs), len(new.token_advs))
    for i in range(len(old.token_advs)):
        assert_list_approx_eq(old.token_advs[i], new.token_advs[i])
    assert_false(new.has_stats, "UniformExpand should not have stats")


# ---------------------------------------------------------------------------
# GTPOTransform
# ---------------------------------------------------------------------------


fn test_gtpo_transform_equivalence() raises:
    """GTPOTransform matches compute_composable_advantages with mode=gtpo."""
    var rewards = _make_rewards()
    var logprobs = _make_logprobs(4, 3)
    var masks = _zero_masks(4, 3)

    var old = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl", transform_mode="gtpo", gtpo_beta=0.5,
    )

    var ep = MaxRLAdvantage()
    var advs = ep.compute(rewards)
    var tx = GTPOTransform(beta=0.5)
    var new = tx.transform(advs, logprobs, masks)

    assert_equal(len(old.token_advs), len(new.token_advs))
    for i in range(len(old.token_advs)):
        assert_list_approx_eq(old.token_advs[i], new.token_advs[i])
    assert_true(new.has_stats, "GTPOTransform should have stats")


# ---------------------------------------------------------------------------
# GTPOHicraTransform
# ---------------------------------------------------------------------------


fn test_gtpo_hicra_transform_equivalence() raises:
    """GTPOHicraTransform matches compute_composable_advantages with mode=gtpo_hicra."""
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

    var old = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl", transform_mode="gtpo_hicra",
        gtpo_beta=0.0, hicra_alpha=0.5,
    )

    var ep = MaxRLAdvantage()
    var advs = ep.compute(rewards)
    var tx = GTPOHicraTransform(beta=0.0, alpha=0.5)
    var new = tx.transform(advs, logprobs, masks)

    assert_equal(len(old.token_advs), len(new.token_advs))
    for i in range(len(old.token_advs)):
        assert_list_approx_eq(old.token_advs[i], new.token_advs[i])


# ---------------------------------------------------------------------------
# GTPOSepaTransform
# ---------------------------------------------------------------------------


fn test_gtpo_sepa_transform_equivalence() raises:
    """GTPOSepaTransform matches compute_composable_advantages with mode=gtpo_sepa."""
    var rewards = _make_rewards()
    var logprobs = _make_logprobs(4, 3)

    var masks = List[List[Int]]()
    for _ in range(4):
        var m = List[Int]()
        m.append(0)
        m.append(0)
        m.append(1)
        masks.append(m^)

    var old = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="maxrl", transform_mode="gtpo_sepa",
        gtpo_beta=0.5, sepa_lambda=0.5,
    )

    var ep = MaxRLAdvantage()
    var advs = ep.compute(rewards)
    var tx = GTPOSepaTransform(beta=0.5, sepa_lambda=0.5)
    var new = tx.transform(advs, logprobs, masks)

    assert_equal(len(old.token_advs), len(new.token_advs))
    for i in range(len(old.token_advs)):
        assert_list_approx_eq(old.token_advs[i], new.token_advs[i])
    assert_true(new.has_stats, "GTPOSepaTransform should have stats")


fn test_gtpo_sepa_lambda_zero_equals_gtpo() raises:
    """GTPOSepaTransform with lambda=0 equals GTPOTransform."""
    var rewards = _make_rewards()
    var logprobs = _make_logprobs(4, 3)
    var masks = _zero_masks(4, 3)

    var ep = MaxRLAdvantage()
    var advs_gtpo = ep.compute(rewards)
    var advs_sepa = ep.compute(rewards)

    var tx_gtpo = GTPOTransform(beta=0.1)
    var tx_sepa = GTPOSepaTransform(beta=0.1, sepa_lambda=0.0)

    var result_gtpo = tx_gtpo.transform(advs_gtpo, logprobs, masks)
    var result_sepa = tx_sepa.transform(advs_sepa, logprobs, masks)

    for i in range(len(result_gtpo.token_advs)):
        assert_list_approx_eq(result_gtpo.token_advs[i], result_sepa.token_advs[i])


fn test_gtpo_sepa_update_lambda() raises:
    """update_sepa_lambda() changes the pooling strength."""
    var rewards = _make_rewards()
    var logprobs = _make_logprobs(4, 3)
    var masks = List[List[Int]]()
    for _ in range(4):
        var m = List[Int]()
        m.append(0)
        m.append(0)
        m.append(1)
        masks.append(m^)

    var ep = MaxRLAdvantage()
    var advs = ep.compute(rewards)

    var tx = GTPOSepaTransform(beta=0.5, sepa_lambda=0.0)
    var result_zero = tx.transform(advs, logprobs, masks)

    tx.update_sepa_lambda(0.5)
    var advs2 = ep.compute(rewards)
    var result_half = tx.transform(advs2, logprobs, masks)

    # Results should differ (SEPA pooling changes execution token entropies)
    var differs = False
    for i in range(len(result_zero.token_advs)):
        for j in range(len(result_zero.token_advs[i])):
            if result_zero.token_advs[i][j] != result_half.token_advs[i][j]:
                differs = True
                break
        if differs:
            break
    assert_true(differs, "Updating SEPA lambda should change advantages")


# ---------------------------------------------------------------------------
# GRPO + all transforms (cross-product coverage)
# ---------------------------------------------------------------------------


fn test_grpo_none_equivalence() raises:
    """GRPO + UniformExpand matches old grpo+none path."""
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

    var old = compute_composable_advantages(
        rewards, logprobs, masks,
        advantage_mode="grpo", transform_mode="none",
    )

    var ep = GRPOAdvantage()
    var advs = ep.compute(rewards)
    var tx = UniformExpand()
    var new = tx.transform(advs, logprobs, masks)

    for i in range(2):
        assert_list_approx_eq(old.token_advs[i], new.token_advs[i])


# ---------------------------------------------------------------------------
# Custom EpisodeAdvantageFn: ConstantAdvantage
# ---------------------------------------------------------------------------


struct ConstantAdvantage(EpisodeAdvantageFn):
    """Always returns the same advantage for all completions."""

    var value: Float64

    fn __init__(out self, value: Float64):
        self.value = value

    fn __moveinit__(out self, deinit take: Self):
        self.value = take.value

    fn compute(self, rewards: List[Float64]) -> List[Float64]:
        var result = List[Float64]()
        for _ in range(len(rewards)):
            result.append(self.value)
        return result^


fn test_constant_advantage() raises:
    """Custom EpisodeAdvantageFn compiles and produces expected output."""
    var ep = ConstantAdvantage(value=42.0)
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)
    rewards.append(0.5)

    var result = ep.compute(rewards)
    assert_equal(len(result), 3)
    for i in range(3):
        assert_almost_equal(result[i], 42.0, atol=1e-6)


fn test_constant_advantage_with_uniform_expand() raises:
    """Custom episode advantage + UniformExpand produces correct token advantages."""
    var ep = ConstantAdvantage(value=2.5)
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(0.0)

    var advs = ep.compute(rewards)
    var logprobs = List[List[Float64]]()
    for _ in range(2):
        var lp = List[Float64]()
        lp.append(-0.5)
        lp.append(-1.0)
        logprobs.append(lp^)

    var masks = _zero_masks(2, 2)
    var tx = UniformExpand()
    var result = tx.transform(advs, logprobs, masks)

    assert_equal(len(result.token_advs), 2)
    for i in range(2):
        for j in range(2):
            assert_almost_equal(result.token_advs[i][j], 2.5, atol=1e-6)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


fn main() raises:
    # GRPOAdvantage
    test_grpo_advantage_equivalence()
    test_grpo_advantage_empty()

    # MaxRLAdvantage
    test_maxrl_advantage_equivalence()
    test_maxrl_advantage_all_zero()

    # UniformExpand
    test_uniform_expand_equivalence()

    # GTPOTransform
    test_gtpo_transform_equivalence()

    # GTPOHicraTransform
    test_gtpo_hicra_transform_equivalence()

    # GTPOSepaTransform
    test_gtpo_sepa_transform_equivalence()
    test_gtpo_sepa_lambda_zero_equals_gtpo()
    test_gtpo_sepa_update_lambda()

    # Cross-product
    test_grpo_none_equivalence()

    # Custom trait
    test_constant_advantage()
    test_constant_advantage_with_uniform_expand()

    print("All 13 advantage_fns tests passed!")
