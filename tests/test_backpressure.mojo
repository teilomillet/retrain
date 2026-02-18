"""Tests for backpressure.mojo: USL math, fitting, roofline, decision logic, NoOp, structs, integration.

30 tests covering:
    1. USL Math (5): p=1 identity, linear scaling, Amdahl, retrograde peak, optimal p*
    2. USL Fitting (5): recover known σ/κ, minimum points guard, clamp negative σ, clamp negative κ, constant throughput
    3. Roofline (4): warmup regime, retrograde detection, regime transitions, skipped preserves regime
    4. Decision Logic (6): hold during warmup, throttle above p*, increase below p*, hold optimal, batch clamped, params propagated
    5. NoOp (3): always hold, observe no-op, reset no-op
    6. Structs (3): StepObservation copy, decision defaults, decision Writable
    7. Integration (4): observe/recommend cycle, reset clears state, validation rejects bad config, custom BackPressure compiles
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from math import abs as math_abs

from src.backpressure import (
    BackPressure,
    BackPressureDecision,
    StepObservation,
    NoOpBackPressure,
    USLBackPressure,
    usl_throughput,
    usl_optimal_p,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


fn make_obs(
    step_time_s: Float64,
    total_tokens: Int,
    batch_size: Int = 8,
    group_size: Int = 16,
    skipped: Bool = False,
) -> StepObservation:
    var obs = StepObservation()
    obs.step_time_s = step_time_s
    obs.sample_time_s = step_time_s * 0.6
    obs.train_time_s = step_time_s * 0.4
    obs.num_datums = batch_size * group_size
    obs.batch_size = batch_size
    obs.group_size = group_size
    obs.total_tokens = total_tokens
    obs.loss = 0.5
    obs.skipped = skipped
    return obs^


# ---------------------------------------------------------------------------
# 1. USL Math (5)
# ---------------------------------------------------------------------------


fn test_usl_p1_identity() raises:
    """C(1) = 1 for any σ, κ."""
    var c = usl_throughput(1.0, 0.5, 0.1)
    assert_almost_equal(c, 1.0, atol=1e-10)


fn test_usl_linear_scaling() raises:
    """σ=κ=0 → C(p) = p (linear scaling)."""
    var c = usl_throughput(8.0, 0.0, 0.0)
    assert_almost_equal(c, 8.0, atol=1e-10)


fn test_usl_amdahl() raises:
    """κ=0 → Amdahl's law: C(p) = p / (1 + σ(p-1))."""
    var sigma = 0.1
    var p = 10.0
    var expected = p / (1.0 + sigma * (p - 1.0))
    var c = usl_throughput(p, sigma, 0.0)
    assert_almost_equal(c, expected, atol=1e-10)


fn test_usl_retrograde_peak() raises:
    """With κ>0, throughput eventually decreases (retrograde)."""
    var sigma = 0.02
    var kappa = 0.001
    # Throughput at p=10 vs p=100
    var c10 = usl_throughput(10.0, sigma, kappa)
    _ = usl_throughput(100.0, sigma, kappa)
    # At some point throughput must drop
    assert_true(c10 > 0.0, "C(10) should be positive")
    # For large p with κ>0, throughput drops
    var c1000 = usl_throughput(1000.0, sigma, kappa)
    assert_true(c1000 < c10, "C(1000) < C(10) when κ > 0 (retrograde)")


fn test_usl_optimal_p_star() raises:
    """p* = sqrt((1-σ)/κ) for known values."""
    var sigma = 0.02
    var kappa = 0.001
    var p_star = usl_optimal_p(sigma, kappa)
    # Expected: sqrt(0.98/0.001) = sqrt(980) ≈ 31.3
    var expected = 31.304951684997057
    assert_almost_equal(p_star, expected, atol=0.01)


# ---------------------------------------------------------------------------
# 2. USL Fitting (5)
# ---------------------------------------------------------------------------


fn test_fit_recovers_known_params() raises:
    """Fit recovers known σ/κ from synthetic USL data."""
    var sigma_true = 0.05
    var kappa_true = 0.002
    var bp = USLBackPressure(warmup_steps=0)

    # Generate observations at various concurrencies
    var ps = List[Float64]()
    ps.append(1.0)
    ps.append(4.0)
    ps.append(8.0)
    ps.append(16.0)
    ps.append(32.0)
    ps.append(64.0)

    for i in range(len(ps)):
        var p = ps[i]
        var tokens_per_sec = usl_throughput(p, sigma_true, kappa_true) * 1000.0
        var total_tokens = Int(tokens_per_sec)
        var batch = Int(p)
        var obs = make_obs(1.0, total_tokens, batch_size=batch, group_size=1)
        bp.observe(obs)

    assert_almost_equal(bp.sigma, sigma_true, atol=0.01)
    assert_almost_equal(bp.kappa, kappa_true, atol=0.001)


fn test_fit_minimum_points_guard() raises:
    """With < 3 observations, σ/κ remain at defaults (0)."""
    var bp = USLBackPressure(warmup_steps=0)

    var obs1 = make_obs(1.0, 1000, batch_size=1, group_size=1)
    bp.observe(obs1)
    assert_almost_equal(bp.sigma, 0.0, atol=1e-10)
    assert_almost_equal(bp.kappa, 0.0, atol=1e-10)

    var obs2 = make_obs(1.0, 2000, batch_size=2, group_size=1)
    bp.observe(obs2)
    assert_almost_equal(bp.sigma, 0.0, atol=1e-10)
    assert_almost_equal(bp.kappa, 0.0, atol=1e-10)


fn test_fit_clamps_negative_sigma() raises:
    """Negative σ from noisy data gets clamped to 0."""
    var bp = USLBackPressure(warmup_steps=0)

    # Super-linear scaling data → raw σ would be negative
    var obs1 = make_obs(1.0, 1000, batch_size=1, group_size=1)
    bp.observe(obs1)
    var obs2 = make_obs(1.0, 2500, batch_size=2, group_size=1)  # > 2x
    bp.observe(obs2)
    var obs3 = make_obs(1.0, 5000, batch_size=4, group_size=1)  # > 4x
    bp.observe(obs3)

    assert_true(bp.sigma >= 0.0, "sigma should be clamped >= 0")


fn test_fit_clamps_negative_kappa() raises:
    """Negative κ from noisy data gets clamped to 0."""
    var bp = USLBackPressure(warmup_steps=0)

    # Continuously improving scaling data → raw κ would be negative
    var obs1 = make_obs(1.0, 100, batch_size=1, group_size=1)
    bp.observe(obs1)
    var obs2 = make_obs(1.0, 300, batch_size=2, group_size=1)
    bp.observe(obs2)
    var obs3 = make_obs(1.0, 1200, batch_size=4, group_size=1)
    bp.observe(obs3)

    assert_true(bp.kappa >= 0.0, "kappa should be clamped >= 0")


fn test_fit_constant_throughput() raises:
    """Constant throughput across concurrencies → σ near 1, κ near 0."""
    var bp = USLBackPressure(warmup_steps=0)

    # Same throughput regardless of concurrency → Amdahl bottleneck
    for p in range(1, 6):
        var obs = make_obs(1.0, 1000, batch_size=p, group_size=1)
        bp.observe(obs)

    # With constant throughput, σ should be high (contention dominates)
    assert_true(bp.sigma > 0.0, "sigma should be positive for constant throughput")
    assert_true(bp.kappa >= 0.0, "kappa should be non-negative")


# ---------------------------------------------------------------------------
# 3. Roofline (4)
# ---------------------------------------------------------------------------


fn test_roofline_warmup_regime() raises:
    """During warmup, regime is 'warmup'."""
    var bp = USLBackPressure(warmup_steps=5)

    for _ in range(3):
        var obs = make_obs(1.0, 1000, batch_size=8, group_size=16)
        bp.observe(obs)

    assert_equal(bp.regime, "warmup")


fn test_roofline_retrograde_detection() raises:
    """Retrograde detected when operating beyond p*."""
    var bp = USLBackPressure(warmup_steps=0, throttle_margin=0.85)

    # Feed data that creates a clear retrograde scenario
    var sigma_true = 0.05
    var kappa_true = 0.01
    # p* ≈ sqrt(0.95/0.01) ≈ 9.7

    var ps = List[Float64]()
    ps.append(1.0)
    ps.append(4.0)
    ps.append(8.0)

    for i in range(len(ps)):
        var p = ps[i]
        var tput = usl_throughput(p, sigma_true, kappa_true) * 1000.0
        var obs = make_obs(1.0, Int(tput), batch_size=Int(p), group_size=1)
        bp.observe(obs)

    # Now feed an observation well beyond p*
    var big_p = 50.0
    var tput = usl_throughput(big_p, sigma_true, kappa_true) * 1000.0
    var obs = make_obs(1.0, Int(tput), batch_size=50, group_size=1)
    bp.observe(obs)

    assert_equal(bp.regime, "retrograde")


fn test_roofline_regime_transitions() raises:
    """Regime transitions from warmup to operational state."""
    var bp = USLBackPressure(warmup_steps=2)

    # During warmup
    var obs1 = make_obs(1.0, 1000, batch_size=4, group_size=1)
    bp.observe(obs1)
    assert_equal(bp.regime, "warmup")

    var obs2 = make_obs(1.0, 2000, batch_size=8, group_size=1)
    bp.observe(obs2)
    assert_equal(bp.regime, "warmup")

    # After warmup, should transition
    var obs3 = make_obs(1.0, 3000, batch_size=12, group_size=1)
    bp.observe(obs3)
    assert_true(bp.regime != "warmup", "Should leave warmup after warmup_steps")


fn test_roofline_skipped_preserves_regime() raises:
    """Skipped observations don't change regime."""
    var bp = USLBackPressure(warmup_steps=0)

    # Build up state
    for p in range(1, 5):
        var obs = make_obs(1.0, p * 1000, batch_size=p, group_size=1)
        bp.observe(obs)

    var regime_before = bp.regime

    # Skipped step should not change regime
    var skipped = make_obs(1.0, 0, batch_size=4, group_size=1, skipped=True)
    bp.observe(skipped)

    assert_equal(bp.regime, regime_before)


# ---------------------------------------------------------------------------
# 4. Decision Logic (6)
# ---------------------------------------------------------------------------


fn test_decision_hold_during_warmup() raises:
    """During warmup, decision is always 'hold'."""
    var bp = USLBackPressure(warmup_steps=10)

    var obs = make_obs(1.0, 1000, batch_size=8, group_size=16)
    bp.observe(obs)

    var decision = bp.recommend()
    assert_equal(decision.action, "hold")


fn test_decision_throttle_above_p_star() raises:
    """Throttle when operating in retrograde region."""
    var bp = USLBackPressure(warmup_steps=0, throttle_margin=0.85)

    # Create retrograde scenario: p* ≈ 9.7
    var sigma_true = 0.05
    var kappa_true = 0.01

    var ps = List[Float64]()
    ps.append(1.0)
    ps.append(4.0)
    ps.append(8.0)

    for i in range(len(ps)):
        var p = ps[i]
        var tput = usl_throughput(p, sigma_true, kappa_true) * 1000.0
        var obs = make_obs(1.0, Int(tput), batch_size=Int(p), group_size=1)
        bp.observe(obs)

    # Observe well beyond p*
    var tput = usl_throughput(50.0, sigma_true, kappa_true) * 1000.0
    var obs = make_obs(1.0, Int(tput), batch_size=50, group_size=1)
    bp.observe(obs)

    var decision = bp.recommend()
    assert_equal(decision.action, "throttle")


fn test_decision_increase_below_p_star() raises:
    """Increase when operating well below p*."""
    var bp = USLBackPressure(warmup_steps=0, increase_margin=0.5)

    # Wide range of data for good fitting, ending at small p
    var sigma_true = 0.02
    var kappa_true = 0.002
    # p* ≈ sqrt(0.98/0.002) ≈ 22.1

    var ps = List[Float64]()
    ps.append(1.0)
    ps.append(4.0)
    ps.append(16.0)
    ps.append(32.0)
    ps.append(8.0)  # last → current_p=8, well below p*×0.5≈11

    for i in range(len(ps)):
        var p = ps[i]
        var tput = usl_throughput(p, sigma_true, kappa_true) * 1000.0
        var obs = make_obs(1.0, Int(tput), batch_size=Int(p), group_size=1)
        bp.observe(obs)

    var decision = bp.recommend()
    assert_equal(decision.action, "increase")


fn test_decision_hold_in_optimal_zone() raises:
    """Hold when operating near p*."""
    var bp = USLBackPressure(warmup_steps=0, throttle_margin=0.85, increase_margin=0.5)

    # Create scenario where we're in the sweet spot
    var sigma_true = 0.05
    var kappa_true = 0.01
    # p* ≈ sqrt(0.95/0.01) ≈ 9.7

    var ps = List[Float64]()
    ps.append(1.0)
    ps.append(4.0)
    ps.append(8.0)

    for i in range(len(ps)):
        var p = ps[i]
        var tput = usl_throughput(p, sigma_true, kappa_true) * 1000.0
        var obs = make_obs(1.0, Int(tput), batch_size=Int(p), group_size=1)
        bp.observe(obs)

    # Observe near p* (≈9.7), specifically at p=7 which is between
    # increase_margin*p* (≈4.9) and throttle_margin*p* (≈8.2)
    var tput = usl_throughput(7.0, sigma_true, kappa_true) * 1000.0
    var obs = make_obs(1.0, Int(tput), batch_size=7, group_size=1)
    bp.observe(obs)

    var decision = bp.recommend()
    assert_equal(decision.action, "hold")


fn test_decision_batch_clamped() raises:
    """Recommended batch_size stays within min/max bounds."""
    var bp = USLBackPressure(
        warmup_steps=0, min_batch_size=2, max_batch_size=16,
    )

    # Create retrograde: wants to throttle down
    var sigma_true = 0.05
    var kappa_true = 0.01
    var ps = List[Float64]()
    ps.append(1.0)
    ps.append(4.0)
    ps.append(8.0)
    for i in range(len(ps)):
        var p = ps[i]
        var tput = usl_throughput(p, sigma_true, kappa_true) * 1000.0
        var obs = make_obs(1.0, Int(tput), batch_size=Int(p), group_size=1)
        bp.observe(obs)

    var tput = usl_throughput(50.0, sigma_true, kappa_true) * 1000.0
    var obs = make_obs(1.0, Int(tput), batch_size=50, group_size=1)
    bp.observe(obs)

    var decision = bp.recommend()
    assert_true(
        decision.recommended_batch_size >= 2,
        "batch_size should be >= min",
    )
    assert_true(
        decision.recommended_batch_size <= 16,
        "batch_size should be <= max",
    )


fn test_decision_params_propagated() raises:
    """Decision contains fitted σ, κ, p*, throughput."""
    var bp = USLBackPressure(warmup_steps=0)

    var sigma_true = 0.05
    var kappa_true = 0.002

    var ps = List[Float64]()
    ps.append(1.0)
    ps.append(4.0)
    ps.append(8.0)
    ps.append(16.0)

    for i in range(len(ps)):
        var p = ps[i]
        var tput = usl_throughput(p, sigma_true, kappa_true) * 1000.0
        var obs = make_obs(1.0, Int(tput), batch_size=Int(p), group_size=1)
        bp.observe(obs)

    var decision = bp.recommend()
    assert_true(decision.sigma >= 0.0, "sigma should be non-negative")
    assert_true(decision.kappa >= 0.0, "kappa should be non-negative")
    assert_true(decision.p_star >= 1.0, "p_star should be >= 1")
    assert_true(decision.throughput > 0.0, "throughput should be positive")


# ---------------------------------------------------------------------------
# 5. NoOp (3)
# ---------------------------------------------------------------------------


fn test_noop_always_hold() raises:
    """NoOpBackPressure always recommends 'hold'."""
    var bp = NoOpBackPressure()
    var decision = bp.recommend()
    assert_equal(decision.action, "hold")


fn test_noop_observe_is_noop() raises:
    """NoOpBackPressure.observe() accepts input without error."""
    var bp = NoOpBackPressure()
    var obs = make_obs(1.0, 1000, batch_size=8, group_size=16)
    bp.observe(obs)
    # Should still return hold after observation
    var decision = bp.recommend()
    assert_equal(decision.action, "hold")


fn test_noop_reset_is_noop() raises:
    """NoOpBackPressure.reset() is a no-op."""
    var bp = NoOpBackPressure()
    bp.reset()
    var decision = bp.recommend()
    assert_equal(decision.action, "hold")


# ---------------------------------------------------------------------------
# 6. Structs (3)
# ---------------------------------------------------------------------------


fn test_step_observation_copy() raises:
    """StepObservation is Copyable."""
    var obs = StepObservation()
    obs.step_time_s = 1.5
    obs.total_tokens = 4096
    obs.batch_size = 8
    obs.group_size = 16
    obs.skipped = True

    var copy = obs.copy()
    assert_almost_equal(copy.step_time_s, 1.5, atol=1e-10)
    assert_equal(copy.total_tokens, 4096)
    assert_equal(copy.batch_size, 8)
    assert_equal(copy.group_size, 16)
    assert_true(copy.skipped, "skipped should be copied")


fn test_decision_defaults() raises:
    """BackPressureDecision defaults are sensible."""
    var d = BackPressureDecision()
    assert_equal(d.action, "hold")
    assert_equal(d.regime, "warmup")
    assert_almost_equal(d.p_star, 1.0, atol=1e-10)
    assert_almost_equal(d.sigma, 0.0, atol=1e-10)
    assert_almost_equal(d.kappa, 0.0, atol=1e-10)
    assert_almost_equal(d.utilization, 0.0, atol=1e-10)
    assert_almost_equal(d.throughput, 0.0, atol=1e-10)


fn test_decision_writable() raises:
    """BackPressureDecision is Writable (can be printed)."""
    var d = BackPressureDecision()
    var s = String(d)
    assert_true(len(s) > 0, "Should produce non-empty string")
    assert_true(s.find("BackPressureDecision") != -1, "Should contain type name")
    assert_true(s.find("hold") != -1, "Should contain action")


# ---------------------------------------------------------------------------
# 7. Integration (4)
# ---------------------------------------------------------------------------


fn test_observe_recommend_cycle() raises:
    """Full observe → recommend cycle produces consistent results."""
    var bp = USLBackPressure(warmup_steps=2)

    # Warmup phase
    for i in range(3):
        var obs = make_obs(1.0, (i + 1) * 1000, batch_size=i + 1, group_size=1)
        bp.observe(obs)

    # Should have left warmup
    var decision = bp.recommend()
    assert_true(
        decision.action == "hold" or decision.action == "throttle" or decision.action == "increase",
        "Action should be one of hold/throttle/increase",
    )
    assert_true(decision.regime != "warmup", "Should have left warmup")


fn test_reset_clears_state() raises:
    """reset() clears all accumulated state."""
    var bp = USLBackPressure(warmup_steps=0)

    # Accumulate state
    for p in range(1, 6):
        var tput = usl_throughput(Float64(p), 0.05, 0.002) * 1000.0
        var obs = make_obs(1.0, Int(tput), batch_size=p, group_size=1)
        bp.observe(obs)

    assert_true(bp.step_count > 0, "Should have observations")
    assert_true(len(bp.obs_p) > 0, "Should have data points")

    bp.reset()

    assert_equal(bp.step_count, 0)
    assert_equal(len(bp.obs_p), 0)
    assert_equal(len(bp.obs_x), 0)
    assert_almost_equal(bp.sigma, 0.0, atol=1e-10)
    assert_almost_equal(bp.kappa, 0.0, atol=1e-10)
    assert_equal(bp.regime, "warmup")


fn test_validation_rejects_bad_config() raises:
    """validate() raises on invalid config."""
    var bp = USLBackPressure(warmup_steps=-1)
    var caught = False
    try:
        bp.validate()
    except:
        caught = True
    assert_true(caught, "Should reject negative warmup_steps")

    var bp2 = USLBackPressure(ema_decay=1.5)
    var caught2 = False
    try:
        bp2.validate()
    except:
        caught2 = True
    assert_true(caught2, "Should reject ema_decay > 1")

    var bp3 = USLBackPressure(min_batch_size=10, max_batch_size=5)
    var caught3 = False
    try:
        bp3.validate()
    except:
        caught3 = True
    assert_true(caught3, "Should reject max < min batch_size")


struct AlwaysThrottleBP(BackPressure):
    """Custom BackPressure impl for testing trait composability."""

    fn __init__(out self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass

    fn observe(mut self, obs: StepObservation) raises:
        pass

    fn recommend(self) -> BackPressureDecision:
        var d = BackPressureDecision()
        d.action = "throttle"
        d.regime = "custom"
        return d^

    fn reset(mut self):
        pass


fn test_custom_backpressure_trait_compiles() raises:
    """Custom BackPressure trait implementation compiles and works."""
    var bp = AlwaysThrottleBP()
    var obs = make_obs(1.0, 1000)
    bp.observe(obs)
    var decision = bp.recommend()
    assert_equal(decision.action, "throttle")
    assert_equal(decision.regime, "custom")
    bp.reset()


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


fn main() raises:
    # 1. USL Math (5)
    test_usl_p1_identity()
    test_usl_linear_scaling()
    test_usl_amdahl()
    test_usl_retrograde_peak()
    test_usl_optimal_p_star()

    # 2. USL Fitting (5)
    test_fit_recovers_known_params()
    test_fit_minimum_points_guard()
    test_fit_clamps_negative_sigma()
    test_fit_clamps_negative_kappa()
    test_fit_constant_throughput()

    # 3. Roofline (4)
    test_roofline_warmup_regime()
    test_roofline_retrograde_detection()
    test_roofline_regime_transitions()
    test_roofline_skipped_preserves_regime()

    # 4. Decision Logic (6)
    test_decision_hold_during_warmup()
    test_decision_throttle_above_p_star()
    test_decision_increase_below_p_star()
    test_decision_hold_in_optimal_zone()
    test_decision_batch_clamped()
    test_decision_params_propagated()

    # 5. NoOp (3)
    test_noop_always_hold()
    test_noop_observe_is_noop()
    test_noop_reset_is_noop()

    # 6. Structs (3)
    test_step_observation_copy()
    test_decision_defaults()
    test_decision_writable()

    # 7. Integration (4)
    test_observe_recommend_cycle()
    test_reset_clears_state()
    test_validation_rejects_bad_config()
    test_custom_backpressure_trait_compiles()

    print("All 30 backpressure tests passed!")
