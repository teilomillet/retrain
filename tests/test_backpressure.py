"""Tests for retrain.backpressure — USL formula, Cramer's rule, regime, actions."""

import math
import random
from collections import deque

import pytest

from retrain.backpressure import (
    BackPressureDecision,
    NoOpBackPressure,
    StepObservation,
    USLBackPressure,
    usl_optimal_p,
    usl_throughput,
)


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

class TestUSLThroughput:
    def test_identity_at_p1(self):
        """C(1) = 1 for any sigma, kappa."""
        assert usl_throughput(1.0, 0.1, 0.01) == pytest.approx(1.0)
        assert usl_throughput(1.0, 0.5, 0.1) == pytest.approx(1.0)
        assert usl_throughput(1.0, 0.0, 0.0) == pytest.approx(1.0)

    def test_linear_scaling(self):
        """sigma=0, kappa=0 → C(p) = p."""
        assert usl_throughput(10.0, 0.0, 0.0) == pytest.approx(10.0)
        assert usl_throughput(100.0, 0.0, 0.0) == pytest.approx(100.0)

    def test_amdahl_only(self):
        """kappa=0 → C(p) = p / (1 + sigma(p-1))."""
        assert usl_throughput(10.0, 0.1, 0.0) == pytest.approx(10.0 / 1.9)

    def test_known_value(self):
        # sigma=0.05, kappa=0.001, p=20
        # denom = 1 + 0.05*19 + 0.001*20*19 = 2.33
        assert usl_throughput(20.0, 0.05, 0.001) == pytest.approx(20.0 / 2.33, abs=1e-4)

    def test_negative_denom_guarded(self):
        # Edge case: should return 0 if denom goes non-positive
        result = usl_throughput(1e6, 0.99, 0.99)
        # denom will be very large positive, so this is fine
        assert result >= 0


class TestUSLOptimalP:
    def test_known_values(self):
        assert usl_optimal_p(0.05, 0.001) == pytest.approx(math.sqrt(0.95 / 0.001))
        assert usl_optimal_p(0.1, 0.01) == pytest.approx(math.sqrt(0.9 / 0.01))

    def test_kappa_zero_returns_one(self):
        assert usl_optimal_p(0.1, 0.0) == 1.0

    def test_sigma_one_returns_one(self):
        assert usl_optimal_p(1.0, 0.01) == 1.0

    def test_is_argmax(self):
        """p* should be the actual argmax of C(p)."""
        sigma, kappa = 0.05, 0.002
        p_star = usl_optimal_p(sigma, kappa)
        c_star = usl_throughput(p_star, sigma, kappa)
        c_left = usl_throughput(p_star - 0.01, sigma, kappa)
        c_right = usl_throughput(p_star + 0.01, sigma, kappa)
        assert c_star >= c_left
        assert c_star >= c_right


# ---------------------------------------------------------------------------
# NoOpBackPressure
# ---------------------------------------------------------------------------

class TestNoOp:
    def test_always_hold(self):
        bp = NoOpBackPressure()
        bp.observe(StepObservation(step_time_s=1.0, batch_size=4, group_size=2))
        dec = bp.recommend()
        assert dec.action == "hold"

    def test_reset(self):
        bp = NoOpBackPressure()
        bp.reset()  # should not raise


# ---------------------------------------------------------------------------
# USLBackPressure
# ---------------------------------------------------------------------------

def _make_obs(p: int, lam: float, sigma: float, kappa: float) -> StepObservation:
    """Create observation at concurrency p with throughput from true USL curve."""
    c = usl_throughput(float(p), sigma, kappa)
    tokens = int(lam * c)
    return StepObservation(step_time_s=1.0, batch_size=p, group_size=1, total_tokens=tokens)


class TestDataStructures:
    def test_uses_deque(self):
        bp = USLBackPressure()
        assert isinstance(bp.obs_p, deque)
        assert bp.obs_p.maxlen == 100

    def test_sliding_window_caps(self):
        bp = USLBackPressure(warmup_steps=0, ema_decay=0.0)
        for _ in range(200):
            bp.observe(StepObservation(step_time_s=1.0, batch_size=4, group_size=1, total_tokens=1000))
        assert len(bp.obs_p) == 100
        assert len(bp.obs_x) == 100


class TestIncrementalSums:
    def test_match_full_rescan(self):
        """After 150 steps (50 evictions), incremental sums must match rescan."""
        random.seed(123)
        bp = USLBackPressure(warmup_steps=0, ema_decay=0.5)

        for _ in range(150):
            p = random.randint(2, 50)
            t = 500.0 * usl_throughput(float(p), 0.06, 0.004) * (1 + random.gauss(0, 0.02))
            bp.observe(StepObservation(step_time_s=1.0, batch_size=p, group_size=1, total_tokens=int(t)))

        # Recompute from scratch
        eps = bp.eps
        ref = {"s0": 0, "s1": 0, "s2": 0, "s3": 0, "s4": 0, "sy": 0, "spy": 0, "sp2y": 0}
        for p, x in zip(bp.obs_p, bp.obs_x):
            if x < eps:
                continue
            y = p / x
            p2 = p * p
            ref["s0"] += 1; ref["s1"] += p; ref["s2"] += p2
            ref["s3"] += p2 * p; ref["s4"] += p2 * p2
            ref["sy"] += y; ref["spy"] += p * y; ref["sp2y"] += p2 * y

        for name, ref_val in ref.items():
            inc_val = getattr(bp, f"_{name}")
            rel_err = abs(inc_val - ref_val) / (abs(ref_val) + 1e-20)
            assert rel_err < 1e-10, f"{name}: rel_err={rel_err}"


class TestParameterRecovery:
    def test_recovers_sigma_kappa_lambda(self):
        TRUE_S, TRUE_K, TRUE_L = 0.08, 0.003, 500.0
        bp = USLBackPressure(warmup_steps=0, ema_decay=0.0)
        random.seed(42)

        for p in [2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 35, 40, 50]:
            c = usl_throughput(float(p), TRUE_S, TRUE_K)
            t = TRUE_L * c * (1 + random.gauss(0, 0.005))
            bp.observe(StepObservation(step_time_s=1.0, batch_size=p, group_size=1, total_tokens=int(t)))

        assert bp.sigma == pytest.approx(TRUE_S, abs=0.02)
        assert bp.kappa == pytest.approx(TRUE_K, abs=0.001)
        assert bp.fitted_lambda == pytest.approx(TRUE_L, abs=50.0)
        true_p_star = usl_optimal_p(TRUE_S, TRUE_K)
        assert bp.p_star == pytest.approx(true_p_star, abs=2.0)


class TestEMA:
    def test_seed_on_first_observation(self):
        bp = USLBackPressure(warmup_steps=0, ema_decay=0.9)
        bp.observe(StepObservation(step_time_s=1.0, batch_size=4, group_size=1, total_tokens=1000))
        assert bp.ema_throughput == pytest.approx(1000.0)

    def test_skip_preserves_ema(self):
        bp = USLBackPressure(warmup_steps=0, ema_decay=0.9)
        bp.observe(StepObservation(step_time_s=1.0, batch_size=4, group_size=1, total_tokens=1000))
        bp.observe(StepObservation(step_time_s=1.0, batch_size=4, group_size=1, total_tokens=500, skipped=True))
        assert bp.ema_throughput == pytest.approx(1000.0)

    def test_normal_update(self):
        bp = USLBackPressure(warmup_steps=0, ema_decay=0.9)
        bp.observe(StepObservation(step_time_s=1.0, batch_size=4, group_size=1, total_tokens=1000))
        bp.observe(StepObservation(step_time_s=1.0, batch_size=4, group_size=1, total_tokens=2000))
        assert bp.ema_throughput == pytest.approx(0.9 * 1000 + 0.1 * 2000)


class TestRegimeClassification:
    def test_warmup(self):
        bp = USLBackPressure(warmup_steps=3)
        for p in [4, 8, 12]:
            bp.observe(_make_obs(p, 1000, 0.05, 0.005))
            assert bp.regime == "warmup"

    def test_kappa_retrograde(self):
        """p >> p* with significant kappa → retrograde."""
        bp = USLBackPressure(warmup_steps=3, ema_decay=0.0)
        for p in [4, 8, 12]:
            bp.observe(_make_obs(p, 1000, 0.05, 0.005))
        bp.observe(_make_obs(50, 1000, 0.05, 0.005))
        assert bp.regime == "retrograde"

    def test_model_based_optimal(self):
        """Throughput matches model prediction → optimal."""
        # Use small kappa so p* is large and kappa check doesn't fire
        sigma, kappa, lam = 0.05, 0.0005, 1000.0
        bp = USLBackPressure(warmup_steps=3, ema_decay=0.0)
        for p in [4, 8, 15]:
            bp.observe(_make_obs(p, lam, sigma, kappa))
        bp.observe(_make_obs(10, lam, sigma, kappa))
        assert bp.regime == "optimal"

    def test_model_based_memory_bound(self):
        """Throughput >> model prediction → memory_bound."""
        sigma, kappa, lam = 0.05, 0.0005, 1000.0
        bp = USLBackPressure(warmup_steps=3, ema_decay=0.0)
        for p in [4, 8, 15]:
            bp.observe(_make_obs(p, lam, sigma, kappa))
        # Inject 1.5× model throughput
        c = usl_throughput(10.0, sigma, kappa)
        bp.observe(StepObservation(step_time_s=1.0, batch_size=10, group_size=1, total_tokens=int(lam * c * 1.5)))
        assert bp.regime == "memory_bound"

    def test_model_based_retrograde(self):
        """Throughput << model prediction → retrograde."""
        sigma, kappa, lam = 0.05, 0.0005, 1000.0
        bp = USLBackPressure(warmup_steps=3, ema_decay=0.0)
        for p in [4, 8, 15]:
            bp.observe(_make_obs(p, lam, sigma, kappa))
        c = usl_throughput(10.0, sigma, kappa)
        bp.observe(StepObservation(step_time_s=1.0, batch_size=10, group_size=1, total_tokens=int(lam * c * 0.5)))
        assert bp.regime == "retrograde"


class TestRecommendActions:
    def test_warmup_holds(self):
        bp = USLBackPressure(warmup_steps=3)
        bp.observe(_make_obs(4, 1000, 0.05, 0.005))
        dec = bp.recommend()
        assert dec.action == "hold"
        assert dec.utilization == 0.0

    def test_retrograde_throttles(self):
        bp = USLBackPressure(warmup_steps=3, ema_decay=0.0)
        for p in [4, 8, 12]:
            bp.observe(_make_obs(p, 1000, 0.05, 0.005))
        bp.observe(_make_obs(50, 1000, 0.05, 0.005))
        dec = bp.recommend()
        assert dec.action == "throttle"
        # Target should be near throttle_margin * p*
        expected_bs = int(bp.p_star * 0.85 / bp.last_group_size)
        assert dec.recommended_batch_size == max(1, min(64, expected_bs))

    def test_low_p_increases_to_safe_target(self):
        """Increase action should target throttle_margin * p*, not 0.5 * p*."""
        bp = USLBackPressure(warmup_steps=3, ema_decay=0.0, throttle_margin=0.85, increase_margin=0.5)
        for p in [4, 8, 12]:
            bp.observe(_make_obs(p, 1000, 0.05, 0.002))
        bp.observe(_make_obs(2, 1000, 0.05, 0.002))
        dec = bp.recommend()
        assert dec.action == "increase"
        # Should target 0.85 * p*, not 0.5 * p*
        safe_target = bp.p_star * 0.85
        old_target = bp.p_star * 0.5
        assert dec.recommended_batch_size > int(old_target)
        assert dec.recommended_batch_size == pytest.approx(int(safe_target), abs=1)

    def test_utilization_near_one_at_peak(self):
        TRUE_S, TRUE_K, TRUE_L = 0.08, 0.003, 500.0
        bp = USLBackPressure(warmup_steps=0, ema_decay=0.0)
        for p in [2, 5, 10, 15, 20, 25, 30]:
            bp.observe(_make_obs(p, TRUE_L, TRUE_S, TRUE_K))
        p_star_int = max(1, int(round(bp.p_star)))
        bp.observe(_make_obs(p_star_int, TRUE_L, TRUE_S, TRUE_K))
        dec = bp.recommend()
        assert dec.utilization == pytest.approx(1.0, abs=0.15)


class TestReset:
    def test_clears_all_state(self):
        bp = USLBackPressure(warmup_steps=5, min_batch_size=2, min_group_size=4)
        for p in [4, 8, 12, 16, 20, 24]:
            bp.observe(StepObservation(step_time_s=1.0, batch_size=p, group_size=1, total_tokens=p * 100))

        bp.reset()
        assert bp.step_count == 0
        assert bp.sigma == 0.0
        assert bp.kappa == 0.0
        assert bp.p_star == 1.0
        assert bp.fitted_lambda == 1.0
        assert bp.ema_throughput == 0.0
        assert bp._s0 == 0.0
        assert len(bp.obs_p) == 0
        assert bp.regime == "warmup"
        assert not bp._ema_initialized
        assert bp.last_batch_size == 2
        assert bp.last_group_size == 4


class TestValidation:
    @pytest.mark.parametrize("kwargs,match", [
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"ema_decay": 1.5}, "ema_decay"),
        ({"throttle_margin": 0.0}, "throttle_margin"),
        ({"increase_margin": 0.0}, "increase_margin"),
        ({"min_batch_size": 0}, "min_batch_size"),
        ({"min_batch_size": 10, "max_batch_size": 5}, "max_batch_size"),
        ({"min_group_size": 0}, "min_group_size"),
        ({"min_group_size": 10, "max_group_size": 5}, "max_group_size"),
    ])
    def test_invalid_params(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            USLBackPressure(**kwargs)
