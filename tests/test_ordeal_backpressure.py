"""Ordeal property tests for retrain.backpressure.

Tests USL math invariants, BackPressure protocol compliance,
and USLBackPressure state machine via ChaosTest.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
from hypothesis import given, settings
from ordeal import ChaosTest, always, invariant, rule, sometimes
from ordeal.invariants import no_nan, no_inf

from retrain.backpressure import (
    BackPressureDecision,
    NoOpBackPressure,
    StepObservation,
    USLBackPressure,
    usl_optimal_p,
    usl_throughput,
)

valid_number = no_nan & no_inf

# ── Strategies ──

sigma_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
# kappa must avoid denormalized values (< 1e-300) which overflow sqrt(1/kappa)
kappa_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False).filter(
    lambda k: k == 0.0 or k >= 1e-300
)
p_st = st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)


# ═══════════════════════════════════════════
# USL Throughput Properties
# ═══════════════════════════════════════════


class TestUSLThroughput:
    @given(sigma=sigma_st, kappa=kappa_st)
    def test_unit_concurrency(self, sigma: float, kappa: float) -> None:
        """C(1) = 1.0 for any sigma, kappa (serial throughput = 1)."""
        result = usl_throughput(1.0, sigma, kappa)
        assert math.isclose(result, 1.0, abs_tol=1e-10)

    @given(p=p_st, sigma=sigma_st, kappa=kappa_st)
    def test_non_negative(self, p: float, sigma: float, kappa: float) -> None:
        """Throughput is always >= 0."""
        result = usl_throughput(p, sigma, kappa)
        assert result >= -1e-10

    @given(p=p_st, sigma=sigma_st, kappa=kappa_st)
    def test_finite(self, p: float, sigma: float, kappa: float) -> None:
        result = usl_throughput(p, sigma, kappa)
        valid_number(result)

    @given(p=p_st)
    def test_zero_contention_linear(self, p: float) -> None:
        """With kappa=0 and sigma=0, throughput scales linearly: C(p) = p."""
        result = usl_throughput(p, sigma=0.0, kappa=0.0)
        assert math.isclose(result, p, rel_tol=1e-8)

    @given(p=p_st, sigma=sigma_st)
    def test_zero_kappa_monotonic(self, p: float, sigma: float) -> None:
        """With kappa=0 (Amdahl's law), throughput is monotonically non-decreasing."""
        t1 = usl_throughput(p, sigma, kappa=0.0)
        t2 = usl_throughput(p + 0.1, sigma, kappa=0.0)
        assert t2 >= t1 - 1e-10

    @given(sigma=sigma_st, kappa=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False))
    def test_retrograde_exists(self, sigma: float, kappa: float) -> None:
        """With kappa > 0, throughput eventually decreases (retrograde region)."""
        t_at_1 = usl_throughput(1.0, sigma, kappa)
        t_at_1000 = usl_throughput(1000.0, sigma, kappa)
        # At very high concurrency, throughput should be less than at p=1
        assert t_at_1000 < t_at_1 + 1e-8

    @given(sigma=sigma_st, kappa=kappa_st)
    def test_deterministic(self, sigma: float, kappa: float) -> None:
        p = 10.0
        assert usl_throughput(p, sigma, kappa) == usl_throughput(p, sigma, kappa)


# ═══════════════════════════════════════════
# USL Optimal Concurrency Properties
# ═══════════════════════════════════════════


class TestUSLOptimalP:
    @given(sigma=sigma_st, kappa=kappa_st)
    def test_positive(self, sigma: float, kappa: float) -> None:
        """Optimal concurrency is always > 0."""
        p_star = usl_optimal_p(sigma, kappa)
        assert p_star > 0.0

    @given(sigma=sigma_st, kappa=kappa_st)
    def test_finite(self, sigma: float, kappa: float) -> None:
        valid_number(usl_optimal_p(sigma, kappa))

    @given(sigma=sigma_st)
    def test_zero_kappa_returns_at_least_one(self, sigma: float) -> None:
        """With no contention, optimal p is unbounded (returns 1.0 as fallback)."""
        p_star = usl_optimal_p(sigma, kappa=0.0)
        assert p_star >= 1.0 - 1e-10

    @given(
        sigma=sigma_st,
        kappa=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_optimal_is_peak(self, sigma: float, kappa: float) -> None:
        """Throughput at p* is >= throughput at p*-1 and p*+1."""
        p_star = usl_optimal_p(sigma, kappa)
        if p_star <= 1.5:
            return  # Can't test neighbors below 1
        t_star = usl_throughput(p_star, sigma, kappa)
        t_below = usl_throughput(max(1.0, p_star - 1.0), sigma, kappa)
        t_above = usl_throughput(p_star + 1.0, sigma, kappa)
        assert t_star >= t_below - 1e-8
        assert t_star >= t_above - 1e-8


# ═══════════════════════════════════════════
# NoOpBackPressure Properties
# ═══════════════════════════════════════════


class TestNoOpBackPressure:
    def test_always_hold(self) -> None:
        bp = NoOpBackPressure()
        bp.observe(StepObservation(step_time_s=1.0, batch_size=8, group_size=16))
        d = bp.recommend()
        assert d.action == "hold"

    @given(n=st.integers(min_value=1, max_value=50))
    def test_hold_after_many_observations(self, n: int) -> None:
        bp = NoOpBackPressure()
        for i in range(n):
            bp.observe(
                StepObservation(step_time_s=0.5, batch_size=8, group_size=16)
            )
        assert bp.recommend().action == "hold"

    def test_reset_is_noop(self) -> None:
        bp = NoOpBackPressure()
        bp.reset()
        assert bp.recommend().action == "hold"


# ═══════════════════════════════════════════
# USLBackPressure ChaosTest
# ═══════════════════════════════════════════


class USLBackPressureChaos(ChaosTest):
    """Stateful chaos test for USLBackPressure."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.bp = USLBackPressure(
            warmup_steps=5,
            min_batch_size=1,
            max_batch_size=32,
            min_group_size=2,
            max_group_size=32,
        )
        self.step_count = 0

    @rule(
        step_time=st.floats(
            min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        batch_size=st.integers(min_value=1, max_value=32),
        group_size=st.integers(min_value=2, max_value=32),
        total_tokens=st.integers(min_value=10, max_value=100000),
    )
    def observe(
        self,
        step_time: float,
        batch_size: int,
        group_size: int,
        total_tokens: int,
    ) -> None:
        obs = StepObservation(
            step_time_s=step_time,
            batch_size=batch_size,
            group_size=group_size,
            total_tokens=total_tokens,
        )
        self.bp.observe(obs)
        self.step_count += 1

    @rule()
    def recommend(self) -> None:
        d = self.bp.recommend()
        always(d.action in ("hold", "throttle", "increase"), "valid action")
        always(0.0 <= d.utilization, "utilization non-negative")
        if d.action != "hold" or d.recommended_batch_size > 0:
            always(
                self.bp.min_batch_size
                <= d.recommended_batch_size
                <= self.bp.max_batch_size,
                "recommended batch size within bounds",
            )

    @rule()
    def reset_and_continue(self) -> None:
        self.bp.reset()
        self.step_count = 0
        d = self.bp.recommend()
        always(d.action == "hold", "hold after reset")

    @invariant()
    def sigma_bounded(self) -> None:
        assert 0.0 <= self.bp.sigma <= 1.0

    @invariant()
    def kappa_non_negative(self) -> None:
        assert self.bp.kappa >= 0.0

    @invariant()
    def p_star_positive(self) -> None:
        assert self.bp.p_star > 0.0

    def teardown(self) -> None:
        super().teardown()


TestUSLBackPressureChaos = USLBackPressureChaos.TestCase


# ═══════════════════════════════════════════
# USLBackPressure Init Validation
# ═══════════════════════════════════════════


class TestUSLBackPressureValidation:
    def test_rejects_negative_warmup(self) -> None:
        with pytest.raises(ValueError):
            USLBackPressure(warmup_steps=-1)

    def test_rejects_min_greater_than_max_batch(self) -> None:
        with pytest.raises(ValueError):
            USLBackPressure(min_batch_size=32, max_batch_size=8)

    def test_rejects_ema_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            USLBackPressure(ema_decay=1.5)

    @given(
        warmup=st.integers(min_value=0, max_value=100),
        ema=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_accepts_valid_params(self, warmup: int, ema: float) -> None:
        bp = USLBackPressure(warmup_steps=warmup, ema_decay=ema)
        assert bp.warmup_steps == warmup


import pytest
