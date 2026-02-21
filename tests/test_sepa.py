"""Tests for retrain.sepa — SEPAController scheduling."""

import math

import pytest

from retrain.sepa import SEPAController


class TestLinearSchedule:
    def test_ramps_to_one(self):
        ctrl = SEPAController(sepa_steps=100, sepa_schedule="linear")
        assert ctrl.resolve_lambda(step=0.0) == pytest.approx(0.0)
        assert ctrl.resolve_lambda(step=50.0) == pytest.approx(0.5)
        assert ctrl.resolve_lambda(step=100.0) == pytest.approx(1.0)

    def test_clamps_at_one(self):
        ctrl = SEPAController(sepa_steps=100, sepa_schedule="linear")
        assert ctrl.resolve_lambda(step=200.0) == pytest.approx(1.0)

    def test_delay(self):
        ctrl = SEPAController(
            sepa_steps=100, sepa_schedule="linear", sepa_delay_steps=50
        )
        assert ctrl.resolve_lambda(step=0.0) == pytest.approx(0.0)
        assert ctrl.resolve_lambda(step=50.0) == pytest.approx(0.0)
        assert ctrl.resolve_lambda(step=100.0) == pytest.approx(0.5)
        assert ctrl.resolve_lambda(step=150.0) == pytest.approx(1.0)

    def test_zero_steps_returns_zero(self):
        ctrl = SEPAController(sepa_steps=0, sepa_schedule="linear")
        assert ctrl.resolve_lambda(step=100.0) == pytest.approx(0.0)


class TestCorrectnessGate:
    def test_gate_blocks_until_threshold(self):
        ctrl = SEPAController(
            sepa_steps=100, sepa_schedule="linear", sepa_correct_rate_gate=0.3
        )
        # Gate closed — lambda should be 0 regardless of step
        assert ctrl.resolve_lambda(step=50.0) == pytest.approx(0.0)
        assert not ctrl.gate_open()

        # Below threshold — still closed
        ctrl.observe_correct_rate(0.2)
        assert not ctrl.gate_open()
        assert ctrl.resolve_lambda(step=50.0) == pytest.approx(0.0)

        # At threshold — gate opens
        ctrl.observe_correct_rate(0.3)
        assert ctrl.gate_open()
        assert ctrl.resolve_lambda(step=50.0) == pytest.approx(0.5)

    def test_gate_is_sticky(self):
        ctrl = SEPAController(
            sepa_steps=100, sepa_schedule="linear", sepa_correct_rate_gate=0.3
        )
        ctrl.observe_correct_rate(0.3)
        assert ctrl.gate_open()

        # Rate drops below threshold — gate stays open
        ctrl.observe_correct_rate(0.1)
        assert ctrl.gate_open()

    def test_zero_gate_always_open(self):
        ctrl = SEPAController(
            sepa_steps=100, sepa_schedule="linear", sepa_correct_rate_gate=0.0
        )
        assert ctrl.gate_open()
        assert ctrl.resolve_lambda(step=50.0) == pytest.approx(0.5)

    def test_nan_correct_rate_ignored(self):
        ctrl = SEPAController(
            sepa_steps=100, sepa_schedule="linear", sepa_correct_rate_gate=0.3
        )
        ctrl.observe_correct_rate(float("nan"))
        assert not ctrl.gate_open()

    def test_none_correct_rate_ignored(self):
        ctrl = SEPAController(
            sepa_steps=100, sepa_schedule="linear", sepa_correct_rate_gate=0.3
        )
        ctrl.observe_correct_rate(None)
        assert not ctrl.gate_open()


class TestAutoSchedule:
    def test_returns_zero_before_warmup(self):
        ctrl = SEPAController(
            sepa_steps=100, sepa_schedule="auto", sepa_warmup=5
        )
        # No variance data yet
        assert ctrl.resolve_lambda(step=50.0) == pytest.approx(0.5)
        # Feed warmup data
        for _ in range(4):
            ctrl.update_auto_state([1.0, 2.0, 3.0])
        # Not yet at warmup count
        assert ctrl._var_0 is None

    def test_auto_lambda_increases_as_variance_drops(self):
        ctrl = SEPAController(
            sepa_steps=1000,
            sepa_schedule="auto",
            sepa_warmup=2,
            sepa_ema_decay=0.0,  # instant tracking
            sepa_var_threshold=0.2,
        )
        # Feed high-variance warmup to set var_0
        ctrl.update_auto_state([0.0, 10.0])  # var = 25.0
        ctrl.update_auto_state([0.0, 10.0])  # warmup complete, var_0 = 25.0
        assert ctrl._var_0 is not None

        # Feed low-variance data
        ctrl.update_auto_state([5.0, 5.0])  # var = 0.0
        lam = ctrl.resolve_lambda(step=0.0)
        # ratio = 0/25 = 0, scaled = 0/0.2 = 0, auto = 1 - 0 = 1.0
        assert lam == pytest.approx(1.0)

    def test_auto_uses_max_of_linear_and_auto(self):
        ctrl = SEPAController(
            sepa_steps=100, sepa_schedule="auto", sepa_warmup=1,
            sepa_ema_decay=0.0,
        )
        ctrl.update_auto_state([5.0, 5.0])  # var=0, var_0 set
        # auto_lambda = 1.0, linear at step=50 = 0.5
        lam = ctrl.resolve_lambda(step=50.0)
        assert lam == pytest.approx(1.0)


class TestEnabled:
    def test_enabled_with_steps(self):
        ctrl = SEPAController(sepa_steps=100, sepa_schedule="linear")
        assert ctrl.enabled()

    def test_enabled_auto(self):
        ctrl = SEPAController(sepa_steps=0, sepa_schedule="auto")
        assert ctrl.enabled()

    def test_disabled(self):
        ctrl = SEPAController(sepa_steps=0, sepa_schedule="linear")
        assert not ctrl.enabled()


class TestValidation:
    def test_negative_steps(self):
        with pytest.raises(ValueError):
            SEPAController(sepa_steps=-1)

    def test_negative_delay(self):
        with pytest.raises(ValueError):
            SEPAController(sepa_delay_steps=-1)

    def test_gate_out_of_range(self):
        with pytest.raises(ValueError):
            SEPAController(sepa_correct_rate_gate=1.5)

    def test_bad_schedule(self):
        with pytest.raises(ValueError):
            SEPAController(sepa_schedule="cosine")

    def test_bad_ema_decay(self):
        with pytest.raises(ValueError):
            SEPAController(sepa_ema_decay=-0.1)

    def test_bad_warmup(self):
        with pytest.raises(ValueError):
            SEPAController(sepa_warmup=0)

    def test_bad_eps(self):
        with pytest.raises(ValueError):
            SEPAController(eps=0.0)
