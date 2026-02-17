"""Tests for SEPAController — port of test_advantages.py TestSEPAController.

Hypotheses tested:
    H9: SEPA controller scheduling matches Python original
        - disabled when zero steps + linear schedule
        - linear ramp from 0 to 1
        - delay steps honored
        - correctness gate blocks then opens
        - auto schedule warmup
"""

from testing import assert_true, assert_false, assert_almost_equal, assert_equal
from math import abs
from collections import Optional

from src.sepa import SEPAController


fn approx_eq(a: Float64, b: Float64, tol: Float64 = 1e-6) -> Bool:
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# H9: SEPA Controller scheduling
# ---------------------------------------------------------------------------


fn test_disabled_when_zero_steps() raises:
    """Steps=0, schedule=linear -> not enabled."""
    var ctrl = SEPAController(sepa_steps=0, sepa_schedule="linear")
    assert_false(ctrl.enabled())


fn test_linear_ramp() raises:
    """Linear schedule: lambda ramps from 0 to 1 over sepa_steps."""
    var ctrl = SEPAController(sepa_steps=100, sepa_schedule="linear")
    assert_almost_equal(ctrl.resolve_lambda(step=0.0), 0.0, atol=1e-6)
    assert_almost_equal(ctrl.resolve_lambda(step=50.0), 0.5, atol=1e-6)
    assert_almost_equal(ctrl.resolve_lambda(step=100.0), 1.0, atol=1e-6)
    assert_almost_equal(ctrl.resolve_lambda(step=200.0), 1.0, atol=1e-6)


fn test_delay_steps() raises:
    """Delay: lambda stays 0 until delay_steps passed."""
    var ctrl = SEPAController(
        sepa_steps=100, sepa_schedule="linear", sepa_delay_steps=50
    )
    assert_almost_equal(ctrl.resolve_lambda(step=0.0), 0.0, atol=1e-6)
    assert_almost_equal(ctrl.resolve_lambda(step=50.0), 0.0, atol=1e-6)
    assert_almost_equal(ctrl.resolve_lambda(step=100.0), 0.5, atol=1e-6)
    assert_almost_equal(ctrl.resolve_lambda(step=150.0), 1.0, atol=1e-6)


fn test_correctness_gate() raises:
    """Lambda stays 0 until correct rate exceeds gate threshold."""
    var ctrl = SEPAController(
        sepa_steps=100,
        sepa_schedule="linear",
        sepa_correct_rate_gate=0.5,
    )
    # Gate closed -> lambda = 0 regardless of step
    assert_almost_equal(ctrl.resolve_lambda(step=50.0), 0.0, atol=1e-6)

    # Below threshold
    ctrl.observe_correct_rate(Optional[Float64](0.3))
    assert_false(ctrl.gate_open())
    assert_almost_equal(ctrl.resolve_lambda(step=50.0), 0.0, atol=1e-6)

    # Meets threshold -> gate opens (sticky)
    ctrl.observe_correct_rate(Optional[Float64](0.5))
    assert_true(ctrl.gate_open())
    assert_true(ctrl.resolve_lambda(step=50.0) > 0.0, "Lambda should be > 0 after gate opens")


fn test_auto_schedule_warmup() raises:
    """Auto schedule: needs warmup observations before producing lambda."""
    var ctrl = SEPAController(
        sepa_steps=0, sepa_schedule="auto", sepa_warmup=3
    )
    assert_true(ctrl.enabled())

    var entropies = List[Float64]()
    entropies.append(1.0)
    entropies.append(2.0)
    entropies.append(3.0)

    # Not enough warmup
    ctrl.update_auto_state(entropies)
    ctrl.update_auto_state(entropies)
    # After 2 updates, var_0 is still None
    assert_true(ctrl._var_0 is None, "var_0 should be None after 2 updates")

    # Third update triggers var_0 latch
    ctrl.update_auto_state(entropies)
    assert_true(ctrl._var_0 is not None, "var_0 should be set after 3 updates")


fn test_correctness_gate_sticky() raises:
    """Once gate opens, it stays open even if correct rate drops."""
    var ctrl = SEPAController(
        sepa_steps=100,
        sepa_schedule="linear",
        sepa_correct_rate_gate=0.5,
    )
    # Open gate
    ctrl.observe_correct_rate(Optional[Float64](0.6))
    assert_true(ctrl.gate_open())

    # Correct rate drops below threshold
    ctrl.observe_correct_rate(Optional[Float64](0.2))
    assert_true(ctrl.gate_open(), "Gate should remain open (sticky)")


fn test_validation_negative_steps() raises:
    """Negative sepa_steps should raise."""
    var raised = False
    try:
        var ctrl = SEPAController(sepa_steps=-1)
    except:
        raised = True
    assert_true(raised, "Should raise on negative sepa_steps")


fn test_validation_bad_schedule() raises:
    """Invalid schedule should raise."""
    var raised = False
    try:
        var ctrl = SEPAController(sepa_schedule="bogus")
    except:
        raised = True
    assert_true(raised, "Should raise on invalid schedule")


fn test_observe_none_rate() raises:
    """Observing None correct rate should be no-op."""
    var ctrl = SEPAController(
        sepa_steps=100,
        sepa_schedule="linear",
        sepa_correct_rate_gate=0.5,
    )
    ctrl.observe_correct_rate(Optional[Float64](None))
    assert_false(ctrl.gate_open())


fn test_auto_empty_entropies() raises:
    """Update with empty entropies is a no-op."""
    var ctrl = SEPAController(sepa_steps=0, sepa_schedule="auto", sepa_warmup=1)
    ctrl.update_auto_state(List[Float64]())
    assert_true(ctrl._var_ema is None, "var_ema should remain None after empty update")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


fn main() raises:
    test_disabled_when_zero_steps()
    test_linear_ramp()
    test_delay_steps()
    test_correctness_gate()
    test_auto_schedule_warmup()
    test_correctness_gate_sticky()
    test_validation_negative_steps()
    test_validation_bad_schedule()
    test_observe_none_rate()
    test_auto_empty_entropies()

    print("All 10 SEPA tests passed!")
