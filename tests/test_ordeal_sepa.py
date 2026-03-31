"""Ordeal ChaosTest for retrain.sepa.

Stateful testing of the SEPAController: step progression, correctness
gating, EMA tracking, state save/load, and mode transitions.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
from ordeal import ChaosTest, always, invariant, rule, sometimes

from retrain.sepa import SEPAController


class SEPAControllerChaos(ChaosTest):
    """Stateful chaos test for SEPAController."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.ctrl = SEPAController(
            sepa_steps=100,
            sepa_schedule="linear",
            sepa_delay_steps=10,
            sepa_correct_rate_gate=0.3,
            sepa_warmup=5,
        )
        self.step_count = 0

    @rule(
        correct_rate=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        )
    )
    def observe_correct_rate(self, correct_rate: float) -> None:
        """Feed a correct rate observation to the gate."""
        self.ctrl.observe_correct_rate(correct_rate)
        if correct_rate >= 0.3:
            always(self.ctrl.gate_open(), "gate must open above threshold")

    @rule()
    def advance_step(self) -> None:
        """Advance step counter and resolve lambda."""
        self.step_count += 1
        lam = self.ctrl.resolve_lambda(float(self.step_count))
        always(0.0 <= lam <= 1.0, "lambda must be in [0, 1]")

    @rule(
        step=st.floats(
            min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False
        )
    )
    def resolve_at_arbitrary_step(self, step: float) -> None:
        """Resolve lambda at an arbitrary step value."""
        lam = self.ctrl.resolve_lambda(step)
        always(0.0 <= lam <= 1.0, "lambda must be in [0, 1]")
        if not self.ctrl.gate_open():
            always(lam == 0.0, "lambda must be 0 when gate closed")

    @rule(
        entropies=st.lists(
            st.floats(
                min_value=0.0,
                max_value=50.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=32,
        )
    )
    def update_auto_state(self, entropies: list[float]) -> None:
        """Feed execution entropies for auto-schedule tracking."""
        self.ctrl.update_auto_state(entropies)

    @rule()
    def roundtrip_state_dict(self) -> None:
        """Save and restore state — lambda must be preserved."""
        state = self.ctrl.state_dict()
        lam_before = self.ctrl.resolve_lambda(float(self.step_count))

        new_ctrl = SEPAController(
            sepa_steps=100,
            sepa_schedule="linear",
            sepa_delay_steps=10,
            sepa_correct_rate_gate=0.3,
            sepa_warmup=5,
        )
        new_ctrl.load_state_dict(state)
        lam_after = new_ctrl.resolve_lambda(float(self.step_count))
        always(
            math.isclose(lam_before, lam_after, abs_tol=1e-12),
            "state roundtrip must preserve lambda",
        )

    @invariant()
    def lambda_bounded(self) -> None:
        lam = self.ctrl.resolve_lambda(float(self.step_count))
        assert 0.0 <= lam <= 1.0

    @invariant()
    def gate_is_sticky(self) -> None:
        """Once open, gate never closes."""
        if self.ctrl.gate_open():
            # Feed a low correct rate — gate should stay open
            self.ctrl.observe_correct_rate(0.0)
            assert self.ctrl.gate_open()

    def teardown(self) -> None:
        sometimes(self.ctrl.gate_open(), "gate opens at least once")
        super().teardown()


TestSEPAControllerChaos = SEPAControllerChaos.TestCase


class SEPAAutoScheduleChaos(ChaosTest):
    """ChaosTest for SEPA auto schedule mode."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.ctrl = SEPAController(
            sepa_steps=50,
            sepa_schedule="auto",
            sepa_correct_rate_gate=0.0,  # Gate always open
            sepa_warmup=3,
        )
        self.step_count = 0

    @rule(
        entropies=st.lists(
            st.floats(
                min_value=0.1,
                max_value=20.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=2,
            max_size=16,
        )
    )
    def feed_entropies(self, entropies: list[float]) -> None:
        """Feed entropy observations and advance."""
        self.ctrl.update_auto_state(entropies)
        self.step_count += 1
        lam = self.ctrl.resolve_lambda(float(self.step_count))
        always(0.0 <= lam <= 1.0, "auto lambda must be in [0, 1]")

    @rule()
    def resolve_lambda(self) -> None:
        lam = self.ctrl.resolve_lambda(float(self.step_count))
        always(0.0 <= lam <= 1.0, "lambda bounded")

    @invariant()
    def auto_lambda_bounded(self) -> None:
        lam = self.ctrl.resolve_lambda(float(self.step_count))
        assert 0.0 <= lam <= 1.0

    def teardown(self) -> None:
        if self.step_count > 5:
            sometimes(
                self.ctrl.resolve_lambda(float(self.step_count)) > 0.0,
                "auto lambda eventually non-zero",
            )
        super().teardown()


TestSEPAAutoScheduleChaos = SEPAAutoScheduleChaos.TestCase
