"""Tests for _apply_advantage_cap — pre-training advantage magnitude capping.

This is NOT ratio clipping. It bounds advantage values before they reach the
backend's loss function. The tests verify the arithmetic is correct AND
document what the mechanism actually measures (our preprocessing, not training
dynamics).
"""

import pytest

from retrain.trainer import _apply_advantage_cap


class TestApplyAdvantageCap:
    def test_no_capping_when_within_bounds(self):
        """Advantages within [-cap, +cap] are unchanged."""
        advs = [[1.0, -2.0, 0.5], [3.0, -1.0]]
        result, frac, mag = _apply_advantage_cap(advs, cap=5.0)
        assert result == advs
        assert frac == 0.0
        assert mag == 0.0

    def test_caps_extreme_values(self):
        """Values beyond cap are clamped to +/- cap."""
        advs = [[10.0, -10.0, 1.0]]
        result, frac, mag = _apply_advantage_cap(advs, cap=5.0)
        assert result == [[5.0, -5.0, 1.0]]
        # 2 of 3 non-zero tokens capped
        assert frac == pytest.approx(2.0 / 3.0)
        # Mean magnitude of capped tokens: (10 + 10) / 2 = 10
        assert mag == pytest.approx(10.0)

    def test_zero_advantages_excluded_from_fraction(self):
        """Zero-valued advantages (prompt padding) don't count."""
        advs = [[0.0, 0.0, 10.0, 1.0]]
        result, frac, mag = _apply_advantage_cap(advs, cap=5.0)
        assert result == [[0.0, 0.0, 5.0, 1.0]]
        # 1 of 2 non-zero tokens capped
        assert frac == pytest.approx(0.5)

    def test_empty_input(self):
        result, frac, mag = _apply_advantage_cap([], cap=5.0)
        assert result == []
        assert frac == 0.0

    def test_all_zeros(self):
        """All-zero sequences (pure padding) produce 0 fraction."""
        advs = [[0.0, 0.0], [0.0]]
        result, frac, mag = _apply_advantage_cap(advs, cap=5.0)
        assert result == advs
        assert frac == 0.0

    def test_preserves_structure(self):
        """Output has same number of sequences and tokens."""
        advs = [[1.0, 2.0], [3.0, 4.0, 5.0]]
        result, _, _ = _apply_advantage_cap(advs, cap=3.0)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 3

    def test_symmetric_capping(self):
        """Positive and negative extremes are capped symmetrically."""
        advs = [[100.0], [-100.0]]
        result, frac, _ = _apply_advantage_cap(advs, cap=1.0)
        assert result == [[1.0], [-1.0]]
        assert frac == pytest.approx(1.0)

    def test_cap_fraction_is_diagnostic_not_causal(self):
        """Demonstrate that cap_fraction measures our preprocessing.

        This test documents the key difference from PPO ratio clipping:
        cap_fraction tells us what fraction of advantages WE modified,
        not what fraction of policy updates were constrained during training.
        A high cap_fraction means we're intervening heavily on the signal,
        which may or may not improve training — the proof is in the
        entropy trajectory and correct_rate, not in this number.
        """
        # Extreme advantages that would cause large gradient updates
        advs = [[50.0, -50.0, 0.1, -0.1]]
        _, frac, mag = _apply_advantage_cap(advs, cap=5.0)
        # We capped 2/4 non-zero tokens — that's our intervention rate
        assert frac == pytest.approx(0.5)
        # Mean pre-cap magnitude of capped tokens: 50.0
        assert mag == pytest.approx(50.0)
        # Whether this improves training is measured by exec_surprisal_var
        # and correct_rate, NOT by cap_fraction itself.
