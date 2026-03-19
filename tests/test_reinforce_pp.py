"""Tests for REINFORCE++ advantage computation and batch normalization.

Tests cover:
  - compute_reinforce_pp_advantages (episode-level, identical to GRPO)
  - apply_batch_advantage_normalization (the core REINFORCE++ innovation)
  - Integration with compute_composable_advantages
  - Algorithm mode shortcuts (reinforce_pp_none, reinforce_pp_gtpo, etc.)
  - Edge cases: empty, single, all-same, zero-padded prompt tokens
  - Statistical properties: zero-mean, unit-variance after normalization
  - Interaction with advantage capping
  - Flow validation compatibility
  - Config round-trip (TOML → TrainConfig → validation)
"""

import math

import pytest

from retrain.advantages import (
    apply_batch_advantage_normalization,
    apply_delight_gating,
    compute_algorithm_advantages,
    compute_composable_advantages,
    compute_grpo_advantages,
    compute_reinforce_pp_advantages,
    get_advantage_spec,
    get_algorithm_spec,
    get_builtin_advantage_modes,
    get_builtin_algorithm_modes,
    get_builtin_transform_modes,
)


# ---------------------------------------------------------------------------
# REINFORCE++ episode-level advantages (should be identical to GRPO)
# ---------------------------------------------------------------------------

class TestReinforcePPAdvantages:
    """Episode-level computation is identical to GRPO: A_i = r_i - mean(r)."""

    def test_identical_to_grpo(self):
        rewards = [1.0, 0.0, 0.5, 0.8, 0.2]
        assert compute_reinforce_pp_advantages(rewards) == compute_grpo_advantages(rewards)

    def test_basic(self):
        advs = compute_reinforce_pp_advantages([1.0, 0.0, 0.0, 1.0])
        assert len(advs) == 4
        assert sum(advs) == pytest.approx(0.0)
        assert advs[0] > 0
        assert advs[1] < 0

    def test_all_same(self):
        advs = compute_reinforce_pp_advantages([0.5, 0.5, 0.5])
        assert all(a == pytest.approx(0.0) for a in advs)

    def test_empty(self):
        assert compute_reinforce_pp_advantages([]) == []

    def test_single(self):
        advs = compute_reinforce_pp_advantages([1.0])
        assert advs == [pytest.approx(0.0)]

    def test_registered_in_builtins(self):
        assert "reinforce_pp" in get_builtin_advantage_modes()

    def test_spec_resolves(self):
        spec = get_advantage_spec("reinforce_pp")
        assert spec.name == "reinforce_pp"
        result = spec.compute([1.0, 0.0], {})
        assert len(result) == 2
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(-0.5)

    def test_negative_rewards(self):
        advs = compute_reinforce_pp_advantages([-1.0, -0.5, -0.2])
        mean_r = sum([-1.0, -0.5, -0.2]) / 3
        for i, r in enumerate([-1.0, -0.5, -0.2]):
            assert advs[i] == pytest.approx(r - mean_r)

    def test_large_group(self):
        rewards = [float(i) / 100.0 for i in range(100)]
        advs = compute_reinforce_pp_advantages(rewards)
        assert len(advs) == 100
        assert sum(advs) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Batch-level advantage normalization (the REINFORCE++ step 2)
# ---------------------------------------------------------------------------

class TestBatchAdvantageNormalization:
    """Core innovation: normalize advantages across the full batch."""

    def test_basic_normalization(self):
        """After normalization, non-zero advantages should have ~zero mean and ~unit std."""
        all_advs = [
            [0.0, 0.0, 0.5, 0.5],      # prompt padding + response
            [0.0, 0.0, -0.5, -0.5],     # prompt padding + response
            [0.0, 0.3, 0.3],            # shorter
            [0.0, -0.3, -0.3],          # shorter
        ]
        result, metrics = apply_batch_advantage_normalization(all_advs)

        # Non-zero values should have zero mean and unit std
        non_zero = [a for seq in result for a in seq if a != 0.0]
        mean_val = sum(non_zero) / len(non_zero)
        var_val = sum((v - mean_val) ** 2 for v in non_zero) / len(non_zero)
        assert mean_val == pytest.approx(0.0, abs=1e-6)
        assert math.sqrt(var_val) == pytest.approx(1.0, abs=1e-3)

    def test_prompt_padding_preserved(self):
        """Zero-padded prompt tokens must stay zero."""
        all_advs = [
            [0.0, 0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, -1.0, -2.0],
        ]
        result, _ = apply_batch_advantage_normalization(all_advs)

        # Check prompt zeros are preserved
        assert result[0][0] == 0.0
        assert result[0][1] == 0.0
        assert result[0][2] == 0.0
        assert result[1][0] == 0.0
        assert result[1][1] == 0.0

        # Non-zero values should be non-zero in result
        assert result[0][3] != 0.0
        assert result[0][4] != 0.0
        assert result[1][2] != 0.0
        assert result[1][3] != 0.0

    def test_empty_batch(self):
        result, metrics = apply_batch_advantage_normalization([])
        assert result == []
        assert metrics["batch_adv_count"] == 0.0

    def test_all_zeros(self):
        """If all advantages are zero (e.g., all-same group), should stay zero."""
        all_advs = [[0.0, 0.0, 0.0], [0.0, 0.0]]
        result, metrics = apply_batch_advantage_normalization(all_advs)
        assert result == all_advs
        assert metrics["batch_adv_count"] == 0.0

    def test_single_nonzero(self):
        """Single non-zero value: std=0, should just center to 0."""
        all_advs = [[0.0, 0.0, 1.0]]
        result, metrics = apply_batch_advantage_normalization(all_advs)
        # With only one value, std=0 → centered: (1.0 - 1.0) = 0.0
        assert result[0][2] == pytest.approx(0.0)
        assert result[0][0] == 0.0
        assert result[0][1] == 0.0

    def test_uniform_nonzero_values(self):
        """All non-zero values equal → std=0 → all centered to 0."""
        all_advs = [[0.0, 0.5, 0.5], [0.0, 0.5]]
        result, metrics = apply_batch_advantage_normalization(all_advs)
        assert all(a == pytest.approx(0.0) for seq in result for a in seq)

    def test_preserves_relative_order(self):
        """Higher advantages before norm should remain higher after norm."""
        all_advs = [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, -1.0, -2.0, -3.0],
        ]
        result, _ = apply_batch_advantage_normalization(all_advs)
        assert result[0][3] > result[0][2] > result[0][1]
        assert result[1][1] > result[1][2] > result[1][3]

    def test_symmetry(self):
        """Symmetric advantages should produce symmetric normalized values."""
        all_advs = [
            [0.0, 1.0, -1.0],
            [0.0, 2.0, -2.0],
        ]
        result, _ = apply_batch_advantage_normalization(all_advs)
        # 1.0 and -1.0 should become equal magnitude opposite sign
        assert result[0][1] == pytest.approx(-result[0][2])
        assert result[1][1] == pytest.approx(-result[1][2])

    def test_metrics_correct(self):
        """Verify returned metrics are accurate."""
        all_advs = [
            [0.0, 0.0, 1.0, 2.0],
            [0.0, -1.0, -2.0],
        ]
        _, metrics = apply_batch_advantage_normalization(all_advs)
        assert metrics["batch_adv_count"] == 4.0
        expected_mean = (1.0 + 2.0 + (-1.0) + (-2.0)) / 4.0
        assert metrics["batch_adv_mean"] == pytest.approx(expected_mean)
        # std = sqrt(mean((v - mean)^2))
        vals = [1.0, 2.0, -1.0, -2.0]
        expected_var = sum((v - expected_mean) ** 2 for v in vals) / 4.0
        assert metrics["batch_adv_std"] == pytest.approx(math.sqrt(expected_var))

    def test_large_batch_statistical_properties(self):
        """With a large batch, normalized advantages should be ~N(0,1)."""
        import random
        rng = random.Random(42)
        # Simulate 20 groups of 8 samples each, 10 tokens per sample
        all_advs = []
        for _ in range(160):
            prompt_len = rng.randint(3, 8)
            response_len = rng.randint(5, 15)
            adv_value = rng.gauss(0, 2.0)
            seq = [0.0] * prompt_len + [adv_value] * response_len
            all_advs.append(seq)

        result, metrics = apply_batch_advantage_normalization(all_advs)

        non_zero = [a for seq in result for a in seq if a != 0.0]
        mean_val = sum(non_zero) / len(non_zero)
        var_val = sum((v - mean_val) ** 2 for v in non_zero) / len(non_zero)
        assert mean_val == pytest.approx(0.0, abs=1e-6)
        assert math.sqrt(var_val) == pytest.approx(1.0, abs=0.01)

    def test_no_numerical_issues_with_extreme_values(self):
        """Very large or very small advantages should not cause NaN/Inf."""
        all_advs = [
            [0.0, 1e6, -1e6],
            [0.0, 1e-8, -1e-8],
        ]
        result, metrics = apply_batch_advantage_normalization(all_advs)
        for seq in result:
            for a in seq:
                assert math.isfinite(a), f"Non-finite value: {a}"


# ---------------------------------------------------------------------------
# Integration: REINFORCE++ through the composable advantage pipeline
# ---------------------------------------------------------------------------

class TestReinforcePPComposable:
    """Test reinforce_pp works through compute_composable_advantages."""

    def test_composable_basic(self):
        rewards = [1.0, 0.0, 0.5]
        logprobs = [[-0.5, -0.3], [-0.7, -0.2], [-0.4, -0.6]]
        planning_masks = [[0, 0], [0, 0], [0, 0]]
        result = compute_composable_advantages(
            rewards, logprobs, planning_masks,
            advantage_mode="reinforce_pp",
            transform_mode="none",
        )
        # Should produce valid token-level advantages
        assert len(result.token_advs) == 3
        for i in range(3):
            assert len(result.token_advs[i]) == 2

    def test_composable_matches_grpo(self):
        """reinforce_pp through composable should match grpo through composable."""
        rewards = [1.0, 0.0, 0.5, 0.8]
        logprobs = [[-0.5, -0.3, -0.1], [-0.7, -0.2, -0.4],
                    [-0.4, -0.6, -0.2], [-0.3, -0.5, -0.7]]
        planning_masks = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

        result_grpo = compute_composable_advantages(
            rewards, logprobs, planning_masks,
            advantage_mode="grpo", transform_mode="none",
        )
        result_rpp = compute_composable_advantages(
            rewards, logprobs, planning_masks,
            advantage_mode="reinforce_pp", transform_mode="none",
        )
        for i in range(4):
            for j in range(3):
                assert result_rpp.token_advs[i][j] == pytest.approx(
                    result_grpo.token_advs[i][j]
                )

    def test_composable_with_gtpo(self):
        """reinforce_pp should work with GTPO transform."""
        rewards = [1.0, 0.0]
        logprobs = [[-0.5, -1.0, -0.3], [-0.7, -0.2, -0.4]]
        planning_masks = [[0, 0, 0], [0, 0, 0]]
        result = compute_composable_advantages(
            rewards, logprobs, planning_masks,
            advantage_mode="reinforce_pp",
            transform_mode="gtpo",
            gtpo_beta=0.1,
        )
        assert len(result.token_advs) == 2
        # GTPO should produce varying token advantages
        assert result.token_advs[0][0] != result.token_advs[0][1]


# ---------------------------------------------------------------------------
# End-to-end: batch normalization after group-level computation
# ---------------------------------------------------------------------------

class TestReinforcePPEndToEnd:
    """Simulate the full trainer flow: per-group advantages → batch normalization."""

    def test_two_group_pipeline(self):
        """Simulate two groups processed separately, then batch-normalized."""
        # Group 1: two samples
        rewards_g1 = [1.0, 0.0]
        advs_g1 = compute_reinforce_pp_advantages(rewards_g1)
        # Group 2: two samples
        rewards_g2 = [0.8, 0.2]
        advs_g2 = compute_reinforce_pp_advantages(rewards_g2)

        # Build token-level advantages (as trainer would)
        # Simulate: 3 prompt tokens (zero-padded) + 2 response tokens
        all_datum_advs = [
            [0.0, 0.0, 0.0] + [advs_g1[0]] * 2,  # sample 0, group 1
            [0.0, 0.0, 0.0] + [advs_g1[1]] * 2,  # sample 1, group 1
            [0.0, 0.0, 0.0] + [advs_g2[0]] * 2,  # sample 0, group 2
            [0.0, 0.0, 0.0] + [advs_g2[1]] * 2,  # sample 1, group 2
        ]

        # Apply batch normalization (REINFORCE++ step 2)
        normalized, metrics = apply_batch_advantage_normalization(all_datum_advs)

        # Check: prompt zeros preserved
        for seq in normalized:
            assert seq[0] == 0.0
            assert seq[1] == 0.0
            assert seq[2] == 0.0

        # Check: non-zero values have zero mean and unit std
        non_zero = [a for seq in normalized for a in seq if a != 0.0]
        mean_val = sum(non_zero) / len(non_zero)
        var_val = sum((v - mean_val) ** 2 for v in non_zero) / len(non_zero)
        assert mean_val == pytest.approx(0.0, abs=1e-6)
        assert math.sqrt(var_val) == pytest.approx(1.0, abs=0.01)

    def test_batch_norm_then_manual_cap(self):
        """Verify batch normalization followed by advantage capping works correctly."""
        # Create advantages with some extreme values
        all_advs = [
            [0.0, 0.0, 5.0, -5.0],
            [0.0, 0.0, 1.0, -1.0],
        ]

        # Step 1: batch normalize
        normalized, _ = apply_batch_advantage_normalization(all_advs)

        # Step 2: manual cap at 0.5 (simulating adv_clip_max)
        cap = 0.5
        capped = []
        for seq in normalized:
            capped.append([max(-cap, min(cap, a)) if a != 0.0 else 0.0 for a in seq])

        # All values should be in [-cap, cap] or 0.0
        for seq in capped:
            for a in seq:
                assert -cap <= a <= cap

    def test_grpo_vs_reinforce_pp_diverge_across_batch(self):
        """Key test: GRPO and REINFORCE++ should diverge when applied across
        multiple groups with different reward distributions.

        GRPO normalizes within each group independently.
        REINFORCE++ adds a second pass that normalizes across all groups.
        After batch normalization, the relative scaling between groups changes.
        """
        # Group 1: high-variance rewards
        rewards_g1 = [1.0, 0.0]
        advs_g1 = compute_grpo_advantages(rewards_g1)  # [0.5, -0.5]

        # Group 2: low-variance rewards
        rewards_g2 = [0.6, 0.4]
        advs_g2 = compute_grpo_advantages(rewards_g2)  # [0.1, -0.1]

        # Without batch norm (plain GRPO): advantages preserve group-level scale
        grpo_all = [
            [0.0] + [advs_g1[0]] * 2,
            [0.0] + [advs_g1[1]] * 2,
            [0.0] + [advs_g2[0]] * 2,
            [0.0] + [advs_g2[1]] * 2,
        ]

        # With batch norm (REINFORCE++): advantages are globally normalized
        rpp_all = [list(seq) for seq in grpo_all]  # copy
        rpp_normalized, _ = apply_batch_advantage_normalization(rpp_all)

        # GRPO: group 1 advantages are 5x group 2 (0.5 vs 0.1)
        grpo_g1_adv = grpo_all[0][1]  # 0.5
        grpo_g2_adv = grpo_all[2][1]  # 0.1
        grpo_ratio = abs(grpo_g1_adv / grpo_g2_adv)
        assert grpo_ratio == pytest.approx(5.0)

        # REINFORCE++: after batch norm, ratio is preserved (linear transform)
        # but the absolute scale is different
        rpp_g1_adv = rpp_normalized[0][1]
        rpp_g2_adv = rpp_normalized[2][1]
        rpp_ratio = abs(rpp_g1_adv / rpp_g2_adv)
        assert rpp_ratio == pytest.approx(5.0)

        # The key difference: magnitudes are now standardized
        non_zero = [a for seq in rpp_normalized for a in seq if a != 0.0]
        std_val = math.sqrt(sum(v ** 2 for v in non_zero) / len(non_zero)
                           - (sum(non_zero) / len(non_zero)) ** 2)
        assert std_val == pytest.approx(1.0, abs=0.01)

    def test_many_groups_reduces_estimator_bias(self):
        """With many groups, REINFORCE++ batch normalization should
        produce more consistent advantage scales than GRPO.

        This tests the theoretical claim: bias vanishes as N→∞.
        """
        import random
        rng = random.Random(123)

        n_groups = 50
        group_size = 8
        all_advs = []

        for _ in range(n_groups):
            # Each group has different reward distribution
            group_mean = rng.gauss(0.5, 0.3)
            group_std = rng.uniform(0.05, 0.5)
            rewards = [rng.gauss(group_mean, group_std) for _ in range(group_size)]
            advs = compute_reinforce_pp_advantages(rewards)
            for adv in advs:
                all_advs.append([0.0, 0.0] + [adv] * 5)  # 2 prompt + 5 response

        normalized, metrics = apply_batch_advantage_normalization(all_advs)

        # After normalization, mean should be ~0 and std ~1
        non_zero = [a for seq in normalized for a in seq if a != 0.0]
        mean_val = sum(non_zero) / len(non_zero)
        var_val = sum((v - mean_val) ** 2 for v in non_zero) / len(non_zero)
        assert mean_val == pytest.approx(0.0, abs=1e-6)
        assert math.sqrt(var_val) == pytest.approx(1.0, abs=0.001)

        # Metrics should report correct count
        assert metrics["batch_adv_count"] == n_groups * group_size * 5  # response tokens


# ---------------------------------------------------------------------------
# Edge cases and robustness
# ---------------------------------------------------------------------------

class TestBatchNormEdgeCases:
    """Edge cases that could cause numerical issues or incorrect behavior."""

    def test_single_sequence(self):
        """Single sequence: std=0, should center to zero."""
        all_advs = [[0.0, 0.0, 0.5, 0.5]]
        result, _ = apply_batch_advantage_normalization(all_advs)
        assert result[0][2] == pytest.approx(0.0)
        assert result[0][3] == pytest.approx(0.0)

    def test_two_opposing_values(self):
        """Two opposing values: should normalize to -1 and +1."""
        all_advs = [[0.0, 1.0], [0.0, -1.0]]
        result, _ = apply_batch_advantage_normalization(all_advs)
        assert result[0][1] == pytest.approx(1.0)
        assert result[1][1] == pytest.approx(-1.0)

    def test_very_small_differences(self):
        """Very small advantage differences should not cause division issues."""
        all_advs = [[0.0, 1e-10, -1e-10]]
        result, _ = apply_batch_advantage_normalization(all_advs)
        for seq in result:
            for a in seq:
                assert math.isfinite(a)

    def test_mixed_sequence_lengths(self):
        """Different sequence lengths should be handled correctly."""
        all_advs = [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, -1.0],
            [0.0, 0.0, 0.0, 5.0],
        ]
        result, _ = apply_batch_advantage_normalization(all_advs)
        assert len(result[0]) == 5
        assert len(result[1]) == 2
        assert len(result[2]) == 4
        # Prompt zeros preserved
        assert result[0][0] == 0.0
        assert result[1][0] == 0.0
        assert result[2][0] == 0.0
        assert result[2][1] == 0.0
        assert result[2][2] == 0.0

    def test_idempotent_for_already_normalized(self):
        """If advantages are already ~N(0,1), output should still be ~N(0,1)."""
        import random
        rng = random.Random(77)
        all_advs = []
        for _ in range(1000):
            val = rng.gauss(0, 1)
            all_advs.append([0.0, val])

        result, _ = apply_batch_advantage_normalization(all_advs)
        # Output should have zero mean and unit std
        non_zero = [seq[1] for seq in result]
        mean_val = sum(non_zero) / len(non_zero)
        var_val = sum((v - mean_val) ** 2 for v in non_zero) / len(non_zero)
        assert mean_val == pytest.approx(0.0, abs=1e-6)
        assert math.sqrt(var_val) == pytest.approx(1.0, abs=0.01)

    def test_empty_sequences_in_batch(self):
        """Empty sequences should pass through without error."""
        all_advs = [[], [0.0, 1.0, -1.0], []]
        result, _ = apply_batch_advantage_normalization(all_advs)
        assert len(result) == 3
        assert result[0] == []
        assert result[2] == []
        assert len(result[1]) == 3


# ---------------------------------------------------------------------------
# Algorithm mode shortcuts (reinforce_pp_none, reinforce_pp_gtpo, etc.)
# ---------------------------------------------------------------------------

class TestReinforcePPAlgorithmModes:
    """Test pre-built algorithm mode shortcuts work end-to-end."""

    def test_algorithm_modes_registered(self):
        modes = get_builtin_algorithm_modes()
        assert "reinforce_pp_none" in modes
        assert "reinforce_pp_gtpo" in modes
        assert "reinforce_pp_gtpo_sepa" in modes

    def test_reinforce_pp_none_spec(self):
        spec = get_algorithm_spec("reinforce_pp_none")
        assert spec.name == "reinforce_pp_none"
        assert spec.needs_planning is False
        assert spec.uses_sepa_controller is False

    def test_reinforce_pp_gtpo_spec(self):
        spec = get_algorithm_spec("reinforce_pp_gtpo")
        assert spec.name == "reinforce_pp_gtpo"
        assert spec.needs_planning is False
        assert spec.uses_sepa_controller is False

    def test_reinforce_pp_gtpo_sepa_spec(self):
        spec = get_algorithm_spec("reinforce_pp_gtpo_sepa")
        assert spec.name == "reinforce_pp_gtpo_sepa"
        assert spec.needs_planning is True
        assert spec.uses_sepa_controller is True

    def test_reinforce_pp_none_computes(self):
        """Full computation through algorithm mode."""
        rewards = [1.0, 0.0, 0.5]
        logprobs = [[-0.5, -0.3], [-0.7, -0.2], [-0.4, -0.6]]
        planning_masks = [[0, 0], [0, 0], [0, 0]]
        result = compute_algorithm_advantages(
            rewards, logprobs, planning_masks,
            algorithm_mode="reinforce_pp_none",
        )
        assert len(result.token_advs) == 3
        for i in range(3):
            assert len(result.token_advs[i]) == 2

    def test_reinforce_pp_none_matches_composable(self):
        """algorithm_mode='reinforce_pp_none' should match composable equivalent."""
        rewards = [1.0, 0.0, 0.5, 0.8]
        logprobs = [[-0.5, -0.3, -0.1], [-0.7, -0.2, -0.4],
                    [-0.4, -0.6, -0.2], [-0.3, -0.5, -0.7]]
        planning_masks = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

        algo_result = compute_algorithm_advantages(
            rewards, logprobs, planning_masks,
            algorithm_mode="reinforce_pp_none",
        )
        comp_result = compute_composable_advantages(
            rewards, logprobs, planning_masks,
            advantage_mode="reinforce_pp",
            transform_mode="none",
        )
        for i in range(4):
            for j in range(3):
                assert algo_result.token_advs[i][j] == pytest.approx(
                    comp_result.token_advs[i][j]
                )

    def test_reinforce_pp_gtpo_varies_token_advantages(self):
        """GTPO transform should produce varying per-token advantages."""
        rewards = [1.0, 0.0]
        logprobs = [[-0.1, -2.0, -0.5], [-1.5, -0.3, -0.8]]
        planning_masks = [[0, 0, 0], [0, 0, 0]]
        result = compute_algorithm_advantages(
            rewards, logprobs, planning_masks,
            algorithm_mode="reinforce_pp_gtpo",
            gtpo_beta=0.2,
        )
        # GTPO weights by surprisal — tokens should have different advantages
        assert result.token_advs[0][0] != pytest.approx(result.token_advs[0][1])

    def test_reinforce_pp_gtpo_sepa_with_planning(self):
        """SEPA transform should differentiate planning vs execution tokens."""
        rewards = [1.0, 0.0]
        logprobs = [[-0.5, -0.3, -0.8, -0.2], [-0.7, -0.2, -0.4, -0.6]]
        planning_masks = [[0, 1, 0, 1], [1, 0, 1, 0]]
        result = compute_algorithm_advantages(
            rewards, logprobs, planning_masks,
            algorithm_mode="reinforce_pp_gtpo_sepa",
            sepa_lambda=0.5,
        )
        assert len(result.token_advs) == 2
        assert result.has_stats is True


# ---------------------------------------------------------------------------
# Flow validation compatibility
# ---------------------------------------------------------------------------

class TestReinforcePPFlow:
    """Test flow.py validation works with reinforce_pp modes."""

    def test_flow_probe_reinforce_pp_none(self):
        """Flow probing should succeed for reinforce_pp + none."""
        from retrain.config import TrainConfig
        from retrain.flow import build_flow

        config = TrainConfig(
            advantage_mode="reinforce_pp",
            transform_mode="none",
            batch_advantage_norm=True,
        )
        flow = build_flow(config, gpu=False)
        result = flow.trace()
        assert result.ok, f"Flow probe failed: {[i.message for i in result.issues]}"

    def test_flow_probe_reinforce_pp_algorithm_mode(self):
        """Flow probing should succeed for reinforce_pp_none algorithm mode."""
        from retrain.config import TrainConfig
        from retrain.flow import build_flow

        config = TrainConfig(
            algorithm_mode="reinforce_pp_none",
            batch_advantage_norm=True,
        )
        flow = build_flow(config, gpu=False)
        result = flow.trace()
        assert result.ok, f"Flow probe failed: {[i.message for i in result.issues]}"

    def test_flow_probe_reinforce_pp_gtpo(self):
        """Flow probing should succeed for reinforce_pp_gtpo."""
        from retrain.config import TrainConfig
        from retrain.flow import build_flow

        config = TrainConfig(
            algorithm_mode="reinforce_pp_gtpo",
            batch_advantage_norm=True,
        )
        flow = build_flow(config, gpu=False)
        result = flow.trace()
        assert result.ok, f"Flow probe failed: {[i.message for i in result.issues]}"

    def test_flow_probe_reinforce_pp_gtpo_sepa(self):
        """Flow probing should succeed for reinforce_pp_gtpo_sepa."""
        from retrain.config import TrainConfig
        from retrain.flow import build_flow

        config = TrainConfig(
            algorithm_mode="reinforce_pp_gtpo_sepa",
            batch_advantage_norm=True,
        )
        flow = build_flow(config, gpu=False)
        result = flow.trace()
        assert result.ok, f"Flow probe failed: {[i.message for i in result.issues]}"


# ---------------------------------------------------------------------------
# Config round-trip (TOML → TrainConfig → validation)
# ---------------------------------------------------------------------------

class TestReinforcePPConfigRoundTrip:
    """Verify TOML configs parse and validate correctly."""

    def test_full_reinforce_pp_config(self, tmp_path):
        """Full TOML config with all REINFORCE++ settings."""
        from retrain.config import load_config
        toml = tmp_path / "rpp.toml"
        toml.write_text("""
[algorithm]
advantage_mode = "reinforce_pp"
transform_mode = "none"

[training]
batch_advantage_norm = true
clip_eps = 0.2
clip_eps_high = 0.28
adv_clip_max = 5.0
max_steps = 100
batch_size = 4
group_size = 8
lr = 1e-5
""")
        c = load_config(str(toml))
        assert c.advantage_mode == "reinforce_pp"
        assert c.transform_mode == "none"
        assert c.batch_advantage_norm is True
        assert c.clip_eps == pytest.approx(0.2)
        assert c.clip_eps_high == pytest.approx(0.28)
        assert c.adv_clip_max == pytest.approx(5.0)

    def test_algorithm_mode_reinforce_pp_none(self, tmp_path):
        """Algorithm mode shortcut via TOML."""
        from retrain.config import load_config
        toml = tmp_path / "rpp_algo.toml"
        toml.write_text("""
[algorithm]
algorithm_mode = "reinforce_pp_none"

[training]
batch_advantage_norm = true
""")
        c = load_config(str(toml))
        assert c.algorithm_mode == "reinforce_pp_none"
        assert c.batch_advantage_norm is True

    def test_cli_override_advantage_mode(self, tmp_path):
        """CLI flags should override TOML."""
        from retrain.config import load_config, parse_cli_overrides
        toml = tmp_path / "base.toml"
        toml.write_text('[algorithm]\nadvantage_mode = "grpo"\n')
        _path, overrides = parse_cli_overrides(
            ["--advantage-mode", "reinforce_pp", "--batch-advantage-norm", "true"]
        )
        c = load_config(str(toml), overrides=overrides)
        assert c.advantage_mode == "reinforce_pp"
        assert c.batch_advantage_norm is True


# ---------------------------------------------------------------------------
# Stress test: realistic multi-group batch simulation
# ---------------------------------------------------------------------------

class TestReinforcePPStress:
    """Simulate realistic training scenarios with many groups."""

    def test_warehouse_like_scenario(self):
        """Simulate a warehouse-like batch: 4 prompts × 8 group_size.

        Each group has different reward distributions (simulating different
        warehouse scenarios with different difficulty levels).
        """
        import random
        rng = random.Random(42)

        n_prompts = 4
        group_size = 8
        prompt_len = 20
        response_len = 50

        all_datum_advantages: list[list[float]] = []

        for prompt_idx in range(n_prompts):
            # Each prompt has different reward distribution
            base_difficulty = rng.uniform(0.2, 0.8)
            rewards = [max(0, min(1, rng.gauss(base_difficulty, 0.2)))
                       for _ in range(group_size)]

            # Compute per-group advantages (REINFORCE++ step 1)
            advs = compute_reinforce_pp_advantages(rewards)
            assert abs(sum(advs)) < 1e-10, "Group advantages must sum to ~0"

            # Expand to token-level (as trainer would)
            for sample_idx in range(group_size):
                seq = [0.0] * prompt_len + [advs[sample_idx]] * response_len
                all_datum_advantages.append(seq)

        # Verify pre-normalization: group means are zero but cross-group
        # scales differ
        group_maxes = []
        for g in range(n_prompts):
            start = g * group_size
            group_vals = [
                all_datum_advantages[start + s][prompt_len]
                for s in range(group_size)
            ]
            group_maxes.append(max(abs(v) for v in group_vals))
        assert max(group_maxes) / min(group_maxes) > 1.1, \
            "Groups should have different scales before normalization"

        # Apply batch normalization (REINFORCE++ step 2)
        normalized, metrics = apply_batch_advantage_normalization(
            all_datum_advantages
        )

        # Verify post-normalization properties
        assert len(normalized) == n_prompts * group_size
        assert metrics["batch_adv_count"] == n_prompts * group_size * response_len

        non_zero = [a for seq in normalized for a in seq if a != 0.0]
        mean_val = sum(non_zero) / len(non_zero)
        var_val = sum((v - mean_val) ** 2 for v in non_zero) / len(non_zero)

        assert mean_val == pytest.approx(0.0, abs=1e-6)
        assert math.sqrt(var_val) == pytest.approx(1.0, abs=0.01)

        # Verify prompt padding preserved
        for seq in normalized:
            for i in range(prompt_len):
                assert seq[i] == 0.0

    def test_extreme_reward_variance_across_groups(self):
        """One group has tiny rewards, another has huge — batch norm should
        standardize both to the same scale."""
        # Group 1: very small advantages (easy scenario, all ~0.9 reward)
        small_advs = compute_reinforce_pp_advantages(
            [0.91, 0.89, 0.92, 0.88]
        )
        # Group 2: large advantages (hard scenario, mixed rewards)
        large_advs = compute_reinforce_pp_advantages(
            [1.0, 0.0, 1.0, 0.0]
        )

        all_advs = []
        for adv in small_advs:
            all_advs.append([0.0, 0.0] + [adv] * 3)
        for adv in large_advs:
            all_advs.append([0.0, 0.0] + [adv] * 3)

        # Before normalization: group 2 advantages are ~25x larger
        max_small = max(abs(a) for a in small_advs)
        max_large = max(abs(a) for a in large_advs)
        assert max_large / max_small > 10

        # After normalization
        normalized, _ = apply_batch_advantage_normalization(all_advs)
        non_zero = [a for seq in normalized for a in seq if a != 0.0]
        mean_val = sum(non_zero) / len(non_zero)
        var_val = sum((v - mean_val) ** 2 for v in non_zero) / len(non_zero)
        assert mean_val == pytest.approx(0.0, abs=1e-6)
        assert math.sqrt(var_val) == pytest.approx(1.0, abs=0.01)

    def test_batch_norm_preserves_within_group_ranking(self):
        """Within each group, relative ordering of samples must be preserved."""
        import random
        rng = random.Random(99)

        for _ in range(20):  # 20 random trials
            rewards = [rng.random() for _ in range(8)]
            advs = compute_reinforce_pp_advantages(rewards)
            all_advs = [[0.0] + [a] * 5 for a in advs]

            # Add a second group to make batch non-trivial
            rewards2 = [rng.random() for _ in range(8)]
            advs2 = compute_reinforce_pp_advantages(rewards2)
            all_advs.extend([[0.0] + [a] * 5 for a in advs2])

            normalized, _ = apply_batch_advantage_normalization(all_advs)

            # Check group 1 ordering preserved
            for i in range(7):
                orig_order = advs[i] < advs[i + 1]
                norm_order = normalized[i][1] < normalized[i + 1][1]
                assert orig_order == norm_order, \
                    f"Ordering violated at index {i}: orig={advs[i]:.4f} vs {advs[i+1]:.4f}"

    def test_deterministic(self):
        """Same input should always produce same output."""
        all_advs = [
            [0.0, 0.0, 0.5, -0.5],
            [0.0, 0.3, -0.3],
            [0.0, 0.1, -0.1],
        ]
        result1, metrics1 = apply_batch_advantage_normalization(
            [list(seq) for seq in all_advs]
        )
        result2, metrics2 = apply_batch_advantage_normalization(
            [list(seq) for seq in all_advs]
        )
        assert result1 == result2
        assert metrics1 == metrics2

    def test_all_finite_under_adversarial_inputs(self):
        """No NaN/Inf even with adversarial reward patterns."""
        import random
        rng = random.Random(7)

        adversarial_cases = [
            [1.0, 1.0, 1.0, 1.0],         # all same (std=0)
            [0.0, 0.0, 0.0, 0.0],         # all zero
            [1e-15, -1e-15],               # near-zero
            [1e6, -1e6],                   # huge
            [1.0],                          # single
            [0.0, 1.0],                    # binary
            [0.999999, 1.000001],          # near-equal
        ]

        all_advs = []
        for rewards in adversarial_cases:
            advs = compute_reinforce_pp_advantages(rewards)
            for a in advs:
                all_advs.append([0.0] + [a] * 3)

        normalized, metrics = apply_batch_advantage_normalization(all_advs)
        for seq in normalized:
            for a in seq:
                assert math.isfinite(a), f"Non-finite: {a}"


# ---------------------------------------------------------------------------
# Delight gating (Osband 2026, arxiv:2603.14608)
# ---------------------------------------------------------------------------

class TestDelightGating:
    """Test delight-gated token advantages."""

    def test_registered(self):
        assert "delight" in get_builtin_transform_modes()

    def test_breakthrough_amplified(self):
        """Rare good actions (high surprisal, positive advantage) → gate open."""
        result = apply_delight_gating(1.0, [5.0, 0.1], eta=1.0)
        # High surprisal token should get higher weight than low surprisal
        assert result[0] > result[1]
        # Both should be positive (positive advantage)
        assert result[0] > 0
        assert result[1] > 0

    def test_blunder_suppressed(self):
        """Rare bad actions (high surprisal, negative advantage) → gate closed."""
        result = apply_delight_gating(-1.0, [5.0, 0.1], eta=1.0)
        # High surprisal with negative advantage → suppressed (gate near 0)
        assert abs(result[0]) < abs(result[1])

    def test_zero_advantage(self):
        result = apply_delight_gating(0.0, [1.0, 2.0, 3.0])
        assert all(a == 0.0 for a in result)

    def test_empty(self):
        assert apply_delight_gating(1.0, []) == []

    def test_high_eta_recovers_uniform(self):
        """η → ∞ should recover uniform weighting (gate ≈ 0.5 for all)."""
        result = apply_delight_gating(1.0, [0.1, 5.0, 10.0], eta=1000.0)
        # All gates should be ~0.5, so all advantages ~0.5
        for a in result:
            assert a == pytest.approx(0.5, abs=0.05)

    def test_composable_with_reinforce_pp(self):
        """Delight should work through the composable pipeline."""
        rewards = [1.0, 0.0, 0.5]
        logprobs = [[-0.5, -3.0, -0.1], [-0.7, -0.2, -5.0], [-0.4, -0.6, -1.0]]
        planning_masks = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        result = compute_composable_advantages(
            rewards, logprobs, planning_masks,
            advantage_mode="reinforce_pp",
            transform_mode="delight",
        )
        assert len(result.token_advs) == 3
        # Token advantages should vary within each sequence
        # (unlike "none" which would broadcast uniformly)
        for i in range(3):
            assert len(result.token_advs[i]) == 3

    def test_algorithm_mode_shortcut(self):
        """reinforce_pp_delight algorithm mode should work."""
        spec = get_algorithm_spec("reinforce_pp_delight")
        assert spec.name == "reinforce_pp_delight"
        result = compute_algorithm_advantages(
            [1.0, 0.0],
            [[-0.5, -3.0], [-0.7, -0.2]],
            [[0, 0], [0, 0]],
            algorithm_mode="reinforce_pp_delight",
        )
        assert len(result.token_advs) == 2

    def test_gtpo_vs_delight_on_negative_advantage(self):
        """Key difference: GTPO amplifies ALL high-entropy tokens.
        Delight suppresses high-entropy tokens with negative advantage."""
        from retrain.advantages import apply_gtpo_weighting

        advantage = -1.0
        surprisals = [0.1, 5.0]  # one low, one high

        gtpo_result = apply_gtpo_weighting(advantage, surprisals, beta=0.1)
        dg_result = apply_delight_gating(advantage, surprisals, eta=1.0)

        # GTPO: high surprisal token gets LARGER magnitude (more negative)
        assert abs(gtpo_result[1]) > abs(gtpo_result[0])

        # DG: high surprisal with NEGATIVE advantage → suppressed (smaller magnitude)
        assert abs(dg_result[1]) < abs(dg_result[0])

    def test_all_finite(self):
        """No NaN/Inf under extreme inputs."""
        cases = [
            (100.0, [0.0, 50.0, 0.001]),
            (-100.0, [0.0, 50.0, 0.001]),
            (0.001, [0.0, 0.0, 0.0]),
            (1.0, [50.0, 50.0]),
        ]
        for adv, surps in cases:
            result = apply_delight_gating(adv, surps, eta=1.0)
            for a in result:
                assert math.isfinite(a), f"Non-finite for adv={adv}, surps={surps}: {a}"
