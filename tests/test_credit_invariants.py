"""Credit assignment invariant tests for the advantage pipeline.

These tests verify mathematical properties that must hold if the GRPO/GTPO
pipeline correctly assigns credit to individual tokens. They are the
advantage-layer analogue of the Rust credit_assignment tests: mathematical
invariants that catch bugs in the gradient signal before training begins.

Invariants tested:
  A. GRPO centering: advantages sum to zero
  B. GTPO mean preservation: mean(token_advs) == episode_advantage
  C. GTPO sign preservation: no sign flips across tokens
  D. GTPO ordering: higher surprisal → higher |token_advantage|
  E. Full pipeline end-to-end invariants
"""

import pytest

from retrain.advantages import (
    apply_gtpo_weighting,
    compute_composable_advantages,
    compute_grpo_advantages,
)


# ---------------------------------------------------------------------------
# A. GRPO centering: advantages must sum to zero
# ---------------------------------------------------------------------------


class TestGRPOCentering:
    """GRPO advantages are mean-centered: sum == 0 for any reward vector."""

    @pytest.mark.parametrize(
        "rewards",
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.3, 0.7, 0.5, 0.1, 0.9],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0],
            [0.42] * 10,
        ],
    )
    def test_sum_is_zero(self, rewards):
        advs = compute_grpo_advantages(rewards)
        assert sum(advs) == pytest.approx(0.0, abs=1e-10)

    def test_positive_rewards_get_positive_advantage(self):
        advs = compute_grpo_advantages([1.0, 0.0, 0.0])
        assert advs[0] > 0  # above mean
        assert advs[1] < 0  # below mean

    def test_uniform_rewards_give_zero_advantages(self):
        advs = compute_grpo_advantages([0.5, 0.5, 0.5, 0.5])
        assert all(a == pytest.approx(0.0) for a in advs)


# ---------------------------------------------------------------------------
# B. GTPO mean preservation: mean(token_advs) == episode_advantage
# ---------------------------------------------------------------------------


class TestGTPOMeanPreservation:
    """When beta < 1 (no weight clamping), GTPO weights average to 1.0,
    so the mean of token advantages equals the episode advantage exactly.

    Proof: weight_j = 1 + beta*(h_j/mean(h) - 1).
    mean(weight_j) = 1 + beta*(mean(h_j/mean(h)) - 1) = 1 + beta*(1-1) = 1.
    """

    @pytest.mark.parametrize("advantage", [-2.0, -0.5, 0.0, 0.5, 2.0])
    def test_mean_preserved_across_magnitudes(self, advantage):
        surprisals = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = apply_gtpo_weighting(advantage, surprisals, beta=0.1)
        mean_result = sum(result) / len(result)
        # Tolerance accounts for the 1e-8 epsilon in h/(mean_h + 1e-8).
        assert mean_result == pytest.approx(advantage, abs=1e-6)

    @pytest.mark.parametrize("beta", [0.01, 0.05, 0.1, 0.5, 0.99])
    def test_mean_preserved_across_betas(self, beta):
        surprisals = [0.2, 0.4, 0.6, 0.8, 1.0]
        result = apply_gtpo_weighting(1.5, surprisals, beta=beta)
        mean_result = sum(result) / len(result)
        assert mean_result == pytest.approx(1.5, abs=1e-6)

    def test_uniform_surprisal_gives_uniform_advantages(self):
        result = apply_gtpo_weighting(1.0, [0.5, 0.5, 0.5, 0.5], beta=0.3)
        assert all(r == pytest.approx(1.0) for r in result)


# ---------------------------------------------------------------------------
# C. GTPO sign preservation: no sign flips
# ---------------------------------------------------------------------------


class TestGTPOSignPreservation:
    """Since GTPO weights are non-negative (max(0, ...)), the sign of
    every token advantage must match the sign of the episode advantage.
    For beta < 1, all weights are strictly positive.
    """

    def test_positive_advantage_nonneg_tokens(self):
        surprisals = [0.1, 0.3, 0.5, 0.7, 0.9, 1.5, 2.0]
        result = apply_gtpo_weighting(1.0, surprisals, beta=0.5)
        assert all(r >= 0 for r in result), f"sign flip: {result}"

    def test_negative_advantage_nonpos_tokens(self):
        surprisals = [0.1, 0.3, 0.5, 0.7, 0.9, 1.5, 2.0]
        result = apply_gtpo_weighting(-1.0, surprisals, beta=0.5)
        assert all(r <= 0 for r in result), f"sign flip: {result}"

    def test_zero_advantage_all_zero(self):
        surprisals = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = apply_gtpo_weighting(0.0, surprisals, beta=0.5)
        assert all(r == 0.0 for r in result)

    @pytest.mark.parametrize("beta", [0.01, 0.1, 0.5, 0.99])
    def test_sign_preserved_across_betas(self, beta):
        surprisals = [0.05, 0.2, 0.5, 1.0, 2.0]
        pos = apply_gtpo_weighting(1.0, surprisals, beta=beta)
        neg = apply_gtpo_weighting(-1.0, surprisals, beta=beta)
        assert all(r >= 0 for r in pos)
        assert all(r <= 0 for r in neg)


# ---------------------------------------------------------------------------
# D. GTPO ordering: higher surprisal → higher |token_advantage|
# ---------------------------------------------------------------------------


class TestGTPOOrdering:
    """For a fixed episode advantage and positive beta, tokens with higher
    surprisal receive higher absolute advantage (monotonic weighting).
    """

    def test_ordering_positive_advantage(self):
        surprisals = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = apply_gtpo_weighting(1.0, surprisals, beta=0.1)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1], (
                f"ordering violated at {i}: {result[i]:.6f} > {result[i + 1]:.6f}"
            )

    def test_ordering_negative_advantage(self):
        surprisals = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = apply_gtpo_weighting(-1.0, surprisals, beta=0.1)
        # Negative advantage: more negative for higher surprisal.
        abs_result = [abs(r) for r in result]
        for i in range(len(abs_result) - 1):
            assert abs_result[i] <= abs_result[i + 1], (
                f"abs ordering violated at {i}: {abs_result[i]:.6f} > {abs_result[i + 1]:.6f}"
            )

    def test_ordering_holds_for_large_beta(self):
        surprisals = [0.2, 0.4, 0.6, 0.8, 1.0]
        result = apply_gtpo_weighting(1.0, surprisals, beta=0.5)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]


# ---------------------------------------------------------------------------
# E. Full pipeline end-to-end invariants
# ---------------------------------------------------------------------------


class TestPipelineCreditInvariants:
    """End-to-end tests through compute_composable_advantages."""

    def test_pipeline_sign_preservation(self):
        """No token should have opposite sign to its episode advantage."""
        rewards = [1.0, 0.0, 0.5, 0.2, 0.8]
        grpo_advs = compute_grpo_advantages(rewards)
        logprobs = [[-0.5, -0.3, -0.1]] * 5
        masks = [[0, 0, 0]] * 5

        result = compute_composable_advantages(
            rewards_G=rewards,
            logprobs_G=logprobs,
            planning_masks_G=masks,
            advantage_mode="grpo",
            transform_mode="gtpo",
            gtpo_beta=0.1,
        )

        for i, (ep_adv, token_advs) in enumerate(
            zip(grpo_advs, result.token_advs)
        ):
            for j, ta in enumerate(token_advs):
                if ep_adv > 0:
                    assert ta >= 0, (
                        f"ep {i} tok {j}: ep_adv={ep_adv:.3f} but ta={ta:.3f}"
                    )
                elif ep_adv < 0:
                    assert ta <= 0, (
                        f"ep {i} tok {j}: ep_adv={ep_adv:.3f} but ta={ta:.3f}"
                    )

    def test_pipeline_mean_preservation(self):
        """Mean of each episode's token advantages == its GRPO advantage."""
        rewards = [1.0, 0.0, 0.3, 0.7]
        grpo_advs = compute_grpo_advantages(rewards)
        logprobs = [[-0.2, -0.5, -0.8, -0.3, -0.6]] * 4
        masks = [[0, 0, 0, 0, 0]] * 4

        result = compute_composable_advantages(
            rewards_G=rewards,
            logprobs_G=logprobs,
            planning_masks_G=masks,
            advantage_mode="grpo",
            transform_mode="gtpo",
            gtpo_beta=0.05,
        )

        for i, (ep_adv, token_advs) in enumerate(
            zip(grpo_advs, result.token_advs)
        ):
            mean_ta = sum(token_advs) / len(token_advs)
            assert mean_ta == pytest.approx(ep_adv, abs=1e-6), (
                f"ep {i}: mean(ta)={mean_ta:.6f} != ep_adv={ep_adv:.6f}"
            )

    def test_pipeline_centering_preserved(self):
        """Cross-episode: mean of all mean-token-advantages ≈ 0."""
        rewards = [1.0, 0.0, 0.5, 0.2, 0.8, 0.3]
        logprobs = [[-0.3, -0.6, -0.9]] * 6
        masks = [[0, 0, 0]] * 6

        result = compute_composable_advantages(
            rewards_G=rewards,
            logprobs_G=logprobs,
            planning_masks_G=masks,
            advantage_mode="grpo",
            transform_mode="gtpo",
            gtpo_beta=0.1,
        )

        episode_means = [
            sum(ta) / len(ta) for ta in result.token_advs
        ]
        grand_mean = sum(episode_means) / len(episode_means)
        assert grand_mean == pytest.approx(0.0, abs=1e-6), (
            f"grand mean of episode means should be 0: {grand_mean:.8f}"
        )

    def test_pipeline_with_hicra_preserves_sign(self):
        """HICRA amplifies planning tokens but should not flip signs."""
        rewards = [1.0, 0.0]
        grpo_advs = compute_grpo_advantages(rewards)
        logprobs = [[-0.3, -0.6, -0.9], [-0.5, -0.2, -0.7]]
        masks = [[1, 0, 0], [0, 1, 0]]

        result = compute_composable_advantages(
            rewards_G=rewards,
            logprobs_G=logprobs,
            planning_masks_G=masks,
            advantage_mode="grpo",
            transform_mode="gtpo_hicra",
            hicra_alpha=0.2,
        )

        for i, (ep_adv, token_advs) in enumerate(
            zip(grpo_advs, result.token_advs)
        ):
            for j, ta in enumerate(token_advs):
                if ep_adv > 0:
                    assert ta >= 0, (
                        f"HICRA sign flip: ep {i} tok {j}: ep_adv={ep_adv:.3f} ta={ta:.3f}"
                    )
                elif ep_adv < 0:
                    assert ta <= 0, (
                        f"HICRA sign flip: ep {i} tok {j}: ep_adv={ep_adv:.3f} ta={ta:.3f}"
                    )

    def test_pipeline_gtpo_differentiates_tokens(self):
        """With non-uniform surprisal, GTPO should produce non-uniform token advantages."""
        rewards = [1.0, 0.0]
        # Different logprobs → different surprisals → different weights.
        logprobs = [[-0.1, -0.5, -1.5], [-0.2, -0.8, -2.0]]
        masks = [[0, 0, 0], [0, 0, 0]]

        result = compute_composable_advantages(
            rewards_G=rewards,
            logprobs_G=logprobs,
            planning_masks_G=masks,
            advantage_mode="grpo",
            transform_mode="gtpo",
            gtpo_beta=0.1,
        )

        # For episodes with non-zero advantage, tokens should differ.
        for i, token_advs in enumerate(result.token_advs):
            if abs(sum(token_advs) / len(token_advs)) > 1e-6:
                unique_values = set(round(a, 10) for a in token_advs)
                assert len(unique_values) > 1, (
                    f"ep {i}: GTPO should differentiate tokens, got uniform {token_advs}"
                )
