"""Tests for retrain.advantages — GRPO, MaxRL, GTPO, HICRA, SEPA, planning tokens."""

import pytest

from retrain.advantages import (
    apply_gtpo_weighting,
    apply_hicra,
    apply_sepa_pooling,
    compute_composable_advantages,
    compute_entropy_stats,
    compute_grpo_advantages,
    compute_maxrl_advantages,
    identify_planning_tokens,
)


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------

class TestGRPO:
    def test_basic(self):
        advs = compute_grpo_advantages([1.0, 0.0, 0.0, 1.0])
        assert len(advs) == 4
        assert sum(advs) == pytest.approx(0.0)
        assert advs[0] > 0  # above mean
        assert advs[1] < 0  # below mean

    def test_all_same(self):
        advs = compute_grpo_advantages([0.5, 0.5, 0.5])
        assert all(a == pytest.approx(0.0) for a in advs)

    def test_empty(self):
        assert compute_grpo_advantages([]) == []

    def test_single(self):
        advs = compute_grpo_advantages([1.0])
        assert advs == [pytest.approx(0.0)]


# ---------------------------------------------------------------------------
# MaxRL
# ---------------------------------------------------------------------------

class TestMaxRL:
    def test_basic(self):
        advs = compute_maxrl_advantages([1.0, 0.0, 1.0, 0.0])
        assert len(advs) == 4
        # mean = 0.5, so correct answers get positive, wrong get negative
        assert advs[0] > 0
        assert advs[1] < 0

    def test_all_zero(self):
        # mean(r) ~ 0 → all advantages zero
        advs = compute_maxrl_advantages([0.0, 0.0, 0.0])
        assert all(a == 0.0 for a in advs)

    def test_empty(self):
        assert compute_maxrl_advantages([]) == []

    def test_inverse_weighting(self):
        # Higher mean → smaller magnitude advantages
        advs_low = compute_maxrl_advantages([1.0, 0.0])   # mean=0.5
        advs_high = compute_maxrl_advantages([1.0, 1.0, 0.0])  # mean=0.667
        # The positive advantage should be smaller when mean is higher
        assert abs(advs_high[0]) < abs(advs_low[0])


# ---------------------------------------------------------------------------
# GTPO weighting
# ---------------------------------------------------------------------------

class TestGTPO:
    def test_uniform_entropy(self):
        result = apply_gtpo_weighting(1.0, [0.5, 0.5, 0.5])
        assert all(r == pytest.approx(1.0) for r in result)

    def test_zero_beta(self):
        result = apply_gtpo_weighting(2.0, [0.1, 0.5, 0.9], beta=0.0)
        assert all(r == pytest.approx(2.0) for r in result)

    def test_high_entropy_gets_higher_weight(self):
        entropies = [0.1, 0.5, 0.9]
        result = apply_gtpo_weighting(1.0, entropies, beta=0.1)
        # Higher entropy → higher weight
        assert result[2] > result[0]

    def test_empty(self):
        assert apply_gtpo_weighting(1.0, []) == []

    def test_zero_entropy(self):
        result = apply_gtpo_weighting(1.0, [0.0, 0.0, 0.0])
        assert all(r == pytest.approx(1.0) for r in result)


# ---------------------------------------------------------------------------
# HICRA
# ---------------------------------------------------------------------------

class TestHICRA:
    def test_planning_tokens_amplified(self):
        advs = [1.0, 1.0, 1.0]
        mask = [0, 1, 0]
        result = apply_hicra(advs, mask, alpha=0.2)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.2)  # 1.0 + 0.2*|1.0|
        assert result[2] == pytest.approx(1.0)

    def test_negative_advantage(self):
        advs = [-1.0, -1.0]
        mask = [0, 1]
        result = apply_hicra(advs, mask, alpha=0.2)
        assert result[0] == pytest.approx(-1.0)
        assert result[1] == pytest.approx(-0.8)  # -1.0 + 0.2*|-1.0|

    def test_zero_alpha(self):
        advs = [1.0, 2.0]
        mask = [1, 1]
        result = apply_hicra(advs, mask, alpha=0.0)
        assert result == advs

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            apply_hicra([1.0, 2.0], [0], alpha=0.2)


# ---------------------------------------------------------------------------
# SEPA pooling
# ---------------------------------------------------------------------------

class TestSEPAPooling:
    def test_exec_tokens_pooled(self):
        entropies = [0.1, 0.5, 0.9]
        mask = [1, 0, 0]  # first is planning, rest are execution
        result = apply_sepa_pooling(entropies, mask, lambda_t=1.0)
        # Planning token unchanged
        assert result[0] == pytest.approx(0.1)
        # Execution tokens pooled toward their mean (0.7)
        exec_mean = (0.5 + 0.9) / 2  # 0.7
        assert result[1] == pytest.approx(exec_mean)
        assert result[2] == pytest.approx(exec_mean)

    def test_lambda_zero_no_change(self):
        entropies = [0.1, 0.5, 0.9]
        mask = [0, 0, 0]
        result = apply_sepa_pooling(entropies, mask, lambda_t=0.0)
        assert result == entropies

    def test_partial_lambda(self):
        entropies = [0.2, 0.8]
        mask = [0, 0]
        exec_mean = 0.5
        result = apply_sepa_pooling(entropies, mask, lambda_t=0.5)
        # 0.5 * 0.5 + 0.5 * 0.2 = 0.35
        assert result[0] == pytest.approx(0.35)
        # 0.5 * 0.5 + 0.5 * 0.8 = 0.65
        assert result[1] == pytest.approx(0.65)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            apply_sepa_pooling([0.1], [0, 1], lambda_t=0.5)


# ---------------------------------------------------------------------------
# Entropy stats
# ---------------------------------------------------------------------------

class TestEntropyStats:
    def test_basic(self):
        stats = compute_entropy_stats([0.1, 0.3, 0.5], [0.2, 0.4])
        assert stats.exec_mean == pytest.approx(0.3)
        assert stats.exec_count == 3.0
        assert stats.plan_mean == pytest.approx(0.3)
        assert stats.plan_count == 2.0

    def test_empty_lists(self):
        stats = compute_entropy_stats([], [])
        assert stats.exec_mean == 0.0
        assert stats.plan_mean == 0.0


# ---------------------------------------------------------------------------
# Planning token identification
# ---------------------------------------------------------------------------

class TestPlanningTokens:
    def test_basic_match(self):
        tokens = ["let", "me", "think", "about", "this"]
        grams = ["let me think"]
        mask = identify_planning_tokens(tokens, grams)
        # "let me think" spans first 3 tokens
        assert mask[:3] == [1, 1, 1]

    def test_no_match(self):
        tokens = ["the", "answer", "is", "42"]
        grams = ["let me think"]
        mask = identify_planning_tokens(tokens, grams)
        assert all(m == 0 for m in mask)

    def test_empty_tokens(self):
        assert identify_planning_tokens([], ["let me think"]) == []

    def test_empty_grams(self):
        mask = identify_planning_tokens(["hello", "world"], [])
        assert mask == [0, 0]

    def test_subword_markers(self):
        # sentencepiece uses \u2581, BPE uses \u0120
        tokens = ["\u2581let", "\u2581me", "\u2581think"]
        grams = ["let me think"]
        mask = identify_planning_tokens(tokens, grams)
        assert mask == [1, 1, 1]


# ---------------------------------------------------------------------------
# Composable pipeline
# ---------------------------------------------------------------------------

class TestComposablePipeline:
    def test_grpo_none(self):
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[0, 0], [0, 0]],
            advantage_mode="grpo",
            transform_mode="none",
        )
        assert len(result.token_advs) == 2
        assert not result.has_stats
        # Each token gets the same episode-level advantage
        assert all(a == result.token_advs[0][0] for a in result.token_advs[0])

    def test_maxrl_gtpo(self):
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0, 1.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6], [-0.4, -0.2]],
            planning_masks_G=[[0, 0], [0, 0], [0, 0]],
            advantage_mode="maxrl",
            transform_mode="gtpo",
        )
        assert len(result.token_advs) == 3
        assert result.has_stats

    def test_gtpo_hicra(self):
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[1, 0], [0, 1]],
            advantage_mode="grpo",
            transform_mode="gtpo_hicra",
            hicra_alpha=0.2,
        )
        assert len(result.token_advs) == 2

    def test_gtpo_sepa(self):
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[1, 0], [0, 1]],
            advantage_mode="grpo",
            transform_mode="gtpo_sepa",
            sepa_lambda=0.5,
        )
        assert len(result.token_advs) == 2
        assert result.has_stats
