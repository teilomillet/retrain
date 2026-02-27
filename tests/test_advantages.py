"""Tests for retrain.advantages — GRPO, MaxRL, GTPO, HICRA, SEPA, planning tokens."""

import math

import pytest

from retrain.advantages import (
    MAX_ENTROPY,
    TransformSpec,
    UncertaintyContext,
    UncertaintySpec,
    apply_entropy_mask,
    apply_gtpo_weighting,
    apply_hicra,
    apply_sepa_amplification,
    apply_sepa_amplification_clamped,
    apply_sepa_pooling,
    compute_composable_advantages,
    compute_entropy_mask_threshold,
    compute_entropy_stats,
    compute_grpo_advantages,
    compute_maxrl_advantages,
    get_uncertainty_spec,
    identify_planning_tokens,
    is_valid_uncertainty_kind_name,
    register_uncertainty_kind,
    _BUILTIN_UNCERTAINTY_SPECS,
    _UNCERTAINTY_SPEC_CACHE,
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
# SEPA amplification
# ---------------------------------------------------------------------------

class TestSEPAAmplification:
    def test_exec_tokens_amplified_away_from_mean(self):
        entropies = [0.1, 0.5, 0.9]
        mask = [1, 0, 0]  # first is planning, rest are execution
        result = apply_sepa_amplification(entropies, mask, lambda_t=1.0)

        # Planning token unchanged.
        assert result[0] == pytest.approx(0.1)
        # Execution mean is 0.7, so values move away from 0.7.
        assert result[1] == pytest.approx(0.3)  # 2*0.5 - 0.7
        assert result[2] == pytest.approx(1.1)  # 2*0.9 - 0.7

    def test_clamped_variant_floors_negative_values(self):
        entropies = [0.05, 0.95]
        mask = [0, 0]
        result = apply_sepa_amplification_clamped(entropies, mask, lambda_t=1.0)
        assert result[0] == pytest.approx(0.0)  # 2*0.05 - 0.5 = -0.4 -> 0
        assert result[1] == pytest.approx(1.4)  # 2*0.95 - 0.5

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            apply_sepa_amplification([0.1], [0, 1], lambda_t=0.5)
        with pytest.raises(ValueError, match="Length mismatch"):
            apply_sepa_amplification_clamped([0.1], [0, 1], lambda_t=0.5)


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
# Entropy masking (Yue et al.)
# ---------------------------------------------------------------------------

class TestEntropyMask:
    def test_basic_masking(self):
        """Top 50% kept, bottom 50% zeroed."""
        entropies = [1.0, 2.0, 3.0, 4.0]
        threshold = compute_entropy_mask_threshold(entropies, rho=0.5)
        advs = [1.0, 1.0, 1.0, 1.0]
        result = apply_entropy_mask(advs, entropies, threshold)
        # Top 50% (3.0, 4.0) kept; bottom 50% (1.0, 2.0) zeroed
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 1.0
        assert result[3] == 1.0

    def test_rho_zero_masks_all(self):
        threshold = compute_entropy_mask_threshold([1.0, 2.0, 3.0], rho=0.0)
        result = apply_entropy_mask([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], threshold)
        assert all(a == 0.0 for a in result)

    def test_rho_one_masks_none(self):
        threshold = compute_entropy_mask_threshold([1.0, 2.0, 3.0], rho=1.0)
        result = apply_entropy_mask([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], threshold)
        assert all(a == 1.0 for a in result)

    def test_empty_entropies(self):
        threshold = compute_entropy_mask_threshold([], rho=0.5)
        assert threshold == 0.0

    def test_all_same_entropy(self):
        entropies = [2.0, 2.0, 2.0, 2.0]
        threshold = compute_entropy_mask_threshold(entropies, rho=0.5)
        # All entropies are equal and >= threshold, so all kept
        result = apply_entropy_mask([1.0, 1.0, 1.0, 1.0], entropies, threshold)
        assert all(a == 1.0 for a in result)

    def test_composable_with_entropy_mask(self):
        """Full pipeline integration with entropy_mask transform."""
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3, -0.1], [-0.8, -0.6, -0.4]],
            planning_masks_G=[[0, 0, 0], [0, 0, 0]],
            advantage_mode="maxrl",
            transform_mode="entropy_mask",
            gtpo_beta=0.0,
            post_process_params={"entropy_mask_rho": 0.5},
        )
        assert len(result.token_advs) == 2
        assert result.extra_metrics["entropy_mask_threshold"] > 0.0
        assert 0.0 < result.extra_metrics["entropy_mask_fraction"] < 1.0
        # Some tokens should be zeroed
        all_advs = [a for seq in result.token_advs for a in seq]
        assert any(a == 0.0 for a in all_advs)
        assert any(a != 0.0 for a in all_advs)

    def test_entropy_mask_rho_zero_is_noop(self):
        """rho=0 should produce identical results to no masking."""
        kwargs = dict(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[0, 0], [0, 0]],
            advantage_mode="grpo",
            transform_mode="gtpo",
            gtpo_beta=0.1,
        )
        result_no_mask = compute_composable_advantages(**kwargs, post_process_params={"entropy_mask_rho": 0.0})
        result_with_mask = compute_composable_advantages(**kwargs, post_process_params={"entropy_mask_rho": 0.0})
        assert result_no_mask.token_advs == result_with_mask.token_advs
        assert result_with_mask.extra_metrics == {}


# ---------------------------------------------------------------------------
# Post-process hooks
# ---------------------------------------------------------------------------

class TestPostProcessHooks:
    def test_post_process_hook_called(self):
        """Custom post_process hook is called and its metrics returned."""
        calls: list[dict] = []

        def _scale_hook(all_token_advs, all_raw_entropies, params):
            calls.append({"params": params})
            factor = params.get("factor", 1.0)
            scaled = [[a * factor for a in seq] for seq in all_token_advs]
            return scaled, {"scale_factor": factor}

        from retrain.advantages import _BUILTIN_TRANSFORM_SPECS, _TRANSFORM_SPEC_CACHE
        spec = TransformSpec(name="test_hook", use_gtpo=True, post_process=_scale_hook)
        _BUILTIN_TRANSFORM_SPECS["test_hook"] = spec
        _TRANSFORM_SPEC_CACHE.pop("test_hook", None)
        try:
            result = compute_composable_advantages(
                rewards_G=[1.0, 0.0],
                logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
                planning_masks_G=[[0, 0], [0, 0]],
                advantage_mode="grpo",
                transform_mode="test_hook",
                post_process_params={"factor": 2.0},
            )
            assert len(calls) == 1
            assert calls[0]["params"] == {"factor": 2.0}
            assert result.extra_metrics == {"scale_factor": 2.0}
        finally:
            _BUILTIN_TRANSFORM_SPECS.pop("test_hook", None)
            _TRANSFORM_SPEC_CACHE.pop("test_hook", None)

    def test_post_process_none_is_noop(self):
        """Specs without a hook produce empty extra_metrics."""
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[0, 0], [0, 0]],
            advantage_mode="grpo",
            transform_mode="gtpo",
        )
        assert result.extra_metrics == {}

    def test_post_process_wrong_sequence_count_raises(self):
        """Hook returning wrong number of sequences is caught."""
        def _bad_hook(all_token_advs, all_raw_entropies, params):
            return all_token_advs[:1], {}  # drop a sequence

        from retrain.advantages import _BUILTIN_TRANSFORM_SPECS, _TRANSFORM_SPEC_CACHE
        spec = TransformSpec(name="bad_seq", use_gtpo=True, post_process=_bad_hook)
        _BUILTIN_TRANSFORM_SPECS["bad_seq"] = spec
        _TRANSFORM_SPEC_CACHE.pop("bad_seq", None)
        try:
            with pytest.raises(ValueError, match="1 sequences, expected 2"):
                compute_composable_advantages(
                    rewards_G=[1.0, 0.0],
                    logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
                    planning_masks_G=[[0, 0], [0, 0]],
                    advantage_mode="grpo",
                    transform_mode="bad_seq",
                )
        finally:
            _BUILTIN_TRANSFORM_SPECS.pop("bad_seq", None)
            _TRANSFORM_SPEC_CACHE.pop("bad_seq", None)

    def test_post_process_wrong_token_count_raises(self):
        """Hook returning wrong token count for a sequence is caught."""
        def _bad_hook(all_token_advs, all_raw_entropies, params):
            return [[0.0], all_token_advs[1]], {}  # truncate first sequence

        from retrain.advantages import _BUILTIN_TRANSFORM_SPECS, _TRANSFORM_SPEC_CACHE
        spec = TransformSpec(name="bad_tok", use_gtpo=True, post_process=_bad_hook)
        _BUILTIN_TRANSFORM_SPECS["bad_tok"] = spec
        _TRANSFORM_SPEC_CACHE.pop("bad_tok", None)
        try:
            with pytest.raises(ValueError, match="1 tokens for sequence 0, expected 2"):
                compute_composable_advantages(
                    rewards_G=[1.0, 0.0],
                    logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
                    planning_masks_G=[[0, 0], [0, 0]],
                    advantage_mode="grpo",
                    transform_mode="bad_tok",
                )
        finally:
            _BUILTIN_TRANSFORM_SPECS.pop("bad_tok", None)
            _TRANSFORM_SPEC_CACHE.pop("bad_tok", None)


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
        assert result.has_stats  # baseline now reports observational stats
        # Each token gets the same episode-level advantage
        assert all(a == result.token_advs[0][0] for a in result.token_advs[0])
        # Stats should reflect the raw surprisal (-logprob) values
        assert result.stats.exec_count > 0

    def test_baseline_stats_match_manual(self):
        """Baseline (none) stats = manual mean/var of -logprob."""
        logprobs = [[-0.5, -0.3], [-0.8, -0.6]]
        masks = [[0, 0], [0, 0]]
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=logprobs,
            planning_masks_G=masks,
            advantage_mode="grpo",
            transform_mode="none",
        )
        # All tokens are exec (masks all 0)
        all_surprisals = [0.5, 0.3, 0.8, 0.6]
        expected_mean = sum(all_surprisals) / len(all_surprisals)
        expected_var = sum((s - expected_mean) ** 2 for s in all_surprisals) / len(all_surprisals)
        assert result.stats.exec_mean == pytest.approx(expected_mean)
        assert result.stats.exec_var == pytest.approx(expected_var)
        assert result.stats.exec_count == 4.0
        assert result.stats.plan_count == 0.0

    def test_baseline_advantages_unchanged(self):
        """Baseline stats don't alter the uniform advantages."""
        rewards = [1.0, 0.0, 0.5]
        logprobs = [[-0.5, -0.3], [-0.8, -0.6], [-0.2, -0.7]]
        masks = [[0, 0], [0, 0], [0, 0]]
        result = compute_composable_advantages(
            rewards_G=rewards,
            logprobs_G=logprobs,
            planning_masks_G=masks,
            advantage_mode="grpo",
            transform_mode="none",
        )
        grpo_advs = compute_grpo_advantages(rewards)
        for i in range(3):
            assert all(a == pytest.approx(grpo_advs[i]) for a in result.token_advs[i])

    def test_post_transform_differs_with_sepa(self):
        """With SEPA transform active, post-transform stats differ from pre-transform."""
        # Need multiple exec tokens per episode so SEPA pooling has effect
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.1, -0.9, -0.3], [-0.5, -0.5, -0.8]],
            planning_masks_G=[[1, 0, 0], [0, 0, 1]],
            advantage_mode="grpo",
            transform_mode="gtpo_sepa",
            sepa_lambda=0.5,
        )
        assert result.has_stats
        # When SEPA is active with non-uniform exec tokens, post-transform
        # variance should be reduced vs pre-transform
        assert result.stats.post_exec_var < result.stats.exec_var

    def test_post_transform_equals_without_sepa(self):
        """Without entropy transform (plain gtpo), post == pre."""
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.1, -0.9], [-0.5, -0.5]],
            planning_masks_G=[[0, 0], [0, 0]],
            advantage_mode="grpo",
            transform_mode="gtpo",
        )
        assert result.has_stats
        assert result.stats.post_exec_mean == pytest.approx(result.stats.exec_mean)
        assert result.stats.post_exec_var == pytest.approx(result.stats.exec_var)

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

    def test_all_transform_modes_produce_correct_shape(self):
        """Every transform mode produces one advantage list per sequence."""
        rewards = [1.0, 0.0, 0.5]
        logprobs = [[-0.5, -0.3, -0.1], [-0.8, -0.6, -0.4], [-0.2, -0.7, -0.9]]
        masks = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for mode in (
            "none",
            "gtpo",
            "entropy_mask",
            "gtpo_hicra",
            "gtpo_sepa",
            "gtpo_sepa_amp",
            "gtpo_sepa_amp_c",
        ):
            result = compute_composable_advantages(
                rewards, logprobs, masks,
                advantage_mode="grpo",
                transform_mode=mode,
                sepa_lambda=0.5,
            )
            assert len(result.token_advs) == 3
            for i, ta in enumerate(result.token_advs):
                assert len(ta) == len(logprobs[i])

    def test_composable_matches_manual_grpo_none(self):
        """Composable pipeline matches manual GRPO + none expansion."""
        rewards = [1.0, 0.0]
        logprobs = [[-0.5, -0.3], [-0.8, -0.6]]
        masks = [[0, 0], [0, 0]]

        result = compute_composable_advantages(
            rewards, logprobs, masks,
            advantage_mode="grpo", transform_mode="none",
        )
        grpo_advs = compute_grpo_advantages(rewards)
        for i in range(2):
            assert all(a == pytest.approx(grpo_advs[i]) for a in result.token_advs[i])

    def test_uncertainty_kind_default_is_surprisal(self):
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[0, 0], [0, 0]],
            advantage_mode="grpo",
            transform_mode="gtpo",
        )
        assert result.has_stats

    def test_uncertainty_kind_shannon_entropy_requires_data(self):
        """Shannon entropy raises when neither precomputed nor distributions are provided."""
        with pytest.raises(ValueError, match="shannon_entropy"):
            compute_composable_advantages(
                rewards_G=[1.0, 0.0],
                logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
                planning_masks_G=[[0, 0], [0, 0]],
                advantage_mode="grpo",
                transform_mode="gtpo",
                transform_params={"uncertainty_kind": "shannon_entropy"},
            )

    def test_uncertainty_kind_varentropy_not_implemented(self):
        with pytest.raises(ValueError, match="Unknown uncertainty_kind 'varentropy'"):
            compute_composable_advantages(
                rewards_G=[1.0, 0.0],
                logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
                planning_masks_G=[[0, 0], [0, 0]],
                advantage_mode="grpo",
                transform_mode="gtpo",
                transform_params={"uncertainty_kind": "varentropy"},
            )

    def test_uncertainty_kind_entropy_alias_maps_to_shannon(self):
        """Alias 'entropy' resolves to shannon_entropy and raises without data."""
        with pytest.raises(ValueError, match="shannon_entropy"):
            compute_composable_advantages(
                rewards_G=[1.0, 0.0],
                logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
                planning_masks_G=[[0, 0], [0, 0]],
                advantage_mode="grpo",
                transform_mode="gtpo",
                transform_params={"uncertainty_kind": "entropy"},
            )


# ---------------------------------------------------------------------------
# Numeric guard tests (inf entropy)
# ---------------------------------------------------------------------------

class TestNumericGuards:
    def test_gtpo_weighting_handles_inf(self):
        """apply_gtpo_weighting clamps inf entropies to MAX_ENTROPY."""
        result = apply_gtpo_weighting(1.0, [float("inf"), 0.5, 0.3], beta=0.1)
        assert all(math.isfinite(r) for r in result)

    def test_sepa_pooling_handles_inf(self):
        """apply_sepa_pooling clamps inf entropies to MAX_ENTROPY."""
        result = apply_sepa_pooling(
            [float("inf"), 0.5, 0.3], [0, 0, 0], lambda_t=0.5
        )
        assert all(math.isfinite(r) for r in result)

    def test_entropy_stats_handles_inf(self):
        """compute_entropy_stats clamps inf values."""
        stats = compute_entropy_stats([float("inf"), 0.5], [float("inf")])
        assert math.isfinite(stats.exec_mean)
        assert math.isfinite(stats.exec_var)
        assert math.isfinite(stats.plan_mean)

    def test_composable_handles_inf_logprobs(self):
        """Composable pipeline handles -inf logprobs (which become +inf entropies)."""
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[float("-inf"), -0.3], [-0.8, -0.6]],
            planning_masks_G=[[0, 0], [0, 0]],
            advantage_mode="grpo",
            transform_mode="gtpo",
        )
        for ta in result.token_advs:
            assert all(math.isfinite(a) for a in ta)
        assert math.isfinite(result.stats.exec_mean)


# ---------------------------------------------------------------------------
# Uncertainty registry
# ---------------------------------------------------------------------------

class TestUncertaintyRegistry:
    def test_builtin_spec_lookup(self):
        spec = get_uncertainty_spec("surprisal")
        assert isinstance(spec, UncertaintySpec)
        assert spec.name == "surprisal"
        assert spec.needs_distributions is False

        spec_pv = get_uncertainty_spec("predictive_variance")
        assert spec_pv.name == "predictive_variance"
        assert spec_pv.needs_distributions is False

        spec_se = get_uncertainty_spec("shannon_entropy")
        assert spec_se.name == "shannon_entropy"
        assert spec_se.needs_distributions is True

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown uncertainty_kind"):
            get_uncertainty_spec("not_a_real_kind")

    def test_is_valid_aliases(self):
        assert is_valid_uncertainty_kind_name("surprisal") is True
        assert is_valid_uncertainty_kind_name("neg_logprob") is True
        assert is_valid_uncertainty_kind_name("pred_var") is True
        assert is_valid_uncertainty_kind_name("bernoulli_variance") is True

    def test_is_valid_builtins(self):
        assert is_valid_uncertainty_kind_name("predictive_variance") is True
        assert is_valid_uncertainty_kind_name("shannon_entropy") is True

    def test_is_valid_dotted_paths(self):
        assert is_valid_uncertainty_kind_name("my_module.my_uncertainty") is True
        assert is_valid_uncertainty_kind_name("nope") is False

    def test_register_and_get_roundtrip(self):
        def _custom_compute(ctx):
            return [0.42] * len(ctx.logprobs)

        register_uncertainty_kind("test_custom_unc", _custom_compute)
        try:
            spec = get_uncertainty_spec("test_custom_unc")
            assert spec.name == "test_custom_unc"
            ctx = UncertaintyContext(logprobs=[-0.5, -0.3])
            assert spec.compute(ctx) == [0.42, 0.42]
        finally:
            _BUILTIN_UNCERTAINTY_SPECS.pop("test_custom_unc", None)
            _UNCERTAINTY_SPEC_CACHE.pop("test_custom_unc", None)

    def test_register_dotted_name_rejected(self):
        with pytest.raises(ValueError, match="cannot contain '\\.'"):
            register_uncertainty_kind("my.dotted.name", lambda ctx: [])


# ---------------------------------------------------------------------------
# Predictive variance signal
# ---------------------------------------------------------------------------

class TestPredictiveVariance:
    def test_value_correctness(self):
        """p * (1 - p) for known logprobs."""
        logprobs = [math.log(0.5), math.log(0.9), math.log(0.1)]
        ctx = UncertaintyContext(logprobs=logprobs)
        spec = get_uncertainty_spec("predictive_variance")
        values = spec.compute(ctx)

        assert values[0] == pytest.approx(0.5 * 0.5, abs=1e-9)
        assert values[1] == pytest.approx(0.9 * 0.1, abs=1e-9)
        assert values[2] == pytest.approx(0.1 * 0.9, abs=1e-9)

    def test_extreme_logprobs(self):
        """Very negative logprobs clamp p near 0, yielding near-zero variance."""
        ctx = UncertaintyContext(logprobs=[-100.0, 0.0])
        spec = get_uncertainty_spec("predictive_variance")
        values = spec.compute(ctx)

        assert values[0] == pytest.approx(0.0, abs=1e-6)
        assert values[1] == pytest.approx(0.0, abs=1e-9)  # p=1 -> 1*(1-1)=0

    def test_pipeline_integration(self):
        """Full pipeline with predictive_variance through compute_composable_advantages."""
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[0, 0], [0, 0]],
            advantage_mode="grpo",
            transform_mode="gtpo",
            transform_params={"uncertainty_kind": "predictive_variance"},
        )
        assert len(result.token_advs) == 2
        assert result.has_stats
        for ta in result.token_advs:
            assert all(math.isfinite(a) for a in ta)

    def test_pipeline_with_sepa(self):
        """Predictive variance works with SEPA transforms."""
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[1, 0], [0, 1]],
            advantage_mode="grpo",
            transform_mode="gtpo_sepa",
            sepa_lambda=0.5,
            transform_params={"uncertainty_kind": "predictive_variance"},
        )
        assert len(result.token_advs) == 2
        assert result.has_stats


# ---------------------------------------------------------------------------
# Precomputed Shannon entropy
# ---------------------------------------------------------------------------

class TestPrecomputedShannonEntropy:
    def test_compute_uses_precomputed_entropy(self):
        """_compute_shannon_entropy uses precomputed values when provided."""
        from retrain.advantages import _compute_shannon_entropy

        precomputed = [0.5, 1.2, 0.8]
        ctx = UncertaintyContext(
            logprobs=[-0.3, -0.7, -0.5],
            precomputed_entropy=precomputed,
        )
        result = _compute_shannon_entropy(ctx)
        assert result == [0.5, 1.2, 0.8]

    def test_precomputed_entropy_clamped_to_max(self):
        """Precomputed entropy values are clamped to MAX_SURPRISAL."""
        from retrain.advantages import _compute_shannon_entropy, MAX_SURPRISAL

        precomputed = [0.5, 100.0, MAX_SURPRISAL + 10.0]
        ctx = UncertaintyContext(
            logprobs=[-0.3, -0.7, -0.5],
            precomputed_entropy=precomputed,
        )
        result = _compute_shannon_entropy(ctx)
        assert result[0] == 0.5
        assert result[1] == MAX_SURPRISAL
        assert result[2] == MAX_SURPRISAL

    def test_falls_back_to_distributions(self):
        """Falls back to token_distributions when no precomputed."""
        from retrain.advantages import _compute_shannon_entropy

        # Simple 2-token vocab: [0.5, 0.5] -> H = log(2) ≈ 0.693
        distributions = [[0.5, 0.5], [0.25, 0.75]]
        ctx = UncertaintyContext(
            logprobs=[-0.3, -0.7],
            token_distributions=distributions,
        )
        result = _compute_shannon_entropy(ctx)
        assert len(result) == 2
        assert result[0] == pytest.approx(math.log(2), abs=1e-6)

    def test_raises_when_no_data(self):
        """Raises clear error when neither precomputed nor distributions."""
        from retrain.advantages import _compute_shannon_entropy

        ctx = UncertaintyContext(logprobs=[-0.3, -0.7])
        with pytest.raises(ValueError, match="shannon_entropy requires"):
            _compute_shannon_entropy(ctx)

    def test_pipeline_with_precomputed_entropies(self):
        """Full pipeline integration: shannon_entropy + precomputed entropies."""
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[0, 0], [0, 0]],
            advantage_mode="grpo",
            transform_mode="gtpo",
            transform_params={"uncertainty_kind": "shannon_entropy"},
            precomputed_entropies_G=[[0.5, 1.2], [0.8, 0.3]],
        )
        assert len(result.token_advs) == 2
        assert result.has_stats
        for ta in result.token_advs:
            assert all(math.isfinite(a) for a in ta)

    def test_pipeline_with_sepa_and_precomputed_entropies(self):
        """Shannon entropy works with SEPA transforms via precomputed path."""
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[1, 0], [0, 1]],
            advantage_mode="grpo",
            transform_mode="gtpo_sepa",
            sepa_lambda=0.5,
            transform_params={"uncertainty_kind": "shannon_entropy"},
            precomputed_entropies_G=[[0.5, 1.2], [0.8, 0.3]],
        )
        assert len(result.token_advs) == 2
        assert result.has_stats
