"""Ordeal property tests for retrain.advantages.

Complements existing unit tests with property-based testing (boundary-biased
quickcheck), composable invariants, and numerical fault injection.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from ordeal.invariants import no_nan, no_inf, bounded

from retrain.advantages import (
    AdvantageResult,
    apply_batch_advantage_normalization,
    apply_gtpo_weighting,
    apply_hard_delight_gating,
    apply_hicra,
    apply_sepa_amplification,
    apply_sepa_amplification_clamped,
    apply_sepa_pooling,
    compute_composable_advantages,
    compute_grpo_advantages,
    compute_maxrl_advantages,
    compute_reinforce_pp_advantages,
)

# ── Strategies ──

binary_rewards = st.lists(st.sampled_from([0.0, 1.0]), min_size=1, max_size=32)

continuous_rewards = st.lists(
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=32,
)

surprisals = st.lists(
    st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=64,
)

logprobs = st.lists(
    st.floats(min_value=-50.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=64,
)

beta_st = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
alpha_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
lambda_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
k_frac_st = st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)

# ── Invariants ──

valid_advantage = no_nan & no_inf


# ═══════════════════════════════════════════
# GRPO Properties
# ═══════════════════════════════════════════


class TestGRPOProperties:
    @given(rewards=binary_rewards)
    def test_sum_is_zero(self, rewards: list[float]) -> None:
        """GRPO advantages are mean-centered: sum must be zero."""
        advs = compute_grpo_advantages(rewards)
        assert math.isclose(sum(advs), 0.0, abs_tol=1e-10)

    @given(rewards=binary_rewards)
    def test_bounded_for_binary(self, rewards: list[float]) -> None:
        """For binary {0,1} rewards, |A_i| < 1.0."""
        advs = compute_grpo_advantages(rewards)
        for a in advs:
            assert abs(a) < 1.0

    @given(rewards=continuous_rewards)
    def test_all_finite(self, rewards: list[float]) -> None:
        advs = compute_grpo_advantages(rewards)
        for a in advs:
            valid_advantage(a)

    @given(rewards=continuous_rewards)
    def test_length_preserved(self, rewards: list[float]) -> None:
        assert len(compute_grpo_advantages(rewards)) == len(rewards)

    @given(rewards=continuous_rewards)
    def test_deterministic(self, rewards: list[float]) -> None:
        assert compute_grpo_advantages(rewards) == compute_grpo_advantages(rewards)

    @given(
        value=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        n=st.integers(min_value=1, max_value=32),
    )
    def test_uniform_rewards_produce_zero(self, value: float, n: int) -> None:
        advs = compute_grpo_advantages([value] * n)
        for a in advs:
            assert math.isclose(a, 0.0, abs_tol=1e-12)

    @given(rewards=continuous_rewards)
    def test_centering_invariant(self, rewards: list[float]) -> None:
        """Mean of advantages is zero (stronger than sum=0 for variable-length)."""
        advs = compute_grpo_advantages(rewards)
        if advs:
            mean_adv = sum(advs) / len(advs)
            assert math.isclose(mean_adv, 0.0, abs_tol=1e-10)

    @given(rewards=binary_rewards)
    def test_grpo_equals_reinforce_pp_per_group(self, rewards: list[float]) -> None:
        """GRPO and REINFORCE++ are identical at the per-group level."""
        grpo = compute_grpo_advantages(rewards)
        rpp = compute_reinforce_pp_advantages(rewards)
        for g, r in zip(grpo, rpp):
            assert math.isclose(g, r, abs_tol=1e-12)


# ═══════════════════════════════════════════
# MaxRL Properties
# ═══════════════════════════════════════════


class TestMaxRLProperties:
    @given(rewards=binary_rewards)
    def test_all_finite(self, rewards: list[float]) -> None:
        advs = compute_maxrl_advantages(rewards)
        for a in advs:
            valid_advantage(a)

    @given(n=st.integers(min_value=1, max_value=32))
    def test_all_zeros_produce_zeros(self, n: int) -> None:
        assert all(a == 0.0 for a in compute_maxrl_advantages([0.0] * n))

    @given(rewards=continuous_rewards)
    def test_length_preserved(self, rewards: list[float]) -> None:
        assert len(compute_maxrl_advantages(rewards)) == len(rewards)

    @given(
        value=st.floats(
            min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        n=st.integers(min_value=1, max_value=32),
    )
    def test_uniform_rewards_produce_zero(self, value: float, n: int) -> None:
        advs = compute_maxrl_advantages([value] * n)
        for a in advs:
            assert math.isclose(a, 0.0, abs_tol=1e-6)

    @given(rewards=binary_rewards)
    def test_correct_positive_negative_sign(self, rewards: list[float]) -> None:
        """Rewards above mean get positive advantage, below get negative."""
        advs = compute_maxrl_advantages(rewards)
        mean_r = sum(rewards) / len(rewards)
        if mean_r <= 1e-6:
            return  # all zeros case
        for r, a in zip(rewards, advs):
            if r > mean_r:
                assert a > 0
            elif r < mean_r:
                assert a < 0


# ═══════════════════════════════════════════
# GTPO Properties
# ═══════════════════════════════════════════


class TestGTPOProperties:
    @given(
        advantage=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        surprisals_=surprisals,
        beta_=beta_st,
    )
    def test_sign_preserved(
        self, advantage: float, surprisals_: list[float], beta_: float
    ) -> None:
        """Token advantages have same sign as episode advantage (or zero)."""
        token_advs = apply_gtpo_weighting(advantage, surprisals_, beta_)
        for ta in token_advs:
            valid_advantage(ta)
            if advantage > 0:
                assert ta >= -1e-10
            elif advantage < 0:
                assert ta <= 1e-10

    @given(
        advantage=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        surprisals_=surprisals,
    )
    def test_beta_zero_is_uniform(
        self, advantage: float, surprisals_: list[float]
    ) -> None:
        token_advs = apply_gtpo_weighting(advantage, surprisals_, beta=0.0)
        for ta in token_advs:
            assert math.isclose(ta, advantage, abs_tol=1e-10)

    @given(
        advantage=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        surprisals_=surprisals,
        beta_=beta_st,
    )
    def test_length_preserved(
        self, advantage: float, surprisals_: list[float], beta_: float
    ) -> None:
        assert len(apply_gtpo_weighting(advantage, surprisals_, beta_)) == len(
            surprisals_
        )

    @given(surprisals_=surprisals, beta_=beta_st)
    def test_zero_advantage_produces_zeros(
        self, surprisals_: list[float], beta_: float
    ) -> None:
        token_advs = apply_gtpo_weighting(0.0, surprisals_, beta_)
        for ta in token_advs:
            assert ta == 0.0

    @given(
        advantage=st.floats(
            min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        beta_=st.floats(
            min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_higher_surprisal_higher_weight(
        self, advantage: float, beta_: float
    ) -> None:
        """For positive advantage, higher surprisal → higher token advantage."""
        surps = [1.0, 5.0, 10.0]
        token_advs = apply_gtpo_weighting(advantage, surps, beta_)
        # Monotonic: ta[0] <= ta[1] <= ta[2]
        assert token_advs[0] <= token_advs[1] + 1e-10
        assert token_advs[1] <= token_advs[2] + 1e-10


# ═══════════════════════════════════════════
# HICRA Properties
# ═══════════════════════════════════════════


class TestHICRAProperties:
    @given(data=st.data())
    def test_sign_preserved(self, data: st.DataObject) -> None:
        n = data.draw(st.integers(min_value=1, max_value=32))
        token_advs = data.draw(
            st.lists(
                st.floats(
                    min_value=-10.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
            )
        )
        mask = data.draw(st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n))
        a = data.draw(alpha_st)
        result = apply_hicra(token_advs, mask, a)
        for orig, out in zip(token_advs, result):
            if orig > 0:
                assert out >= -1e-10
            elif orig < 0:
                assert out <= 1e-10

    @given(data=st.data())
    def test_alpha_zero_is_identity(self, data: st.DataObject) -> None:
        n = data.draw(st.integers(min_value=1, max_value=32))
        token_advs = data.draw(
            st.lists(
                st.floats(
                    min_value=-10.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
            )
        )
        mask = data.draw(st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n))
        result = apply_hicra(token_advs, mask, alpha=0.0)
        for orig, out in zip(token_advs, result):
            assert math.isclose(orig, out, abs_tol=1e-10)

    @given(data=st.data())
    def test_execution_tokens_unchanged(self, data: st.DataObject) -> None:
        """Execution tokens (mask=0) are never modified."""
        n = data.draw(st.integers(min_value=1, max_value=32))
        token_advs = data.draw(
            st.lists(
                st.floats(
                    min_value=-10.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
            )
        )
        a = data.draw(alpha_st)
        result = apply_hicra(token_advs, [0] * n, a)
        for orig, out in zip(token_advs, result):
            assert math.isclose(orig, out, abs_tol=1e-10)

    @given(data=st.data())
    def test_planning_positive_amplified(self, data: st.DataObject) -> None:
        """Positive planning tokens get amplified: A_HICRA >= A when A > 0."""
        n = data.draw(st.integers(min_value=1, max_value=32))
        token_advs = data.draw(
            st.lists(
                st.floats(
                    min_value=0.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
            )
        )
        a = data.draw(alpha_st)
        result = apply_hicra(token_advs, [1] * n, a)
        for orig, out in zip(token_advs, result):
            assert out >= orig - 1e-10

    @given(data=st.data())
    def test_planning_negative_dampened(self, data: st.DataObject) -> None:
        """Negative planning tokens move toward zero: |A_HICRA| <= |A|."""
        n = data.draw(st.integers(min_value=1, max_value=32))
        token_advs = data.draw(
            st.lists(
                st.floats(
                    min_value=-10.0,
                    max_value=0.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
            )
        )
        a = data.draw(alpha_st)
        result = apply_hicra(token_advs, [1] * n, a)
        for orig, out in zip(token_advs, result):
            # HICRA adds α|A|, so negative values move toward zero
            assert out >= orig - 1e-10


# ═══════════════════════════════════════════
# SEPA Properties
# ═══════════════════════════════════════════


class TestSEPAProperties:
    @given(data=st.data())
    def test_pooling_lambda_zero_is_identity(self, data: st.DataObject) -> None:
        n = data.draw(st.integers(min_value=1, max_value=32))
        surps = data.draw(
            st.lists(
                st.floats(
                    min_value=0.0,
                    max_value=50.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
            )
        )
        mask = data.draw(st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n))
        result = apply_sepa_pooling(surps, mask, lambda_t=0.0)
        for orig, out in zip(surps, result):
            assert math.isclose(orig, out, abs_tol=1e-10)

    @given(data=st.data())
    def test_amplification_lambda_zero_is_identity(self, data: st.DataObject) -> None:
        n = data.draw(st.integers(min_value=1, max_value=32))
        surps = data.draw(
            st.lists(
                st.floats(
                    min_value=0.0,
                    max_value=50.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
            )
        )
        mask = data.draw(st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n))
        result = apply_sepa_amplification(surps, mask, lambda_t=0.0)
        for orig, out in zip(surps, result):
            assert math.isclose(orig, out, abs_tol=1e-10)

    @given(data=st.data())
    def test_planning_tokens_unchanged_by_pooling(self, data: st.DataObject) -> None:
        n = data.draw(st.integers(min_value=1, max_value=32))
        surps = data.draw(
            st.lists(
                st.floats(
                    min_value=0.0,
                    max_value=50.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
            )
        )
        lam = data.draw(lambda_st)
        result = apply_sepa_pooling(surps, [1] * n, lam)
        for orig, out in zip(surps, result):
            assert math.isclose(orig, out, abs_tol=1e-10)

    @given(data=st.data())
    def test_planning_tokens_unchanged_by_amplification(
        self, data: st.DataObject
    ) -> None:
        n = data.draw(st.integers(min_value=1, max_value=32))
        surps = data.draw(
            st.lists(
                st.floats(
                    min_value=0.0,
                    max_value=50.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
            )
        )
        lam = data.draw(lambda_st)
        result = apply_sepa_amplification(surps, [1] * n, lam)
        for orig, out in zip(surps, result):
            assert math.isclose(orig, out, abs_tol=1e-10)

    @given(data=st.data())
    def test_clamped_amplification_non_negative(self, data: st.DataObject) -> None:
        n = data.draw(st.integers(min_value=1, max_value=32))
        surps = data.draw(
            st.lists(
                st.floats(
                    min_value=0.0,
                    max_value=50.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
            )
        )
        mask = data.draw(st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n))
        lam = data.draw(lambda_st)
        result = apply_sepa_amplification_clamped(surps, mask, lam)
        for v in result:
            assert v >= -1e-10

    @given(data=st.data())
    def test_pooling_reduces_variance(self, data: st.DataObject) -> None:
        """SEPA pooling reduces variance of execution tokens toward mean."""
        n = data.draw(st.integers(min_value=3, max_value=32))
        surps = data.draw(
            st.lists(
                st.floats(
                    min_value=0.1,
                    max_value=50.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n,
                max_size=n,
            )
        )
        mask = [0] * n  # All execution tokens
        lam = data.draw(
            st.floats(
                min_value=0.01,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        result = apply_sepa_pooling(surps, mask, lam)

        def variance(xs: list[float]) -> float:
            m = sum(xs) / len(xs)
            return sum((x - m) ** 2 for x in xs) / len(xs)

        # Pooling toward mean should reduce or maintain variance
        assert variance(result) <= variance(surps) + 1e-8


# ═══════════════════════════════════════════
# Hard Delight Gating Properties
# ═══════════════════════════════════════════


class TestDelightGatingProperties:
    @given(
        advantage=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        surprisals_=st.lists(
            st.floats(
                min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=32,
        ),
        k=k_frac_st,
    )
    def test_binary_output(
        self, advantage: float, surprisals_: list[float], k: float
    ) -> None:
        """Output is binary: each token is either A or 0.0."""
        result = apply_hard_delight_gating(advantage, surprisals_, k)
        for v in result:
            assert v == 0.0 or math.isclose(v, advantage, abs_tol=1e-10)

    @given(
        surprisals_=st.lists(
            st.floats(
                min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=32,
        ),
        k=k_frac_st,
    )
    def test_zero_advantage_all_zeros(
        self, surprisals_: list[float], k: float
    ) -> None:
        result = apply_hard_delight_gating(0.0, surprisals_, k)
        assert all(v == 0.0 for v in result)

    @given(
        advantage=st.floats(
            min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        surprisals_=st.lists(
            st.floats(
                min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=32,
        ),
        k=k_frac_st,
    )
    def test_kept_count(
        self, advantage: float, surprisals_: list[float], k: float
    ) -> None:
        """Exactly max(1, int(n*k_frac)) tokens are kept."""
        result = apply_hard_delight_gating(advantage, surprisals_, k)
        expected_k = max(1, int(len(surprisals_) * k))
        kept = sum(1 for v in result if v != 0.0)
        assert kept == expected_k

    @given(
        advantage=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        surprisals_=st.lists(
            st.floats(
                min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=32,
        ),
        k=k_frac_st,
    )
    def test_length_preserved(
        self, advantage: float, surprisals_: list[float], k: float
    ) -> None:
        result = apply_hard_delight_gating(advantage, surprisals_, k)
        assert len(result) == len(surprisals_)


# ═══════════════════════════════════════════
# Batch Normalization Properties
# ═══════════════════════════════════════════


class TestBatchNormProperties:
    @given(data=st.data())
    def test_zero_tokens_unchanged(self, data: st.DataObject) -> None:
        """Prompt-padding tokens (0.0) are never modified."""
        n_seqs = data.draw(st.integers(min_value=1, max_value=8))
        all_advs = []
        for _ in range(n_seqs):
            seq_len = data.draw(st.integers(min_value=2, max_value=16))
            # Mix of zeros (padding) and non-zeros (response tokens)
            seq = [0.0] + data.draw(
                st.lists(
                    st.floats(
                        min_value=-5.0,
                        max_value=5.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=seq_len - 1,
                    max_size=seq_len - 1,
                )
            )
            all_advs.append(seq)
        normed, _ = apply_batch_advantage_normalization(all_advs)
        for orig_seq, norm_seq in zip(all_advs, normed):
            for orig, norm in zip(orig_seq, norm_seq):
                if orig == 0.0:
                    assert norm == 0.0

    @given(data=st.data())
    def test_shape_preserved(self, data: st.DataObject) -> None:
        n_seqs = data.draw(st.integers(min_value=1, max_value=8))
        all_advs = []
        for _ in range(n_seqs):
            seq_len = data.draw(st.integers(min_value=1, max_value=16))
            seq = data.draw(
                st.lists(
                    st.floats(
                        min_value=-5.0,
                        max_value=5.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=seq_len,
                    max_size=seq_len,
                )
            )
            all_advs.append(seq)
        normed, _ = apply_batch_advantage_normalization(all_advs)
        assert len(normed) == len(all_advs)
        for orig_seq, norm_seq in zip(all_advs, normed):
            assert len(norm_seq) == len(orig_seq)

    @given(data=st.data())
    def test_all_finite(self, data: st.DataObject) -> None:
        n_seqs = data.draw(st.integers(min_value=1, max_value=8))
        all_advs = []
        for _ in range(n_seqs):
            seq_len = data.draw(st.integers(min_value=1, max_value=16))
            seq = data.draw(
                st.lists(
                    st.floats(
                        min_value=-5.0,
                        max_value=5.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=seq_len,
                    max_size=seq_len,
                )
            )
            all_advs.append(seq)
        normed, _ = apply_batch_advantage_normalization(all_advs)
        for seq in normed:
            for v in seq:
                valid_advantage(v)


# ═══════════════════════════════════════════
# Composable Pipeline Properties
# ═══════════════════════════════════════════


@st.composite
def pipeline_inputs(draw: st.DrawFn, g_max: int = 8, seq_max: int = 16):
    """Generate valid inputs for compute_composable_advantages."""
    g = draw(st.integers(min_value=1, max_value=g_max))
    rewards = draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=g, max_size=g))
    logprobs_G = []
    planning_masks_G = []
    for _ in range(g):
        seq_len = draw(st.integers(min_value=1, max_value=seq_max))
        lps = draw(
            st.lists(
                st.floats(
                    min_value=-50.0,
                    max_value=0.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=seq_len,
                max_size=seq_len,
            )
        )
        logprobs_G.append(lps)
        planning_masks_G.append(
            draw(st.lists(st.sampled_from([0, 1]), min_size=seq_len, max_size=seq_len))
        )
    return rewards, logprobs_G, planning_masks_G


class TestComposablePipelineProperties:
    @given(
        inputs=pipeline_inputs(),
        mode=st.sampled_from(["grpo", "maxrl"]),
        transform=st.sampled_from(["none", "gtpo"]),
    )
    @settings(max_examples=50, deadline=None)
    def test_output_shape_matches_logprobs(
        self,
        inputs: tuple[list[float], list[list[float]], list[list[int]]],
        mode: str,
        transform: str,
    ) -> None:
        rewards, logprobs_G, planning_masks_G = inputs
        result = compute_composable_advantages(
            rewards,
            logprobs_G,
            planning_masks_G,
            advantage_mode=mode,
            transform_mode=transform,
        )
        assert len(result.token_advs) == len(rewards)
        for i in range(len(rewards)):
            assert len(result.token_advs[i]) == len(logprobs_G[i])

    @given(
        inputs=pipeline_inputs(),
        mode=st.sampled_from(["grpo", "maxrl"]),
        transform=st.sampled_from(["none", "gtpo", "gtpo_hicra"]),
    )
    @settings(max_examples=50, deadline=None)
    def test_all_token_advs_finite(
        self,
        inputs: tuple[list[float], list[list[float]], list[list[int]]],
        mode: str,
        transform: str,
    ) -> None:
        rewards, logprobs_G, planning_masks_G = inputs
        result = compute_composable_advantages(
            rewards,
            logprobs_G,
            planning_masks_G,
            advantage_mode=mode,
            transform_mode=transform,
        )
        for seq in result.token_advs:
            for v in seq:
                valid_advantage(v)

    @given(inputs=pipeline_inputs())
    @settings(max_examples=30, deadline=None)
    def test_grpo_none_preserves_centering(
        self, inputs: tuple[list[float], list[list[float]], list[list[int]]]
    ) -> None:
        """GRPO + none transform: each episode's token advs are uniform and
        sum across episodes is zero."""
        rewards, logprobs_G, planning_masks_G = inputs
        result = compute_composable_advantages(
            rewards,
            logprobs_G,
            planning_masks_G,
            advantage_mode="grpo",
            transform_mode="none",
        )
        # With transform=none, each token gets the episode advantage uniformly
        episode_advs = [
            result.token_advs[i][0] if result.token_advs[i] else 0.0
            for i in range(len(rewards))
        ]
        assert math.isclose(sum(episode_advs), 0.0, abs_tol=1e-8)

    @given(
        inputs=pipeline_inputs(),
        mode=st.sampled_from(["grpo", "maxrl"]),
        transform=st.sampled_from(["none", "gtpo", "gtpo_hicra"]),
    )
    @settings(max_examples=30, deadline=None)
    def test_deterministic(
        self,
        inputs: tuple[list[float], list[list[float]], list[list[int]]],
        mode: str,
        transform: str,
    ) -> None:
        rewards, logprobs_G, planning_masks_G = inputs
        r1 = compute_composable_advantages(
            rewards,
            logprobs_G,
            planning_masks_G,
            advantage_mode=mode,
            transform_mode=transform,
        )
        r2 = compute_composable_advantages(
            rewards,
            logprobs_G,
            planning_masks_G,
            advantage_mode=mode,
            transform_mode=transform,
        )
        assert r1.token_advs == r2.token_advs


# ═══════════════════════════════════════════
# Differential: GRPO ≡ MaxRL under uniform rewards
# ═══════════════════════════════════════════


class TestDifferentialProperties:
    @given(
        value=st.floats(
            min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        n=st.integers(min_value=2, max_value=32),
    )
    def test_grpo_maxrl_equivalent_uniform(self, value: float, n: int) -> None:
        """Under uniform rewards, GRPO and MaxRL both produce all-zero advantages."""
        rewards = [value] * n
        grpo = compute_grpo_advantages(rewards)
        maxrl = compute_maxrl_advantages(rewards)
        for g, m in zip(grpo, maxrl):
            assert math.isclose(g, 0.0, abs_tol=1e-10)
            assert math.isclose(m, 0.0, abs_tol=1e-6)


# ═══════════════════════════════════════════
# Numerical Stability Under Extreme Values
# ═══════════════════════════════════════════


class TestNumericalStability:
    def test_grpo_extreme_rewards(self) -> None:
        """GRPO handles extreme but finite reward magnitudes."""
        rewards = [1e10, -1e10, 1e10]
        advs = compute_grpo_advantages(rewards)
        for a in advs:
            valid_advantage(a)
        assert math.isclose(sum(advs), 0.0, abs_tol=1.0)

    def test_maxrl_tiny_mean(self) -> None:
        """MaxRL with near-zero mean returns zeros (not inf)."""
        rewards = [1e-8, 0.0, 0.0, 0.0]
        advs = compute_maxrl_advantages(rewards)
        for a in advs:
            valid_advantage(a)

    def test_gtpo_extreme_surprisal_spread(self) -> None:
        """GTPO handles extreme surprisal spread without overflow."""
        surps = [1e-8, 49.99, 0.001, 50.0]
        advs = apply_gtpo_weighting(1.0, surps, beta=1.0)
        for a in advs:
            valid_advantage(a)

    def test_sepa_all_same_surprisals(self) -> None:
        """SEPA with uniform surprisals is a no-op."""
        surps = [5.0] * 10
        mask = [0] * 10
        pooled = apply_sepa_pooling(surps, mask, lambda_t=1.0)
        for orig, out in zip(surps, pooled):
            assert math.isclose(orig, out, abs_tol=1e-10)

    def test_batch_norm_single_nonzero(self) -> None:
        """Batch normalization with a single non-zero value doesn't crash."""
        all_advs = [[0.0, 0.0, 1.5, 0.0]]
        normed, metrics = apply_batch_advantage_normalization(all_advs)
        assert len(normed) == 1
        assert len(normed[0]) == 4
        for v in normed[0]:
            valid_advantage(v)

    @given(
        rewards=st.lists(
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=32,
        )
    )
    def test_grpo_never_produces_nan(self, rewards: list[float]) -> None:
        """GRPO never produces NaN for any finite input."""
        advs = compute_grpo_advantages(rewards)
        for a in advs:
            assert not math.isnan(a)
