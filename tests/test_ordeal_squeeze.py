"""Ordeal property tests for retrain.squeeze.

Tests SVD mathematical invariants: singular values non-negative,
sorted descending, variance monotonic, compression fidelity.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

torch = pytest.importorskip("torch")

from retrain.squeeze import LayerSqueeze, compress_layer, squeeze_layer


# ── Strategies ──

# Matrix dimensions: m, n are outer dims, r is rank (m, n >> r typical)
rank_st = st.integers(min_value=1, max_value=16)
outer_st = st.integers(min_value=4, max_value=64)


def random_lora_pair(
    m: int, n: int, r: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random LoRA pair A (m, r) and B (r, n)."""
    A = torch.randn(m, r)
    B = torch.randn(r, n)
    return A, B


# ═══════════════════════════════════════════
# SVD Invariants
# ═══════════════════════════════════════════


class TestSVDInvariants:
    @given(m=outer_st, n=outer_st, r=rank_st)
    @settings(max_examples=30, deadline=None)
    def test_singular_values_non_negative(self, m: int, n: int, r: int) -> None:
        r = min(r, m, n)
        A, B = random_lora_pair(m, n, r)
        result = squeeze_layer(A, B, target_ranks=[r])
        for sv in result.singular_values.tolist():
            assert sv >= -1e-6, f"singular value {sv} is negative"

    @given(m=outer_st, n=outer_st, r=rank_st)
    @settings(max_examples=30, deadline=None)
    def test_singular_values_sorted_descending(
        self, m: int, n: int, r: int
    ) -> None:
        r = min(r, m, n)
        A, B = random_lora_pair(m, n, r)
        result = squeeze_layer(A, B, target_ranks=[r])
        svs = result.singular_values.tolist()
        for i in range(len(svs) - 1):
            assert svs[i] >= svs[i + 1] - 1e-6

    @given(m=outer_st, n=outer_st, r=rank_st)
    @settings(max_examples=30, deadline=None)
    def test_source_rank_matches(self, m: int, n: int, r: int) -> None:
        r = min(r, m, n)
        A, B = random_lora_pair(m, n, r)
        result = squeeze_layer(A, B, target_ranks=[r])
        assert result.source_rank == r

    @given(m=outer_st, n=outer_st, r=rank_st)
    @settings(max_examples=30, deadline=None)
    def test_singular_values_count(self, m: int, n: int, r: int) -> None:
        r = min(r, m, n)
        A, B = random_lora_pair(m, n, r)
        result = squeeze_layer(A, B, target_ranks=[r])
        assert len(result.singular_values) == r


# ═══════════════════════════════════════════
# Variance Invariants
# ═══════════════════════════════════════════


class TestVarianceInvariants:
    @given(m=outer_st, n=outer_st, r=st.integers(min_value=2, max_value=16))
    @settings(max_examples=30, deadline=None)
    def test_full_rank_variance_is_one(self, m: int, n: int, r: int) -> None:
        """Variance at full rank = 1.0 (100% of variance explained)."""
        r = min(r, m, n)
        A, B = random_lora_pair(m, n, r)
        result = squeeze_layer(A, B, target_ranks=[r])
        assert math.isclose(result.variance_at_rank[r], 1.0, abs_tol=1e-4)

    @given(m=outer_st, n=outer_st, r=st.integers(min_value=3, max_value=16))
    @settings(max_examples=30, deadline=None)
    def test_variance_monotonically_non_decreasing(
        self, m: int, n: int, r: int
    ) -> None:
        r = min(r, m, n)
        if r < 3:
            return
        targets = list(range(1, r + 1))
        A, B = random_lora_pair(m, n, r)
        result = squeeze_layer(A, B, target_ranks=targets)
        prev = 0.0
        for k in targets:
            v = result.variance_at_rank[k]
            assert v >= prev - 1e-6, f"variance decreased at rank {k}"
            prev = v

    @given(m=outer_st, n=outer_st, r=st.integers(min_value=2, max_value=16))
    @settings(max_examples=30, deadline=None)
    def test_variance_bounded_zero_one(self, m: int, n: int, r: int) -> None:
        """Variance is in [0, 1]."""
        r = min(r, m, n)
        targets = list(range(1, r + 1))
        A, B = random_lora_pair(m, n, r)
        result = squeeze_layer(A, B, target_ranks=targets)
        for k in targets:
            v = result.variance_at_rank[k]
            assert -1e-6 <= v <= 1.0 + 1e-6, f"variance {v} out of bounds"

    def test_zero_matrix_variance(self) -> None:
        """Zero matrices → variance = 1.0 (degenerate case)."""
        A = torch.zeros(8, 4)
        B = torch.zeros(4, 8)
        result = squeeze_layer(A, B, target_ranks=[1, 2, 4])
        for k in [1, 2, 4]:
            assert math.isclose(result.variance_at_rank[k], 1.0, abs_tol=1e-4)


# ═══════════════════════════════════════════
# Compression Invariants
# ═══════════════════════════════════════════


class TestCompressionInvariants:
    @given(m=outer_st, n=outer_st, r=st.integers(min_value=2, max_value=16))
    @settings(max_examples=20, deadline=None)
    def test_full_rank_exact_reconstruction(
        self, m: int, n: int, r: int
    ) -> None:
        """Compressing at full rank reproduces A @ B exactly."""
        r = min(r, m, n)
        A, B = random_lora_pair(m, n, r)
        original = A @ B
        A_c, B_c = compress_layer(A, B, target_rank=r)
        compressed = A_c @ B_c
        assert torch.allclose(original, compressed, atol=1e-4, rtol=1e-4)

    @given(
        m=outer_st,
        n=outer_st,
        r=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=20, deadline=None)
    def test_compressed_shapes(self, m: int, n: int, r: int) -> None:
        """Compressed output has correct shapes."""
        r = min(r, m, n)
        target = max(1, r // 2)
        A, B = random_lora_pair(m, n, r)
        A_c, B_c = compress_layer(A, B, target_rank=target)
        assert A_c.shape == (m, target)
        assert B_c.shape == (target, n)

    @given(m=outer_st, n=outer_st, r=st.integers(min_value=3, max_value=16))
    @settings(max_examples=20, deadline=None)
    def test_error_decreases_with_rank(self, m: int, n: int, r: int) -> None:
        """Higher target rank → lower reconstruction error."""
        r = min(r, m, n)
        if r < 3:
            return
        A, B = random_lora_pair(m, n, r)
        original = A @ B

        errors = []
        for target in range(1, r + 1):
            A_c, B_c = compress_layer(A, B, target_rank=target)
            err = (original - A_c @ B_c).norm().item()
            errors.append(err)

        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-4

    @given(m=outer_st, n=outer_st, r=rank_st)
    @settings(max_examples=20, deadline=None)
    def test_target_rank_clamped_to_source(
        self, m: int, n: int, r: int
    ) -> None:
        """Target rank > source rank is clamped — same as full rank."""
        r = min(r, m, n)
        A, B = random_lora_pair(m, n, r)
        original = A @ B
        A_c, B_c = compress_layer(A, B, target_rank=r + 10)
        compressed = A_c @ B_c
        assert torch.allclose(original, compressed, atol=1e-4, rtol=1e-4)
