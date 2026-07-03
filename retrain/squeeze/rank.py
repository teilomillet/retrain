"""RSVD rank analysis and compression math for LoRA layers.

Based on LoRA-Squeeze (arXiv 2602.10993): train at high rank, then use
memory-efficient RSVD to measure explained variance at each target rank,
compress to the optimal one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerSqueeze:
    """Per-layer squeeze analysis result."""

    name: str
    source_rank: int
    singular_values: torch.Tensor  # shape (r,)
    variance_at_rank: dict[int, float]  # rank -> cumulative variance ratio


@dataclass
class SqueezeAnalysis:
    """Aggregate squeeze analysis across all layers."""

    layers: list[LayerSqueeze]
    target_ranks: list[int]
    mean_variance: dict[int, float]  # rank -> mean variance across layers
    recommended_rank: int
    min_variance_retention: float


# ---------------------------------------------------------------------------
# Tinker checkpoint download
# ---------------------------------------------------------------------------


def squeeze_layer(
    A: torch.Tensor, B: torch.Tensor, target_ranks: list[int]
) -> LayerSqueeze:
    """Memory-efficient SVD analysis of a single LoRA layer.

    Algorithm 2 from LoRA-Squeeze: operates on the small r×r core matrix,
    never forms the full m×n product.

    Args:
        A: (m, r) matrix
        B: (r, n) matrix
        target_ranks: list of ranks to evaluate variance at

    Returns:
        LayerSqueeze with singular values and variance at each target rank.
    """
    r = A.shape[1]
    assert B.shape[0] == r, f"Rank mismatch: A has {A.shape[1]} cols, B has {B.shape[0]} rows"

    # QR decompositions (memory-efficient: never form m×n)
    Q_A, R_A = torch.linalg.qr(A)  # Q_A: (m, r), R_A: (r, r)
    Q_B, R_B = torch.linalg.qr(B.t())  # Q_B: (n, r), R_B: (r, r)

    # Core SVD on small r×r matrix
    M = R_A @ R_B.t()  # (r, r)
    _, S, _ = torch.linalg.svd(M)  # S: (r,)

    # Variance retention: V(k) = sum(s²[:k]) / sum(s²)
    s_squared = S ** 2
    total_var = s_squared.sum().item()

    variance_at_rank: dict[int, float] = {}
    for k in target_ranks:
        k_clamped = min(k, r)
        if total_var > 0:
            variance_at_rank[k] = s_squared[:k_clamped].sum().item() / total_var
        else:
            variance_at_rank[k] = 1.0

    return LayerSqueeze(
        name="",  # filled in by caller
        source_rank=r,
        singular_values=S,
        variance_at_rank=variance_at_rank,
    )


def compress_layer(
    A: torch.Tensor, B: torch.Tensor, target_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress a LoRA layer to a target rank via SVD.

    Same QR + core SVD as squeeze_layer, but reconstructs truncated factors:
      A_tgt = Q_A @ U_k @ sqrt(S_k)   shape (m, target_rank)
      B_tgt = sqrt(S_k) @ V_k^T @ Q_B^T  shape (target_rank, n)

    So A_tgt @ B_tgt ≈ A @ B.
    """
    r = A.shape[1]
    k = min(target_rank, r)

    Q_A, R_A = torch.linalg.qr(A)
    Q_B, R_B = torch.linalg.qr(B.t())

    M = R_A @ R_B.t()
    U, S, Vh = torch.linalg.svd(M)

    # Truncate to target rank
    U_k = U[:, :k]       # (r, k)
    S_k = S[:k]           # (k,)
    Vh_k = Vh[:k, :]      # (k, r)

    sqrt_S = torch.sqrt(S_k)

    # Reconstruct compressed factors
    A_tgt = Q_A @ U_k @ torch.diag(sqrt_S)          # (m, k)
    B_tgt = torch.diag(sqrt_S) @ Vh_k @ Q_B.t()     # (k, n)

    return A_tgt, B_tgt


# ---------------------------------------------------------------------------
# Adapter-level analysis and compression
# ---------------------------------------------------------------------------
