"""LoRA-Squeeze: optimal rank analysis and compression.

Based on LoRA-Squeeze (arXiv 2602.10993): train at high rank, then use
memory-efficient RSVD to measure explained variance at each target rank,
compress to the optimal one.

Key insight: "it is better to first learn an expressive, higher-rank
solution and then compress it."
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

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

def _resolve_tinker_path(tinker_path: str, local_dir: str | None = None) -> str:
    """Download a tinker:// checkpoint to a local directory.

    Uses the Tinker SDK to get a signed archive URL, downloads and extracts
    the PEFT adapter files (adapter_model.safetensors + adapter_config.json).

    Args:
        tinker_path: tinker:// URI (e.g. tinker://run-id/weights/checkpoint-name)
        local_dir: where to extract (default: tempdir under /tmp/retrain_squeeze/)

    Returns:
        Local directory path containing the extracted adapter files.
    """
    try:
        import tinker
    except ImportError:
        raise RuntimeError(
            "Tinker SDK required for tinker:// paths. Install with: uv add tinker"
        )

    if local_dir is None:
        local_dir = tempfile.mkdtemp(prefix="retrain_squeeze_")

    out = Path(local_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (out / "adapter_model.safetensors").is_file():
        print(f"Using cached adapter at {out}")
        return str(out)

    print(f"Downloading Tinker checkpoint: {tinker_path}")
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()

    url_response = rest_client.get_checkpoint_archive_url_from_tinker_path(
        tinker_path
    ).result()

    # Download archive
    archive_path = out / "checkpoint.tar"
    with urllib.request.urlopen(url_response.url, timeout=300) as response:
        with open(archive_path, "wb") as f:
            f.write(response.read())

    # Extract
    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(path=out, filter="data")

    archive_path.unlink()

    # Verify expected files
    if not (out / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(
            f"Downloaded checkpoint missing adapter_model.safetensors: {out}"
        )

    print(f"Checkpoint extracted to {out}")
    return str(out)


def _resolve_adapter_path(adapter_path: str) -> str:
    """Resolve adapter_path: download if tinker://, otherwise return as-is."""
    if adapter_path.startswith("tinker://"):
        return _resolve_tinker_path(adapter_path)
    return adapter_path


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------

def load_adapter_matrices(
    adapter_path: str, device: str = "cpu"
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Load PEFT safetensors, pair lora_A/lora_B by module name.

    PEFT stores:
      lora_A.weight: (r, in_features)   — we transpose to (in_features, r)
      lora_B.weight: (out_features, r)   — kept as-is

    Returns list of (module_name, A, B) where:
      A: (m, r) = in_features × rank
      B: (r, n) = rank × out_features  (transposed from PEFT convention)

    So the effective weight delta is A @ B = (m, n).
    """
    safetensors_path = Path(adapter_path) / "adapter_model.safetensors"
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_path}")

    state_dict = load_file(str(safetensors_path), device=device)

    # Group by module name
    a_matrices: dict[str, torch.Tensor] = {}
    b_matrices: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if "lora_A" in key:
            # Extract module name: everything before .lora_A
            module_name = key.split(".lora_A")[0]
            # PEFT stores lora_A as (r, in_features), transpose to (in_features, r)
            a_matrices[module_name] = tensor.float().t()
        elif "lora_B" in key:
            module_name = key.split(".lora_B")[0]
            # PEFT stores lora_B as (out_features, r), transpose to (r, out_features)
            b_matrices[module_name] = tensor.float().t()

    # Pair them up
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for name in sorted(a_matrices.keys()):
        if name not in b_matrices:
            continue
        pairs.append((name, a_matrices[name], b_matrices[name]))

    if not pairs:
        raise ValueError(f"No lora_A/lora_B pairs found in {adapter_path}")

    return pairs


# ---------------------------------------------------------------------------
# Core algorithm (Algorithm 2 from paper)
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

def analyze_adapter(
    adapter_path: str,
    source_rank: int = 0,
    target_ranks: list[int] | None = None,
    min_variance_retention: float = 0.95,
    device: str = "cpu",
) -> SqueezeAnalysis:
    """Run squeeze analysis on all LoRA layers in a saved adapter.

    Args:
        adapter_path: path to PEFT adapter directory
        source_rank: expected rank (0 = detect from weights)
        target_ranks: ranks to evaluate (empty = auto power-of-2)
        min_variance_retention: threshold for recommendation
        device: torch device string

    Returns:
        SqueezeAnalysis with per-layer and aggregate results.
    """
    adapter_path = _resolve_adapter_path(adapter_path)
    pairs = load_adapter_matrices(adapter_path, device)

    # Detect source rank from first layer if not specified
    if source_rank == 0:
        source_rank = pairs[0][1].shape[1]

    # Auto target ranks: powers of 2 up to source rank
    if not target_ranks:
        target_ranks = []
        r = 1
        while r < source_rank:
            target_ranks.append(r)
            r *= 2
        target_ranks.append(source_rank)

    layers: list[LayerSqueeze] = []
    for name, A, B in pairs:
        result = squeeze_layer(A, B, target_ranks)
        result.name = name
        layers.append(result)

    # Aggregate: mean variance across layers
    mean_variance: dict[int, float] = {}
    for k in target_ranks:
        vals = [layer.variance_at_rank[k] for layer in layers]
        mean_variance[k] = sum(vals) / len(vals)

    # Recommend: lowest rank with mean variance >= threshold
    recommended_rank = source_rank
    for k in sorted(target_ranks):
        if mean_variance[k] >= min_variance_retention:
            recommended_rank = k
            break

    return SqueezeAnalysis(
        layers=layers,
        target_ranks=target_ranks,
        mean_variance=mean_variance,
        recommended_rank=recommended_rank,
        min_variance_retention=min_variance_retention,
    )


def compress_adapter(
    adapter_path: str,
    output_path: str,
    target_rank: int,
    device: str = "cpu",
) -> None:
    """Compress all LoRA layers and save as a valid PEFT adapter.

    Updates adapter_config.json with new rank and scaled alpha:
      alpha_new = alpha_old * r_tgt / r_src
    """
    adapter_path = _resolve_adapter_path(adapter_path)
    pairs = load_adapter_matrices(adapter_path, device)

    # Build compressed state dict (in PEFT naming convention)
    compressed: dict[str, torch.Tensor] = {}
    source_rank = pairs[0][1].shape[1]

    for name, A, B in pairs:
        A_tgt, B_tgt = compress_layer(A, B, target_rank)
        # Convert back to PEFT convention:
        # lora_A.weight: (r, in_features) = A_tgt.t()
        # lora_B.weight: (out_features, r) = B_tgt.t()
        a_key = f"{name}.lora_A.weight"
        b_key = f"{name}.lora_B.weight"
        compressed[a_key] = A_tgt.t().contiguous()
        compressed[b_key] = B_tgt.t().contiguous()

    # Save safetensors
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    save_file(compressed, str(out / "adapter_model.safetensors"))

    # Copy and update adapter_config.json
    config_src = Path(adapter_path) / "adapter_config.json"
    if config_src.is_file():
        with open(config_src) as f:
            config = json.load(f)

        old_rank = config.get("r", source_rank)
        old_alpha = config.get("lora_alpha", old_rank * 2)

        # Scale alpha proportionally
        config["r"] = target_rank
        config["lora_alpha"] = int(old_alpha * target_rank / old_rank)

        with open(out / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
    else:
        # Create minimal config
        alpha = target_rank * 2
        config = {"r": target_rank, "lora_alpha": alpha}
        with open(out / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

    print(f"Compressed adapter saved to {output_path} (rank {source_rank} -> {target_rank})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_squeeze(config_path: str) -> None:
    """Entry point: load TOML, analyze adapter, print table, optionally compress."""
    from retrain.config import load_squeeze_config

    cfg = load_squeeze_config(config_path)

    print(f"LoRA-Squeeze: analyzing {cfg.adapter_path}")
    print(f"  min_variance_retention = {cfg.min_variance_retention}")

    analysis = analyze_adapter(
        adapter_path=cfg.adapter_path,
        source_rank=cfg.source_rank,
        target_ranks=cfg.target_ranks if cfg.target_ranks else None,
        min_variance_retention=cfg.min_variance_retention,
        device=cfg.device,
    )

    # Print variance table
    print(f"\nSource rank: {analysis.layers[0].source_rank}")
    print(f"Layers analyzed: {len(analysis.layers)}")
    print()

    header = f"{'Rank':>6}  {'Mean Var%':>9}  {'Min Var%':>9}  {'Max Var%':>9}"
    print(header)
    print("-" * len(header))

    for k in analysis.target_ranks:
        vals = [layer.variance_at_rank[k] for layer in analysis.layers]
        mean_v = analysis.mean_variance[k]
        min_v = min(vals)
        max_v = max(vals)
        marker = " <--" if k == analysis.recommended_rank else ""
        print(
            f"{k:>6}  {mean_v * 100:>8.2f}%  {min_v * 100:>8.2f}%  {max_v * 100:>8.2f}%{marker}"
        )

    print(f"\nRecommended rank: {analysis.recommended_rank} "
          f"(>= {analysis.min_variance_retention * 100:.0f}% variance retained)")

    # Compress if requested
    target = cfg.compress_to if cfg.compress_to > 0 else analysis.recommended_rank
    if cfg.output_path:
        print(f"\nCompressing to rank {target}...")
        compress_adapter(
            adapter_path=cfg.adapter_path,
            output_path=cfg.output_path,
            target_rank=target,
            device=cfg.device,
        )
