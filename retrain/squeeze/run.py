"""Squeeze workflows: analyze, compress, and the `[squeeze]` TOML entry."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from retrain.squeeze.adapter import _resolve_adapter_path, load_adapter_matrices
from retrain.squeeze.rank import (
    LayerSqueeze,
    SqueezeAnalysis,
    compress_layer,
    squeeze_layer,
)


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
