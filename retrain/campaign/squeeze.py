"""Post-campaign LoRA squeeze analysis and wandb logging."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from retrain.campaign.parse import float_from_object, int_from_object


class _SqueezeLayerLike(Protocol):
    source_rank: int
    variance_at_rank: Mapping[int, float]


class _SqueezeAnalysisLike(Protocol):
    layers: Sequence[_SqueezeLayerLike]
    target_ranks: Sequence[int]
    mean_variance: Mapping[int, float]
    recommended_rank: int
    min_variance_retention: float


def auto_squeeze(
    adapter_path: str,
    squeeze_cfg: Mapping[str, object],
    lora_rank: int,
    wandb_project: str = "",
    wandb_entity: str = "",
) -> int:
    """Run squeeze analysis, print results, log to wandb. Returns recommended rank."""
    from retrain.squeeze.run import analyze_adapter

    min_var = float_from_object(
        squeeze_cfg.get("min_variance_retention", 0.95),
        "squeeze.min_variance_retention",
    )
    source_rank = (
        int_from_object(
            squeeze_cfg.get("source_rank", 0),
            "squeeze.source_rank",
        )
        or lora_rank
    )

    print(f"\n{'=' * 60}")
    print(f"Auto-squeeze: analyzing {adapter_path}")
    print(f"  source_rank={source_rank}, min_variance_retention={min_var}")

    analysis = analyze_adapter(
        adapter_path=adapter_path,
        source_rank=source_rank,
        min_variance_retention=min_var,
    )

    # Print variance table
    print(f"\nSource rank: {analysis.layers[0].source_rank}")
    print(f"Layers analyzed: {len(analysis.layers)}\n")

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

    print(
        f"\nRecommended rank: {analysis.recommended_rank} "
        f"(>= {min_var * 100:.0f}% variance retained)"
    )
    print(f"{'=' * 60}\n")

    # Log to wandb
    if wandb_project:
        _log_squeeze_to_wandb(
            cast(_SqueezeAnalysisLike, analysis),
            wandb_project,
            wandb_entity,
        )

    return analysis.recommended_rank


def _log_squeeze_to_wandb(
    analysis: _SqueezeAnalysisLike,
    wandb_project: str,
    wandb_entity: str = "",
) -> None:
    """Log squeeze analysis to wandb as a dedicated run with table + summary."""
    try:
        import wandb
    except ImportError:
        print("wandb not installed, skipping squeeze logging")
        return

    run = wandb.init(
        project=wandb_project,
        name="squeeze-analysis",
        job_type="squeeze",
        tags=["squeeze", f"rank-{analysis.recommended_rank}"],
        config={
            "source_rank": analysis.layers[0].source_rank,
            "recommended_rank": analysis.recommended_rank,
            "min_variance_retention": analysis.min_variance_retention,
            "num_layers": len(analysis.layers),
        },
        entity=wandb_entity or None,
    )

    # Variance table
    columns = ["rank", "mean_variance", "min_variance", "max_variance", "recommended"]
    table = wandb.Table(columns=columns)

    for k in analysis.target_ranks:
        vals = [layer.variance_at_rank[k] for layer in analysis.layers]
        mean_v = analysis.mean_variance[k]
        min_v = min(vals)
        max_v = max(vals)
        table.add_data(k, mean_v, min_v, max_v, k == analysis.recommended_rank)

    run.log({"squeeze/variance_table": table})

    # Line chart data: log each rank as a step for a clean variance curve
    for k in analysis.target_ranks:
        vals = [layer.variance_at_rank[k] for layer in analysis.layers]
        run.log(
            {
                "squeeze/mean_variance": analysis.mean_variance[k],
                "squeeze/min_variance": min(vals),
                "squeeze/max_variance": max(vals),
                "squeeze/rank": k,
            }
        )

    # Summary metrics
    run.summary["squeeze/recommended_rank"] = analysis.recommended_rank
    run.summary["squeeze/source_rank"] = analysis.layers[0].source_rank
    run.summary["squeeze/min_variance_retention"] = analysis.min_variance_retention
    run.summary["squeeze/num_layers"] = len(analysis.layers)

    run.finish()
    print(f"Squeeze results logged to wandb: {wandb_project}/squeeze-analysis")
