"""Shared metrics and W&B projections for supervised fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping

from retrain.backends import RuntimeMetric
from retrain.config import TrainConfig
from retrain.training.telemetry import build_runtime_wandb_metrics


_SFT_SCHEDULE_WANDB_FIELDS = (
    ("sft_epoch", "train/sft/epoch"),
    ("sft_epoch_end", "train/sft/epoch_end"),
    ("sft_epoch_seed", "train/sft/epoch_seed"),
    ("sft_epoch_end_seed", "train/sft/epoch_end_seed"),
    ("sft_epoch_sample_offset", "train/sft/epoch_sample_offset"),
    ("sft_reshuffle_each_epoch", "train/sft/reshuffle_each_epoch"),
    ("sft_batch_indices_sha256", "train/sft/batch_indices_sha256"),
    ("sft_epoch_start_order_sha256", "train/sft/epoch_start_order_sha256"),
)

_SFT_BATCH_WANDB_KEYS = (
    "sft_sequence_length_min",
    "sft_sequence_length_mean",
    "sft_sequence_length_p50",
    "sft_sequence_length_p95",
    "sft_sequence_length_max",
    "sft_logical_padded_tokens",
    "sft_logical_padding_tokens",
    "sft_logical_padding_fraction",
    "sft_supervised_token_fraction",
)


def sft_runtime_metric_key(key: str) -> str:
    """Keep optimizer evidence canonical; namespace ordinary backend metrics."""

    return key if key.startswith("optimizer/") else f"backend/{key}"


def build_sft_step_metrics(
    config: TrainConfig,
    *,
    step: int,
    loss: float,
    datums: int,
    total_tokens: int,
    supervised_tokens: int,
    batch_size: int,
    example_count: int,
    elapsed: float,
    lr: float,
) -> dict[str, int | float | str]:
    """Build fields shared by standalone and warmup SFT step logs."""

    return {
        "step": step,
        "loss": loss,
        "phase": "sft",
        "datums": datums,
        "tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
        "sft_batch_order": config.sft_batch_order,
        "sft_length_bucket_size": int(config.sft_length_bucket_size),
        "sft_reshuffle_each_epoch": int(config.sft_reshuffle_each_epoch),
        "sft_unique_examples_seen": min(example_count, (step + 1) * batch_size),
        "sft_dataset_coverage": min(
            1.0,
            ((step + 1) * batch_size) / max(example_count, 1),
        ),
        "time_s": round(elapsed, 2),
        "lr": lr,
    }


def _schedule_wandb_metrics(
    metrics: Mapping[str, object],
) -> dict[str, object]:
    projected = {
        target: metrics[source] for source, target in _SFT_SCHEDULE_WANDB_FIELDS
    }
    if "sft_epoch_end_order_sha256" in metrics:
        projected["train/sft/epoch_end_order_sha256"] = metrics[
            "sft_epoch_end_order_sha256"
        ]
    return projected


def build_standalone_sft_wandb_metrics(
    metrics: Mapping[str, object],
    *,
    runtime_metrics: Mapping[str, RuntimeMetric],
    recoverability_metrics: Mapping[str, object],
) -> dict[str, object]:
    """Project one standalone SFT step onto its stable W&B fields."""

    wandb_metrics: dict[str, object] = {
        "train/loss": metrics["loss"],
        "train/sft": 1,
        "train/step": metrics["step"],
        "train/lr": metrics["lr"],
        "train/datums": metrics["datums"],
        "train/tokens": metrics["tokens"],
        "train/supervised_tokens": metrics["supervised_tokens"],
        "train/sft_dataset_coverage": metrics["sft_dataset_coverage"],
        "train/step_time_s": metrics["step_time_s"],
        "train/train_time_s": metrics["train_time_s"],
        "train/train_time_semantics": metrics["train_time_semantics"],
    }
    wandb_metrics.update(_schedule_wandb_metrics(metrics))
    if "train_submit_enqueue_share" in metrics:
        wandb_metrics["train/train_submit_enqueue_time_s"] = metrics[
            "train_submit_enqueue_time_s"
        ]
        wandb_metrics["train/train_submit_enqueue_share"] = metrics[
            "train_submit_enqueue_share"
        ]
    else:
        wandb_metrics["train/train_share"] = metrics["train_share"]
    wandb_metrics.update(
        {
            f"train/sft/{key.removeprefix('sft_')}": metrics[key]
            for key in _SFT_BATCH_WANDB_KEYS
        }
    )
    if "process_max_rss_mb" in metrics:
        wandb_metrics["train/process_max_rss_mb"] = metrics["process_max_rss_mb"]
    wandb_metrics.update(build_runtime_wandb_metrics(runtime_metrics))
    wandb_metrics.update(recoverability_metrics)
    wandb_metrics.update(
        {
            f"train/{key}": value
            for key, value in metrics.items()
            if key.startswith("optimizer/")
        }
    )
    return wandb_metrics


def build_warmup_sft_wandb_metrics(
    metrics: Mapping[str, object],
    *,
    recoverability_metrics: Mapping[str, object],
) -> dict[str, object]:
    """Project one warmup SFT step onto its stable W&B fields."""

    wandb_metrics: dict[str, object] = {
        "train/loss": metrics["loss"],
        "train/sft_signal": metrics["sft_signal"],
        "train/sft_warmup": 1,
        "train/step": metrics["step"],
        "train/lr": metrics["lr"],
    }
    wandb_metrics.update(_schedule_wandb_metrics(metrics))
    wandb_metrics.update(recoverability_metrics)
    return wandb_metrics
