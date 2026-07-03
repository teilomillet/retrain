"""Scaffolding shared by every local optimizer step.

Each step variant (RL, SFT, hybrid ECHO) loops over microbatches, applies
one optimizer update, snapshots LoRA weights for cross-thread sync, and
records the same timing/telemetry row. The shared pieces live here so the
step modules contain only their loss logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from retrain.backends.local import metrics as local_metrics
from retrain.backends.local import sync as local_sync
from retrain.backends.local.memory import saved_tensors_context
from retrain.backends.torch import timer_start, timer_stop

if TYPE_CHECKING:
    from retrain.backends.local.train import LocalTrainHelper


def saved_tensors(helper: "LocalTrainHelper"):
    """The helper's configured activation-offload context for one backward."""
    return saved_tensors_context(
        enabled=bool(getattr(helper, "train_save_on_cpu", False)),
        train_device=helper.train_device,
        pin_memory=bool(getattr(helper, "train_save_on_cpu_pin_memory", True)),
        min_numel=int(getattr(helper, "train_save_on_cpu_min_numel", 0)),
    )


def apply_optimizer(helper: "LocalTrainHelper") -> float:
    """Run the (scaled) optimizer step + update; return elapsed seconds."""
    timer = timer_start(helper.train_device)
    helper.scaler.step(helper.optimizer)
    helper.scaler.update()
    return timer_stop(timer)


def policy_total_tokens(helper: "LocalTrainHelper", advantages, attention_mask):
    """Number of loss-bearing tokens for policy steps (selective or dense)."""
    if getattr(helper, "train_selective_suffix_logits", False):
        return (
            ((attention_mask[:, 1:] > 0) & (advantages[:, 1:] != 0))
            .sum()
            .clamp(min=1)
        )
    return attention_mask[:, 1:].sum().clamp(min=1)


def snapshot_and_record(
    helper: "LocalTrainHelper",
    *,
    kind: str,
    wall_start_s: float,
    forward_s: float,
    backward_s: float,
    optimizer_s: float,
    microbatches: int,
    total_tokens: float,
    batch_size: int,
) -> None:
    """Snapshot LoRA weights for cross-thread sync and record step telemetry."""
    snapshot_s = local_sync.snapshot_lora_weights_if_needed(helper)
    local_metrics.record_train(
        helper,
        kind=kind,
        wall_start_s=wall_start_s,
        forward_s=forward_s,
        backward_s=backward_s,
        optimizer_s=optimizer_s,
        snapshot_s=snapshot_s,
        microbatches=microbatches,
        total_tokens=total_tokens,
        batch_size=batch_size,
    )
