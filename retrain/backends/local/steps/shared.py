"""Scaffolding shared by every local optimizer step.

Each step variant (RL, SFT, hybrid ECHO) loops over microbatches, applies
one optimizer update, snapshots LoRA weights for cross-thread sync, and
records the same timing/telemetry row. The shared pieces live here so the
step modules contain only their loss logic.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    """Count sampled action positions independently of the logits fast path."""
    _ = helper
    return ((attention_mask[:, 1:] > 0) & (advantages[:, 1:] != 0)).sum().clamp(min=1)


def policy_total_tokens_from_rows(
    helper: "LocalTrainHelper",
    tokens: Sequence[Sequence[int]],
    advantages: Sequence[Sequence[float]],
) -> float:
    """Count sampled action positions before tensors are padded."""
    _ = helper
    count = sum(
        1
        for token_row, advantage_row in zip(tokens, advantages, strict=True)
        for value in advantage_row[1 : len(token_row)]
        if value != 0.0
    )
    return float(max(count, 1))


def record_sequence_padding(
    helper: "LocalTrainHelper",
    lengths: Sequence[int],
    ranges: Sequence[tuple[int, int]],
) -> None:
    """Record exact padding and a quadratic attention-work proxy."""
    metrics = getattr(helper, "_last_train_metrics", {})
    if not isinstance(metrics, dict) or not lengths:
        return

    max_len = max(lengths)
    global_padded = len(lengths) * max_len
    global_attention = len(lengths) * max_len * max_len
    microbatch_padded = 0
    microbatch_attention = 0
    for start, stop in ranges:
        chunk = lengths[start:stop]
        if not chunk:
            continue
        chunk_max = max(chunk)
        microbatch_padded += len(chunk) * chunk_max
        microbatch_attention += len(chunk) * chunk_max * chunk_max

    sorted_lengths = sorted(lengths)
    p50 = sorted_lengths[max(0, (50 * len(lengths) + 99) // 100 - 1)]
    p95 = sorted_lengths[max(0, (95 * len(lengths) + 99) // 100 - 1)]
    avoided = max(0, global_padded - microbatch_padded)
    attention_avoided = max(0, global_attention - microbatch_attention)
    metrics.update(
        {
            "local_train_microbatch_local_padding": 1,
            "local_train_unpadded_tokens": sum(lengths),
            "local_train_global_padded_tokens": global_padded,
            "local_train_microbatch_padded_tokens": microbatch_padded,
            "local_train_padding_tokens_avoided": avoided,
            "local_train_padding_avoidance_fraction": (
                avoided / global_padded if global_padded else 0.0
            ),
            "local_train_global_attention_proxy": global_attention,
            "local_train_microbatch_attention_proxy": microbatch_attention,
            "local_train_attention_proxy_avoided": attention_avoided,
            "local_train_attention_proxy_avoidance_fraction": (
                attention_avoided / global_attention if global_attention else 0.0
            ),
            "local_train_sequence_length_min": min(lengths),
            "local_train_sequence_length_p50": p50,
            "local_train_sequence_length_p95": p95,
            "local_train_sequence_length_max": max_len,
            "local_train_avg_microbatch_examples": (
                len(lengths) / len(ranges) if ranges else 0.0
            ),
        }
    )


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
