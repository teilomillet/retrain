"""SFT / ECHO cross-entropy optimizer steps.

Two shapes: ``run_padded`` consumes one pre-padded tensor batch;
``run_sequence`` pads per microbatch so a single long row does not force
the whole batch to its width (long-context SFT).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from retrain.backends.local import sft as local_sft
from retrain.backends.local.memory import empty_cuda_cache_if_requested
from retrain.backends.local.steps import shared
from retrain.backends.torch import reset_cuda_peak, timer_start, timer_stop

if TYPE_CHECKING:
    from retrain.backends.local.train import LocalTrainHelper


def compute_loss(helper: "LocalTrainHelper", input_ids, advantages, attention_mask):
    """Compute weighted next-token cross-entropy for SFT/ECHO datums."""
    with helper._autocast_context():
        weights = torch.clamp(advantages[:, 1:], min=0.0)
        weights = weights * attention_mask[:, 1:].float()
        sft_target_mask = weights > 0
        target_mask = None
        if getattr(helper, "train_selective_suffix_logits", False):
            target_mask = sft_target_mask

        fused_loss = helper._maybe_compute_unsloth_fused_sft_loss(
            input_ids,
            attention_mask,
            weights,
            sft_target_mask,
            sft_target_mask.float().sum().clamp(min=1),
        )
        if fused_loss is not None:
            token_count = sft_target_mask.float().sum().clamp(min=1)
            return fused_loss, token_count

        new_logprobs = helper._shifted_token_logprobs(
            input_ids,
            attention_mask,
            target_mask=target_mask,
        )

        token_mask = (weights > 0).float()
        token_count = token_mask.sum().clamp(min=1)
        loss = (-new_logprobs * weights).sum() / token_count

    return loss, token_count


def backward_microbatch(
    helper: "LocalTrainHelper",
    input_ids,
    advantages,
    attention_mask,
    total_tokens,
):
    """Backward one SFT microbatch with the optional fused-CE retry policy."""
    retried_fused_ce_runtime_failure = False
    forward_s_total = 0.0
    backward_s_total = 0.0
    while True:
        fused_ce_attempts_before = int(getattr(helper, "_unsloth_fused_ce_attempts", 0))
        loss_counts = getattr(helper, "_loss_path_counts", {})
        fused_ce_batches_before = int(loss_counts.get("unsloth_fused_ce", 0))
        timer = timer_start(helper.train_device)
        try:
            with shared.saved_tensors(helper):
                masked_loss, token_count = helper._compute_sft_loss(
                    input_ids,
                    advantages,
                    attention_mask,
                )
                forward_s_total += timer_stop(timer)
                token_count_value = float(token_count.item())
                scaled_loss = masked_loss * (token_count / total_tokens)
                timer = timer_start(helper.train_device)
                helper.scaler.scale(scaled_loss).backward()
                backward_s_total += timer_stop(timer)
        except Exception as exc:
            fused_ce_attempted = (
                int(getattr(helper, "_unsloth_fused_ce_attempts", 0))
                > fused_ce_attempts_before
            )
            if (
                helper._is_cuda_oom_exception(exc)
                or not fused_ce_attempted
                or getattr(helper, "train_unsloth_fused_ce", "off") != "auto"
                or retried_fused_ce_runtime_failure
            ):
                raise
            loss_counts = getattr(helper, "_loss_path_counts", {})
            if int(loss_counts.get("unsloth_fused_ce", 0)) > fused_ce_batches_before:
                loss_counts["unsloth_fused_ce"] = fused_ce_batches_before
            helper._disable_unsloth_fused_ce_after_runtime_failure(exc)
            helper.optimizer.zero_grad()
            retried_fused_ce_runtime_failure = True
            continue
        return masked_loss, token_count_value, forward_s_total, backward_s_total


def run_padded(
    helper: "LocalTrainHelper", input_ids, advantages, attention_mask
) -> float:
    """Execute one weighted cross-entropy SFT/ECHO update synchronously."""
    helper.train_model.train()
    helper.optimizer.zero_grad()
    wall_start = time.perf_counter()
    reset_cuda_peak(helper.train_device)
    forward_s = 0.0
    backward_s = 0.0
    optimizer_s = 0.0
    microbatches = 0
    microbatch_ranges: list[tuple[int, int]] = []

    try:
        batch_size = int(input_ids.shape[0])
        microbatch_size = helper.train_microbatch_size or batch_size
        weights = torch.clamp(advantages[:, 1:], min=0.0)
        total_tokens = (weights * attention_mask[:, 1:].float() > 0).sum().clamp(min=1)
        total_tokens_value = float(total_tokens.item())
        loss_sum = 0.0

        for start in range(0, batch_size, microbatch_size):
            stop = min(start + microbatch_size, batch_size)
            microbatches += 1
            microbatch_ranges.append((start, stop))
            masked_loss, token_count_value, mb_forward_s, mb_backward_s = (
                backward_microbatch(
                    helper,
                    input_ids[start:stop],
                    advantages[start:stop],
                    attention_mask[start:stop],
                    total_tokens,
                )
            )
            forward_s += mb_forward_s
            backward_s += mb_backward_s
            loss_sum += float(masked_loss.detach().item()) * token_count_value

        optimizer_s += shared.apply_optimizer(helper)
        loss_val = loss_sum / total_tokens_value

        shared.snapshot_and_record(
            helper,
            kind="sft",
            wall_start_s=wall_start,
            forward_s=forward_s,
            backward_s=backward_s,
            optimizer_s=optimizer_s,
            microbatches=microbatches,
            total_tokens=total_tokens_value,
            batch_size=batch_size,
        )
        lengths = [int(value) for value in attention_mask.sum(dim=1).tolist()]
        shared.record_sequence_padding(helper, lengths, microbatch_ranges)

        return loss_val
    finally:
        helper.optimizer.zero_grad()
        empty_cuda_cache_if_requested(helper.cuda_empty_cache)


def run_sequence(helper: "LocalTrainHelper", all_tokens, all_advantages) -> float:
    """Execute one SFT update while padding only each microbatch.

    Preserves the logical batch and optimizer-step semantics of
    ``run_padded`` but avoids allocating ``[batch, global_max_len]`` on the
    training device before microbatching. It targets long-context SFT rows
    where a single outlier otherwise forces every row to the same width.
    """
    helper.train_model.train()
    helper.optimizer.zero_grad()
    wall_start = time.perf_counter()
    reset_cuda_peak(helper.train_device)
    forward_s = 0.0
    backward_s = 0.0
    optimizer_s = 0.0
    microbatches = 0
    batch_size = len(all_tokens)
    microbatch_size = helper.train_microbatch_size or batch_size
    token_budget = getattr(helper, "train_sft_microbatch_token_budget", 0)
    lengths = [len(row) for row in all_tokens]
    microbatch_ranges = local_sft.microbatch_ranges(
        lengths,
        microbatch_size,
        token_budget,
    )

    try:
        total_tokens_value = float(
            sum(1 for row in all_advantages for value in row[1:] if value > 0.0) or 1
        )
        loss_sum = 0.0

        for start, stop in microbatch_ranges:
            microbatches += 1
            input_ids, advantages, attention_mask = local_sft.pad_microbatch(
                all_tokens,
                all_advantages,
                start,
                stop,
                device=helper.train_device,
            )
            masked_loss, token_count_value, mb_forward_s, mb_backward_s = (
                backward_microbatch(
                    helper,
                    input_ids,
                    advantages,
                    attention_mask,
                    total_tokens_value,
                )
            )
            forward_s += mb_forward_s
            backward_s += mb_backward_s
            loss_sum += float(masked_loss.detach().item()) * token_count_value

        optimizer_s += shared.apply_optimizer(helper)
        loss_val = loss_sum / total_tokens_value

        shared.snapshot_and_record(
            helper,
            kind="sft",
            wall_start_s=wall_start,
            forward_s=forward_s,
            backward_s=backward_s,
            optimizer_s=optimizer_s,
            microbatches=microbatches,
            total_tokens=total_tokens_value,
            batch_size=batch_size,
        )
        shared.record_sequence_padding(helper, lengths, microbatch_ranges)
        metrics = getattr(helper, "_last_train_metrics", {})
        if isinstance(metrics, dict):
            metrics["local_train_sft_microbatch_token_budget"] = int(token_budget)
            metrics["local_train_sft_avg_microbatch_examples"] = metrics.get(
                "local_train_avg_microbatch_examples",
                0.0,
            )
        return loss_val
    finally:
        helper.optimizer.zero_grad()
        empty_cuda_cache_if_requested(helper.cuda_empty_cache)
