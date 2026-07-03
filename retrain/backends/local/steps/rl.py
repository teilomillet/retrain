"""The RL (importance-sampling policy loss) optimizer step."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from retrain.backends.local.memory import empty_cuda_cache_if_requested
from retrain.backends.local.steps import shared
from retrain.backends.torch import reset_cuda_peak, timer_start, timer_stop
from retrain.training.loss import compute_policy_loss

if TYPE_CHECKING:
    from retrain.backends.local.train import LocalTrainHelper


def compute_loss(
    helper: "LocalTrainHelper",
    input_ids,
    old_logprobs,
    advantages,
    attention_mask,
):
    """Compute masked policy loss for one already-padded microbatch."""
    with helper._autocast_context():
        old_lp = old_logprobs[:, 1:]  # [N, max_len-1]
        adv = advantages[:, 1:]       # [N, max_len-1]
        mask = attention_mask[:, 1:]   # [N, max_len-1] — exclude padding
        loss_mask = mask
        target_mask = None
        if getattr(helper, "train_selective_suffix_logits", False):
            target_mask = (mask > 0) & (adv != 0)
            loss_mask = target_mask.to(mask.dtype)

        new_logprobs = helper._shifted_token_logprobs(
            input_ids,
            attention_mask,
            target_mask=target_mask,
        )

        masked_loss, clip_frac, cov_frac, abs_kl = compute_policy_loss(
            old_lp,
            new_logprobs,
            adv,
            loss_mask,
            helper.clip_eps,
            helper.clip_eps_high,
            getattr(helper, "policy_loss_mode", "standard"),
            getattr(helper, "kl_cov_percent", 0.2),
            getattr(helper, "kl_cov_coef", 1.0),
            getattr(helper, "clip_cov_ratio", 0.0002),
            getattr(helper, "clip_cov_min", 1.0),
            getattr(helper, "clip_cov_max", 5.0),
        )
        token_count = loss_mask.sum().clamp(min=1)

    return masked_loss, token_count, clip_frac, cov_frac, abs_kl


def run(
    helper: "LocalTrainHelper",
    input_ids,
    old_logprobs,
    advantages,
    attention_mask,
) -> float:
    """Execute one RL forward/backward/step on pre-prepared tensors.

    After the optimizer step, clones LoRA params into the weight snapshot
    for safe cross-thread syncing. Runs on a background thread in split mode.

    Returns:
        Scalar loss value.
    """
    helper.train_model.train()
    helper.optimizer.zero_grad()
    wall_start = time.perf_counter()
    reset_cuda_peak(helper.train_device)
    forward_s = 0.0
    backward_s = 0.0
    optimizer_s = 0.0
    microbatches = 0

    try:
        batch_size = int(input_ids.shape[0])
        microbatch_size = helper.train_microbatch_size or batch_size
        total_tokens = shared.policy_total_tokens(helper, advantages, attention_mask)
        total_tokens_value = float(total_tokens.item())
        loss_sum = 0.0
        clip_count = 0.0
        cov_count = 0.0
        abs_kl_sum = 0.0

        for start in range(0, batch_size, microbatch_size):
            stop = min(start + microbatch_size, batch_size)
            microbatches += 1
            timer = timer_start(helper.train_device)
            with shared.saved_tensors(helper):
                masked_loss, token_count, clip_frac, cov_frac, abs_kl = (
                    compute_loss(
                        helper,
                        input_ids[start:stop],
                        old_logprobs[start:stop],
                        advantages[start:stop],
                        attention_mask[start:stop],
                    )
                )
                forward_s += timer_stop(timer)
                token_count_value = float(token_count.item())
                scaled_loss = masked_loss * (token_count / total_tokens)
                timer = timer_start(helper.train_device)
                helper.scaler.scale(scaled_loss).backward()
                backward_s += timer_stop(timer)
            loss_sum += float(masked_loss.detach().item()) * token_count_value
            clip_count += clip_frac * token_count_value
            cov_count += cov_frac * token_count_value
            abs_kl_sum += abs_kl * token_count_value

        optimizer_s += shared.apply_optimizer(helper)

        helper._clip_fraction = clip_count / total_tokens_value
        helper._policy_cov_fraction = cov_count / total_tokens_value
        helper._policy_abs_kl = abs_kl_sum / total_tokens_value
        loss_val = loss_sum / total_tokens_value

        shared.snapshot_and_record(
            helper,
            kind="rl",
            wall_start_s=wall_start,
            forward_s=forward_s,
            backward_s=backward_s,
            optimizer_s=optimizer_s,
            microbatches=microbatches,
            total_tokens=total_tokens_value,
            batch_size=batch_size,
        )

        return loss_val
    finally:
        helper.optimizer.zero_grad()
        empty_cuda_cache_if_requested(helper.cuda_empty_cache)
