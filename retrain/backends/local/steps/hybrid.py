"""Hybrid step: RL policy loss + ECHO cross-entropy in one optimizer update.

Both losses share a single forward pass over the same rollout rows — the
policy loss covers model-sampled tokens, the ECHO term covers observation
tokens — so world-model supervision comes at no extra forward cost.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from retrain.backends.local.memory import empty_cuda_cache_if_requested
from retrain.backends.local.steps import shared
from retrain.backends.torch import reset_cuda_peak, timer_start, timer_stop
from retrain.training.loss import compute_policy_loss

if TYPE_CHECKING:
    from retrain.backends.local.train import LocalTrainHelper


def run(
    helper: "LocalTrainHelper",
    input_ids,
    old_logprobs,
    advantages,
    attention_mask,
    echo_advantages,
    echo_full_observation_counts,
    echo_loss_fn,
) -> tuple[float, float]:
    """Execute RL + ECHO on the same rollout rows in one optimizer step."""
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
        echo_loss_sum = 0.0

        for start in range(0, batch_size, microbatch_size):
            stop = min(start + microbatch_size, batch_size)
            microbatches += 1
            mb_input_ids = input_ids[start:stop]
            mb_old_logprobs = old_logprobs[start:stop]
            mb_advantages = advantages[start:stop]
            mb_attention_mask = attention_mask[start:stop]
            mb_echo_advantages = echo_advantages[start:stop]
            mb_echo_counts = echo_full_observation_counts[start:stop]

            timer = timer_start(helper.train_device)
            with shared.saved_tensors(helper):
                with helper._autocast_context():
                    old_lp = mb_old_logprobs[:, 1:]
                    adv = mb_advantages[:, 1:]
                    mask = mb_attention_mask[:, 1:]
                    echo_weights = torch.clamp(
                        mb_echo_advantages[:, 1:],
                        min=0.0,
                    )
                    echo_weights = echo_weights * mask.float()
                    loss_mask = mask
                    target_mask = None
                    if getattr(helper, "train_selective_suffix_logits", False):
                        target_mask = (mask > 0) & (
                            (adv != 0) | (echo_weights > 0.0)
                        )
                        loss_mask = ((mask > 0) & (adv != 0)).to(mask.dtype)
                    token_logprobs = helper._shifted_token_logprobs(
                        mb_input_ids,
                        mb_attention_mask,
                        target_mask=target_mask,
                    )
                    masked_loss, clip_frac, cov_frac, abs_kl = (
                        compute_policy_loss(
                            old_lp,
                            token_logprobs,
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
                    )
                    token_count = loss_mask.sum().clamp(min=1)
                    token_count_value = float(token_count.item())
                    scaled_loss = masked_loss * (token_count / total_tokens)

                    if echo_loss_fn != "cross_entropy":
                        raise ValueError(
                            "echo_loss_fn must be 'cross_entropy' for "
                            "paper-faithful ECHO."
                        )
                    echo_selected = (echo_weights > 0.0).float()
                    if echo_selected.sum() > 0:
                        denom = mb_echo_counts.float().clamp(min=1e-3).unsqueeze(1)
                        echo_loss = (
                            (-token_logprobs * echo_weights) / denom
                        ).sum()
                    else:
                        echo_loss = token_logprobs.sum() * 0.0

                    scaled_loss = scaled_loss + echo_loss / max(batch_size, 1)
                forward_s += timer_stop(timer)

                timer = timer_start(helper.train_device)
                helper.scaler.scale(scaled_loss).backward()
                backward_s += timer_stop(timer)
            loss_sum += float(masked_loss.detach().item()) * token_count_value
            clip_count += clip_frac * token_count_value
            cov_count += cov_frac * token_count_value
            abs_kl_sum += abs_kl * token_count_value
            echo_loss_sum += float(echo_loss.detach().item())

        optimizer_s += shared.apply_optimizer(helper)

        helper._clip_fraction = clip_count / total_tokens_value
        helper._policy_cov_fraction = cov_count / total_tokens_value
        helper._policy_abs_kl = abs_kl_sum / total_tokens_value
        loss_val = loss_sum / total_tokens_value
        echo_loss_val = echo_loss_sum / max(batch_size, 1)

        shared.snapshot_and_record(
            helper,
            kind="hybrid_echo",
            wall_start_s=wall_start,
            forward_s=forward_s,
            backward_s=backward_s,
            optimizer_s=optimizer_s,
            microbatches=microbatches,
            total_tokens=total_tokens_value,
            batch_size=batch_size,
        )

        return loss_val, echo_loss_val
    finally:
        helper.optimizer.zero_grad()
        empty_cuda_cache_if_requested(helper.cuda_empty_cache)
