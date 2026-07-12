"""Imperative executor for pure local training-step plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from retrain.backends.local import batch as local_batch
from retrain.backends.local import sft as local_sft
from retrain.backends.local.steps import planning
from retrain.training.batch_digest import (
    local_rl_effective_rows_sha256,
    local_sft_effective_rows_sha256,
)

if TYPE_CHECKING:
    from retrain.backends.local.train import LocalTrainHelper


def crop_supervised_context(
    helper: "LocalTrainHelper",
    all_tokens,
    all_logprobs=None,
    all_advantages=None,
    echo_advantages=None,
):
    """Compatibility effect for the protected helper method."""

    context_tokens = int(getattr(helper, "train_supervised_context_tokens", 0))
    result = local_sft.crop_supervised_context(
        all_tokens,
        all_logprobs=all_logprobs,
        all_advantages=all_advantages,
        echo_advantages=echo_advantages,
        context_tokens=context_tokens,
        enabled=(
            context_tokens > 0
            and getattr(helper, "train_selective_suffix_logits", False)
        ),
    )
    helper._last_context_crop_metrics = result.metrics
    return result.tokens, result.logprobs, result.advantages, result.echo_advantages


def clear_effective_optimizer_rows(helper: "LocalTrainHelper") -> None:
    """Prevent a failed or empty step from emitting a previous row digest."""

    helper._last_effective_optimizer_rows_sha256 = ""


def record_effective_rl_rows(
    helper: "LocalTrainHelper",
    all_tokens,
    all_logprobs,
    all_advantages,
    *,
    echo_advantages=None,
    echo_full_observation_counts=None,
    echo_rollout_denominator=None,
) -> None:
    """Compatibility effect for the protected helper method."""

    helper._last_effective_optimizer_rows_sha256 = local_rl_effective_rows_sha256(
        all_tokens,
        all_logprobs,
        all_advantages,
        echo_observation_masks=echo_advantages,
        echo_full_observation_counts=echo_full_observation_counts,
        echo_rollout_denominator=echo_rollout_denominator,
    )


def record_effective_sft_rows(
    helper: "LocalTrainHelper", all_tokens, all_advantages
) -> None:
    """Compatibility effect for the protected helper method."""

    helper._last_effective_optimizer_rows_sha256 = local_sft_effective_rows_sha256(
        all_tokens, all_advantages
    )


def train_step(
    helper: "LocalTrainHelper",
    all_tokens,
    all_logprobs,
    all_advantages,
    lr,
    weight_decay,
):
    helper._clear_effective_optimizer_rows()
    plan = planning.plan_rl_step(
        _config(helper),
        all_tokens,
        all_logprobs,
        all_advantages,
        learning_rate=lr,
        weight_decay=weight_decay,
    )
    if plan.execution == "empty":
        return 0.0

    _apply_plan_metadata(helper, plan)
    helper._clear_inference_prefix_cache()
    if helper.split_mode:
        _collect_pending_step(helper)
    _set_optimizer_options(helper, plan.optimizer)

    if plan.execution == "sequence":
        if helper.split_mode:
            helper._train_future = helper._train_executor.submit(
                helper._do_train_sequence_impl,
                plan.rows.tokens,
                plan.rows.logprobs,
                plan.rows.advantages,
            )
            return helper._pending_loss
        return helper._do_train_sequence_impl(
            plan.rows.tokens,
            plan.rows.logprobs,
            plan.rows.advantages,
        )

    batch = local_batch.policy(
        plan.rows.tokens,
        plan.rows.logprobs,
        plan.rows.advantages,
        device=helper.train_device,
    )
    if helper.split_mode:
        helper._train_future = helper._train_executor.submit(
            helper._do_train_impl,
            batch.input_ids,
            batch.old_logprobs,
            batch.advantages,
            batch.attention_mask,
        )
        return helper._pending_loss
    return helper._do_train_impl(
        batch.input_ids,
        batch.old_logprobs,
        batch.advantages,
        batch.attention_mask,
    )


def sft_train_step(
    helper: "LocalTrainHelper",
    all_tokens,
    all_advantages,
    lr,
    weight_decay,
):
    helper._clear_effective_optimizer_rows()
    loss_fn = planning.validate_sft_loss_fn(
        getattr(helper, "sft_loss_fn", "importance_sampling")
    )
    if loss_fn == "importance_sampling":
        return helper.train_step(
            all_tokens,
            planning.zero_logprob_rows(all_tokens),
            all_advantages,
            lr,
            weight_decay,
        )

    plan = planning.plan_sft_step(
        _config(helper),
        all_tokens,
        all_advantages,
        learning_rate=lr,
        weight_decay=weight_decay,
    )
    if plan.execution == "empty":
        return 0.0

    _apply_plan_metadata(helper, plan)
    helper._clear_inference_prefix_cache()
    _collect_pending_step(helper)
    _set_optimizer_options(helper, plan.optimizer)

    if plan.execution == "sequence":
        return helper._do_sft_sequence_impl(plan.rows.tokens, plan.rows.advantages)

    batch = local_batch.sft(
        plan.rows.tokens,
        plan.rows.advantages,
        device=helper.train_device,
    )
    return helper._do_sft_impl(
        batch.input_ids,
        batch.advantages,
        batch.attention_mask,
    )


def train_step_with_echo_masks(
    helper: "LocalTrainHelper",
    all_tokens,
    all_logprobs,
    all_advantages,
    echo_advantages,
    echo_full_observation_counts,
    echo_loss_fn,
    lr,
    weight_decay,
    echo_rollout_denominator=0,
):
    helper._clear_effective_optimizer_rows()
    if not echo_advantages:
        return helper.train_step(
            all_tokens,
            all_logprobs,
            all_advantages,
            lr,
            weight_decay,
        ), 0.0

    plan = planning.plan_echo_step(
        _config(helper),
        all_tokens,
        all_logprobs,
        all_advantages,
        echo_advantages,
        echo_full_observation_counts,
        echo_loss_fn=echo_loss_fn,
        echo_rollout_denominator=echo_rollout_denominator,
        learning_rate=lr,
        weight_decay=weight_decay,
    )
    _apply_plan_metadata(helper, plan)
    helper._clear_inference_prefix_cache()
    _collect_pending_step(helper)
    _set_optimizer_options(helper, plan.optimizer)

    if plan.execution == "sequence":
        return helper._do_hybrid_mask_sequence_impl(
            plan.rows.tokens,
            plan.rows.logprobs,
            plan.rows.advantages,
            plan.rows.echo_advantages,
            plan.rows.full_observation_counts,
            echo_loss_fn,
            plan.rows.rollout_denominator,
        )

    batch = local_batch.echo(
        plan.rows.tokens,
        plan.rows.logprobs,
        plan.rows.advantages,
        plan.rows.echo_advantages,
        plan.rows.full_observation_counts,
        device=helper.train_device,
    )
    return helper._do_hybrid_mask_impl(
        batch.input_ids,
        batch.old_logprobs,
        batch.advantages,
        batch.attention_mask,
        batch.echo_advantages,
        batch.echo_counts,
        echo_loss_fn,
        plan.rows.rollout_denominator,
    )


def _config(helper: "LocalTrainHelper") -> planning.StepConfig:
    return planning.config_from_values(
        train_microbatch_size=helper.train_microbatch_size,
        train_sft_microbatch_token_budget=getattr(
            helper, "train_sft_microbatch_token_budget", 0
        ),
        train_supervised_context_tokens=getattr(
            helper, "train_supervised_context_tokens", 0
        ),
        train_selective_suffix_logits=getattr(
            helper, "train_selective_suffix_logits", False
        ),
        split_mode=helper.split_mode,
    )


def _apply_plan_metadata(helper: "LocalTrainHelper", plan) -> None:
    helper._last_context_crop_metrics = dict(plan.context_crop_metrics)
    helper._last_effective_optimizer_rows_sha256 = plan.effective_rows_sha256


def _set_optimizer_options(
    helper: "LocalTrainHelper", options: planning.OptimizerOptions
) -> None:
    for param_group in helper.optimizer.param_groups:
        param_group["lr"] = options.learning_rate
        param_group["weight_decay"] = options.weight_decay


def _collect_pending_step(helper: "LocalTrainHelper") -> None:
    if helper._train_future is not None:
        helper._pending_loss = helper._train_future.result()
        helper._train_future = None
