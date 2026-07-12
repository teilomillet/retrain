"""Pure planning functions for local RL, SFT, and hybrid training steps.

Nothing in this module owns a model, optimizer, engine, CUDA resource, future,
or mutable backend helper. Given immutable configuration and row values, the
planners return immutable descriptions of the work the imperative dispatcher
must perform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from retrain.backends.local import sft as local_sft
from retrain.training.batch_digest import (
    local_rl_effective_rows_sha256,
    local_sft_effective_rows_sha256,
)

IntRows: TypeAlias = tuple[tuple[int, ...], ...]
FloatRows: TypeAlias = tuple[tuple[float, ...], ...]
ExecutionMode: TypeAlias = Literal["empty", "padded", "sequence"]
MetricItems: TypeAlias = tuple[tuple[str, float | int], ...]


@dataclass(frozen=True)
class StepConfig:
    train_microbatch_size: int
    train_sft_microbatch_token_budget: int
    train_supervised_context_tokens: int
    train_selective_suffix_logits: bool
    split_mode: bool


@dataclass(frozen=True)
class OptimizerOptions:
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class RlRows:
    tokens: IntRows
    logprobs: FloatRows
    advantages: FloatRows


@dataclass(frozen=True)
class SftRows:
    tokens: IntRows
    advantages: FloatRows


@dataclass(frozen=True)
class EchoRows:
    tokens: IntRows
    logprobs: FloatRows
    advantages: FloatRows
    echo_advantages: FloatRows
    full_observation_counts: tuple[int, ...]
    rollout_denominator: int


@dataclass(frozen=True)
class RlStepPlan:
    execution: ExecutionMode
    rows: RlRows
    optimizer: OptimizerOptions
    effective_rows_sha256: str
    context_crop_metrics: MetricItems
    snapshot_for_async: bool


@dataclass(frozen=True)
class SftStepPlan:
    execution: ExecutionMode
    rows: SftRows
    optimizer: OptimizerOptions
    effective_rows_sha256: str
    context_crop_metrics: MetricItems


@dataclass(frozen=True)
class EchoStepPlan:
    execution: ExecutionMode
    rows: EchoRows
    optimizer: OptimizerOptions
    effective_rows_sha256: str
    context_crop_metrics: MetricItems


def config_from_values(
    *,
    train_microbatch_size: int,
    train_sft_microbatch_token_budget: int = 0,
    train_supervised_context_tokens: int = 0,
    train_selective_suffix_logits: bool = False,
    split_mode: bool = False,
) -> StepConfig:
    return StepConfig(
        train_microbatch_size=max(0, int(train_microbatch_size)),
        train_sft_microbatch_token_budget=max(
            0, int(train_sft_microbatch_token_budget)
        ),
        train_supervised_context_tokens=max(0, int(train_supervised_context_tokens)),
        train_selective_suffix_logits=bool(train_selective_suffix_logits),
        split_mode=bool(split_mode),
    )


def plan_rl_step(
    config: StepConfig,
    all_tokens,
    all_logprobs,
    all_advantages,
    *,
    learning_rate: float,
    weight_decay: float,
) -> RlStepPlan:
    optimizer = OptimizerOptions(float(learning_rate), float(weight_decay))
    if not all_tokens:
        return RlStepPlan(
            execution="empty",
            rows=RlRows((), (), ()),
            optimizer=optimizer,
            effective_rows_sha256="",
            context_crop_metrics=(),
            snapshot_for_async=False,
        )

    crop = _crop(
        config,
        all_tokens,
        all_logprobs=all_logprobs,
        all_advantages=all_advantages,
    )
    rows = RlRows(
        _int_rows(crop.tokens),
        _float_rows(crop.logprobs),
        _float_rows(crop.advantages),
    )
    execution = _execution_mode(config.train_microbatch_size, len(rows.tokens))
    return RlStepPlan(
        execution=execution,
        rows=rows,
        optimizer=optimizer,
        effective_rows_sha256=local_rl_effective_rows_sha256(
            rows.tokens,
            rows.logprobs,
            rows.advantages,
        ),
        context_crop_metrics=_metric_items(crop.metrics),
        snapshot_for_async=config.split_mode and execution == "sequence",
    )


def plan_sft_step(
    config: StepConfig,
    all_tokens,
    all_advantages,
    *,
    learning_rate: float,
    weight_decay: float,
) -> SftStepPlan:
    optimizer = OptimizerOptions(float(learning_rate), float(weight_decay))
    if not all_tokens:
        return SftStepPlan(
            execution="empty",
            rows=SftRows((), ()),
            optimizer=optimizer,
            effective_rows_sha256="",
            context_crop_metrics=(),
        )

    crop = _crop(config, all_tokens, all_advantages=all_advantages)
    rows = SftRows(_int_rows(crop.tokens), _float_rows(crop.advantages))
    sequence = (
        0 < config.train_microbatch_size < len(rows.tokens)
        or config.train_sft_microbatch_token_budget > 0
    )
    return SftStepPlan(
        execution="sequence" if sequence else "padded",
        rows=rows,
        optimizer=optimizer,
        effective_rows_sha256=local_sft_effective_rows_sha256(
            rows.tokens, rows.advantages
        ),
        context_crop_metrics=_metric_items(crop.metrics),
    )


def plan_echo_step(
    config: StepConfig,
    all_tokens,
    all_logprobs,
    all_advantages,
    echo_advantages,
    echo_full_observation_counts,
    *,
    echo_loss_fn: str,
    echo_rollout_denominator: int,
    learning_rate: float,
    weight_decay: float,
) -> EchoStepPlan:
    if echo_loss_fn != "cross_entropy":
        raise ValueError(
            "echo_loss_fn must be 'cross_entropy' for paper-faithful ECHO."
        )
    if len(echo_advantages) != len(all_tokens):
        raise ValueError("echo_advantages must have one row per training datum.")
    if len(echo_full_observation_counts) != len(all_tokens):
        raise ValueError(
            "echo_full_observation_counts must have one value per training datum."
        )

    crop = _crop(
        config,
        all_tokens,
        all_logprobs=all_logprobs,
        all_advantages=all_advantages,
        echo_advantages=echo_advantages,
    )
    denominator = int(echo_rollout_denominator) or len(crop.tokens)
    rows = EchoRows(
        tokens=_int_rows(crop.tokens),
        logprobs=_float_rows(crop.logprobs),
        advantages=_float_rows(crop.advantages),
        echo_advantages=_float_rows(crop.echo_advantages),
        full_observation_counts=tuple(
            int(value) for value in echo_full_observation_counts
        ),
        rollout_denominator=denominator,
    )
    execution = _execution_mode(config.train_microbatch_size, len(rows.tokens))
    return EchoStepPlan(
        execution=execution,
        rows=rows,
        optimizer=OptimizerOptions(float(learning_rate), float(weight_decay)),
        effective_rows_sha256=local_rl_effective_rows_sha256(
            rows.tokens,
            rows.logprobs,
            rows.advantages,
            echo_observation_masks=rows.echo_advantages,
            echo_full_observation_counts=rows.full_observation_counts,
            echo_rollout_denominator=rows.rollout_denominator,
        ),
        context_crop_metrics=_metric_items(crop.metrics),
    )


def validate_sft_loss_fn(
    raw: object,
) -> Literal["importance_sampling", "cross_entropy"]:
    value = str(raw)
    if value not in {"importance_sampling", "cross_entropy"}:
        raise ValueError(
            "sft_loss_fn must be 'importance_sampling' or 'cross_entropy'."
        )
    return value  # type: ignore[return-value]


def zero_logprob_rows(all_tokens) -> FloatRows:
    return tuple(tuple(0.0 for _ in tokens) for tokens in all_tokens)


def _crop(config: StepConfig, all_tokens, **aligned_rows):
    return local_sft.crop_supervised_context(
        all_tokens,
        **aligned_rows,
        context_tokens=config.train_supervised_context_tokens,
        enabled=(
            config.train_supervised_context_tokens > 0
            and config.train_selective_suffix_logits
        ),
    )


def _execution_mode(microbatch_size: int, row_count: int) -> ExecutionMode:
    if row_count == 0:
        return "empty"
    if 0 < microbatch_size < row_count:
        return "sequence"
    return "padded"


def _int_rows(rows) -> IntRows:
    return tuple(tuple(int(value) for value in row) for row in rows)


def _float_rows(rows) -> FloatRows:
    return tuple(tuple(float(value) for value in row) for row in rows)


def _metric_items(metrics: dict[str, float | int]) -> MetricItems:
    return tuple(sorted(metrics.items()))
