"""Supervised warmup phase for the main RL trainer."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from retrain.backends import TrainHelper, run_sft_train_step
from retrain.config import TrainConfig
from retrain.io.log import JsonlLogger
from retrain.process.metrics import max_rss_mb
from retrain.training.log import WandbRunLike
from retrain.training.recoverability import checkpoint_recoverability_wandb_metrics
from retrain.training.sft_data import (
    SftExample,
    load_sft_dataset,
    verify_sft_data_contract,
)
from retrain.training.sft_schedule import (
    build_sft_example_order,
    build_sft_resume_schedule_contract,
    build_sft_schedule_metrics,
    select_sft_batch_indices,
    verify_sft_resume_schedule_contract,
)
from retrain.training.sft_telemetry import (
    build_sft_step_metrics,
    build_warmup_sft_wandb_metrics,
)
from retrain.training.sft_tokenization import (
    SftTokenizedExample,
    build_sft_tokenized_batch,
    tokenize_sft_dataset,
)


@dataclass
class SftWarmupData:
    """SFT warmup dataset, loaded and tokenized once before the step loop."""

    examples: list[SftExample]
    tokenized: list[SftTokenizedExample]
    order: list[int]
    lengths: list[int]
    epoch_order_cache: dict[int, list[int]]
    epoch_order_sha256_cache: dict[int, str]
    schedule_contract: dict[str, object]


def load_sft_warmup_data(
    config: TrainConfig,
    tokenizer: object,
) -> SftWarmupData | None:
    """Load the SFT warmup dataset, or None when unconfigured or missing."""
    if config.sft_warmup_steps <= 0 or not config.sft_data_path:
        return None
    sft_path = Path(config.sft_data_path)
    if not sft_path.exists():
        if (
            config.sft_data_sha256
            or config.sft_data_rows > 0
            or config.sft_audit_path
            or config.sft_audit_sha256
            or config.sft_token_audit_path
            or config.sft_token_audit_sha256
        ):
            raise FileNotFoundError(
                f"SFT data path {sft_path} not found; cannot verify configured "
                "data/audit pins."
            )
        print(f"WARNING: SFT data path {sft_path} not found, skipping warmup")
        return None
    dataset = load_sft_dataset(sft_path)
    verify_sft_data_contract(config, dataset.provenance)
    examples = dataset.examples
    token_limit = (
        config.sft_max_tokens
        if config.sft_max_tokens > 0
        else config.max_tokens + 512
    )
    tokenized = tokenize_sft_dataset(tokenizer, examples, max_tokens=token_limit)
    lengths = [example.total_tokens for example in tokenized]
    order = build_sft_example_order(
        len(tokenized),
        config.seed,
        lengths=lengths,
        batch_order=config.sft_batch_order,
        length_bucket_size=config.sft_length_bucket_size,
    )
    batch_size = (
        config.sft_batch_size
        if config.sft_batch_size > 0
        else min(16, len(examples))
    )
    schedule_contract = build_sft_resume_schedule_contract(
        config,
        dataset.provenance,
        batch_size=batch_size,
        max_tokens=token_limit,
        example_order=order,
    )
    print(
        f"Loaded {len(examples)} SFT warmup examples from {sft_path} "
        f"(order={config.sft_batch_order}, "
        f"reshuffle_each_epoch={config.sft_reshuffle_each_epoch})"
    )
    return SftWarmupData(
        examples=examples,
        tokenized=tokenized,
        order=order,
        lengths=lengths,
        epoch_order_cache={0: order},
        epoch_order_sha256_cache={},
        schedule_contract=schedule_contract,
    )


def verify_sft_warmup_resume_schedule(
    saved_schedule: Mapping[str, object] | None,
    sft_data: SftWarmupData | None,
    *,
    start_step: int,
    warmup_steps: int,
) -> None:
    """Fail closed when a resume would re-enter an unverifiable SFT warmup."""

    saved_warmup_steps = _saved_warmup_steps(saved_schedule)
    latest_boundary = max(warmup_steps, saved_warmup_steps or 0)
    if start_step >= latest_boundary:
        return
    if saved_warmup_steps is not None and saved_warmup_steps != warmup_steps:
        raise ValueError(
            "SFT resume schedule contract mismatch; refusing to change the "
            "warmup phase boundary:\n- sft_warmup_steps: "
            f"saved {saved_warmup_steps!r}, current {warmup_steps!r}"
        )
    if sft_data is None or not sft_data.examples:
        raise ValueError(
            "Cannot resume inside SFT warmup because the configured warmup "
            "dataset is unavailable or empty. Restore the original data and "
            "audit contract, or restart training after the warmup boundary."
        )
    verify_sft_resume_schedule_contract(
        saved_schedule,
        sft_data.schedule_contract,
    )


def _saved_warmup_steps(
    saved_schedule: Mapping[str, object] | None,
) -> int | None:
    if saved_schedule is None:
        return None
    value = saved_schedule.get("sft_warmup_steps")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def run_sft_warmup_step(
    helper: TrainHelper,
    config: TrainConfig,
    sft_data: SftWarmupData,
    step: int,
    *,
    metrics_logger: JsonlLogger,
    steps_logger: JsonlLogger,
    wandb_run: WandbRunLike | None,
) -> tuple[str, str] | None:
    """Run one supervised warmup step from oracle demonstrations."""
    step_start = time.perf_counter()
    helper.checkpoint(f"step_{step}")

    batch_size = (
        config.sft_batch_size
        if config.sft_batch_size > 0
        else min(16, len(sft_data.examples))
    )
    batch_indices = select_sft_batch_indices(
        sft_data.order,
        batch_size=batch_size,
        step=step,
        seed=config.seed,
        lengths=sft_data.lengths,
        batch_order=config.sft_batch_order,
        length_bucket_size=config.sft_length_bucket_size,
        reshuffle_each_epoch=config.sft_reshuffle_each_epoch,
        epoch_order_cache=sft_data.epoch_order_cache,
    )
    batch = [sft_data.tokenized[idx] for idx in batch_indices]
    tokenized = build_sft_tokenized_batch(batch)

    effective_lr = config.sft_lr if config.sft_lr > 0 else config.lr
    print(
        f"Step {step} [SFT warmup]: {len(batch)} examples "
        f"(lr={effective_lr:.1e})...",
        flush=True,
    )
    loss = run_sft_train_step(
        helper,
        tokenized.tokens,
        tokenized.advantages,
        effective_lr,
        config.weight_decay,
    )
    elapsed = time.perf_counter() - step_start
    # Tinker's importance_sampling loss with logprobs=0 produces negative
    # values that get MORE negative as the model learns. Report both the raw
    # IS loss and flipped signal so the learning curve is intuitive.
    sft_signal = -loss
    print(
        f"Step {step} [SFT warmup] | is_loss={loss:.4f} | "
        f"sft_signal={sft_signal:.4f} | "
        f"datums={len(batch)} | time={elapsed:.1f}s",
        flush=True,
    )

    metrics: dict[str, int | float | str] = {
        **build_sft_step_metrics(
            config,
            step=step,
            loss=loss,
            datums=len(batch),
            total_tokens=tokenized.total_tokens,
            supervised_tokens=tokenized.supervised_tokens,
            batch_size=batch_size,
            example_count=len(sft_data.examples),
            elapsed=elapsed,
            lr=effective_lr,
        ),
        "sft_signal": sft_signal,
        "advantage_mode": config.advantage_mode,
    }
    metrics.update(
        build_sft_schedule_metrics(
            sft_data.order,
            batch_indices,
            batch_size=batch_size,
            step=step,
            seed=config.seed,
            lengths=sft_data.lengths,
            batch_order=config.sft_batch_order,
            length_bucket_size=config.sft_length_bucket_size,
            reshuffle_each_epoch=config.sft_reshuffle_each_epoch,
            epoch_order_cache=sft_data.epoch_order_cache,
            epoch_order_sha256_cache=sft_data.epoch_order_sha256_cache,
        )
    )
    rss_mb = max_rss_mb()
    if rss_mb is not None:
        metrics["process_max_rss_mb"] = round(rss_mb, 3)
    metrics_logger.log(metrics)
    steps_logger.log(metrics)

    if wandb_run is not None:
        wandb_metrics = build_warmup_sft_wandb_metrics(
            metrics,
            recoverability_metrics=checkpoint_recoverability_wandb_metrics(
                config,
                wandb_run,
            ),
        )
        wandb_run.log(wandb_metrics, step=step)

    if config.save_every > 0 and (step + 1) % config.save_every == 0:
        ckpt_name = f"checkpoint_step_{step + 1}"
        checkpoint_path = helper.save_adapter(config.adapter_path, ckpt_name)
        print(f"Saved checkpoint: {ckpt_name}")
        return ckpt_name, checkpoint_path
    return None
