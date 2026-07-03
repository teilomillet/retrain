"""Supervised warmup phase for the main RL trainer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from retrain.backends import TrainHelper, run_sft_train_step
from retrain.config import TrainConfig
from retrain.io.log import JsonlLogger
from retrain.process.metrics import max_rss_mb
from retrain.training.log import WandbRunLike
from retrain.training.sft import (
    SftExample,
    SftTokenizedExample,
    build_sft_example_order,
    build_sft_tokenized_batch,
    load_sft_jsonl,
    select_sft_batch_indices,
    tokenize_sft_dataset,
)


@dataclass
class SftWarmupData:
    """SFT warmup dataset, loaded and tokenized once before the step loop."""

    examples: list[SftExample]
    tokenized: list[SftTokenizedExample]
    order: list[int]


def load_sft_warmup_data(
    config: TrainConfig,
    tokenizer: object,
) -> SftWarmupData | None:
    """Load the SFT warmup dataset, or None when unconfigured or missing."""
    if config.sft_warmup_steps <= 0 or not config.sft_data_path:
        return None
    sft_path = Path(config.sft_data_path)
    if not sft_path.exists():
        print(f"WARNING: SFT data path {sft_path} not found, skipping warmup")
        return None
    examples = load_sft_jsonl(sft_path)
    token_limit = (
        config.sft_max_tokens
        if config.sft_max_tokens > 0
        else config.max_tokens + 512
    )
    tokenized = tokenize_sft_dataset(tokenizer, examples, max_tokens=token_limit)
    order = build_sft_example_order(
        len(tokenized),
        config.seed,
        lengths=[example.total_tokens for example in tokenized],
        batch_order=config.sft_batch_order,
        length_bucket_size=config.sft_length_bucket_size,
    )
    print(
        f"Loaded {len(examples)} SFT warmup examples from {sft_path} "
        f"(order={config.sft_batch_order})"
    )
    return SftWarmupData(examples=examples, tokenized=tokenized, order=order)


def run_sft_warmup_step(
    helper: TrainHelper,
    config: TrainConfig,
    sft_data: SftWarmupData,
    step: int,
    *,
    metrics_logger: JsonlLogger,
    steps_logger: JsonlLogger,
    wandb_run: WandbRunLike | None,
) -> None:
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
        "step": step,
        "loss": loss,
        "sft_signal": sft_signal,
        "phase": "sft",
        "datums": len(batch),
        "tokens": tokenized.total_tokens,
        "supervised_tokens": tokenized.supervised_tokens,
        "sft_batch_order": config.sft_batch_order,
        "sft_length_bucket_size": int(config.sft_length_bucket_size),
        "sft_unique_examples_seen": min(
            len(sft_data.examples),
            (step + 1) * batch_size,
        ),
        "sft_dataset_coverage": min(
            1.0,
            ((step + 1) * batch_size) / max(len(sft_data.examples), 1),
        ),
        "time_s": round(elapsed, 2),
        "advantage_mode": config.advantage_mode,
        "lr": effective_lr,
    }
    rss_mb = max_rss_mb()
    if rss_mb is not None:
        metrics["process_max_rss_mb"] = round(rss_mb, 3)
    metrics_logger.log(metrics)
    steps_logger.log(metrics)

    if wandb_run is not None:
        wandb_run.log(
            {
                "train/loss": loss,
                "train/sft_signal": sft_signal,
                "train/sft_warmup": 1,
                "train/step": step,
                "train/lr": effective_lr,
            },
            step=step,
        )

    if config.save_every > 0 and (step + 1) % config.save_every == 0:
        ckpt_name = f"checkpoint_step_{step + 1}"
        helper.save_adapter(config.adapter_path, ckpt_name)
        print(f"Saved checkpoint: {ckpt_name}")
