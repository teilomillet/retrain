#!/usr/bin/env python3
"""Probe local PyTorch train-step capacity by sequence length.

This intentionally exercises the current LocalTrainHelper training loss path.
It is meant to find the length at which the implementation OOMs or becomes
impractical, not to produce a quality training update.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoTokenizer

from retrain.backend_definitions import get_builtin_backend_definitions
from retrain.config import TrainConfig, load_config


def _parse_bool(raw: str) -> bool:
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {raw!r}")


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value <= 1 for value in values):
        raise argparse.ArgumentTypeError("sequence lengths must be > 1")
    return values


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_config(base: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    options = dict(base.backend_options)
    options.update(
        {
            "cuda_empty_cache": args.cuda_empty_cache,
            "gradient_checkpointing": args.gradient_checkpointing,
            "sample_use_cache": args.sample_use_cache,
            "train_microbatch_size": args.microbatch_size,
            "train_logprob_chunk_size": args.train_logprob_chunk_size,
            "liger_kernel": args.liger_kernel,
            "liger_fused_linear_ce": args.liger_fused_linear_ce,
        }
    )
    return replace(
        base,
        backend="local",
        inference_engine="pytorch",
        inference_url="",
        backend_options=options,
        prefix_caching=args.prefix_caching,
        attention_kernel=args.attention_kernel,
        seed=args.seed,
        sft_warmup_steps=0,
        sft_data_path="",
    )


def _fixed_token_row(tokenizer, length: int) -> list[int]:
    text = (
        "Reason carefully about the schema and produce the SQL action. "
        "Keep the trajectory valid and submit only when the query is ready. "
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        fallback = tokenizer.eos_token_id
        ids = [int(fallback) if fallback is not None else 0]
    return [int(token_id) for token_id in (ids * ((length // len(ids)) + 1))[:length]]


def _cuda_peak_mb(device: str) -> dict[str, float]:
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return {}
    torch_device = torch.device(device)
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated(torch_device)
        / (1024.0 * 1024.0),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(torch_device)
        / (1024.0 * 1024.0),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe retrain local train-step sequence length capacity.",
    )
    parser.add_argument("config", help="Retrain TOML config")
    parser.add_argument("--output", default="", help="Write JSON result here")
    parser.add_argument("--sequence-lengths", type=_parse_int_list, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--train-logprob-chunk-size", type=int, default=0)
    parser.add_argument("--attention-kernel", default="default")
    parser.add_argument("--liger-kernel", type=_parse_bool, default=True)
    parser.add_argument("--liger-fused-linear-ce", type=_parse_bool, default=True)
    parser.add_argument("--active-tail-tokens", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=-1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--cuda-empty-cache", type=_parse_bool, default=True)
    parser.add_argument("--gradient-checkpointing", type=_parse_bool, default=True)
    parser.add_argument("--sample-use-cache", type=_parse_bool, default=True)
    parser.add_argument("--prefix-caching", type=_parse_bool, default=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.microbatch_size < 0:
        raise SystemExit("--microbatch-size must be >= 0")
    if args.train_logprob_chunk_size < 0:
        raise SystemExit("--train-logprob-chunk-size must be >= 0")
    if args.active_tail_tokens < 0:
        raise SystemExit("--active-tail-tokens must be >= 0")

    _set_seed(args.seed)
    base_config = load_config(args.config)
    config = _build_config(base_config, args)
    lr = config.lr if args.lr < 0 else args.lr
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    helper = get_builtin_backend_definitions()["local"].factory(config)
    results: list[dict[str, object]] = []

    try:
        for seq_len in args.sequence_lengths:
            row = _fixed_token_row(tokenizer, seq_len)
            active_tail = args.active_tail_tokens or seq_len - 1
            active_start = max(1, seq_len - active_tail)
            all_tokens = [row[:] for _ in range(args.batch_size)]
            all_logprobs = [[0.0] * seq_len for _ in range(args.batch_size)]
            all_advantages = []
            for _ in range(args.batch_size):
                advantages = [0.0] * seq_len
                for idx in range(active_start, seq_len):
                    advantages[idx] = 1.0
                all_advantages.append(advantages)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(torch.device(helper.train_device))
                torch.cuda.synchronize()
            start_s = time.perf_counter()
            record: dict[str, object] = {
                "sequence_length": seq_len,
                "batch_size": args.batch_size,
                "microbatch_size": args.microbatch_size,
                "active_tail_tokens": active_tail,
            }
            try:
                loss = helper.train_step(
                    all_tokens,
                    all_logprobs,
                    all_advantages,
                    lr,
                    args.weight_decay,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                record.update(
                    {
                        "status": "ok",
                        "loss": float(loss),
                        "wall_s": time.perf_counter() - start_s,
                        **helper.runtime_metrics(),
                        **_cuda_peak_mb(helper.train_device),
                    }
                )
            except torch.cuda.OutOfMemoryError as exc:
                record.update(
                    {
                        "status": "oom",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "wall_s": time.perf_counter() - start_s,
                        **_cuda_peak_mb(helper.train_device),
                    }
                )
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                is_oom = "out of memory" in str(exc).lower()
                record.update(
                    {
                        "status": "oom" if is_oom else "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "wall_s": time.perf_counter() - start_s,
                        **_cuda_peak_mb(helper.train_device),
                    }
                )
                if is_oom and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results.append(record)
            if record["status"] != "ok":
                break
    finally:
        helper.shutdown()

    payload = {
        "config": str(Path(args.config)),
        "model": config.model,
        "seed": args.seed,
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
            ),
        },
        "results": results,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")

    print("seq_len,status,wall_s,peak_alloc_mb,train_tokens_per_s")
    for result in results:
        print(
            f"{result['sequence_length']},"
            f"{result['status']},"
            f"{float(result.get('wall_s', 0.0)):.6f},"
            f"{float(result.get('peak_allocated_mb', 0.0)):.3f},"
            f"{float(result.get('local_train_tokens_per_s', 0.0)):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
