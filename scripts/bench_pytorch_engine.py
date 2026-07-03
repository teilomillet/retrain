#!/usr/bin/env python3
"""Controlled one-GPU PyTorch inference benchmark.

This isolates the local PyTorch sampling path from verifier, dbt, and trainer
step noise while preserving the single-GPU LocalTrainHelper topology.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import torch
from transformers import AutoTokenizer

from retrain.backends.catalog import get_builtin_backend_definitions
from retrain.config import TrainConfig, load_config


class _TokenizerLike(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


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
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all integer values must be > 0")
    return values


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    if any(value <= 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("top-p values must be in (0, 1]")
    return values


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fixed_prompt_ids(
    tokenizer: _TokenizerLike,
    target_len: int,
    prompt_idx: int,
) -> list[int]:
    text = (
        "You are solving a DuckDB SQL task. Read the schema, reason about the "
        "requested result, write the SQL query, and only submit once you are "
        "confident. Use exact column names and avoid unsupported syntax. "
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        fallback = tokenizer.eos_token_id
        ids = [int(fallback) if fallback is not None else 0]
    rotated = ids[prompt_idx % len(ids) :] + ids[: prompt_idx % len(ids)]
    repeated = (rotated * ((target_len // len(rotated)) + 1))[:target_len]
    return [int(token_id) for token_id in repeated]


def _build_config(base: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    options = dict(base.backend_options)
    options.update(
        {
            "cuda_empty_cache": args.cuda_empty_cache,
            "gradient_checkpointing": args.gradient_checkpointing,
            "sample_use_cache": args.sample_use_cache,
            "train_microbatch_size": args.microbatch_size,
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
    )


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _summarize(runs: list[dict[str, object]]) -> dict[str, float]:
    numeric_keys = [
        "bench_wall_s",
        "local_sample_wall_s",
        "engine_generation_wall_s",
        "engine_prompt_prefill_s",
        "engine_decode_s",
        "engine_generated_tokens",
        "engine_generation_tokens_per_s",
        "local_sample_generation_tokens_per_s",
        "local_sample_gpu_peak_memory_allocated_mb",
        "local_sample_gpu_peak_memory_reserved_mb",
    ]
    summary: dict[str, float] = {}
    for key in numeric_keys:
        values = [float(run[key]) for run in runs if key in run and run[key] is not None]
        if not values:
            continue
        summary[f"{key}_mean"] = _mean(values)
        summary[f"{key}_stdev"] = _stdev(values)
        summary[f"{key}_min"] = min(values)
        summary[f"{key}_max"] = max(values)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark retrain's local one-GPU PyTorch inference hot path.",
    )
    parser.add_argument("config", help="Retrain TOML config")
    parser.add_argument("--output", default="", help="Write JSON results here")
    parser.add_argument("--prompt-lengths", type=_parse_int_list, default=[512])
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=-1.0)
    parser.add_argument("--top-p-values", type=_parse_float_list, default=[0.95, 1.0])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--attention-kernel", default="default")
    parser.add_argument("--liger-kernel", type=_parse_bool, default=True)
    parser.add_argument("--liger-fused-linear-ce", type=_parse_bool, default=True)
    parser.add_argument("--cuda-empty-cache", type=_parse_bool, default=True)
    parser.add_argument("--gradient-checkpointing", type=_parse_bool, default=True)
    parser.add_argument("--sample-use-cache", type=_parse_bool, default=True)
    parser.add_argument("--prefix-caching", type=_parse_bool, default=True)
    parser.add_argument(
        "--clear-prefix-cache",
        type=_parse_bool,
        default=True,
        help="Clear the local PyTorch prefix cache before each measured repeat.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.num_prompts <= 0:
        raise SystemExit("--num-prompts must be > 0")
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be > 0")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be > 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be > 0")
    if args.microbatch_size < 0:
        raise SystemExit("--microbatch-size must be >= 0")

    base_config = load_config(args.config)
    config = _build_config(base_config, args)
    temperature = config.temperature if args.temperature < 0 else args.temperature

    _set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    helper = get_builtin_backend_definitions()["local"].factory(config)
    conditions: list[dict[str, object]] = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        for prompt_len in args.prompt_lengths:
            prompt_ids_list = [
                _fixed_prompt_ids(tokenizer, prompt_len, idx)
                for idx in range(args.num_prompts)
            ]
            for top_p in args.top_p_values:
                condition_runs: list[dict[str, object]] = []
                for run_idx in range(args.warmup + args.repeat):
                    _set_seed(args.seed + run_idx)
                    if args.clear_prefix_cache:
                        engine = getattr(helper, "engine", None)
                        if engine is not None and hasattr(engine, "clear_prefix_cache"):
                            engine.clear_prefix_cache()
                    _cuda_sync()
                    bench_start = time.perf_counter()
                    helper.sample(
                        prompt_ids_list,
                        args.num_samples,
                        args.max_tokens,
                        temperature,
                        top_p,
                    )
                    _cuda_sync()
                    bench_wall = time.perf_counter() - bench_start
                    if run_idx < args.warmup:
                        continue
                    metrics = dict(helper.runtime_metrics())
                    metrics.update(
                        {
                            "bench_wall_s": bench_wall,
                            "run_idx": run_idx - args.warmup + 1,
                            "prompt_len": prompt_len,
                            "num_prompts": args.num_prompts,
                            "num_samples": args.num_samples,
                            "max_tokens": args.max_tokens,
                            "temperature": float(temperature),
                            "top_p": float(top_p),
                        }
                    )
                    condition_runs.append(metrics)
                conditions.append(
                    {
                        "prompt_len": prompt_len,
                        "num_prompts": args.num_prompts,
                        "num_samples": args.num_samples,
                        "max_tokens": args.max_tokens,
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "runs": condition_runs,
                        "summary": _summarize(condition_runs),
                    }
                )
    finally:
        helper.shutdown()

    payload = {
        "started_at": started_at,
        "config": str(Path(args.config)),
        "model": config.model,
        "seed": args.seed,
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
            ),
        },
        "options": {
            "cuda_empty_cache": args.cuda_empty_cache,
            "gradient_checkpointing": args.gradient_checkpointing,
            "sample_use_cache": args.sample_use_cache,
            "prefix_caching": args.prefix_caching,
            "clear_prefix_cache": args.clear_prefix_cache,
            "microbatch_size": args.microbatch_size,
            "attention_kernel": args.attention_kernel,
            "liger_kernel": args.liger_kernel,
            "liger_fused_linear_ce": args.liger_fused_linear_ce,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "conditions": conditions,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")

    print("prompt_len,top_p,wall_s_mean,tok_s_mean,decode_s_mean,peak_alloc_mb_mean")
    for condition in conditions:
        summary = condition["summary"]
        print(
            f"{condition['prompt_len']},"
            f"{condition['top_p']:.3f},"
            f"{summary.get('bench_wall_s_mean', 0.0):.6f},"
            f"{summary.get('engine_generation_tokens_per_s_mean', 0.0):.6f},"
            f"{summary.get('engine_decode_s_mean', 0.0):.6f},"
            f"{summary.get('local_sample_gpu_peak_memory_allocated_mb_mean', 0.0):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
