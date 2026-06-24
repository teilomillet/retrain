#!/usr/bin/env python3
"""Smoke-test standalone Unsloth SFT through retrain's trainer path.

This is intentionally a capability/resource smoke, not a quality benchmark.
It creates or consumes a tiny JSONL SFT dataset, runs ``trainer = "sft"`` with
``backend = "unsloth"``, verifies the saved PEFT adapter artifacts, and emits
JSON evidence including CUDA peak-memory metrics when available.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path


def _json_default(value: object) -> str:
    return str(value)


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _bool_arg(raw: str) -> bool:
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {raw!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a standalone Unsloth SFT smoke through retrain.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--adapter-path", default="/tmp/retrain_unsloth_sft_adapter")
    parser.add_argument("--log-dir", default="/tmp/retrain_unsloth_sft_logs")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--prompt", default="Question: what is 1 + 1?\nAnswer:")
    parser.add_argument("--completion", default=" 2")
    parser.add_argument("--synthetic-prompt-tokens", type=int, default=0)
    parser.add_argument("--synthetic-token-text", default=" x")
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--train-logprob-chunk-size", type=int, default=0)
    parser.add_argument("--train-selective-suffix-logits", type=_bool_arg, default=True)
    parser.add_argument("--train-save-on-cpu", type=_bool_arg, default=False)
    parser.add_argument("--train-save-on-cpu-pin-memory", type=_bool_arg, default=True)
    parser.add_argument("--train-save-on-cpu-min-numel", type=int, default=0)
    parser.add_argument("--train-supervised-context-tokens", type=int, default=0)
    parser.add_argument("--liger-fused-linear-ce", type=_bool_arg, default=True)
    parser.add_argument("--gradient-checkpointing", type=_bool_arg, default=True)
    parser.add_argument("--sample-use-cache", type=_bool_arg, default=True)
    parser.add_argument("--qwen35-gated-delta-chunk-size", default="auto")
    parser.add_argument("--unsloth-tiled-mlp", type=_bool_arg, default=False)
    parser.add_argument("--unsloth-tiled-mlp-mode", default="")
    parser.add_argument("--offload-embedding", type=_bool_arg, default=False)
    parser.add_argument("--load-in-4bit", type=_bool_arg, default=True)
    parser.add_argument("--load-in-8bit", type=_bool_arg, default=False)
    parser.add_argument("--load-in-16bit", type=_bool_arg, default=False)
    parser.add_argument("--fast-inference", type=_bool_arg, default=False)
    parser.add_argument("--trust-remote-code", type=_bool_arg, default=False)
    parser.add_argument("--require-cuda", type=_bool_arg, default=True)
    parser.add_argument("--max-peak-reserved-mb", type=float, default=0.0)
    parser.add_argument("--compare-to", default="", help="Existing RL smoke JSON to compare peak VRAM against.")
    parser.add_argument("--cleanup", type=_bool_arg, default=False)
    parser.add_argument("--include-traceback", type=_bool_arg, default=False)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _torch_status() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - diagnostic path.
        return {
            "torch_version": "unavailable",
            "cuda_available": False,
            "torch_import_error": str(exc),
        }
    return {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _synthetic_prompt(args: argparse.Namespace) -> str:
    if args.synthetic_prompt_tokens <= 0:
        return args.prompt
    return args.synthetic_token_text * max(1, args.synthetic_prompt_tokens)


def _ensure_dataset(args: argparse.Namespace) -> Path:
    if args.data_path:
        path = Path(args.data_path)
        if not path.is_file():
            raise FileNotFoundError(f"SFT data file not found: {path}")
        return path

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "synthetic_sft.jsonl"
    rows = [
        {
            "prompt": _synthetic_prompt(args),
            "completion": args.completion,
        }
        for _ in range(max(1, args.batch_size))
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _last_jsonl(path: Path) -> dict[str, object]:
    last = ""
    if path.is_file():
        with open(path) as f:
            for line in f:
                if line.strip():
                    last = line
    if not last:
        return {}
    payload = json.loads(last)
    return payload if isinstance(payload, dict) else {}


def _metric_number(metrics: dict[str, object], *keys: str) -> float:
    for key in keys:
        raw = metrics.get(key)
        if isinstance(raw, (int, float)):
            return float(raw)
    return 0.0


def _read_compare_peak(path: str) -> float:
    if not path:
        return 0.0
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        return 0.0
    memory = payload.get("memory")
    if isinstance(memory, dict):
        raw = memory.get("cuda_peak_reserved_mb")
        if isinstance(raw, (int, float)):
            return float(raw)
    runtime_metrics = payload.get("runtime_metrics")
    if isinstance(runtime_metrics, dict):
        return _metric_number(
            runtime_metrics,
            "local_train_gpu_peak_memory_reserved_mb",
            "backend/local_train_gpu_peak_memory_reserved_mb",
        )
    return 0.0


def _check_unsloth_api() -> dict[str, object]:
    unsloth = importlib.import_module("unsloth")
    unsloth_backend = importlib.import_module("retrain.unsloth_backend")
    fast_language_model = getattr(unsloth, "FastLanguageModel")
    payload = unsloth_backend.validate_fast_language_model_api(fast_language_model)
    return {
        "unsloth_version": getattr(unsloth, "__version__", "unknown"),
        **payload,
    }


def _comparison_payload(compare_peak: float, peak_reserved: float) -> dict[str, object]:
    if compare_peak <= 0:
        return {}
    payload: dict[str, object] = {
        "baseline_peak_reserved_mb": compare_peak,
        "comparison_available": peak_reserved > 0,
    }
    if peak_reserved > 0:
        payload.update(
            {
                "sft_peak_reserved_mb": peak_reserved,
                "sft_peak_reserved_delta_mb": peak_reserved - compare_peak,
                "sft_reserved_fraction_of_baseline": peak_reserved / compare_peak,
                "sft_reserved_below_baseline": peak_reserved < compare_peak,
                "sft_reserved_not_above_baseline": peak_reserved <= compare_peak,
            }
        )
    return payload


def _adapter_checks(policy_ref: str) -> dict[str, object]:
    if not policy_ref:
        return {
            "adapter_dir": "",
            "exists": False,
            "files": [],
            "has_peft_config": False,
            "has_adapter_weights": False,
            "has_retrain_manifest": False,
        }
    adapter_dir = Path(policy_ref)
    files = sorted(path.name for path in adapter_dir.iterdir()) if adapter_dir.is_dir() else []
    has_weights = any(name in files for name in ("adapter_model.safetensors", "adapter_model.bin"))
    return {
        "adapter_dir": str(adapter_dir),
        "exists": adapter_dir.is_dir(),
        "files": files,
        "has_peft_config": "adapter_config.json" in files,
        "has_adapter_weights": has_weights,
        "has_retrain_manifest": "retrain_sft_manifest.json" in files,
    }


def _run(args: argparse.Namespace) -> dict[str, object]:
    api = _check_unsloth_api()

    import torch

    from retrain.config import TrainConfig
    from retrain.registry import get_registry

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Unsloth SFT smoke, but unavailable")
    if sum(bool(v) for v in (args.load_in_4bit, args.load_in_8bit, args.load_in_16bit)) > 1:
        raise RuntimeError("set only one of --load-in-4bit, --load-in-8bit, --load-in-16bit")

    data_path = _ensure_dataset(args)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    config = TrainConfig(
        trainer="sft",
        backend="unsloth",
        devices=args.device,
        adapter_path=args.adapter_path,
        backend_options={
            "max_seq_length": args.max_seq_length,
            "load_in_4bit": args.load_in_4bit,
            "load_in_8bit": args.load_in_8bit,
            "load_in_16bit": args.load_in_16bit,
            "fast_inference": args.fast_inference,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "train_microbatch_size": 1,
            "train_logprob_chunk_size": args.train_logprob_chunk_size,
            "train_selective_suffix_logits": args.train_selective_suffix_logits,
            "train_save_on_cpu": args.train_save_on_cpu,
            "train_save_on_cpu_pin_memory": args.train_save_on_cpu_pin_memory,
            "train_save_on_cpu_min_numel": args.train_save_on_cpu_min_numel,
            "train_supervised_context_tokens": args.train_supervised_context_tokens,
            "liger_kernel": False,
            "liger_fused_linear_ce": args.liger_fused_linear_ce,
            "cuda_empty_cache": True,
            "sample_use_cache": args.sample_use_cache,
            "gradient_checkpointing": args.gradient_checkpointing,
            "qwen35_gated_delta_chunk_size": args.qwen35_gated_delta_chunk_size,
            "unsloth_tiled_mlp": args.unsloth_tiled_mlp,
            "unsloth_tiled_mlp_mode": args.unsloth_tiled_mlp_mode,
            "offload_embedding": args.offload_embedding,
            "trust_remote_code": args.trust_remote_code,
        },
        model=args.model,
        lora_rank=args.lora_rank,
        max_steps=args.steps,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_every=args.save_every,
        seed=args.seed,
        sft_data_path=str(data_path),
        sft_batch_size=args.batch_size,
        sft_max_tokens=args.max_tokens,
        sft_loss_fn="auto",
        log_dir=args.log_dir,
    )
    runner = get_registry("trainer").create("sft", config)
    result = runner.run(config)
    metrics = _last_jsonl(Path(args.log_dir) / "metrics.jsonl")
    peak_reserved = _metric_number(
        metrics,
        "backend/local_train_gpu_peak_memory_reserved_mb",
        "local_train_gpu_peak_memory_reserved_mb",
    )
    compare_peak = _read_compare_peak(args.compare_to)
    peak_limit_ok = args.max_peak_reserved_mb <= 0 or peak_reserved <= args.max_peak_reserved_mb
    comparison = _comparison_payload(compare_peak, peak_reserved)

    checks = _adapter_checks(result.policy_ref)
    artifacts_ok = (
        bool(checks["exists"])
        and bool(checks["has_adapter_weights"])
        and bool(checks["has_retrain_manifest"])
    )
    return {
        "ok": bool(result.ok and artifacts_ok and peak_limit_ok),
        "result": result.to_dict(),
        "model": args.model,
        "data_path": str(data_path),
        "metrics": metrics,
        "adapter_checks": checks,
        "peak_limit_ok": peak_limit_ok,
        "max_peak_reserved_mb": args.max_peak_reserved_mb,
        "comparison": comparison,
        "torch": _torch_status(),
        "unsloth": api,
    }


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    payload: dict[str, object]
    try:
        payload = _run(args)
    except Exception as exc:  # noqa: BLE001 - this script reports evidence JSON.
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        payload.update(_torch_status())
        if args.include_traceback:
            payload["traceback"] = traceback.format_exc()
        status = 1
    else:
        status = 0 if payload.get("ok") is True else 1
    finally:
        if args.cleanup:
            shutil.rmtree(args.adapter_path, ignore_errors=True)
            shutil.rmtree(args.log_dir, ignore_errors=True)
    payload["wall_s"] = time.perf_counter() - started
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
            + "\n"
        )
    _print_json(payload)
    return status


if __name__ == "__main__":
    sys.exit(main())
