#!/usr/bin/env python3
"""Smoke-test the real Unsloth retrain backend on a CUDA host.

This is intentionally small but end-to-end:
- import the installed Unsloth package and check the FastLanguageModel API
- instantiate retrain's UnslothTrainHelper
- sample one short completion through the shared PyTorch engine
- run one RL + ECHO train step through train_step_with_echo_masks
- emit JSON evidence for the run

It is not a quality benchmark. It exists to prove that the real package/model
path works before running expensive Quaero training.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType


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
        description="Run a real installed-Unsloth smoke for retrain.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--adapter-path", default="/tmp/retrain_unsloth_smoke_adapter")
    parser.add_argument("--prompt", default="Reason briefly: 1 + 1 =")
    parser.add_argument(
        "--synthetic-prompt-tokens",
        type=int,
        default=0,
        help=(
            "Use a repeated tokenizer token prompt of this length. "
            "This is for memory-capacity smoke tests, not quality evaluation."
        ),
    )
    parser.add_argument("--synthetic-token-text", default=" x")
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--sample-tokens", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
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
    parser.add_argument("--cleanup-adapter", type=_bool_arg, default=False)
    parser.add_argument("--include-traceback", type=_bool_arg, default=False)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _torch_status() -> dict[str, object]:
    try:
        torch = importlib.import_module("torch")
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


def _set_seed(torch: ModuleType, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _check_unsloth_api() -> dict[str, object]:
    unsloth = importlib.import_module("unsloth")
    unsloth_backend = importlib.import_module("retrain.unsloth_backend")
    fast_language_model = getattr(unsloth, "FastLanguageModel")
    payload = unsloth_backend.validate_fast_language_model_api(fast_language_model)
    return {
        "unsloth_version": getattr(unsloth, "__version__", "unknown"),
        **payload,
    }


def _cuda_memory(torch: ModuleType, device: str) -> dict[str, float]:
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return {}
    torch_device = torch.device(device)
    return {
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(torch_device)
        / (1024.0 * 1024.0),
        "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved(torch_device)
        / (1024.0 * 1024.0),
    }


def _build_train_rows(
    prompt_ids: list[int],
    completion_ids: list[int],
    completion_logprobs: list[float],
) -> tuple[list[int], list[float], list[float], list[float]]:
    if not completion_ids:
        raise RuntimeError("sample produced no completion tokens; cannot train smoke row")
    tokens = [int(token_id) for token_id in prompt_ids + completion_ids]
    logprobs = [0.0] * len(prompt_ids) + [float(lp) for lp in completion_logprobs]
    advantages = [0.0] * len(prompt_ids) + [1.0] * len(completion_ids)
    echo_advantages = [0.0] * len(tokens)
    echo_index = min(max(1, len(prompt_ids) - 1), len(tokens) - 1)
    echo_advantages[echo_index] = 0.2
    return tokens, logprobs, advantages, echo_advantages


def _build_prompt_ids(tokenizer, args: argparse.Namespace) -> list[int]:
    if args.synthetic_prompt_tokens > 0:
        piece_ids = tokenizer.encode(args.synthetic_token_text, add_special_tokens=False)
        if not piece_ids:
            fallback = tokenizer.eos_token_id
            piece_ids = [int(fallback) if fallback is not None else 0]
        repeats = (args.synthetic_prompt_tokens + len(piece_ids) - 1) // len(piece_ids)
        return [int(token_id) for token_id in (piece_ids * repeats)[: args.synthetic_prompt_tokens]]

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt_ids:
        fallback = tokenizer.eos_token_id
        prompt_ids = [int(fallback) if fallback is not None else 0]
    return [int(token_id) for token_id in prompt_ids]


def _run(args: argparse.Namespace) -> dict[str, object]:
    stage = "import"
    helper = None
    try:
        torch = importlib.import_module("torch")
        unsloth_backend = importlib.import_module("retrain.unsloth_backend")
        auto_tokenizer = importlib.import_module("transformers").AutoTokenizer
        unsloth_train_helper = unsloth_backend.UnslothTrainHelper

        stage = "seed_and_validate"
        _set_seed(torch, args.seed)
        if args.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this Unsloth smoke, but unavailable")
        if sum(bool(v) for v in (args.load_in_4bit, args.load_in_8bit, args.load_in_16bit)) > 1:
            raise RuntimeError("set only one of --load-in-4bit, --load-in-8bit, --load-in-16bit")

        stage = "unsloth_api_check"
        api = _check_unsloth_api()
        stage = "tokenizer_load"
        tokenizer = auto_tokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
        )
        prompt_ids = _build_prompt_ids(tokenizer, args)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        stage = "model_load"
        helper = unsloth_train_helper(
            args.model,
            args.adapter_path,
            args.device,
            lora_rank=args.lora_rank,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            load_in_16bit=args.load_in_16bit,
            fast_inference=args.fast_inference,
            gpu_memory_utilization=args.gpu_memory_utilization,
            train_microbatch_size=1,
            train_logprob_chunk_size=args.train_logprob_chunk_size,
            train_selective_suffix_logits=args.train_selective_suffix_logits,
            train_save_on_cpu=args.train_save_on_cpu,
            train_save_on_cpu_pin_memory=args.train_save_on_cpu_pin_memory,
            train_save_on_cpu_min_numel=args.train_save_on_cpu_min_numel,
            train_supervised_context_tokens=args.train_supervised_context_tokens,
            liger_kernel=False,
            liger_fused_linear_ce=args.liger_fused_linear_ce,
            cuda_empty_cache=True,
            sample_use_cache=args.sample_use_cache,
            gradient_checkpointing=args.gradient_checkpointing,
            trust_remote_code=args.trust_remote_code,
            offload_embedding=args.offload_embedding,
            unsloth_tiled_mlp=args.unsloth_tiled_mlp,
            unsloth_tiled_mlp_mode=args.unsloth_tiled_mlp_mode,
            qwen35_gated_delta_chunk_size=args.qwen35_gated_delta_chunk_size,
        )
    except Exception as exc:
        setattr(exc, "_smoke_stage", stage)
        if helper is not None:
            try:
                setattr(exc, "_smoke_runtime_metrics", helper.runtime_metrics())
            except Exception:  # noqa: BLE001 - diagnostic best effort.
                pass
        raise

    try:
        stage = "sample"
        sample_start = time.perf_counter()
        samples = helper.sample(
            [prompt_ids],
            num_samples=1,
            max_tokens=args.sample_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        sample_s = time.perf_counter() - sample_start
        completion_ids, completion_logprobs = samples[0][0]
        tokens, logprobs, advantages, echo_advantages = _build_train_rows(
            prompt_ids,
            completion_ids,
            completion_logprobs,
        )

        stage = "train"
        train_start = time.perf_counter()
        rl_loss, echo_loss = helper.train_step_with_echo_masks(
            all_tokens=[tokens],
            all_logprobs=[logprobs],
            all_advantages=[advantages],
            echo_advantages=[echo_advantages],
            echo_full_observation_counts=[1],
            echo_loss_fn="cross_entropy",
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        train_s = time.perf_counter() - train_start
        metrics = helper.runtime_metrics()
        if not math.isfinite(float(rl_loss)) or not math.isfinite(float(echo_loss)):
            raise RuntimeError(f"non-finite losses: rl={rl_loss}, echo={echo_loss}")

        return {
            "ok": True,
            "model": args.model,
            "device": helper.train_device,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(torch.device(helper.train_device))
                if helper.train_device.startswith("cuda") and torch.cuda.is_available()
                else ""
            ),
            "unsloth": api,
            "prompt_tokens": len(prompt_ids),
            "synthetic_prompt_tokens": int(args.synthetic_prompt_tokens),
            "generated_tokens": len(completion_ids),
            "total_train_tokens": len(tokens),
            "sample_s": sample_s,
            "train_s": train_s,
            "rl_loss": float(rl_loss),
            "echo_loss": float(echo_loss),
            "runtime_metrics": metrics,
            "memory": _cuda_memory(torch, helper.train_device),
        }
    except Exception as exc:
        setattr(exc, "_smoke_stage", stage)
        if helper is not None:
            try:
                setattr(exc, "_smoke_runtime_metrics", helper.runtime_metrics())
            except Exception:  # noqa: BLE001 - diagnostic best effort.
                pass
        raise
    finally:
        helper.shutdown()


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    try:
        payload = _run(args)
    except Exception as exc:  # noqa: BLE001 - this script reports evidence JSON.
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "stage": getattr(exc, "_smoke_stage", "unknown"),
        }
        runtime_metrics = getattr(exc, "_smoke_runtime_metrics", None)
        if runtime_metrics is not None:
            payload["runtime_metrics"] = runtime_metrics
        if args.include_traceback:
            payload["traceback"] = traceback.format_exc()
        payload.update(_torch_status())
        status = 1
    else:
        status = 0
    finally:
        if args.cleanup_adapter:
            shutil.rmtree(args.adapter_path, ignore_errors=True)
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
