"""Run a real local-backend LoRA smoke on CPU or MPS.

The probe is intentionally independent from pytest so it can be used as a
preflight command on a developer Mac. By default it creates a tiny local Llama
checkpoint with no EOS token, which makes the generated-token count assertion
stable. Pass --model to probe a real local/Hugging Face checkpoint instead.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import tempfile
import time
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from retrain.local_train_helper import LocalTrainHelper, _mps_is_available


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify retrain local PyTorch LoRA train+sample on MPS.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=("mps", "cpu"),
        help="Local backend device to probe.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional existing local path or Hugging Face model id.",
    )
    parser.add_argument(
        "--expected-new-tokens",
        type=int,
        default=5,
        help="Required exact number of generated tokens per sample.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of completions to sample for the single prompt.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=2,
        help="LoRA rank for the probe adapter.",
    )
    parser.add_argument(
        "--train-microbatch-size",
        type=int,
        default=1,
        help="Microbatch size used for the one optimizer step.",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="Optional directory for generated tiny model and adapter files.",
    )
    return parser


def _tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )


def _prepare_model(model_arg: str, work_dir: Path) -> tuple[str, str]:
    if model_arg:
        return model_arg, "user"

    model_dir = work_dir / "tiny-llama-no-eos"
    model_dir.mkdir(parents=True, exist_ok=True)
    LlamaForCausalLM(_tiny_llama_config()).save_pretrained(model_dir)
    return str(model_dir), "tiny_llama_no_eos"


def _lora_snapshot(train: LocalTrainHelper) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().float().cpu().clone()
        for name, param in train.train_model.named_parameters()
        if "lora_" in name
    }


def _any_lora_changed(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> bool:
    return any(
        (after[name] - before[name]).abs().max().item() > 0
        for name in before
    )


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    if args.expected_new_tokens <= 0:
        raise ValueError("--expected-new-tokens must be > 0")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")
    if args.device == "mps" and not _mps_is_available():
        raise RuntimeError("MPS requested, but PyTorch reports MPS unavailable")

    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="retrain-mps-lora-probe-"))

    model_name, model_source = _prepare_model(args.model, work_dir)
    adapter_dir = work_dir / "adapter"

    started = time.perf_counter()
    train = LocalTrainHelper(
        model_name,
        str(adapter_dir),
        args.device,
        lora_rank=args.lora_rank,
        engine_type="pytorch",
        sample_use_cache=False,
        train_microbatch_size=args.train_microbatch_size,
    )

    before = _lora_snapshot(train)
    loss = train.train_step(
        [[1, 3, 4, 5], [1, 6, 7, 2]],
        [[0.0, 0.0, -0.1, -0.2], [0.0, 0.0, -0.3, -0.4]],
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, -1.0, -1.0]],
        lr=1e-4,
        weight_decay=0.0,
    )
    after = _lora_snapshot(train)
    samples = train.sample(
        [[1, 3, 4]],
        num_samples=args.num_samples,
        max_tokens=args.expected_new_tokens,
        temperature=0.8,
        top_p=0.95,
    )
    elapsed_s = time.perf_counter() - started

    token_counts = [
        len(token_ids)
        for group in samples
        for token_ids, _logprobs in group
    ]
    logprob_counts = [
        len(logprobs)
        for group in samples
        for _token_ids, logprobs in group
    ]
    expected = int(args.expected_new_tokens)
    exact_tokens = token_counts and all(count == expected for count in token_counts)
    logprobs_aligned = token_counts == logprob_counts
    lora_changed = _any_lora_changed(before, after)

    result = {
        "ok": bool(
            exact_tokens
            and logprobs_aligned
            and lora_changed
            and math.isfinite(float(loss))
        ),
        "device_requested": args.device,
        "train_device": train.train_device,
        "autocast_dtype": str(train.autocast_dtype),
        "use_amp": bool(train.use_amp),
        "grad_scaler_enabled": bool(train.scaler.is_enabled()),
        "model_source": model_source,
        "model": model_name,
        "work_dir": str(work_dir),
        "expected_new_tokens": expected,
        "num_samples": int(args.num_samples),
        "observed_new_token_counts": token_counts,
        "observed_logprob_counts": logprob_counts,
        "logprobs_aligned": bool(logprobs_aligned),
        "loss": float(loss),
        "lora_changed": bool(lora_changed),
        "elapsed_s": elapsed_s,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "mps_available": bool(_mps_is_available()),
    }
    if not result["ok"]:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    result = run_probe(_build_parser().parse_args())
    print("MPS_LOCAL_LORA_PROBE_RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
