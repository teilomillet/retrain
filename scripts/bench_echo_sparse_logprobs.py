#!/usr/bin/env python3
"""Benchmark the sparse ECHO logprob path.

This script has two checks:
- a direct comparison of an old suffix-style LM-head/log-softmax computation,
  the previous selected-token log-softmax+gather path, and the current
  selected-token cross-entropy path on the same sparse targets, optionally
  including the torch.compile selected-CE path;
- a real LocalTrainHelper.train_step_with_echo_masks smoke that verifies sparse
  ECHO targets route through the hidden logprob path.

It uses a tiny CausalLM-shaped module so the benchmark is reproducible without
downloading a model. The measured region is exactly the allocation avoided by
the sparse suffix guard in retrain.local_train_helper.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from retrain.local_train_helper import LocalTrainHelper


def _selected_linear_ce_logprobs(hidden, weight, target_ids):  # noqa: ANN001
    logits = hidden @ weight.T
    return -F.cross_entropy(logits.float(), target_ids, reduction="none")


class _BenchBackbone(torch.nn.Module):
    def __init__(self, embed: torch.nn.Embedding) -> None:
        super().__init__()
        self.embed = embed

    def forward(self, input_ids, attention_mask=None, use_cache=False):  # noqa: ANN001
        _ = attention_mask, use_cache
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class _BenchLM(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.model = _BenchBackbone(self.embed)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):  # noqa: ANN001
        hidden = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        logits_to_keep = kwargs.get("logits_to_keep")
        if logits_to_keep is not None:
            hidden = hidden[:, -int(logits_to_keep) :, :]
        return SimpleNamespace(logits=self.lm_head(hidden))


def _device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _dtype(raw: str, device: torch.device) -> torch.dtype:
    if raw == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if raw == "float16":
        return torch.float16
    if raw == "bfloat16":
        return torch.bfloat16
    if raw == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {raw}")


def _cuda_peak(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _measure(fn, device: torch.device, repeats: int) -> dict[str, object]:  # noqa: ANN001
    times: list[float] = []
    peaks: list[dict[str, float]] = []
    values: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize(device)
        start = time.perf_counter()
        value = fn()
        _synchronize(device)
        times.append(time.perf_counter() - start)
        values.append(float(value))
        peaks.append(_cuda_peak(device))

    payload: dict[str, object] = {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "value": values[-1],
    }
    if peaks and peaks[-1]:
        payload["median_peak_allocated_mb"] = statistics.median(
            peak["peak_allocated_mb"] for peak in peaks
        )
        payload["median_peak_reserved_mb"] = statistics.median(
            peak["peak_reserved_mb"] for peak in peaks
        )
    return payload


def _median_seconds(result: dict[str, object]) -> float:
    value = result["median_s"]
    if not isinstance(value, int | float):
        raise TypeError(f"median_s must be numeric, got {type(value).__name__}")
    return float(value)


def _region_benchmark(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    torch.manual_seed(args.seed)
    suffix_slots = args.seq_len - 1 - args.early_target_pos
    selected_offsets = torch.linspace(
        0,
        suffix_slots - 1,
        args.selected_tokens,
        dtype=torch.long,
        device=device,
    )
    selected_positions = selected_offsets + args.early_target_pos
    suffix_targets = torch.randint(
        0,
        args.vocab_size,
        (suffix_slots,),
        device=device,
    )
    selected_targets = suffix_targets[selected_offsets]
    base_suffix_hidden = torch.randn(
        suffix_slots,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    base_weight = torch.randn(
        args.vocab_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.02

    def suffix_path() -> torch.Tensor:
        hidden = base_suffix_hidden.detach().clone().requires_grad_(True)
        weight = base_weight.detach().clone().requires_grad_(True)
        logits = hidden @ weight.T
        logprobs = F.log_softmax(logits.float(), dim=-1)
        loss = -logprobs[selected_offsets, selected_targets].mean()
        loss.backward()
        return loss.detach()

    def selected_logsoftmax_path() -> torch.Tensor:
        hidden = (
            base_suffix_hidden[selected_offsets]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        weight = base_weight.detach().clone().requires_grad_(True)
        logits = hidden @ weight.T
        logprobs = F.log_softmax(logits.float(), dim=-1)
        loss = -logprobs[
            torch.arange(args.selected_tokens, device=device),
            selected_targets,
        ].mean()
        loss.backward()
        return loss.detach()

    def selected_cross_entropy_path() -> torch.Tensor:
        hidden = (
            base_suffix_hidden[selected_offsets]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        weight = base_weight.detach().clone().requires_grad_(True)
        logits = hidden @ weight.T
        loss = F.cross_entropy(logits.float(), selected_targets, reduction="mean")
        loss.backward()
        return loss.detach()

    compiled_selected_ce = None
    compiled_selected_error = ""
    if args.compile_selective_ce != "off":
        if device.type != "cuda" or not torch.cuda.is_available():
            compiled_selected_error = "non_cuda"
            if args.compile_selective_ce == "require":
                raise RuntimeError(
                    "--compile-selective-ce=require requires a CUDA device"
                )
        elif not hasattr(torch, "compile"):
            compiled_selected_error = "torch_compile_unavailable"
            if args.compile_selective_ce == "require":
                raise RuntimeError("torch.compile is unavailable")
        else:
            try:
                compiled_selected_ce = torch.compile(
                    _selected_linear_ce_logprobs,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            except Exception as exc:  # noqa: BLE001 - optional benchmark path.
                compiled_selected_error = type(exc).__name__
                if args.compile_selective_ce == "require":
                    raise

    def selected_compiled_cross_entropy_path() -> torch.Tensor:
        if compiled_selected_ce is None:
            raise RuntimeError(compiled_selected_error or "compiled path unavailable")
        hidden = (
            base_suffix_hidden[selected_offsets]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        weight = base_weight.detach().clone().requires_grad_(True)
        logprobs = compiled_selected_ce(hidden, weight, selected_targets)
        loss = -logprobs.mean()
        loss.backward()
        return loss.detach()

    suffix_result = _measure(suffix_path, device, args.repeats)
    selected_logsoftmax_result = _measure(
        selected_logsoftmax_path,
        device,
        args.repeats,
    )
    selected_cross_entropy_result = _measure(
        selected_cross_entropy_path,
        device,
        args.repeats,
    )
    selected_compiled_result = None
    if compiled_selected_ce is not None:
        selected_compiled_result = _measure(
            selected_compiled_cross_entropy_path,
            device,
            args.repeats,
        )
    suffix_elements = suffix_slots * args.vocab_size
    selected_elements = args.selected_tokens * args.vocab_size
    selected_ce_median_s = _median_seconds(selected_cross_entropy_result)
    payload: dict[str, object] = {
        "suffix_slots": int(suffix_slots),
        "selected_positions_first_last": [
            int(selected_positions[0].item()),
            int(selected_positions[-1].item()),
        ],
        "suffix_logits_elements": int(suffix_elements),
        "selected_logits_elements": int(selected_elements),
        "element_ratio": float(suffix_elements / selected_elements),
        "suffix_logits_approx_mib_fp32": float(suffix_elements * 4 / 1024**2),
        "selected_logits_approx_mib_fp32": float(selected_elements * 4 / 1024**2),
        "suffix_path": suffix_result,
        "selected_logsoftmax_path": selected_logsoftmax_result,
        "selected_cross_entropy_path": selected_cross_entropy_result,
        "selected_ce_speedup_vs_logsoftmax": float(
            _median_seconds(selected_logsoftmax_result) / selected_ce_median_s
        ),
        "selected_ce_speedup_vs_suffix": float(
            _median_seconds(suffix_result) / selected_ce_median_s
        ),
    }
    if selected_compiled_result is not None:
        payload["selected_compiled_cross_entropy_path"] = selected_compiled_result
        payload["selected_compiled_ce_speedup_vs_eager_ce"] = float(
            selected_ce_median_s / _median_seconds(selected_compiled_result)
        )
    elif args.compile_selective_ce != "off":
        payload["selected_compiled_ce_error"] = compiled_selected_error
    return payload


def _make_helper(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> LocalTrainHelper:
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    helper.scaler = torch.amp.GradScaler(enabled=False)
    helper.train_microbatch_size = 1
    helper.train_device = str(device)
    helper.use_amp = False
    helper.clip_eps = 0.0
    helper.clip_eps_high = 0.0
    helper.split_mode = False
    helper._external_engine = False
    helper._train_future = None
    helper.cuda_empty_cache = False
    helper._clip_fraction = 0.0
    helper._policy_cov_fraction = 0.0
    helper._policy_abs_kl = 0.0
    helper.policy_loss_mode = "standard"
    helper.kl_cov_percent = 0.2
    helper.kl_cov_coef = 1.0
    helper.clip_cov_ratio = 0.0002
    helper.clip_cov_min = 1.0
    helper.clip_cov_max = 5.0
    helper.train_selective_suffix_logits = True
    helper.train_logprob_chunk_size = 0
    helper.train_save_on_cpu = False
    helper.train_save_on_cpu_pin_memory = True
    helper.train_save_on_cpu_min_numel = 0
    helper.train_supervised_context_tokens = 0
    helper.train_unsloth_fused_ce = "off"
    helper.train_unsloth_fused_ce_target_gb = 0.0
    helper.train_unsloth_fused_ce_torch_compile = False
    helper.train_compile_selective_ce = args.compile_selective_ce
    helper.train_compile_selective_ce_min_tokens = args.compile_selective_ce_min_tokens
    helper._last_sample_metrics = {}
    helper._last_train_metrics = {}
    helper._last_sync_metrics = {}
    helper._last_context_crop_metrics = {}
    helper._train_logits_to_keep_supported = None
    helper._selective_logprob_path_counts = {}
    helper._loss_path_counts = {}
    helper._last_unsloth_fused_ce_effective_target_gb = 0.0
    helper._unsloth_fused_ce_available = None
    helper._unsloth_fused_ce_unavailable_reason = ""
    helper._unsloth_fused_ce_fallback_reason = ""
    helper._compiled_selective_ce_available = None
    helper._compiled_selective_ce_fallback_reason = ""
    return helper


def _echo_train_smoke(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    torch.manual_seed(args.seed + 1)
    model = _BenchLM(args.vocab_size, args.hidden_size).to(device=device, dtype=dtype)
    helper = _make_helper(model, device, args)
    input_ids = (
        torch.arange(args.seq_len, dtype=torch.long, device=device)
        .unsqueeze(0)
        .remainder(args.vocab_size)
    )
    tokens = [int(value) for value in input_ids.squeeze(0).detach().cpu().tolist()]
    logprobs = [0.0] * args.seq_len
    advantages = [0.0] * args.seq_len
    advantages[-1] = 1.0
    echo_advantages = [0.0] * args.seq_len
    echo_advantages[args.early_target_pos + 1] = 0.2

    def train_step() -> torch.Tensor:
        helper.optimizer.zero_grad()
        rl_loss, echo_loss = helper.train_step_with_echo_masks(
            all_tokens=[tokens],
            all_logprobs=[logprobs],
            all_advantages=[advantages],
            echo_advantages=[echo_advantages],
            echo_full_observation_counts=[1],
            echo_loss_fn="cross_entropy",
            lr=1e-6,
            weight_decay=0.0,
        )
        return torch.tensor(float(rl_loss) + float(echo_loss), device=device)

    result = _measure(train_step, device, max(1, min(args.repeats, 3)))
    return {
        "result": result,
        "runtime_metrics": helper.runtime_metrics(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--early-target-pos", type=int, default=16)
    parser.add_argument("--selected-tokens", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--compile-selective-ce",
        default="off",
        choices=("off", "auto", "require"),
        help="Benchmark/request the optional torch.compile selected-CE path.",
    )
    parser.add_argument(
        "--compile-selective-ce-min-tokens",
        type=int,
        default=128,
        help="Minimum selected targets before the helper uses compiled CE.",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = _device(args.device)
    dtype = _dtype(args.dtype, device)
    if args.seq_len <= args.early_target_pos + 1:
        raise SystemExit("--seq-len must be greater than --early-target-pos + 1")
    if args.selected_tokens <= 0:
        raise SystemExit("--selected-tokens must be positive")
    if args.selected_tokens > args.seq_len - 1 - args.early_target_pos:
        raise SystemExit("--selected-tokens cannot exceed suffix slots")
    if args.compile_selective_ce_min_tokens < 0:
        raise SystemExit("--compile-selective-ce-min-tokens must be non-negative")

    payload = {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" and torch.cuda.is_available()
            else ""
        ),
        "config": {
            "seq_len": args.seq_len,
            "early_target_pos": args.early_target_pos,
            "selected_tokens": args.selected_tokens,
            "vocab_size": args.vocab_size,
            "hidden_size": args.hidden_size,
            "repeats": args.repeats,
            "compile_selective_ce": args.compile_selective_ce,
            "compile_selective_ce_min_tokens": args.compile_selective_ce_min_tokens,
        },
        "region_benchmark": _region_benchmark(args, device, dtype),
        "echo_train_smoke": _echo_train_smoke(args, device, dtype),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
