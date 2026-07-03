#!/usr/bin/env python3
"""Run a USL-oriented Unsloth SFT batch/microbatch sweep.

The sweep isolates each condition in a child Python process by invoking
``scripts/smoke_unsloth_sft.py``.  It then fits the Universal Scalability Law
over batch size for each microbatch strategy and reports bottleneck signals.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from retrain.backpressure import StepObservation, USLBackPressure, usl_throughput


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("values must be non-negative")
    return values


def _bool_arg(raw: str) -> bool:
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {raw!r}")


@dataclass(frozen=True)
class UslFit:
    count: int
    sigma: float
    kappa: float
    p_star: float
    fitted_lambda: float
    r2: float
    relative_spread: float
    peak_p_observed: int
    peak_throughput_observed: float
    peak_p_predicted: float
    peak_throughput_predicted: float
    classification: str


def _metric_number(metrics: dict[str, object], *keys: str) -> float:
    for key in keys:
        raw = metrics.get(key)
        if isinstance(raw, (int, float)):
            return float(raw)
    return 0.0


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _object_dict(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _throughput_from_payload(payload: dict[str, object]) -> float:
    metrics = _object_dict(payload.get("metrics"))
    if not metrics:
        return 0.0
    datums = _metric_number(metrics, "datums")
    wall_s = _metric_number(metrics, "backend/local_train_wall_s", "local_train_wall_s")
    if datums > 0 and wall_s > 0:
        return datums / wall_s
    return _metric_number(
        metrics,
        "backend/local_train_tokens_per_s",
        "local_train_tokens_per_s",
    )


def _fit_usl(rows: list[dict[str, object]]) -> UslFit | None:
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "succeeded"
        and _as_float(row.get("throughput_datums_per_s")) > 0.0
        and _as_int(row.get("batch_size")) > 0
    ]
    if len(ok_rows) < 3:
        return None

    bp = USLBackPressure(
        warmup_steps=0,
        ema_decay=0.0,
        min_batch_size=1,
        max_batch_size=max(_as_int(row.get("batch_size")) for row in ok_rows),
        min_group_size=1,
        max_group_size=1,
    )
    for row in sorted(ok_rows, key=lambda item: _as_int(item.get("batch_size"))):
        p = _as_int(row.get("batch_size"))
        throughput = _as_float(row.get("throughput_datums_per_s"))
        bp.observe(
            StepObservation(
                step_time_s=1_000_000.0,
                batch_size=p,
                group_size=1,
                total_tokens=max(1, int(round(throughput * 1_000_000.0))),
            )
        )

    observed = [
        (
            _as_float(row.get("batch_size")),
            _as_float(row.get("throughput_datums_per_s")),
        )
        for row in ok_rows
    ]
    mean_y = sum(y for _, y in observed) / len(observed)
    relative_spread = (
        (max(y for _, y in observed) - min(y for _, y in observed)) / mean_y
        if mean_y > 0
        else 0.0
    )
    ss_tot = sum((y - mean_y) ** 2 for _, y in observed)
    ss_res = 0.0
    for p, y in observed:
        predicted = bp.fitted_lambda * usl_throughput(p, bp.sigma, bp.kappa)
        ss_res += (y - predicted) ** 2
    r2 = 1.0 if ss_tot <= 0.0 else max(0.0, 1.0 - ss_res / ss_tot)

    peak_row = max(
        ok_rows,
        key=lambda row: _as_float(row.get("throughput_datums_per_s")),
    )
    max_observed_p = max(_as_float(row.get("batch_size")) for row in ok_rows)
    predicted_peak_p = bp.p_star if bp.kappa > 0 else max_observed_p
    predicted_peak_p = max(1.0, min(predicted_peak_p, max_observed_p))
    predicted_peak = bp.fitted_lambda * usl_throughput(
        predicted_peak_p,
        bp.sigma,
        bp.kappa,
    )
    classification = _classify_fit(
        bp.sigma,
        bp.kappa,
        bp.p_star,
        max_observed_p,
        r2,
        relative_spread,
    )
    return UslFit(
        count=len(ok_rows),
        sigma=bp.sigma,
        kappa=bp.kappa,
        p_star=bp.p_star,
        fitted_lambda=bp.fitted_lambda,
        r2=r2,
        relative_spread=relative_spread,
        peak_p_observed=_as_int(peak_row.get("batch_size")),
        peak_throughput_observed=_as_float(
            peak_row.get("throughput_datums_per_s")
        ),
        peak_p_predicted=predicted_peak_p,
        peak_throughput_predicted=predicted_peak,
        classification=classification,
    )


def _classify_fit(
    sigma: float,
    kappa: float,
    p_star: float,
    max_observed_p: float,
    r2: float,
    relative_spread: float,
) -> str:
    if relative_spread < 0.05 and max_observed_p > 1:
        return "serialization_or_launch_overhead_flat_throughput"
    if r2 < 0.5:
        return "weak_fit_use_raw_measurements"
    if kappa > 1e-6 and p_star < max_observed_p:
        return "coherency_or_memory_pressure_retrograde"
    if kappa > 1e-6:
        return "coherency_or_memory_pressure"
    if sigma > 0.25:
        return "serialization_or_launch_overhead"
    if sigma > 0.05:
        return "contention_limited"
    return "near_linear_in_measured_range"


def _stage_summary(metrics: dict[str, object]) -> dict[str, float]:
    wall_s = _metric_number(metrics, "backend/local_train_wall_s", "local_train_wall_s")
    forward_s = _metric_number(metrics, "backend/local_train_forward_s", "local_train_forward_s")
    backward_s = _metric_number(metrics, "backend/local_train_backward_s", "local_train_backward_s")
    optimizer_s = _metric_number(metrics, "backend/local_train_optimizer_s", "local_train_optimizer_s")
    peak_mb = _metric_number(
        metrics,
        "backend/local_train_gpu_peak_memory_reserved_mb",
        "local_train_gpu_peak_memory_reserved_mb",
    )
    if wall_s <= 0:
        return {
            "train_wall_s": 0.0,
            "forward_share": 0.0,
            "backward_share": 0.0,
            "optimizer_share": 0.0,
            "peak_reserved_mb": peak_mb,
        }
    return {
        "train_wall_s": wall_s,
        "forward_share": forward_s / wall_s,
        "backward_share": backward_s / wall_s,
        "optimizer_share": optimizer_s / wall_s,
        "peak_reserved_mb": peak_mb,
    }


def _condition_label(batch_size: int, microbatch_size: int) -> str:
    micro = "full" if microbatch_size == 0 else str(microbatch_size)
    return f"b{batch_size}-mb{micro}"


def _default_output_root() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("logs") / f"unsloth-sft-usl-{ts}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a USL batch/microbatch sweep for Unsloth SFT.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to run smoke script")
    parser.add_argument("--output-root", default="", help="Directory for condition JSON and summary")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=_parse_int_list("1,2,4,8"))
    parser.add_argument(
        "--microbatch-sizes",
        type=_parse_int_list,
        default=_parse_int_list("1,0"),
        help="Comma-separated train microbatch sizes. 0 means full batch.",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--synthetic-prompt-tokens", type=int, default=128)
    parser.add_argument("--synthetic-token-text", default=" x")
    parser.add_argument("--synthetic-completion-tokens", type=int, default=0)
    parser.add_argument("--synthetic-completion-token-text", default=" y")
    parser.add_argument("--completion", default="2")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--load-in-4bit", type=_bool_arg, default=True)
    parser.add_argument("--gradient-checkpointing", type=_bool_arg, default=True)
    parser.add_argument("--train-selective-suffix-logits", type=_bool_arg, default=True)
    parser.add_argument("--train-save-on-cpu", type=_bool_arg, default=False)
    parser.add_argument("--train-save-on-cpu-pin-memory", type=_bool_arg, default=True)
    parser.add_argument("--train-save-on-cpu-min-numel", type=int, default=0)
    parser.add_argument("--train-supervised-context-tokens", type=int, default=0)
    parser.add_argument(
        "--train-unsloth-fused-ce",
        choices=("off", "auto", "require"),
        default="auto",
    )
    parser.add_argument("--train-unsloth-fused-ce-target-gb", type=float, default=0.0)
    parser.add_argument("--train-unsloth-fused-ce-torch-compile", type=_bool_arg, default=True)
    parser.add_argument("--liger-fused-linear-ce", type=_bool_arg, default=True)
    parser.add_argument("--continue-on-error", type=_bool_arg, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run_condition(args: argparse.Namespace, output_root: Path, batch_size: int, microbatch_size: int) -> dict[str, object]:
    label = _condition_label(batch_size, microbatch_size)
    output_path = output_root / f"{label}.json"
    adapter_path = output_root / label / "adapter"
    log_dir = output_root / label / "logs"
    cmd = [
        args.python,
        "scripts/smoke_unsloth_sft.py",
        "--model",
        args.model,
        "--max-seq-length",
        str(args.max_seq_length),
        "--max-tokens",
        str(args.max_tokens),
        "--batch-size",
        str(batch_size),
        "--steps",
        str(args.steps),
        "--lora-rank",
        str(args.lora_rank),
        "--synthetic-prompt-tokens",
        str(args.synthetic_prompt_tokens),
        "--synthetic-token-text",
        args.synthetic_token_text,
        "--synthetic-completion-tokens",
        str(args.synthetic_completion_tokens),
        "--synthetic-completion-token-text",
        args.synthetic_completion_token_text,
        "--completion",
        args.completion,
        "--train-microbatch-size",
        str(microbatch_size),
        "--load-in-4bit",
        str(args.load_in_4bit).lower(),
        "--gradient-checkpointing",
        str(args.gradient_checkpointing).lower(),
        "--train-selective-suffix-logits",
        str(args.train_selective_suffix_logits).lower(),
        "--train-save-on-cpu",
        str(args.train_save_on_cpu).lower(),
        "--train-save-on-cpu-pin-memory",
        str(args.train_save_on_cpu_pin_memory).lower(),
        "--train-save-on-cpu-min-numel",
        str(args.train_save_on_cpu_min_numel),
        "--train-supervised-context-tokens",
        str(args.train_supervised_context_tokens),
        "--train-unsloth-fused-ce",
        args.train_unsloth_fused_ce,
        "--train-unsloth-fused-ce-target-gb",
        str(args.train_unsloth_fused_ce_target_gb),
        "--train-unsloth-fused-ce-torch-compile",
        str(args.train_unsloth_fused_ce_torch_compile).lower(),
        "--liger-fused-linear-ce",
        str(args.liger_fused_linear_ce).lower(),
        "--adapter-path",
        str(adapter_path),
        "--log-dir",
        str(log_dir),
        "--output",
        str(output_path),
        "--include-traceback",
        "true",
    ]
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload: dict[str, object]
    if output_path.is_file():
        payload = _object_dict(json.loads(output_path.read_text()))
    else:
        payload = {
            "ok": False,
            "error_type": "MissingOutput",
            "error": f"{output_path} was not written",
        }
    metrics = _object_dict(payload.get("metrics"))
    result = _object_dict(payload.get("result"))
    row: dict[str, object] = {
        "label": label,
        "batch_size": batch_size,
        "train_microbatch_size": microbatch_size,
        "status": "succeeded" if payload.get("ok") is True else "failed",
        "returncode": completed.returncode,
        "command": cmd,
        "started_at": started,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "payload_path": str(output_path),
        "adapter_path": str(adapter_path / "final"),
        "log_dir": str(log_dir),
        "error_type": payload.get("error_type", result.get("failure_status", "")),
        "error_message": payload.get("error", result.get("error_message", "")),
        "throughput_datums_per_s": _throughput_from_payload(payload),
        "train_tokens_per_s": _metric_number(
            metrics,
            "backend/local_train_tokens_per_s",
            "local_train_tokens_per_s",
        ),
        "datums": _metric_number(metrics, "datums"),
        "tokens": _metric_number(metrics, "tokens"),
        "supervised_tokens": _metric_number(metrics, "supervised_tokens"),
        "context_rows_cropped": _metric_number(
            metrics,
            "backend/local_train_context_rows_cropped",
            "local_train_context_rows_cropped",
        ),
        "context_tokens_removed": _metric_number(
            metrics,
            "backend/local_train_context_tokens_removed",
            "local_train_context_tokens_removed",
        ),
        "context_cropped_max_tokens": _metric_number(
            metrics,
            "backend/local_train_context_cropped_max_tokens",
            "local_train_context_cropped_max_tokens",
        ),
        "logits_to_keep_supported": _metric_number(
            metrics,
            "backend/local_train_logits_to_keep_supported",
            "local_train_logits_to_keep_supported",
        ),
        "selective_suffix_batches": _metric_number(
            metrics,
            "backend/local_train_selective_suffix_logprob_batches",
            "local_train_selective_suffix_logprob_batches",
        ),
        "selective_hidden_batches": _metric_number(
            metrics,
            "backend/local_train_selective_hidden_logprob_batches",
            "local_train_selective_hidden_logprob_batches",
        ),
        "selective_fallback_batches": _metric_number(
            metrics,
            "backend/local_train_selective_fallback_logprob_batches",
            "local_train_selective_fallback_logprob_batches",
        ),
        "unsloth_fused_ce_batches": _metric_number(
            metrics,
            "backend/local_train_unsloth_fused_ce_batches",
            "local_train_unsloth_fused_ce_batches",
        ),
        "unsloth_fused_ce_attempts": _metric_number(
            metrics,
            "backend/local_train_unsloth_fused_ce_attempts",
            "local_train_unsloth_fused_ce_attempts",
        ),
        "unsloth_fused_ce_available": _metric_number(
            metrics,
            "backend/local_train_unsloth_fused_ce_available",
            "local_train_unsloth_fused_ce_available",
        ),
        "unsloth_fused_ce_effective_target_gb": _metric_number(
            metrics,
            "backend/local_train_unsloth_fused_ce_effective_target_gb",
            "local_train_unsloth_fused_ce_effective_target_gb",
        ),
        "unsloth_fused_ce_fallback_reason": str(
            metrics.get(
                "backend/local_train_unsloth_fused_ce_fallback_reason",
                metrics.get("local_train_unsloth_fused_ce_fallback_reason", ""),
            )
        ),
        "microbatches": _metric_number(
            metrics,
            "backend/local_train_microbatches",
            "local_train_microbatches",
        ),
        **_stage_summary(metrics),
    }
    return row


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_microbatch: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        key = str(row["train_microbatch_size"])
        by_microbatch.setdefault(key, []).append(row)

    fits: dict[str, object] = {}
    for key, group in by_microbatch.items():
        fit = _fit_usl(group)
        fits[key] = asdict(fit) if fit is not None else None

    improvements: list[dict[str, object]] = []
    by_batch: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_batch.setdefault(_as_int(row.get("batch_size")), []).append(row)
    for batch_size, group in sorted(by_batch.items()):
        successful = [
            row
            for row in group
            if row.get("status") == "succeeded"
            and _as_float(row.get("throughput_datums_per_s")) > 0
        ]
        if len(successful) < 2:
            continue
        baseline = next(
            (
                row
                for row in successful
                if _as_int(row.get("train_microbatch_size")) == 1
            ),
            None,
        )
        best = max(
            successful,
            key=lambda row: _as_float(row.get("throughput_datums_per_s")),
        )
        if baseline is None:
            continue
        baseline_tput = _as_float(baseline.get("throughput_datums_per_s"))
        best_tput = _as_float(best.get("throughput_datums_per_s"))
        gain = (best_tput / baseline_tput) - 1.0 if baseline_tput > 0 else 0.0
        improvements.append(
            {
                "batch_size": batch_size,
                "baseline_microbatch_size": baseline["train_microbatch_size"],
                "baseline_throughput_datums_per_s": baseline_tput,
                "best_microbatch_size": best["train_microbatch_size"],
                "best_throughput_datums_per_s": best_tput,
                "gain_fraction": gain,
                "gain_percent": gain * 100.0,
                "meets_15_percent": gain >= 0.15,
            }
        )

    best_improvement = (
        max(improvements, key=lambda item: float(item["gain_fraction"]))
        if improvements
        else None
    )
    return {
        "fits_by_microbatch_size": fits,
        "improvements_vs_microbatch_1": improvements,
        "best_improvement": best_improvement,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be > 0")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be > 0")
    if args.synthetic_prompt_tokens < 0:
        raise SystemExit("--synthetic-prompt-tokens must be >= 0")
    if args.synthetic_completion_tokens < 0:
        raise SystemExit("--synthetic-completion-tokens must be >= 0")
    if args.train_save_on_cpu_min_numel < 0:
        raise SystemExit("--train-save-on-cpu-min-numel must be >= 0")
    if args.train_supervised_context_tokens < 0:
        raise SystemExit("--train-supervised-context-tokens must be >= 0")
    if args.train_unsloth_fused_ce_target_gb < 0:
        raise SystemExit("--train-unsloth-fused-ce-target-gb must be >= 0")

    output_root = Path(args.output_root) if args.output_root else _default_output_root()
    conditions = [
        {
            "label": _condition_label(batch_size, microbatch_size),
            "batch_size": batch_size,
            "train_microbatch_size": microbatch_size,
        }
        for microbatch_size in args.microbatch_sizes
        for batch_size in args.batch_sizes
    ]
    manifest = {
        "model": args.model,
        "output_root": str(output_root),
        "axis": "batch_size",
        "target": ">=15% datums/s improvement over train_microbatch_size=1 at matching batch size",
        "steps": args.steps,
        "max_seq_length": args.max_seq_length,
        "max_tokens": args.max_tokens,
        "synthetic_prompt_tokens": args.synthetic_prompt_tokens,
        "synthetic_token_text": args.synthetic_token_text,
        "synthetic_completion_tokens": args.synthetic_completion_tokens,
        "synthetic_completion_token_text": args.synthetic_completion_token_text,
        "train_save_on_cpu": args.train_save_on_cpu,
        "train_save_on_cpu_pin_memory": args.train_save_on_cpu_pin_memory,
        "train_save_on_cpu_min_numel": args.train_save_on_cpu_min_numel,
        "train_supervised_context_tokens": args.train_supervised_context_tokens,
        "train_unsloth_fused_ce": args.train_unsloth_fused_ce,
        "train_unsloth_fused_ce_target_gb": args.train_unsloth_fused_ce_target_gb,
        "train_unsloth_fused_ce_torch_compile": args.train_unsloth_fused_ce_torch_compile,
        "conditions": conditions,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows: list[dict[str, object]] = []
    for condition in conditions:
        print(f"== {condition['label']} ==")
        row = _run_condition(
            args,
            output_root,
            int(condition["batch_size"]),
            int(condition["train_microbatch_size"]),
        )
        rows.append(row)
        summary = {
            **manifest,
            "status": "running",
            "results": rows,
            "analysis": _summarize(rows),
        }
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if row["status"] != "succeeded":
            print(f"failed: {row['label']} {row.get('error_type')}: {row.get('error_message')}")
            if not args.continue_on_error:
                return 1
            continue
        print(
            f"{row['label']}: "
            f"datums/s={_as_float(row.get('throughput_datums_per_s')):.4f} "
            f"wall={_as_float(row.get('train_wall_s')):.3f}s "
            f"peak={_as_float(row.get('peak_reserved_mb')):.1f}MB"
        )

    final = {
        **manifest,
        "status": "complete",
        "results": rows,
        "analysis": _summarize(rows),
    }
    (output_root / "summary.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final["analysis"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
