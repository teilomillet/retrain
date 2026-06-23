#!/usr/bin/env python3
"""Run a reproducible one-GPU local-backend benchmark sweep.

The default axes are intentionally explicit:
- sample KV-cache off/on
- cuda_empty_cache off/on
- gradient checkpointing off/on
- train microbatch sizes 0/1/2/4
- group sizes 1/2/4/6/8

Use --dry-run first; the full Cartesian product is expensive.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import multiprocessing as mp
import traceback
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from retrain.benchmark import format_suite_summary, run_benchmark_suite
from retrain.config import TrainConfig, load_config
from retrain.registry import get_registry


def _cleanup_after_condition() -> None:
    """Release benchmark objects before the next condition starts."""
    gc.collect()
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch import is optional for dry-run/help.
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - after OOM the context can be mid-recovery.
        pass
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001 - cleanup must not hide the benchmark failure.
        pass


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("values must be non-negative")
    return values


def _parse_bool_list(raw: str) -> list[bool]:
    values: list[bool] = []
    for part in raw.split(","):
        text = part.strip().lower()
        if not text:
            continue
        if text in {"1", "true", "yes", "on"}:
            values.append(True)
        elif text in {"0", "false", "no", "off"}:
            values.append(False)
        else:
            raise argparse.ArgumentTypeError(f"invalid boolean value: {part!r}")
    if not values:
        raise argparse.ArgumentTypeError("expected at least one boolean")
    return values


def _default_output_root(config_path: str, config: TrainConfig) -> Path:
    stem = Path(config_path).stem or "config"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base_log_dir = Path(config.log_dir)
    return base_log_dir.parent / f"{base_log_dir.name}-{stem}-one-gpu-sweep-{ts}"


def _condition_label(
    *,
    sample_use_cache: bool,
    cuda_empty_cache: bool,
    gradient_checkpointing: bool,
    train_microbatch_size: int,
    group_size: int,
) -> str:
    return (
        f"cache_{int(sample_use_cache)}"
        f"-empty_{int(cuda_empty_cache)}"
        f"-gc_{int(gradient_checkpointing)}"
        f"-mb_{train_microbatch_size}"
        f"-g_{group_size}"
    )


def _build_config(
    base: TrainConfig,
    *,
    sample_use_cache: bool,
    cuda_empty_cache: bool,
    gradient_checkpointing: bool,
    train_microbatch_size: int,
    group_size: int,
    max_steps: int,
    rollout_env_workers: int,
    rollout_buffer_size: int,
) -> TrainConfig:
    backend_options = dict(base.backend_options)
    backend_options.update(
        {
            "sample_use_cache": sample_use_cache,
            "cuda_empty_cache": cuda_empty_cache,
            "gradient_checkpointing": gradient_checkpointing,
            "train_microbatch_size": train_microbatch_size,
        }
    )
    replacements = {
        "backend_options": backend_options,
        "group_size": group_size,
    }
    if max_steps > 0:
        replacements["max_steps"] = max_steps
    if rollout_env_workers > 0:
        replacements["environment_rollout_env_workers"] = rollout_env_workers
    if rollout_buffer_size > 0:
        replacements["environment_rollout_buffer_size"] = rollout_buffer_size
    return replace(base, **replacements)


def _run_condition(
    base: TrainConfig,
    condition: dict[str, object],
    *,
    repeat: int,
    max_steps: int,
    rollout_env_workers: int,
    rollout_buffer_size: int,
) -> dict[str, object]:
    config = _build_config(
        base,
        sample_use_cache=bool(condition["sample_use_cache"]),
        cuda_empty_cache=bool(condition["cuda_empty_cache"]),
        gradient_checkpointing=bool(condition["gradient_checkpointing"]),
        train_microbatch_size=int(condition["train_microbatch_size"]),
        group_size=int(condition["group_size"]),
        max_steps=max_steps,
        rollout_env_workers=rollout_env_workers,
        rollout_buffer_size=rollout_buffer_size,
    )
    condition_dir = Path(str(condition["output_dir"]))
    try:
        summary = run_benchmark_suite(
            config,
            repeats=repeat,
            output_dir=condition_dir,
            runner_factory=lambda cfg: get_registry("trainer").create(
                cfg.trainer, cfg
            ),
            disable_wandb=True,
        )
        result: dict[str, object] = {
            **condition,
            "status": "succeeded",
            "summary_path": str(condition_dir / "benchmark_summary.json"),
            "summary": asdict(summary),
        }
        print(format_suite_summary(summary))
    except Exception as exc:  # noqa: BLE001 - benchmark failures are data.
        result = {
            **condition,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        condition_dir.mkdir(parents=True, exist_ok=True)
    (condition_dir / "condition_status.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def _run_condition_child(
    base: TrainConfig,
    condition: dict[str, object],
    *,
    repeat: int,
    max_steps: int,
    rollout_env_workers: int,
    rollout_buffer_size: int,
    result_path: str,
) -> None:
    result = _run_condition(
        base,
        condition,
        repeat=repeat,
        max_steps=max_steps,
        rollout_env_workers=rollout_env_workers,
        rollout_buffer_size=rollout_buffer_size,
    )
    Path(result_path).write_text(json.dumps(result, indent=2), encoding="utf-8")


def _run_condition_isolated(
    base: TrainConfig,
    condition: dict[str, object],
    *,
    repeat: int,
    max_steps: int,
    rollout_env_workers: int,
    rollout_buffer_size: int,
) -> dict[str, object]:
    condition_dir = Path(str(condition["output_dir"]))
    condition_dir.mkdir(parents=True, exist_ok=True)
    result_path = condition_dir / "condition_result.json"
    if result_path.exists():
        result_path.unlink()

    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=_run_condition_child,
        kwargs={
            "base": base,
            "condition": condition,
            "repeat": repeat,
            "max_steps": max_steps,
            "rollout_env_workers": rollout_env_workers,
            "rollout_buffer_size": rollout_buffer_size,
            "result_path": str(result_path),
        },
    )
    process.start()
    process.join()

    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))

    result: dict[str, object] = {
        **condition,
        "status": "failed",
        "error_type": "ChildProcessError",
        "error_message": f"condition process exited with code {process.exitcode}",
        "traceback": "",
    }
    (condition_dir / "condition_status.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local one-GPU benchmark sweeps for cache/memory axes.",
    )
    parser.add_argument("config", help="Base retrain TOML config")
    parser.add_argument("--output-root", default="", help="Sweep output root")
    parser.add_argument("--repeat", type=int, default=1, help="Benchmark repeats")
    parser.add_argument(
        "--sample-use-cache",
        type=_parse_bool_list,
        default=_parse_bool_list("false,true"),
        help="Comma-separated booleans for local PyTorch generation KV-cache",
    )
    parser.add_argument(
        "--cuda-empty-cache",
        type=_parse_bool_list,
        default=_parse_bool_list("false,true"),
        help="Comma-separated booleans for torch.cuda.empty_cache calls",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        type=_parse_bool_list,
        default=_parse_bool_list("false,true"),
        help="Comma-separated booleans for train-time gradient checkpointing",
    )
    parser.add_argument(
        "--microbatch-sizes",
        type=_parse_int_list,
        default=_parse_int_list("0,1,2,4"),
        help="Comma-separated train microbatch sizes; 0 disables microbatching",
    )
    parser.add_argument(
        "--group-sizes",
        type=_parse_int_list,
        default=_parse_int_list("1,2,4,6,8"),
        help="Comma-separated rollout group sizes",
    )
    parser.add_argument(
        "--max-conditions",
        type=int,
        default=0,
        help="Optional cap for smoke subsets; 0 means no cap",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Override max_steps per condition; 0 preserves the base config",
    )
    parser.add_argument(
        "--rollout-env-workers",
        type=int,
        default=0,
        help="Override [environment].rollout_env_workers; 0 preserves config",
    )
    parser.add_argument(
        "--rollout-buffer-size",
        type=int,
        default=0,
        help="Override [environment].rollout_buffer_size; 0 preserves config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the manifest without running benchmarks",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed conditions and keep sweeping instead of aborting.",
    )
    parser.add_argument(
        "--isolate-conditions",
        action="store_true",
        help="Run each condition in a fresh child process to isolate CUDA state.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.repeat <= 0:
        raise SystemExit("--repeat must be >= 1")
    if args.max_conditions < 0:
        raise SystemExit("--max-conditions must be >= 0")
    if args.max_steps < 0:
        raise SystemExit("--max-steps must be >= 0")
    if args.rollout_env_workers < 0:
        raise SystemExit("--rollout-env-workers must be >= 0")
    if args.rollout_buffer_size < 0:
        raise SystemExit("--rollout-buffer-size must be >= 0")

    base = load_config(args.config)
    if base.backend != "local":
        raise SystemExit(
            "one_gpu_local_sweep requires [backend].backend = 'local' in the base config"
        )

    output_root = (
        Path(args.output_root)
        if args.output_root
        else _default_output_root(args.config, base)
    )

    raw_conditions = list(
        itertools.product(
            args.sample_use_cache,
            args.cuda_empty_cache,
            args.gradient_checkpointing,
            args.microbatch_sizes,
            args.group_sizes,
        )
    )
    if args.max_conditions:
        raw_conditions = raw_conditions[: args.max_conditions]

    conditions = []
    for sample_use_cache, cuda_empty_cache, gradient_checkpointing, microbatch, group in raw_conditions:
        label = _condition_label(
            sample_use_cache=sample_use_cache,
            cuda_empty_cache=cuda_empty_cache,
            gradient_checkpointing=gradient_checkpointing,
            train_microbatch_size=microbatch,
            group_size=group,
        )
        conditions.append(
            {
                "label": label,
                "sample_use_cache": sample_use_cache,
                "cuda_empty_cache": cuda_empty_cache,
                "gradient_checkpointing": gradient_checkpointing,
                "train_microbatch_size": microbatch,
                "group_size": group,
                "output_dir": str(output_root / label),
            }
        )

    manifest = {
        "config": args.config,
        "output_root": str(output_root),
        "repeat": args.repeat,
        "max_steps": args.max_steps,
        "rollout_env_workers": args.rollout_env_workers,
        "rollout_buffer_size": args.rollout_buffer_size,
        "continue_on_error": args.continue_on_error,
        "isolate_conditions": args.isolate_conditions,
        "conditions": conditions,
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    condition_results = []
    for condition in conditions:
        print(f"== {condition['label']} ==")
        if args.isolate_conditions:
            result = _run_condition_isolated(
                base,
                condition,
                repeat=args.repeat,
                max_steps=args.max_steps,
                rollout_env_workers=args.rollout_env_workers,
                rollout_buffer_size=args.rollout_buffer_size,
            )
        else:
            result = _run_condition(
                base,
                condition,
                repeat=args.repeat,
                max_steps=args.max_steps,
                rollout_env_workers=args.rollout_env_workers,
                rollout_buffer_size=args.rollout_buffer_size,
            )

        condition_results.append(result)
        if result.get("status") == "failed":
            (output_root / "sweep_manifest.json").write_text(
                json.dumps(
                    {
                        **manifest,
                        "status": "running" if args.continue_on_error else "failed",
                        "results": condition_results,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if not args.continue_on_error:
                _cleanup_after_condition()
                raise RuntimeError(result.get("error_message", "condition failed"))
            print(
                f"condition failed: {condition['label']} "
                f"({result.get('error_type')}: {result.get('error_message')})"
            )
            _cleanup_after_condition()
            continue

        (output_root / "sweep_manifest.json").write_text(
            json.dumps(
                {
                    **manifest,
                    "status": "running",
                    "results": condition_results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _cleanup_after_condition()

    failed = [result for result in condition_results if result["status"] != "succeeded"]
    status = "complete" if not failed else "partial_failures"
    (output_root / "sweep_manifest.json").write_text(
        json.dumps(
            {
                **manifest,
                "status": status,
                "succeeded": len(condition_results) - len(failed),
                "failed": len(failed),
                "results": condition_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"sweep {status}: {output_root}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
