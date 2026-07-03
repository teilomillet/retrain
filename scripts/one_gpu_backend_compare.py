#!/usr/bin/env python3
"""Compare local PyTorch, vLLM, SGLang, and TensorRT-LLM inference in one-GPU retrain runs."""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from retrain.benchmark.format import format_suite_summary
from retrain.benchmark.summary import BenchmarkSuiteSummary
from retrain.benchmark.run import run_benchmark_suite
from retrain.config import TrainConfig, load_config
from retrain.registry import get_registry


_TOKEN_NATIVE_ENGINES = frozenset({"vllm", "sglang"})
_SERVER_ENGINES = frozenset({"vllm", "sglang", "trtllm"})
_ADAPTER_FRESHNESS_ENGINES = frozenset({"vllm", "sglang", "trtllm"})


def _parse_bool(raw: str) -> bool:
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {raw!r}")


def _parse_engines(raw: str) -> list[str]:
    engines = [part.strip().lower() for part in raw.split(",") if part.strip()]
    valid = {"pytorch", "vllm", "sglang", "trtllm"}
    unknown = [engine for engine in engines if engine not in valid]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown engine(s): {', '.join(unknown)}; expected pytorch,vllm,sglang,trtllm"
        )
    if not engines:
        raise argparse.ArgumentTypeError("expected at least one engine")
    return engines


def _default_output_root(config_path: str, config: TrainConfig) -> Path:
    stem = Path(config_path).stem or "config"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base_log_dir = Path(config.log_dir)
    return base_log_dir.parent / f"{base_log_dir.name}-{stem}-backend-compare-{ts}"


def _engine_url(args: argparse.Namespace, engine: str) -> str:
    if engine == "vllm":
        return args.vllm_url
    if engine == "sglang":
        return args.sglang_url
    if engine == "trtllm":
        return args.trtllm_url
    return ""


def _check_server(engine: str, base_url: str, timeout_s: float) -> None:
    if engine not in _SERVER_ENGINES:
        return
    if not base_url:
        raise RuntimeError(f"{engine} requires an inference URL")

    root = base_url.rstrip("/")
    errors: list[str] = []
    for suffix in ("/health", "/v1/models"):
        url = f"{root}{suffix}"
        try:
            with urlopen(url, timeout=timeout_s) as response:
                status = int(getattr(response, "status", 0))
                if 200 <= status < 300:
                    return
                errors.append(f"{url} -> HTTP {status}")
        except HTTPError as exc:
            errors.append(f"{url} -> HTTP {exc.code}")
        except URLError as exc:
            errors.append(f"{url} -> {exc.reason}")
        except TimeoutError:
            errors.append(f"{url} -> timeout")
    raise RuntimeError(f"{engine} server preflight failed: {'; '.join(errors)}")


def _build_config(base: TrainConfig, args: argparse.Namespace, engine: str) -> TrainConfig:
    backend_options = dict(base.backend_options)
    backend_options.update(
        {
            "cuda_empty_cache": args.cuda_empty_cache,
            "gradient_checkpointing": args.gradient_checkpointing,
            "sample_use_cache": args.sample_use_cache,
            "train_microbatch_size": args.microbatch_size,
        }
    )
    replacements = {
        "backend": "local",
        "inference_engine": engine,
        "inference_url": _engine_url(args, engine),
        "backend_options": backend_options,
        "group_size": args.group_size,
        "prefix_caching": args.prefix_caching,
    }
    if args.rollout_env_workers > 0:
        replacements["environment_rollout_env_workers"] = args.rollout_env_workers
    if args.rollout_buffer_size > 0:
        replacements["environment_rollout_buffer_size"] = args.rollout_buffer_size
    if args.max_steps > 0:
        replacements["max_steps"] = args.max_steps
    if args.seed >= 0:
        replacements["seed"] = args.seed
    return replace(base, **replacements)


def _summary_mean(summary: BenchmarkSuiteSummary, key: str) -> float | None:
    stat = summary.aggregates.get(key)
    return None if stat is None else stat.mean


def _quality_gates(
    *,
    engine: str,
    summary: BenchmarkSuiteSummary,
    min_correct_rate: float,
    require_token_native: bool,
    require_adapter_reload: bool,
) -> dict[str, dict[str, object]]:
    final_rates = [
        run.final_correct_rate
        for run in summary.runs
        if run.final_correct_rate is not None
    ]
    final_correct_rate = final_rates[-1] if final_rates else None
    gates: dict[str, dict[str, object]] = {
        "min_correct_rate": {
            "required": min_correct_rate,
            "observed": final_correct_rate,
            "passed": (
                final_correct_rate is not None
                and final_correct_rate >= min_correct_rate
            ),
        }
    }
    if require_token_native and engine in _TOKEN_NATIVE_ENGINES:
        token_calls = _summary_mean(summary, "engine_token_prompt_calls")
        fallbacks = _summary_mean(summary, "engine_token_prompt_fallbacks")
        enabled = _summary_mean(summary, "engine_token_native_prompt_enabled")
        gates["token_native_prompt"] = {
            "calls": token_calls,
            "fallbacks": fallbacks,
            "enabled": enabled,
            "passed": bool(token_calls and token_calls > 0)
            and fallbacks == 0
            and enabled == 1,
        }
    if require_adapter_reload and engine in _ADAPTER_FRESHNESS_ENGINES:
        reload_calls = _summary_mean(summary, "engine_adapter_reload_calls")
        reload_failures = _summary_mean(summary, "engine_adapter_reload_failures")
        min_calls = 1 if any(run.steps > 1 for run in summary.runs) else 0
        gates["adapter_reload"] = {
            "min_calls": min_calls,
            "calls": reload_calls,
            "failures": reload_failures,
            "passed": (
                reload_failures == 0
                and (
                    min_calls == 0
                    or bool(reload_calls and reload_calls >= min_calls)
                )
            ),
        }
    return gates


def _gates_passed(gates: Mapping[str, Mapping[str, object]]) -> bool:
    for gate in gates.values():
        if gate.get("passed") is not True:
            return False
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch, vLLM, SGLang, and TensorRT-LLM inference on one GPU.",
    )
    parser.add_argument("config", help="Base retrain TOML config")
    parser.add_argument("--output-root", default="", help="Comparison output root")
    parser.add_argument("--engines", type=_parse_engines, default=_parse_engines("pytorch,vllm,sglang"))
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    parser.add_argument("--trtllm-url", default="http://127.0.0.1:31000")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--cuda-empty-cache", type=_parse_bool, default=True)
    parser.add_argument("--gradient-checkpointing", type=_parse_bool, default=True)
    parser.add_argument("--sample-use-cache", type=_parse_bool, default=True)
    parser.add_argument("--prefix-caching", type=_parse_bool, default=True)
    parser.add_argument("--rollout-env-workers", type=int, default=0)
    parser.add_argument("--rollout-buffer-size", type=int, default=0)
    parser.add_argument("--min-correct-rate", type=float, default=0.0)
    parser.add_argument("--server-preflight-timeout", type=float, default=5.0)
    parser.add_argument(
        "--skip-server-preflight",
        action="store_true",
        help="Do not check external inference server health before benchmarking.",
    )
    parser.add_argument(
        "--allow-token-prompt-fallback",
        action="store_true",
        help="Do not fail vLLM/SGLang runs that reject token-id prompts.",
    )
    parser.add_argument(
        "--allow-adapter-reload-failure",
        action="store_true",
        help="Do not fail vLLM/SGLang/TensorRT-LLM runs when dynamic LoRA freshness is missing or fails.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed engines and keep comparing instead of aborting.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.repeat <= 0:
        raise SystemExit("--repeat must be >= 1")
    if args.group_size <= 0:
        raise SystemExit("--group-size must be >= 1")
    if args.microbatch_size < 0:
        raise SystemExit("--microbatch-size must be >= 0")
    if args.max_steps < 0:
        raise SystemExit("--max-steps must be >= 0")
    if args.rollout_env_workers < 0:
        raise SystemExit("--rollout-env-workers must be >= 0")
    if args.rollout_buffer_size < 0:
        raise SystemExit("--rollout-buffer-size must be >= 0")

    base = load_config(args.config)
    output_root = (
        Path(args.output_root)
        if args.output_root
        else _default_output_root(args.config, base)
    )
    conditions = [
        {
            "label": engine,
            "engine": engine,
            "inference_url": _engine_url(args, engine),
            "output_dir": str(output_root / engine),
        }
        for engine in args.engines
    ]
    manifest = {
        "config": args.config,
        "output_root": str(output_root),
        "repeat": args.repeat,
        "group_size": args.group_size,
        "microbatch_size": args.microbatch_size,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "cuda_empty_cache": args.cuda_empty_cache,
        "gradient_checkpointing": args.gradient_checkpointing,
        "sample_use_cache": args.sample_use_cache,
        "prefix_caching": args.prefix_caching,
        "rollout_env_workers": args.rollout_env_workers,
        "rollout_buffer_size": args.rollout_buffer_size,
        "min_correct_rate": args.min_correct_rate,
        "server_preflight_timeout": args.server_preflight_timeout,
        "skip_server_preflight": args.skip_server_preflight,
        "require_token_native": not args.allow_token_prompt_fallback,
        "require_adapter_reload": not args.allow_adapter_reload_failure,
        "continue_on_error": args.continue_on_error,
        "conditions": conditions,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "backend_compare_manifest.json"
    manifest_path.write_text(
        json.dumps({**manifest, "status": "running"}, indent=2),
        encoding="utf-8",
    )

    results = []
    for condition in conditions:
        engine = str(condition["engine"])
        print(f"== {engine} ==")
        config = _build_config(base, args, engine)
        condition_dir = Path(str(condition["output_dir"]))
        try:
            if not args.skip_server_preflight:
                _check_server(
                    engine,
                    str(condition["inference_url"]),
                    args.server_preflight_timeout,
                )
            summary = run_benchmark_suite(
                config,
                repeats=args.repeat,
                output_dir=condition_dir,
                runner_factory=lambda cfg: get_registry("trainer").create(
                    cfg.trainer, cfg
                ),
                disable_wandb=True,
            )
            gates = _quality_gates(
                engine=engine,
                summary=summary,
                min_correct_rate=args.min_correct_rate,
                require_token_native=not args.allow_token_prompt_fallback,
                require_adapter_reload=not args.allow_adapter_reload_failure,
            )
            status = "succeeded" if _gates_passed(gates) else "gate_failed"
            result = {
                **condition,
                "status": status,
                "quality_gates": gates,
                "summary_path": str(condition_dir / "benchmark_summary.json"),
                "summary": summary.to_dict(),
            }
            print(format_suite_summary(summary))
        except Exception as exc:  # Backend comparison failures are data.
            condition_dir.mkdir(parents=True, exist_ok=True)
            result = {
                **condition,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            if not args.continue_on_error:
                (condition_dir / "condition_status.json").write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )
                raise
            print(f"engine failed: {engine} ({type(exc).__name__}: {exc})")

        (condition_dir / "condition_status.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        results.append(result)
        manifest_path.write_text(
            json.dumps({**manifest, "status": "running", "results": results}, indent=2),
            encoding="utf-8",
        )

    bad = [result for result in results if result["status"] != "succeeded"]
    status = "complete" if not bad else "partial_failures"
    manifest_path.write_text(
        json.dumps(
            {
                **manifest,
                "status": status,
                "succeeded": len(results) - len(bad),
                "failed": len(bad),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"backend comparison {status}: {output_root}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
