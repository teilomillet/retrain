"""Runtime metrics for the local training backend."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

from retrain.backends.torch import cuda_peak_metrics


MetricValue = float | int | str


class SampleResultLike(Protocol):
    token_ids: Sequence[int]


def runtime_metrics(owner: object) -> dict[str, object]:
    """Build the local backend telemetry snapshot exposed to trainers."""
    metrics: dict[str, object] = {
        "local_gradient_checkpointing_enabled": int(
            getattr(owner, "gradient_checkpointing", False)
        ),
        "local_gradient_checkpointing_use_reentrant": str(
            getattr(owner, "gradient_checkpointing_use_reentrant", "auto")
        ),
        "local_gradient_checkpointing_skip_last_n": int(
            getattr(owner, "gradient_checkpointing_skip_last_n", 0)
        ),
        "local_gradient_checkpointing_layer_count": int(
            getattr(owner, "_gradient_checkpointing_layer_metrics", {}).get(
                "total",
                0,
            )
        ),
        "local_gradient_checkpointing_enabled_layers": int(
            getattr(owner, "_gradient_checkpointing_layer_metrics", {}).get(
                "enabled",
                0,
            )
        ),
        "local_gradient_checkpointing_skipped_last_layers": int(
            getattr(owner, "_gradient_checkpointing_layer_metrics", {}).get(
                "skipped_last_n",
                0,
            )
        ),
        "local_cuda_expandable_segments_enabled": int(
            (getattr(owner, "_cuda_allocator_metrics", None) or {}).get(
                "enabled",
                0,
            )
        ),
        "local_cuda_expandable_segments_env_preset": int(
            (getattr(owner, "_cuda_allocator_metrics", None) or {}).get(
                "env_preset",
                0,
            )
        ),
        "local_cuda_expandable_segments_set_failed": int(
            (getattr(owner, "_cuda_allocator_metrics", None) or {}).get(
                "set_failed",
                0,
            )
        ),
        "local_train_amp_dtype": str(getattr(owner, "amp_dtype", "")).replace(
            "torch.",
            "",
        ),
        "local_train_grad_scaler_enabled": _grad_scaler_enabled(owner),
        "local_sample_use_cache": int(getattr(owner, "sample_use_cache", True)),
        "local_sample_kv_quantization": str(
            getattr(owner, "sample_kv_quantization", "off")
        ),
        "local_sample_kv_quantization_oscar": int(
            getattr(owner, "sample_kv_quantization", "off") == "oscar"
        ),
        "local_train_microbatch_size": int(getattr(owner, "train_microbatch_size", 0)),
        "local_train_sft_microbatch_token_budget": int(
            getattr(owner, "train_sft_microbatch_token_budget", 0)
        ),
        "local_train_selective_suffix_logits": int(
            getattr(owner, "train_selective_suffix_logits", False)
        ),
        "local_train_save_on_cpu": int(getattr(owner, "train_save_on_cpu", False)),
        "local_train_save_on_cpu_pin_memory": int(
            getattr(owner, "train_save_on_cpu_pin_memory", True)
        ),
        "local_train_save_on_cpu_min_numel": int(
            getattr(owner, "train_save_on_cpu_min_numel", 0)
        ),
        "local_train_supervised_context_tokens": int(
            getattr(owner, "train_supervised_context_tokens", 0)
        ),
        "local_train_logprob_chunk_size": int(
            getattr(owner, "train_logprob_chunk_size", 0)
        ),
        "local_train_unsloth_fused_ce_mode": str(
            getattr(owner, "train_unsloth_fused_ce", "off")
        ),
        "local_train_unsloth_fused_ce_target_gb": float(
            getattr(owner, "train_unsloth_fused_ce_target_gb", 0.0)
        ),
        "local_train_unsloth_fused_ce_effective_target_gb": float(
            getattr(owner, "_last_unsloth_fused_ce_effective_target_gb", 0.0)
        ),
        "local_train_unsloth_fused_ce_torch_compile": int(
            getattr(owner, "train_unsloth_fused_ce_torch_compile", True)
        ),
        "local_train_compile_selective_ce_mode": str(
            getattr(owner, "train_compile_selective_ce", "off")
        ),
        "local_train_compile_selective_ce_min_tokens": int(
            getattr(owner, "train_compile_selective_ce_min_tokens", 0)
        ),
        "local_train_compile_selective_ce_available": _flag_or_unknown(
            getattr(owner, "_compiled_selective_ce_available", None)
        ),
        "local_train_compile_selective_ce_fallback_reason": str(
            getattr(owner, "_compiled_selective_ce_fallback_reason", "")
        ),
        "local_train_unsloth_fused_ce_available": _flag_or_unknown(
            getattr(owner, "_unsloth_fused_ce_available", None)
        ),
        "local_train_unsloth_fused_ce_unavailable_reason": str(
            getattr(owner, "_unsloth_fused_ce_unavailable_reason", "")
        ),
        "local_train_unsloth_fused_ce_fallback_reason": str(
            getattr(owner, "_unsloth_fused_ce_fallback_reason", "")
        ),
        "local_train_unsloth_fused_ce_attempts": int(
            getattr(owner, "_unsloth_fused_ce_attempts", 0)
        ),
        "local_train_unsloth_fused_ce_batches": _counter(
            owner,
            "_loss_path_counts",
            "unsloth_fused_ce",
        ),
        "local_train_liger_fused_ce_batches": _counter(
            owner,
            "_loss_path_counts",
            "liger_fused_ce",
        ),
        "local_train_dense_logprob_batches": _counter(
            owner,
            "_loss_path_counts",
            "dense_logprob",
        ),
        "local_train_chunked_logprob_batches": _counter(
            owner,
            "_loss_path_counts",
            "chunked_logprob",
        ),
        "local_train_packed_quantized_lm_head_batches": _counter(
            owner,
            "_loss_path_counts",
            "packed_quantized_lm_head",
        ),
        "local_train_logits_to_keep_supported": _flag_or_unknown(
            getattr(owner, "_train_logits_to_keep_supported", None)
        ),
        "local_train_selective_suffix_logprob_batches": _counter(
            owner,
            "_selective_logprob_path_counts",
            "suffix",
        ),
        "local_train_selective_sparse_suffix_skips": _counter(
            owner,
            "_selective_logprob_path_counts",
            "sparse_suffix_skip",
        ),
        "local_train_selective_hidden_logprob_batches": _counter(
            owner,
            "_selective_logprob_path_counts",
            "hidden",
        ),
        "local_train_selective_compiled_ce_batches": _counter(
            owner,
            "_selective_logprob_path_counts",
            "compiled_ce",
        ),
        "local_train_selective_packed_quantized_lm_head_batches": _counter(
            owner,
            "_selective_logprob_path_counts",
            "packed_quantized_lm_head",
        ),
        "local_train_selective_fallback_logprob_batches": _counter(
            owner,
            "_selective_logprob_path_counts",
            "fallback",
        ),
        "local_liger_kernel_enabled": int(getattr(owner, "liger_kernel", False)),
        "local_liger_fused_linear_ce_enabled": int(
            getattr(owner, "liger_fused_linear_ce", False)
        ),
        "local_prefix_caching": int(getattr(owner, "prefix_caching", True)),
    }
    effective_rows_digest = str(
        getattr(owner, "_last_effective_optimizer_rows_sha256", "")
    )
    if effective_rows_digest:
        metrics["optimizer/local_effective_rows_sha256"] = effective_rows_digest
    metrics.update(getattr(owner, "_determinism_metrics", {}))
    metrics.update(getattr(owner, "_lora_model_metrics", {}))
    metrics.update(getattr(owner, "_accelerator_metrics", {}))
    engine = getattr(owner, "engine", None)
    if hasattr(engine, "performance_counters"):
        counters = engine.performance_counters()
        if isinstance(counters, dict):
            metrics.update(counters)
    metrics.update(getattr(owner, "_last_context_crop_metrics", {}))
    metrics.update(getattr(owner, "_last_sample_metrics", {}))
    metrics.update(getattr(owner, "_last_train_metrics", {}))
    metrics.update(getattr(owner, "_last_sync_metrics", {}))
    return metrics


def record_sample(
    owner: object,
    *,
    start_s: float,
    prompt_ids_list: Sequence[Sequence[int]],
    num_samples: int,
    engine_results: Sequence[Sequence[SampleResultLike]],
) -> None:
    """Record sampling latency, token counts, and GPU peak metrics."""
    wall_s = time.perf_counter() - start_s
    prompt_tokens = sum(len(prompt) for prompt in prompt_ids_list) * int(num_samples)
    generated_tokens = sum(
        len(result.token_ids) for group in engine_results for result in group
    )
    metrics: dict[str, float | int] = {
        "local_sample_wall_s": wall_s,
        "local_sample_prompt_tokens": prompt_tokens,
        "local_sample_generated_tokens": generated_tokens,
        "local_sample_generation_tokens_per_s": (
            generated_tokens / wall_s if wall_s > 0 else 0.0
        ),
        "local_sample_gc_disabled_for_cache": int(
            getattr(owner, "_last_sample_gc_disabled_for_cache", 0)
        ),
    }
    metrics.update(
        cuda_peak_metrics(
            "local_sample_gpu",
            getattr(owner, "infer_device", getattr(owner, "train_device", "cpu")),
        )
    )
    setattr(owner, "_last_sample_metrics", metrics)


def record_train(
    owner: object,
    *,
    kind: str,
    wall_start_s: float,
    forward_s: float,
    backward_s: float,
    optimizer_s: float,
    snapshot_s: float,
    microbatches: int,
    total_tokens: float,
    batch_size: int,
) -> None:
    """Record training latency, throughput, microbatching, and GPU peaks."""
    wall_s = time.perf_counter() - wall_start_s
    metrics: dict[str, MetricValue] = {
        "local_train_kind": kind,
        "local_train_wall_s": wall_s,
        "local_train_forward_s": forward_s,
        "local_train_backward_s": backward_s,
        "local_train_optimizer_s": optimizer_s,
        "local_train_snapshot_s": snapshot_s,
        "local_train_microbatches": microbatches,
        "local_train_tokens": total_tokens,
        "local_train_batch_size": batch_size,
        "local_train_tokens_per_s": total_tokens / wall_s if wall_s > 0 else 0.0,
    }
    metrics.update(cuda_peak_metrics("local_train_gpu", getattr(owner, "train_device")))
    setattr(owner, "_last_train_metrics", metrics)


def record_selective_logprob_path(owner: object, path: str) -> None:
    """Increment the selective-logprob path counter."""
    _increment(owner, "_selective_logprob_path_counts", path)


def record_loss_path(owner: object, path: str) -> None:
    """Increment the train-loss path counter."""
    _increment(owner, "_loss_path_counts", path)


def _counter(owner: object, attr: str, path: str) -> int:
    counts = getattr(owner, attr, None)
    return int(counts.get(path, 0)) if isinstance(counts, dict) else 0


def _flag_or_unknown(value: object) -> int:
    if value is None:
        return -1
    return int(bool(value))


def _grad_scaler_enabled(owner: object) -> int:
    scaler = getattr(owner, "scaler", None)
    if scaler is None:
        return 0
    is_enabled = getattr(scaler, "is_enabled", None)
    return int(callable(is_enabled) and bool(is_enabled()))


def _increment(owner: object, attr: str, path: str) -> None:
    counts = getattr(owner, attr, None)
    if not isinstance(counts, dict):
        counts = {}
    counts[path] = int(counts.get(path, 0)) + 1
    setattr(owner, attr, counts)
