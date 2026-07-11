"""Ragged safetensors codec and structural validation for optimizer rows."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from retrain.training.batch_digest import logical_optimizer_batch_sha256
from retrain.training.optimizer_batch.types import OptimizerBatch, TorchRngState


def batch_tensors(batch: OptimizerBatch) -> dict[str, np.ndarray]:
    """Flatten ragged rows without losing binary64 logical float values."""

    offsets = _row_offsets(batch.tokens)
    tensors = {
        "row_offsets": np.asarray(offsets, dtype=np.int64),
        "tokens": np.asarray(_flatten(batch.tokens), dtype=np.int64),
        "old_logprobs": np.asarray(_flatten(batch.old_logprobs), dtype=np.float64),
        "advantages": np.asarray(_flatten(batch.advantages), dtype=np.float64),
        "torch_cpu_rng": np.frombuffer(batch.torch_rng.cpu, dtype=np.uint8).copy(),
        "torch_cuda_rng_offsets": np.asarray(
            _byte_offsets(batch.torch_rng.cuda),
            dtype=np.int64,
        ),
        "torch_cuda_rng": np.frombuffer(
            b"".join(batch.torch_rng.cuda),
            dtype=np.uint8,
        ).copy(),
    }
    if batch.echo_advantages is not None:
        assert batch.echo_full_observation_counts is not None
        assert batch.echo_rollout_denominator is not None
        tensors["echo_advantages"] = np.asarray(
            _flatten(batch.echo_advantages),
            dtype=np.float64,
        )
        tensors["echo_full_observation_counts"] = np.asarray(
            batch.echo_full_observation_counts,
            dtype=np.int64,
        )
        tensors["echo_rollout_denominator"] = np.asarray(
            [batch.echo_rollout_denominator],
            dtype=np.int64,
        )
    return tensors


def batch_from_tensors(tensors: dict[str, np.ndarray]) -> OptimizerBatch:
    """Reconstruct ragged rows and reject malformed payload tensors."""

    required = {
        "row_offsets",
        "tokens",
        "old_logprobs",
        "advantages",
        "torch_cpu_rng",
        "torch_cuda_rng_offsets",
        "torch_cuda_rng",
    }
    missing = sorted(required - set(tensors))
    if missing:
        raise ValueError(f"optimizer-batch payload missing tensors: {missing}.")
    offsets = _validated_offsets(tensors["row_offsets"], len(tensors["tokens"]))
    _require_dtype(tensors, "tokens", np.dtype(np.int64))
    _require_dtype(tensors, "old_logprobs", np.dtype(np.float64))
    _require_dtype(tensors, "advantages", np.dtype(np.float64))
    _require_dtype(tensors, "torch_cpu_rng", np.dtype(np.uint8))
    flat_tokens = tensors["tokens"].tolist()
    flat_logprobs = tensors["old_logprobs"].tolist()
    flat_advantages = tensors["advantages"].tolist()
    if not (len(flat_tokens) == len(flat_logprobs) == len(flat_advantages)):
        raise ValueError("optimizer-batch flat tensor lengths do not match.")

    echo_advantages, echo_counts, echo_denominator = _decode_echo(
        tensors,
        offsets=offsets,
        flat_token_count=len(flat_tokens),
    )
    cuda_offsets = _validated_offsets(
        tensors["torch_cuda_rng_offsets"],
        len(tensors["torch_cuda_rng"]),
    )
    cuda_flat = tensors["torch_cuda_rng"].tobytes()
    cuda_states = tuple(
        cuda_flat[cuda_offsets[index] : cuda_offsets[index + 1]]
        for index in range(len(cuda_offsets) - 1)
    )
    return OptimizerBatch(
        tokens=_unflatten(flat_tokens, offsets),
        old_logprobs=_unflatten(flat_logprobs, offsets),
        advantages=_unflatten(flat_advantages, offsets),
        echo_advantages=echo_advantages,
        echo_full_observation_counts=echo_counts,
        echo_rollout_denominator=echo_denominator,
        torch_rng=TorchRngState(
            cpu=tensors["torch_cpu_rng"].tobytes(),
            cuda=cuda_states,
        ),
    )


def validate_batch(batch: OptimizerBatch) -> None:
    """Enforce dimensions, finite floats, token IDs, ECHO, and RNG state."""

    if not batch.tokens:
        raise ValueError("optimizer-batch capture requires at least one row.")
    rows = len(batch.tokens)
    if len(batch.old_logprobs) != rows or len(batch.advantages) != rows:
        raise ValueError("optimizer-batch token/logprob/advantage row counts differ.")
    for index, (tokens, logprobs, advantages) in enumerate(
        zip(batch.tokens, batch.old_logprobs, batch.advantages, strict=True)
    ):
        if not tokens:
            raise ValueError(f"optimizer-batch row {index} is empty.")
        if len(tokens) != len(logprobs) or len(tokens) != len(advantages):
            raise ValueError(f"optimizer-batch row {index} lengths differ.")
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in tokens
        ):
            raise ValueError(f"optimizer-batch row {index} has an invalid token id.")
        if any(
            not math.isfinite(float(value))
            for value in (*logprobs, *advantages)
        ):
            raise ValueError(f"optimizer-batch row {index} has a non-finite float.")
    _validate_echo(batch, rows=rows)
    if not batch.torch_rng.cpu:
        raise ValueError("optimizer-batch capture requires Torch CPU RNG state.")


def logical_sha(batch: OptimizerBatch) -> str:
    return logical_optimizer_batch_sha256(
        batch.tokens,
        batch.old_logprobs,
        batch.advantages,
        echo_observation_masks=batch.echo_advantages,
        echo_full_observation_counts=batch.echo_full_observation_counts,
        echo_rollout_denominator=batch.echo_rollout_denominator,
    )


def batch_summary(batch: OptimizerBatch) -> dict[str, int | bool]:
    return {
        "rows": len(batch.tokens),
        "tokens": sum(len(row) for row in batch.tokens),
        "nonzero_advantage_tokens": sum(
            value != 0.0 for row in batch.advantages for value in row
        ),
        "echo_present": batch.echo_advantages is not None,
        "echo_nonzero_tokens": sum(
            value != 0.0
            for row in (batch.echo_advantages or [])
            for value in row
        ),
        "torch_cpu_rng_bytes": len(batch.torch_rng.cpu),
        "torch_cuda_rng_devices": len(batch.torch_rng.cuda),
    }


def _decode_echo(
    tensors: dict[str, np.ndarray],
    *,
    offsets: list[int],
    flat_token_count: int,
) -> tuple[list[list[float]] | None, list[int] | None, int | None]:
    echo_keys = {
        "echo_advantages",
        "echo_full_observation_counts",
        "echo_rollout_denominator",
    }
    if not (echo_keys & set(tensors)):
        return None, None, None
    if not echo_keys <= set(tensors):
        raise ValueError("optimizer-batch payload has incomplete ECHO tensors.")
    _require_dtype(tensors, "echo_advantages", np.dtype(np.float64))
    _require_dtype(tensors, "echo_full_observation_counts", np.dtype(np.int64))
    _require_dtype(tensors, "echo_rollout_denominator", np.dtype(np.int64))
    flat_echo = tensors["echo_advantages"].tolist()
    if len(flat_echo) != flat_token_count:
        raise ValueError("optimizer-batch flat ECHO tensor length does not match.")
    counts = [
        int(value)
        for value in tensors["echo_full_observation_counts"].tolist()
    ]
    denominator_values = tensors["echo_rollout_denominator"].tolist()
    if len(denominator_values) != 1:
        raise ValueError("optimizer-batch ECHO denominator must be scalar.")
    return _unflatten(flat_echo, offsets), counts, int(denominator_values[0])


def _validate_echo(batch: OptimizerBatch, *, rows: int) -> None:
    echo_values = (
        batch.echo_advantages,
        batch.echo_full_observation_counts,
        batch.echo_rollout_denominator,
    )
    if any(value is not None for value in echo_values) != all(
        value is not None for value in echo_values
    ):
        raise ValueError("optimizer-batch ECHO fields must be supplied together.")
    if batch.echo_advantages is None:
        return
    assert batch.echo_full_observation_counts is not None
    assert batch.echo_rollout_denominator is not None
    if len(batch.echo_advantages) != rows:
        raise ValueError("optimizer-batch ECHO row count differs.")
    if len(batch.echo_full_observation_counts) != rows:
        raise ValueError("optimizer-batch ECHO count row count differs.")
    for index, (tokens, masks) in enumerate(
        zip(batch.tokens, batch.echo_advantages, strict=True)
    ):
        if len(tokens) != len(masks) or any(
            not math.isfinite(float(value)) for value in masks
        ):
            raise ValueError(f"optimizer-batch ECHO row {index} is invalid.")
    if any(value < 0 for value in batch.echo_full_observation_counts):
        raise ValueError("optimizer-batch ECHO counts must be non-negative.")
    if batch.echo_rollout_denominator <= 0:
        raise ValueError("optimizer-batch ECHO denominator must be positive.")


def _row_offsets(rows: Sequence[Sequence[object]]) -> list[int]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return offsets


def _byte_offsets(rows: tuple[bytes, ...]) -> list[int]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return offsets


def _validated_offsets(raw: np.ndarray, flat_length: int) -> list[int]:
    if raw.dtype != np.dtype(np.int64) or raw.ndim != 1:
        raise ValueError(
            "optimizer-batch offsets must be a one-dimensional int64 tensor."
        )
    offsets = [int(value) for value in raw.tolist()]
    if not offsets or offsets[0] != 0 or offsets[-1] != flat_length:
        raise ValueError("optimizer-batch offsets do not span the flat tensor.")
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("optimizer-batch offsets must be monotonic.")
    return offsets


def _flatten[T](rows: Sequence[Sequence[T]]) -> list[T]:
    return [value for row in rows for value in row]


def _unflatten[T](values: list[T], offsets: list[int]) -> list[list[T]]:
    return [
        list(values[offsets[index] : offsets[index + 1]])
        for index in range(len(offsets) - 1)
    ]


def _require_dtype(
    tensors: dict[str, np.ndarray],
    key: str,
    dtype: np.dtype,
) -> None:
    if tensors[key].dtype != dtype or tensors[key].ndim != 1:
        raise ValueError(
            f"optimizer-batch tensor {key!r} must be one-dimensional {dtype}."
        )
