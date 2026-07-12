"""Deterministic supervised fine-tuning schedule helpers."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from typing import TYPE_CHECKING

from retrain.training.sft_audit import SFT_AUDIT_SCHEMA

if TYPE_CHECKING:
    from retrain.config import TrainConfig
    from retrain.training.sft import SftDataProvenance

SFT_RESUME_SCHEDULE_CONTRACT_VERSION = 3
SFT_RESUME_SCHEDULE_ALGORITHM = "absolute_sample_seed_plus_epoch_v1"


def build_sft_resume_schedule_contract(
    config: "TrainConfig",
    provenance: SftDataProvenance,
    *,
    batch_size: int,
    max_tokens: int,
    example_order: list[int],
) -> dict[str, object]:
    """Bind every input that can change standalone-SFT example traversal.

    The epoch-zero order fingerprint also binds tokenizer-derived lengths for
    length-aware policies. Dataset paths are intentionally excluded so an
    identical, hash-pinned dataset remains relocatable.
    """

    configured_bucket_size = int(config.sft_length_bucket_size)
    effective_bucket_size = max(
        1,
        configured_bucket_size or len(example_order),
    )
    return {
        "version": SFT_RESUME_SCHEDULE_CONTRACT_VERSION,
        "algorithm": SFT_RESUME_SCHEDULE_ALGORITHM,
        "seed": int(config.seed),
        "batch_size": int(batch_size),
        "batch_order": str(config.sft_batch_order),
        "length_bucket_size": configured_bucket_size,
        "effective_length_bucket_size": effective_bucket_size,
        "reshuffle_each_epoch": bool(config.sft_reshuffle_each_epoch),
        "data_sha256": provenance.data_sha256,
        "data_rows": int(provenance.data_rows),
        "audit_sha256": config.sft_audit_sha256.strip().lower(),
        "audit_schema": (SFT_AUDIT_SCHEMA if config.sft_audit_sha256 else ""),
        "token_audit_sha256": config.sft_token_audit_sha256.strip().lower(),
        "sft_warmup_steps": int(config.sft_warmup_steps),
        "example_count": len(example_order),
        "model": str(config.model),
        "model_revision": str(config.model_revision),
        "model_local_files_only": bool(config.model_local_files_only),
        "max_tokens": int(max_tokens),
        "epoch_zero_order_sha256": sft_indices_sha256(example_order),
    }


def verify_sft_resume_schedule_contract(
    saved: Mapping[str, object] | None,
    current: Mapping[str, object],
) -> None:
    """Fail closed when checkpoint continuation would change SFT traversal."""

    if saved is None:
        raise ValueError(
            "SFT resume schedule contract is missing from trainer_state.json; "
            "refusing checkpoint continuation because the original traversal "
            "cannot be verified. Restart SFT, or use the checkpoint adapter "
            "directly as step-0 initialization instead."
        )

    errors: list[str] = []
    for key, expected in current.items():
        if key not in saved:
            errors.append(f"{key}: missing (expected {expected!r})")
            continue
        actual = saved[key]
        if type(actual) is not type(expected) or actual != expected:
            errors.append(f"{key}: saved {actual!r}, current {expected!r}")
    if errors:
        raise ValueError(
            "SFT resume schedule contract mismatch; refusing to change example "
            "traversal:\n- " + "\n- ".join(errors)
        )


def build_sft_example_order(
    example_count: int,
    seed: int,
    *,
    lengths: list[int] | None = None,
    batch_order: str = "shuffle",
    length_bucket_size: int = 0,
) -> list[int]:
    """Return a deterministic SFT traversal.

    ``shuffle`` preserves the historical behavior. Length-aware modes operate
    after tokenization, matching the cost signal that controls padding and VRAM.
    """
    if example_count <= 0:
        return []
    order = list(range(example_count))
    rng = random.Random(seed)
    rng.shuffle(order)
    if batch_order == "shuffle":
        return order
    if lengths is None or len(lengths) != example_count:
        raise ValueError(
            "length-aware SFT ordering requires one token length per example."
        )
    if batch_order in ("length", "length_asc"):
        return sorted(order, key=lambda idx: (lengths[idx], idx))
    if batch_order == "length_desc":
        return sorted(order, key=lambda idx: (-lengths[idx], idx))
    if batch_order == "length_bucket":
        bucket_size = int(length_bucket_size or example_count)
        bucket_size = max(1, bucket_size)
        bucketed: list[int] = []
        for start in range(0, example_count, bucket_size):
            bucket = order[start : start + bucket_size]
            bucketed.extend(sorted(bucket, key=lambda idx: (lengths[idx], idx)))
        return bucketed
    raise ValueError(
        "batch_order must be 'shuffle', 'length', 'length_asc', "
        "'length_desc', or 'length_bucket'."
    )


def build_sft_epoch_order(
    example_order: list[int],
    *,
    epoch: int,
    seed: int,
    lengths: list[int] | None = None,
    batch_order: str = "shuffle",
    length_bucket_size: int = 0,
    reshuffle_each_epoch: bool = False,
    epoch_order_cache: dict[int, list[int]] | None = None,
) -> list[int]:
    """Resolve one deterministic epoch order, optionally through a cache."""
    if epoch < 0:
        raise ValueError("SFT epoch must be >= 0.")
    if epoch == 0 or not reshuffle_each_epoch:
        return example_order
    if epoch_order_cache is not None and epoch in epoch_order_cache:
        return epoch_order_cache[epoch]
    order = build_sft_example_order(
        len(example_order),
        seed + epoch,
        lengths=lengths,
        batch_order=batch_order,
        length_bucket_size=length_bucket_size,
    )
    if epoch_order_cache is not None:
        epoch_order_cache[epoch] = order
    return order


def select_sft_batch_indices(
    example_order: list[int],
    *,
    batch_size: int,
    step: int,
    seed: int = 0,
    lengths: list[int] | None = None,
    batch_order: str = "shuffle",
    length_bucket_size: int = 0,
    reshuffle_each_epoch: bool = False,
    epoch_order_cache: dict[int, list[int]] | None = None,
) -> list[int]:
    """Select a deterministic SFT batch by absolute sample position.

    With ``reshuffle_each_epoch=False`` this exactly preserves the historical
    fixed-permutation cycle. When enabled, epoch zero uses ``example_order``
    and each later epoch rebuilds the same ordering policy with ``seed +
    epoch``. Deriving the epoch and in-epoch offset from ``step * batch_size``
    makes a resumed step select the same examples without serialized RNG
    state, including batches that cross a non-divisible epoch boundary.
    """
    if batch_size <= 0 or not example_order:
        return []
    if step < 0:
        raise ValueError("SFT batch step must be >= 0.")
    start = step * batch_size
    size = len(example_order)
    epoch_orders = epoch_order_cache if epoch_order_cache is not None else {}
    epoch_orders.setdefault(0, example_order)
    indices: list[int] = []
    for absolute_position in range(start, start + batch_size):
        epoch, epoch_offset = divmod(absolute_position, size)
        order = build_sft_epoch_order(
            example_order,
            epoch=epoch,
            seed=seed,
            lengths=lengths,
            batch_order=batch_order,
            length_bucket_size=length_bucket_size,
            reshuffle_each_epoch=reshuffle_each_epoch,
            epoch_order_cache=epoch_orders,
        )
        indices.append(order[epoch_offset])
    if epoch_order_cache is not None:
        start_epoch = start // size
        end_epoch = (start + batch_size - 1) // size
        keep_epochs = {0, start_epoch, end_epoch}
        for cached_epoch in tuple(epoch_orders):
            if cached_epoch not in keep_epochs:
                del epoch_orders[cached_epoch]
    return indices


def describe_sft_batch_position(
    example_count: int,
    *,
    batch_size: int,
    step: int,
) -> dict[str, int]:
    """Describe the absolute epoch span of one deterministic SFT batch."""
    if example_count <= 0 or batch_size <= 0:
        return {
            "sft_epoch": 0,
            "sft_epoch_end": 0,
            "sft_epoch_sample_offset": 0,
            "sft_absolute_sample": 0,
        }
    if step < 0:
        raise ValueError("SFT batch step must be >= 0.")
    start = step * batch_size
    end = start + batch_size - 1
    epoch, epoch_offset = divmod(start, example_count)
    return {
        "sft_epoch": epoch,
        "sft_epoch_end": end // example_count,
        "sft_epoch_sample_offset": epoch_offset,
        "sft_absolute_sample": start,
    }


def sft_indices_sha256(indices: list[int]) -> str:
    """Hash an exact index sequence as concatenated unsigned 64-bit big endian."""
    digest = hashlib.sha256()
    for index in indices:
        if index < 0 or index >= 1 << 64:
            raise ValueError("SFT schedule indices must fit unsigned 64-bit encoding.")
        digest.update(index.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def build_sft_schedule_metrics(
    example_order: list[int],
    selected_indices: list[int],
    *,
    batch_size: int,
    step: int,
    seed: int,
    lengths: list[int] | None = None,
    batch_order: str = "shuffle",
    length_bucket_size: int = 0,
    reshuffle_each_epoch: bool = False,
    epoch_order_cache: dict[int, list[int]] | None = None,
    epoch_order_sha256_cache: dict[int, str] | None = None,
) -> dict[str, int | str]:
    """Return reconstructable position and SHA256 evidence for one SFT batch."""
    position = describe_sft_batch_position(
        len(example_order),
        batch_size=batch_size,
        step=step,
    )
    start_epoch = position["sft_epoch"]
    end_epoch = position["sft_epoch_end"]
    start_order = build_sft_epoch_order(
        example_order,
        epoch=start_epoch,
        seed=seed,
        lengths=lengths,
        batch_order=batch_order,
        length_bucket_size=length_bucket_size,
        reshuffle_each_epoch=reshuffle_each_epoch,
        epoch_order_cache=epoch_order_cache,
    )
    start_order_sha256 = None
    if epoch_order_sha256_cache is not None:
        start_order_sha256 = epoch_order_sha256_cache.get(start_epoch)
    if start_order_sha256 is None:
        start_order_sha256 = sft_indices_sha256(start_order)
        if epoch_order_sha256_cache is not None:
            epoch_order_sha256_cache[start_epoch] = start_order_sha256
    metrics: dict[str, int | str] = {
        **position,
        "sft_epoch_seed": seed + start_epoch if reshuffle_each_epoch else seed,
        "sft_epoch_end_seed": seed + end_epoch if reshuffle_each_epoch else seed,
        "sft_batch_indices_sha256": sft_indices_sha256(selected_indices),
        "sft_epoch_start_order_sha256": start_order_sha256,
    }
    if end_epoch != start_epoch:
        end_order = build_sft_epoch_order(
            example_order,
            epoch=end_epoch,
            seed=seed,
            lengths=lengths,
            batch_order=batch_order,
            length_bucket_size=length_bucket_size,
            reshuffle_each_epoch=reshuffle_each_epoch,
            epoch_order_cache=epoch_order_cache,
        )
        end_order_sha256 = None
        if epoch_order_sha256_cache is not None:
            end_order_sha256 = epoch_order_sha256_cache.get(end_epoch)
        if end_order_sha256 is None:
            end_order_sha256 = sft_indices_sha256(end_order)
            if epoch_order_sha256_cache is not None:
                epoch_order_sha256_cache[end_epoch] = end_order_sha256
        metrics["sft_epoch_end_order_sha256"] = end_order_sha256
    if epoch_order_sha256_cache is not None:
        keep_epochs = {0, start_epoch, end_epoch}
        for cached_epoch in tuple(epoch_order_sha256_cache):
            if cached_epoch not in keep_epochs:
                del epoch_order_sha256_cache[cached_epoch]
    return metrics
