"""LoRA weight synchronization between the train model and the engine.

Split mode trains on one device while sampling on another, so syncs read
from a post-step weight snapshot (never live weights) to stay safe across
threads. External engines (server-based / MAX) sync via adapter save +
reload instead of tensor copies.
"""

from __future__ import annotations

import os
import time
from typing import Protocol, cast

import torch

from retrain.backends.local import state as local_state
from retrain.backends.torch import timer_start, timer_stop


class _SyncEngine(Protocol):
    def sync_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None: ...
    def reload_weights(self, save_dir: str) -> None: ...


def initial_sync(helper: object) -> None:
    """Copy LoRA weights train_model -> engine once at construction time.

    No snapshot exists yet; later syncs go through sync_lora_weights and
    read the snapshot instead.
    """
    engine = cast(_SyncEngine, getattr(helper, "engine"))
    engine.sync_from_state_dict(
        local_state.lora_state_dict(getattr(helper, "train_model"), clone=False)
    )


def sync_lora_weights(helper: object) -> None:
    """Sync engine weights from the latest snapshot; record timing metrics.

    No-op outside split/external modes or before the first completed step.
    """
    sync_start = time.perf_counter()
    save_s = 0.0
    reload_s = 0.0
    copied = 0
    try:
        snapshot = getattr(helper, "_weight_snapshot", None)
        if getattr(helper, "_external_engine", False):
            if getattr(helper, "_weights_dirty", False) and snapshot is not None:
                # Save adapter to disk, then tell the engine to reload.
                save_dir = os.path.join(
                    str(getattr(helper, "adapter_path")), "_live_adapter"
                )
                os.makedirs(save_dir, exist_ok=True)
                save_start = time.perf_counter()
                getattr(helper, "train_model").save_pretrained(save_dir)
                save_s = time.perf_counter() - save_start
                reload_start = time.perf_counter()
                cast(_SyncEngine, getattr(helper, "engine")).reload_weights(save_dir)
                reload_s = time.perf_counter() - reload_start
                setattr(helper, "_weights_dirty", False)
                copied = 1
            return

        if not getattr(helper, "split_mode", False):
            return
        if snapshot is None:
            return

        cast(_SyncEngine, getattr(helper, "engine")).sync_from_state_dict(snapshot)
        copied = 1
    finally:
        setattr(
            helper,
            "_last_sync_metrics",
            {
                "local_adapter_sync_s": time.perf_counter() - sync_start,
                "local_adapter_save_s": save_s,
                "local_adapter_reload_s": reload_s,
                "local_adapter_sync_copied": copied,
            },
        )


def snapshot_lora_weights_if_needed(helper: object) -> float:
    """Clone LoRA params for cross-thread sync; return the time spent."""
    if not (
        getattr(helper, "split_mode", False)
        or getattr(helper, "_external_engine", False)
    ):
        return 0.0
    timer = timer_start(str(getattr(helper, "train_device")))
    setattr(
        helper,
        "_weight_snapshot",
        local_state.lora_state_dict(getattr(helper, "train_model"), clone=True),
    )
    setattr(helper, "_weights_dirty", True)
    return timer_stop(timer)


def clear_inference_prefix_cache(helper: object) -> None:
    """Invalidate the engine prefix cache after a weight update."""
    engine = getattr(helper, "engine", None)
    if engine is not None and hasattr(engine, "clear_prefix_cache"):
        engine.clear_prefix_cache()
