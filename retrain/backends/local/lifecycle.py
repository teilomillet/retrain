"""Lifecycle operations for the local training backend."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import torch

from retrain.backends.local import state as local_state

if TYPE_CHECKING:
    from retrain.backends.local.train import LocalTrainHelper


def checkpoint(helper: "LocalTrainHelper", name: str) -> None:
    """Collect completed work and expose the latest weights for sampling."""

    del name  # Checkpoint names are part of the backend contract, not local sync.
    if helper._train_future is not None and helper._train_future.done():
        helper._pending_loss = helper._train_future.result()
        helper._train_future = None
    helper._sync_lora_weights()
    helper._clear_inference_prefix_cache()


def shutdown(helper: "LocalTrainHelper") -> None:
    """Release model, optimizer, executor, and CUDA allocator state."""

    future = getattr(helper, "_train_future", None)
    if future is not None:
        try:
            future.result()
        except Exception:  # Cleanup should not mask failures.
            pass

    executor = getattr(helper, "_train_executor", None)
    if executor is not None:
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=True)

    engine = getattr(helper, "engine", None)
    if engine is not None:
        try:
            engine.shutdown()
        except Exception:  # Best-effort cleanup.
            pass

    for name in (
        "engine",
        "train_model",
        "optimizer",
        "scaler",
        "_weight_snapshot",
        "_train_future",
    ):
        if hasattr(helper, name):
            try:
                delattr(helper, name)
            except Exception:  # Best-effort cleanup.
                pass

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:  # CUDA may be recovering from OOM.
            pass
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:  # Cleanup must not raise.
            pass


def load_state(helper: "LocalTrainHelper", name: str) -> None:
    """Load adapter weights and mark split/external engines for re-sync."""

    save_dir = local_state.load_into_model(
        helper.train_model,
        adapter_path=helper.adapter_path,
        name=name,
        train_device=helper.train_device,
    )
    if helper.split_mode or helper._external_engine:
        helper._weight_snapshot = local_state.lora_state_dict(
            helper.train_model,
            clone=True,
        )
        helper._weights_dirty = True
    print(f"Loaded adapter checkpoint: {save_dir} (optimizer state not restored)")


def save_adapter(helper: "LocalTrainHelper", path: str, name: str) -> str:
    """Flush pending training and save the latest LoRA adapter."""

    if helper._train_future is not None:
        helper._pending_loss = helper._train_future.result()
        helper._train_future = None
    save_dir = local_state.save_model(helper.train_model, path=path, name=name)
    print(f"Adapter saved to {save_dir}")
    return save_dir
