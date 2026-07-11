"""Deterministic optimizer-batch capture and one-step replay."""

from __future__ import annotations

from retrain.training.optimizer_batch.adapter import (
    adapter_provenance,
    preflight_optimizer_batch_capture,
    resolve_initial_adapter,
    validate_capture_resume_step,
)
from retrain.training.optimizer_batch.artifact import (
    load_optimizer_batch_capture,
    save_optimizer_batch_capture,
)
from retrain.training.optimizer_batch.contract import (
    config_snapshot,
    optimizer_contract,
    sha256_json,
    validate_replay_contract,
)
from retrain.training.optimizer_batch.types import (
    AdapterProvenance,
    CapturedOptimizerBatch,
    LoadedOptimizerBatch,
    OptimizerBatch,
    ReplayContract,
    TorchRngState,
)
from retrain.training.optimizer_batch.rng import (
    capture_torch_rng_state,
    restore_torch_rng_state,
)

__all__ = [
    "AdapterProvenance",
    "CapturedOptimizerBatch",
    "LoadedOptimizerBatch",
    "OptimizerBatch",
    "ReplayContract",
    "TorchRngState",
    "adapter_provenance",
    "capture_torch_rng_state",
    "config_snapshot",
    "load_optimizer_batch_capture",
    "optimizer_contract",
    "preflight_optimizer_batch_capture",
    "resolve_initial_adapter",
    "restore_torch_rng_state",
    "save_optimizer_batch_capture",
    "sha256_json",
    "validate_capture_resume_step",
    "validate_replay_contract",
]
