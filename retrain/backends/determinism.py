"""Fail-closed deterministic-update controls for the local PyTorch backend.

This module deliberately separates two guarantees:

* PyTorch/CUDA deterministic-algorithm controls are established and audited.
* Third-party kernels are *not* declared deterministic by configuration alone.

The latter still requires a repeated, same-input post-update adapter hash on
the target GPU before a causal training claim is sound.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Protocol, cast

from retrain.config.constants import _MAX_REPRODUCIBLE_SEED


_CUBLAS_WORKSPACE_ENV = "CUBLAS_WORKSPACE_CONFIG"
_DEFAULT_CUBLAS_WORKSPACE = ":4096:8"
_VALID_CUBLAS_WORKSPACES = frozenset({_DEFAULT_CUBLAS_WORKSPACE, ":16:8"})
_STRICT_SETUP_ESTABLISHED_BEFORE_CUDA = False


class _CudaLike(Protocol):
    def is_initialized(self) -> bool: ...

    def is_available(self) -> bool: ...

    def manual_seed_all(self, seed: int) -> None: ...


class _CudnnLike(Protocol):
    deterministic: bool
    benchmark: bool


class _BackendsLike(Protocol):
    cudnn: _CudnnLike


class _TorchLike(Protocol):
    __version__: str
    cuda: _CudaLike
    backends: _BackendsLike

    def use_deterministic_algorithms(
        self,
        mode: bool,
        *,
        warn_only: bool = False,
    ) -> None: ...

    def are_deterministic_algorithms_enabled(self) -> bool: ...

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool: ...

    def manual_seed(self, seed: int) -> object: ...


def prepare_strict_determinism_environment(*, enabled: bool) -> dict[str, object]:
    """Set deterministic CUDA process environment before CUDA construction."""

    current = os.environ.get(_CUBLAS_WORKSPACE_ENV, "")
    if not enabled:
        return {
            "local_strict_deterministic_requested": 0,
            "local_cublas_workspace_config": current,
            "local_cublas_workspace_config_valid": int(
                current in _VALID_CUBLAS_WORKSPACES
            ),
        }
    if current and current not in _VALID_CUBLAS_WORKSPACES:
        raise RuntimeError(
            "strict_deterministic requires CUBLAS_WORKSPACE_CONFIG to be "
            f"one of {sorted(_VALID_CUBLAS_WORKSPACES)} before CUDA setup; "
            f"got {current!r}."
        )
    was_preconfigured = bool(current)
    resolved = current or _DEFAULT_CUBLAS_WORKSPACE
    os.environ[_CUBLAS_WORKSPACE_ENV] = resolved
    if os.environ.get(_CUBLAS_WORKSPACE_ENV) != resolved:
        raise RuntimeError(
            "strict_deterministic could not establish CUBLAS_WORKSPACE_CONFIG."
        )
    return {
        "local_strict_deterministic_requested": 1,
        "local_cublas_workspace_config": resolved,
        "local_cublas_workspace_config_valid": 1,
        "local_cublas_workspace_config_preconfigured": int(was_preconfigured),
    }


def establish_strict_determinism(*, enabled: bool) -> dict[str, object]:
    """Enable and verify strict PyTorch controls, failing before model creation."""

    environment = prepare_strict_determinism_environment(enabled=enabled)
    import torch

    runtime = cast(_TorchLike, torch)
    if not enabled and _STRICT_SETUP_ESTABLISHED_BEFORE_CUDA:
        raise RuntimeError(
            "strict_deterministic=false cannot follow strict_deterministic=true "
            "in the same process because PyTorch, cuDNN, and cuBLAS controls are "
            "process-global. Run mixed-strictness campaign arms in separate "
            "processes."
        )
    if enabled:
        _enable_and_verify(runtime, environment)
    proof = _runtime_proof(runtime, requested=enabled)
    proof.update(environment)
    return proof


def seed_strict_determinism(seed: int) -> dict[str, object]:
    """Seed every supported RNG before strict local-model construction."""

    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
        or seed > _MAX_REPRODUCIBLE_SEED
    ):
        raise RuntimeError(
            "strict_deterministic requires [training] seed between 0 and "
            f"{_MAX_REPRODUCIBLE_SEED} inclusive so every supported RNG can "
            f"be seeded; got {seed}."
        )
    import random

    random.seed(seed)
    numpy_seeded = 0
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_seeded = 1
    except ImportError:
        pass

    import torch

    runtime = cast(_TorchLike, torch)
    runtime.manual_seed(seed)
    cuda_seeded = 0
    if runtime.cuda.is_available():
        runtime.cuda.manual_seed_all(seed)
        cuda_seeded = 1
    return {
        "local_strict_deterministic_seed": seed,
        "local_strict_deterministic_python_seeded": 1,
        "local_strict_deterministic_numpy_seeded": numpy_seeded,
        "local_strict_deterministic_torch_seeded": 1,
        "local_strict_deterministic_cuda_seeded": cuda_seeded,
    }


def add_model_attention_proof(
    proof: Mapping[str, object],
    *,
    model: object,
    requested_attention_kernel: str,
) -> dict[str, object]:
    """Bind deterministic runtime proof to the model's resolved attention path."""

    resolved = _resolved_attention_implementation(model)
    requested = bool(proof.get("local_strict_deterministic_requested", 0))
    if requested and not resolved:
        raise RuntimeError(
            "strict_deterministic could not resolve the loaded model's attention "
            "implementation; refusing an unauditable deterministic run."
        )
    deterministic_algorithms = bool(
        proof.get("local_torch_deterministic_algorithms_enabled", 0)
    )
    warn_only = bool(proof.get("local_torch_deterministic_warn_only", 0))
    result = dict(proof)
    result.update(
        {
            "local_attention_kernel_requested": str(
                requested_attention_kernel or "default"
            ),
            "local_attention_implementation_resolved": resolved or "unknown",
            "local_sdpa_strict_torch_guard_enabled": int(
                "sdpa" in resolved.lower()
                and deterministic_algorithms
                and not warn_only
            ),
        }
    )
    return result


def _enable_and_verify(
    torch: _TorchLike,
    environment: Mapping[str, object],
) -> None:
    global _STRICT_SETUP_ESTABLISHED_BEFORE_CUDA

    cuda_initialized = bool(torch.cuda.is_initialized())
    already_established = _torch_controls_established(torch)
    workspace_was_present = bool(
        environment.get("local_cublas_workspace_config_preconfigured", 0)
    )
    if cuda_initialized and not (
        already_established
        and _STRICT_SETUP_ESTABLISHED_BEFORE_CUDA
        and workspace_was_present
    ):
        raise RuntimeError(
            "strict_deterministic must be established before CUDA is initialized; "
            "restart the process and enable [backend.options] "
            "strict_deterministic = true before constructing a local backend."
        )

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not _torch_controls_established(torch):
        raise RuntimeError(
            "strict_deterministic requested, but PyTorch deterministic controls "
            "did not remain enabled."
        )
    workspace = os.environ.get(_CUBLAS_WORKSPACE_ENV, "")
    if workspace not in _VALID_CUBLAS_WORKSPACES:
        raise RuntimeError(
            "strict_deterministic lost its deterministic cuBLAS workspace "
            f"configuration: {workspace!r}."
        )
    if not cuda_initialized:
        _STRICT_SETUP_ESTABLISHED_BEFORE_CUDA = True


def _torch_controls_established(torch: _TorchLike) -> bool:
    return (
        bool(torch.are_deterministic_algorithms_enabled())
        and not bool(torch.is_deterministic_algorithms_warn_only_enabled())
        and bool(torch.backends.cudnn.deterministic)
        and not bool(torch.backends.cudnn.benchmark)
    )


def _runtime_proof(torch: _TorchLike, *, requested: bool) -> dict[str, object]:
    established = _torch_controls_established(torch)
    return {
        "local_strict_deterministic_requested": int(requested),
        "local_strict_deterministic_established": int(requested and established),
        "local_torch_version": str(torch.__version__),
        "local_torch_deterministic_algorithms_enabled": int(
            torch.are_deterministic_algorithms_enabled()
        ),
        "local_torch_deterministic_warn_only": int(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "local_cudnn_deterministic": int(torch.backends.cudnn.deterministic),
        "local_cudnn_benchmark": int(torch.backends.cudnn.benchmark),
        "local_strict_deterministic_setup_before_cuda": int(
            requested and _STRICT_SETUP_ESTABLISHED_BEFORE_CUDA
        ),
        # PyTorch's guard cannot attest arbitrary Triton/custom kernels.
        "local_third_party_kernel_determinism_guaranteed": 0,
        "local_two_run_adapter_hash_canary_required": 1,
    }


def _resolved_attention_implementation(model: object) -> str:
    config = getattr(model, "config", None)
    candidates = (
        getattr(config, "_attn_implementation", None),
        getattr(getattr(config, "text_config", None), "_attn_implementation", None),
    )
    for value in candidates:
        if isinstance(value, str) and value:
            return value
        if isinstance(value, Mapping) and value:
            return json.dumps(
                {str(key): nested for key, nested in value.items()},
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
    return ""
