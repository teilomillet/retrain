"""Dependency checks and runtime probes for `retrain doctor`."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

from retrain.backends.catalog import get_backend_dependency_map
from retrain.config import TrainConfig


_DEPENDENCY_MAP: dict[str, tuple[str, str]] = {
    **get_backend_dependency_map(),
    "math": ("math_verify", "pip install retrain[verifiers]"),
    "judge": ("verifiers", "pip install retrain[verifiers]"),
    "semantic": ("sentence_transformers", "pip install retrain[semantic]"),
    "verifiers_env": ("verifiers", "pip install retrain[verifiers]"),
    "openenv_env": ("websockets", "pip install retrain[openenv]"),
}


def check_environment(
    config: TrainConfig | None = None,
) -> list[tuple[str, str, str, bool]]:
    """Check whether dependencies for configured (or all) components are importable.

    Returns a list of ``(component_name, import_name, install_hint, available)``
    tuples.  When *config* is ``None``, checks **all** known dependencies
    (``retrain doctor`` mode).  When *config* is given, checks only the
    components actually referenced by the config.
    """
    if config is None:
        names_to_check = list(_DEPENDENCY_MAP)
    else:
        names_to_check = []
        for name in (config.backend, config.reward_type, config.planning_detector):
            if name in _DEPENDENCY_MAP:
                names_to_check.append(name)
        if config.environment_provider == "verifiers":
            names_to_check.append("verifiers_env")
        if config.environment_provider == "openenv":
            names_to_check.append("openenv_env")

    results: list[tuple[str, str, str, bool]] = []
    for name in names_to_check:
        import_name, hint = _DEPENDENCY_MAP[name]
        try:
            importlib.import_module(import_name)
            available = True
        except ImportError:
            available = False
        results.append((name, import_name, hint, available))

    return results


@dataclass(frozen=True)
class BackendRuntimeProbe:
    """Runtime probe result for doctor diagnostics."""

    backend: str
    probe: str
    status: str  # ok | fail | skip
    detail: str


def _probe_http_endpoint(
    base_url: str,
    paths: tuple[str, ...],
    timeout_s: float = 0.8,
) -> BackendRuntimeProbe:
    """Probe an HTTP endpoint quickly; returns ok/fail with details."""
    try:
        import requests
    except ImportError:
        return BackendRuntimeProbe(
            backend="",
            probe="http",
            status="fail",
            detail="requests not installed",
        )

    clean_base = base_url.rstrip("/")
    last_err = "no response"
    for path in paths:
        url = f"{clean_base}{path}"
        try:
            resp = requests.get(url, timeout=timeout_s)
        except Exception as exc:
            last_err = f"{url} ({type(exc).__name__}: {exc})"
            continue
        code = int(resp.status_code)
        if code in {200, 204, 405}:
            return BackendRuntimeProbe(
                backend="",
                probe="http",
                status="ok",
                detail=f"{url} status={code}",
            )
        if code == 404:
            last_err = f"{url} status=404"
            continue
        if code < 500:
            return BackendRuntimeProbe(
                backend="",
                probe="http",
                status="ok",
                detail=f"{url} status={code}",
            )
        last_err = f"{url} status={code}"
    return BackendRuntimeProbe(
        backend="",
        probe="http",
        status="fail",
        detail=last_err,
    )


def probe_backend_runtime(config: TrainConfig | None = None) -> list[BackendRuntimeProbe]:
    """Run lightweight runtime probes for built-in backends."""
    probes: list[BackendRuntimeProbe] = []

    # Local backend: torch import + CUDA visibility.
    try:
        torch = importlib.import_module("torch")
        cuda_ok = bool(torch.cuda.is_available())
        probes.append(
            BackendRuntimeProbe(
                backend="local",
                probe="torch_runtime",
                status="ok",
                detail=f"torch import ok, cuda_available={cuda_ok}",
            )
        )
    except Exception as exc:
        probes.append(
            BackendRuntimeProbe(
                backend="local",
                probe="torch_runtime",
                status="fail",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )

    # Unsloth backend: optional import plus torch runtime visibility.
    try:
        unsloth = importlib.import_module("unsloth")
        from retrain.backends.unsloth import validate_fast_language_model_api

        validate_fast_language_model_api(unsloth.FastLanguageModel)
        torch = importlib.import_module("torch")
        cuda_ok = bool(torch.cuda.is_available())
        probes.append(
            BackendRuntimeProbe(
                backend="unsloth",
                probe="unsloth_runtime",
                status="ok",
                detail=(
                    "unsloth import ok, FastLanguageModel API ok, "
                    f"cuda_available={cuda_ok}"
                ),
            )
        )
    except Exception as exc:
        probes.append(
            BackendRuntimeProbe(
                backend="unsloth",
                probe="unsloth_runtime",
                status="fail",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )

    # Tinker backend: SDK import + optional endpoint reachability.
    try:
        importlib.import_module("tinker")
        tinker_url = ""
        if config is not None:
            tinker_url = config.inference_url or config.base_url
        tinker_url = tinker_url or os.getenv("RETRAIN_TINKER_URL", "")
        if not tinker_url:
            probes.append(
                BackendRuntimeProbe(
                    backend="tinker",
                    probe="service_reachability",
                    status="skip",
                    detail=(
                        "set RETRAIN_TINKER_URL (or [model].base_url/[inference].url) "
                        "to enable endpoint probing"
                    ),
                )
            )
        else:
            hit = _probe_http_endpoint(tinker_url, ("/health", "/"))
            probes.append(
                BackendRuntimeProbe(
                    backend="tinker",
                    probe="service_reachability",
                    status=hit.status,
                    detail=hit.detail,
                )
            )
    except Exception as exc:
        probes.append(
            BackendRuntimeProbe(
                backend="tinker",
                probe="service_reachability",
                status="fail",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )

    # PRIME-RL backend: import + endpoint compatibility probe.
    try:
        importlib.import_module("prime_rl")
        prime_url = ""
        if config is not None:
            prime_url = config.inference_url or config.base_url
        prime_url = prime_url or os.getenv("RETRAIN_PRIME_RL_URL", "http://localhost:8000")
        hit = _probe_http_endpoint(prime_url, ("/health", "/v1/models", "/"))
        probes.append(
            BackendRuntimeProbe(
                backend="prime_rl",
                probe="endpoint_compat",
                status=hit.status,
                detail=hit.detail,
            )
        )
    except Exception as exc:
        probes.append(
            BackendRuntimeProbe(
                backend="prime_rl",
                probe="endpoint_compat",
                status="fail",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )

    return probes
