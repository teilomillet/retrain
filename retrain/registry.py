"""Plug-and-play component registry for retrain.

Each component type (backend, inference_engine, reward, planning_detector,
data_source, backpressure) has its own Registry instance.  Built-in
implementations are registered via lazy factory functions so heavy imports
(torch, sentence-transformers, …) only happen when actually used.

Unknown names containing a ``.`` are treated as dotted-path imports
(``mypackage.module.MyClass``), enabling third-party plugins with zero
boilerplate.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Callable, Generic, Literal, TypeVar, cast, overload

from retrain.backends import TrainHelper
from retrain.backend_definitions import (
    get_backend_dependency_map,
    get_builtin_backend_definitions,
)
from retrain.backpressure import BackPressure
from retrain.config import TrainConfig
from retrain.data import DataSource
from retrain.plugin_resolver import resolve_dotted_attribute
from retrain.planning import PlanningDetector
from retrain.rewards import RewardFunction
from retrain.training_runner import TrainingRunner


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------

class Registry(Generic[T]):
    """Dict-based registry for a single component type.

    Stores *lazy factory functions* ``(config: TrainConfig) -> object`` keyed
    by short name (e.g. ``"local"``, ``"match"``).
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[[TrainConfig], T]] = {}

    # -- public API --------------------------------------------------------

    def register(
        self, name: str
    ) -> Callable[[Callable[[TrainConfig], T]], Callable[[TrainConfig], T]]:
        """Decorator to register a lazy factory under *name*."""
        def decorator(
            fn: Callable[[TrainConfig], T],
        ) -> Callable[[TrainConfig], T]:
            self._factories[name] = fn
            return fn
        return decorator

    def create(self, name: str, config: TrainConfig) -> T:
        """Look up *name* and call its factory with *config*.

        Falls back to dotted-path import when *name* contains a ``.``.
        """
        factory = self._factories.get(name)
        if factory is not None:
            return factory(config)

        # Dotted-path fallback: ``mypackage.module.ClassName``
        if "." in name:
            return self._import_dotted(name, config)

        available = sorted(self._factories)
        raise ValueError(
            f"Unknown {self.kind} '{name}'. "
            f"Built-in options: {available}. "
            f"For a third-party plugin, use a dotted import path "
            f"(e.g. 'mypackage.MyClass')."
        )

    @property
    def builtin_names(self) -> list[str]:
        """Sorted list of registered built-in names."""
        return sorted(self._factories)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _import_dotted(dotted: str, config: TrainConfig) -> T:
        """Import ``module.attr`` and call ``attr(config)``."""
        resolved = resolve_dotted_attribute(
            dotted,
            selector="registry component",
            expected="a callable factory(config)",
        )
        factory = resolved.obj
        if not callable(factory):
            raise TypeError(
                f"Dotted import target '{dotted}' is not callable."
            )
        typed_factory = cast(Callable[[TrainConfig], T], factory)
        return typed_factory(config)


# ---------------------------------------------------------------------------
# Six global registries
# ---------------------------------------------------------------------------

backend = Registry[TrainHelper]("backend")
inference_engine = Registry[None]("inference_engine")
reward = Registry[RewardFunction]("reward")
planning_detector = Registry[PlanningDetector]("planning_detector")
data_source = Registry[DataSource]("data_source")
backpressure = Registry[BackPressure]("backpressure")
trainer = Registry[TrainingRunner]("trainer")

KnownRegistry = (
    Registry[TrainHelper]
    | Registry[None]
    | Registry[RewardFunction]
    | Registry[PlanningDetector]
    | Registry[DataSource]
    | Registry[BackPressure]
    | Registry[TrainingRunner]
)

_ALL_REGISTRIES: dict[str, KnownRegistry] = {
    "backend": backend,
    "inference_engine": inference_engine,
    "reward": reward,
    "planning_detector": planning_detector,
    "data_source": data_source,
    "backpressure": backpressure,
    "trainer": trainer,
}


@overload
def get_registry(name: Literal["backend"]) -> Registry[TrainHelper]: ...


@overload
def get_registry(name: Literal["inference_engine"]) -> Registry[None]: ...


@overload
def get_registry(name: Literal["reward"]) -> Registry[RewardFunction]: ...


@overload
def get_registry(name: Literal["planning_detector"]) -> Registry[PlanningDetector]: ...


@overload
def get_registry(name: Literal["data_source"]) -> Registry[DataSource]: ...


@overload
def get_registry(name: Literal["backpressure"]) -> Registry[BackPressure]: ...


@overload
def get_registry(name: Literal["trainer"]) -> Registry[TrainingRunner]: ...


def get_registry(name: str) -> KnownRegistry:
    """Return the named registry, or raise ``KeyError``."""
    try:
        return _ALL_REGISTRIES[name]
    except KeyError:
        raise KeyError(
            f"No registry '{name}'. "
            f"Available: {sorted(_ALL_REGISTRIES)}"
        ) from None


# ---------------------------------------------------------------------------
# Built-in registrations (lazy factories — heavy imports deferred)
# ---------------------------------------------------------------------------

# -- backend ---------------------------------------------------------------

for _backend_name, _backend_definition in get_builtin_backend_definitions().items():
    backend.register(_backend_name)(_backend_definition.factory)


# -- inference_engine (for doctor diagnostics) -----------------------------

@inference_engine.register("pytorch")
def _ie_pytorch(config: TrainConfig) -> None:
    pass

@inference_engine.register("max")
def _ie_max(config: TrainConfig) -> None:
    pass

@inference_engine.register("vllm")
def _ie_vllm(config: TrainConfig) -> None:
    pass

@inference_engine.register("sglang")
def _ie_sglang(config: TrainConfig) -> None:
    pass

@inference_engine.register("mlx")
def _ie_mlx(config: TrainConfig) -> None:
    pass

@inference_engine.register("openai")
def _ie_openai(config: TrainConfig) -> None:
    pass


# -- reward ----------------------------------------------------------------

@reward.register("match")
def _reward_match(config: TrainConfig) -> RewardFunction:
    from retrain.rewards import BoxedMathReward
    return BoxedMathReward()


@reward.register("math")
def _reward_math(config: TrainConfig) -> RewardFunction:
    try:
        from retrain.rewards import VerifiersMathReward
        return VerifiersMathReward()
    except ImportError:
        raise ImportError(
            "Reward type 'math' requires the verifiers library.\n"
            "Install it with: pip install retrain[verifiers]"
        ) from None


@reward.register("judge")
def _reward_judge(config: TrainConfig) -> RewardFunction:
    try:
        from retrain.rewards import VerifiersJudgeReward
        model = config.reward_judge_model or "gpt-4o-mini"
        return VerifiersJudgeReward(model=model)
    except ImportError:
        raise ImportError(
            "Reward type 'judge' requires the verifiers library.\n"
            "Install it with: pip install retrain[verifiers]"
        ) from None


@reward.register("custom")
def _reward_custom(config: TrainConfig) -> RewardFunction:
    from retrain.rewards import CustomReward
    if not config.reward_custom_module:
        raise ValueError(
            "Reward type 'custom' requires [reward] custom_module to be set."
        )
    return CustomReward(
        config.reward_custom_module,
        config.reward_custom_function or "score",
    )


# -- planning_detector ----------------------------------------------------

@planning_detector.register("regex")
def _detector_regex(config: TrainConfig) -> PlanningDetector:
    from retrain.planning import create_planning_detector
    # Reuse existing factory for regex — it handles strategic_grams parsing
    return create_planning_detector(config)


@planning_detector.register("semantic")
def _detector_semantic(config: TrainConfig) -> PlanningDetector:
    from retrain.planning import SemanticPlanningDetector
    return SemanticPlanningDetector(
        model_name=config.planning_model,
        threshold=config.planning_threshold,
    )


# -- data_source -----------------------------------------------------------

@data_source.register("math")
def _data_math(config: TrainConfig) -> DataSource:
    from retrain.data import MathDataSource
    return MathDataSource(config.max_examples)


# -- backpressure ----------------------------------------------------------

@backpressure.register("noop")
def _bp_noop(config: TrainConfig) -> BackPressure:
    from retrain.backpressure import NoOpBackPressure
    return NoOpBackPressure()


@backpressure.register("usl")
def _bp_usl(config: TrainConfig) -> BackPressure:
    from retrain.backpressure import USLBackPressure
    return USLBackPressure(
        warmup_steps=config.bp_warmup_steps,
        ema_decay=config.bp_ema_decay,
        throttle_margin=config.bp_throttle_margin,
        increase_margin=config.bp_increase_margin,
        min_batch_size=config.bp_min_batch_size,
        max_batch_size=config.bp_max_batch_size,
        peak_gflops=config.bp_peak_gflops,
        peak_bw_gb_s=config.bp_peak_bw_gb_s,
    )


# -- trainer ---------------------------------------------------------------

@trainer.register("retrain")
def _trainer_retrain(config: TrainConfig) -> TrainingRunner:
    from retrain.training_runner import RetainRunner
    return RetainRunner()


@trainer.register("command")
def _trainer_command(config: TrainConfig) -> TrainingRunner:
    from retrain.training_runner import CommandRunner
    if not config.trainer_command:
        raise ValueError(
            "trainer='command' requires [training] trainer_command to be set."
        )
    return CommandRunner(config.trainer_command)


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

_DEPENDENCY_MAP: dict[str, tuple[str, str]] = {
    **get_backend_dependency_map(),
    "math": ("math_verify", "pip install retrain[verifiers]"),
    "judge": ("verifiers", "pip install retrain[verifiers]"),
    "semantic": ("sentence_transformers", "pip install retrain[semantic]"),
    "verifiers_env": ("verifiers", "pip install retrain[verifiers]"),
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
