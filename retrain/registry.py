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
from typing import Any, Callable

from retrain.config import TrainConfig


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------

class Registry:
    """Dict-based registry for a single component type.

    Stores *lazy factory functions* ``(config: TrainConfig) -> object`` keyed
    by short name (e.g. ``"local"``, ``"match"``).
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[[TrainConfig], Any]] = {}

    # -- public API --------------------------------------------------------

    def register(self, name: str) -> Callable:
        """Decorator to register a lazy factory under *name*."""
        def decorator(fn: Callable[[TrainConfig], Any]) -> Callable[[TrainConfig], Any]:
            self._factories[name] = fn
            return fn
        return decorator

    def create(self, name: str, config: TrainConfig) -> Any:
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
    def _import_dotted(dotted: str, config: TrainConfig) -> Any:
        """Import ``module.attr`` and call ``attr(config)``."""
        module_path, _, attr_name = dotted.rpartition(".")
        if not module_path or not attr_name:
            raise ValueError(
                f"Invalid dotted import path '{dotted}'. "
                f"Expected format: 'mypackage.module.ClassName'."
            )
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"Could not import module '{module_path}' "
                f"for dotted path '{dotted}': {exc}"
            ) from exc
        cls = getattr(mod, attr_name, None)
        if cls is None:
            raise AttributeError(
                f"Module '{module_path}' has no attribute '{attr_name}'."
            )
        return cls(config)


# ---------------------------------------------------------------------------
# Six global registries
# ---------------------------------------------------------------------------

backend = Registry("backend")
inference_engine = Registry("inference_engine")
reward = Registry("reward")
planning_detector = Registry("planning_detector")
data_source = Registry("data_source")
backpressure = Registry("backpressure")

_ALL_REGISTRIES: dict[str, Registry] = {
    "backend": backend,
    "inference_engine": inference_engine,
    "reward": reward,
    "planning_detector": planning_detector,
    "data_source": data_source,
    "backpressure": backpressure,
}


def get_registry(name: str) -> Registry:
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

@backend.register("local")
def _create_local(config: TrainConfig) -> Any:
    try:
        from retrain.local_train_helper import LocalTrainHelper
    except ImportError:
        raise RuntimeError(
            "Backend 'local' requires PyTorch.\n"
            "Install it with: pip install retrain[local]"
        ) from None
    return LocalTrainHelper(
        config.model,
        config.adapter_path,
        config.devices,
        config.lora_rank,
        config.inference_engine,
        config.inference_url,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        optim_beta1=config.optim_beta1,
        optim_beta2=config.optim_beta2,
        optim_eps=config.optim_eps,
    )


@backend.register("tinker")
def _create_tinker(config: TrainConfig) -> Any:
    try:
        from retrain.tinker_backend import TinkerTrainHelper
    except ImportError:
        raise RuntimeError(
            "Backend 'tinker' requires the tinker SDK.\n"
            "Install it with: pip install retrain[tinker]"
        ) from None
    # Prefer [inference].url, but keep [model].base_url as backward-compatible fallback.
    tinker_url = config.inference_url or config.base_url
    return TinkerTrainHelper(
        config.model,
        tinker_url,
        config.lora_rank,
        optim_beta1=config.optim_beta1,
        optim_beta2=config.optim_beta2,
        optim_eps=config.optim_eps,
    )


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

@inference_engine.register("openai")
def _ie_openai(config: TrainConfig) -> None:
    pass


# -- reward ----------------------------------------------------------------

@reward.register("match")
def _reward_match(config: TrainConfig) -> Any:
    from retrain.rewards import BoxedMathReward
    return BoxedMathReward()


@reward.register("math")
def _reward_math(config: TrainConfig) -> Any:
    try:
        from retrain.rewards import VerifiersMathReward
        return VerifiersMathReward()
    except ImportError:
        raise ImportError(
            "Reward type 'math' requires the verifiers library.\n"
            "Install it with: pip install retrain[verifiers]"
        ) from None


@reward.register("judge")
def _reward_judge(config: TrainConfig) -> Any:
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
def _reward_custom(config: TrainConfig) -> Any:
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
def _detector_regex(config: TrainConfig) -> Any:
    from retrain.planning import create_planning_detector
    # Reuse existing factory for regex — it handles strategic_grams parsing
    return create_planning_detector(config)


@planning_detector.register("semantic")
def _detector_semantic(config: TrainConfig) -> Any:
    from retrain.planning import SemanticPlanningDetector
    return SemanticPlanningDetector(
        model_name=config.planning_model,
        threshold=config.planning_threshold,
    )


# -- data_source -----------------------------------------------------------

@data_source.register("math")
def _data_math(config: TrainConfig) -> Any:
    from retrain.data import MathDataSource
    return MathDataSource(config.max_examples)


# -- backpressure ----------------------------------------------------------

@backpressure.register("noop")
def _bp_noop(config: TrainConfig) -> Any:
    from retrain.backpressure import NoOpBackPressure
    return NoOpBackPressure()


@backpressure.register("usl")
def _bp_usl(config: TrainConfig) -> Any:
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


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

_DEPENDENCY_MAP: dict[str, tuple[str, str]] = {
    "local": ("torch", "pip install retrain[local]"),
    "tinker": ("tinker", "pip install retrain[tinker]"),
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
