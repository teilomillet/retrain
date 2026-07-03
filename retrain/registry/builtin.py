"""The global registries and their built-in factories.

Built-ins are registered via lazy factory functions so heavy imports
(torch, sentence-transformers, ...) only happen when actually used.
"""

from __future__ import annotations

from typing import Literal, overload

from retrain.backends import TrainHelper
from retrain.backends.catalog import get_builtin_backend_definitions
from retrain.config import TrainConfig
from retrain.data.source import DataSource
from retrain.planning.types import PlanningDetector
from retrain.registry.core import Registry
from retrain.rewards.types import RewardFunction
from retrain.training.backpressure import BackPressure
from retrain.training.runner import TrainingRunner


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

def _noop_inference_engine(config: TrainConfig) -> None:
    """Reserve inference-engine names; construction lives in retrain.inference_engine."""


for _engine_name in ("pytorch", "max", "vllm", "sglang", "trtllm", "mlx", "openai"):
    inference_engine.register(_engine_name)(_noop_inference_engine)


# -- reward ----------------------------------------------------------------

@reward.register("match")
def _reward_match(config: TrainConfig) -> RewardFunction:
    from retrain.rewards.boxed import BoxedMathReward
    return BoxedMathReward()


@reward.register("math")
def _reward_math(config: TrainConfig) -> RewardFunction:
    try:
        from retrain.rewards.verifiers import VerifiersMathReward
        return VerifiersMathReward()
    except ImportError:
        raise ImportError(
            "Reward type 'math' requires the verifiers library.\n"
            "Install it with: pip install retrain[verifiers]"
        ) from None


@reward.register("judge")
def _reward_judge(config: TrainConfig) -> RewardFunction:
    try:
        from retrain.rewards.verifiers import VerifiersJudgeReward
        model = config.reward_judge_model or "gpt-4o-mini"
        return VerifiersJudgeReward(model=model)
    except ImportError:
        raise ImportError(
            "Reward type 'judge' requires the verifiers library.\n"
            "Install it with: pip install retrain[verifiers]"
        ) from None


@reward.register("custom")
def _reward_custom(config: TrainConfig) -> RewardFunction:
    from retrain.rewards.custom import CustomReward
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
    from retrain.planning.create import create_planning_detector
    # Reuse existing factory for regex — it handles strategic_grams parsing
    return create_planning_detector(config)


@planning_detector.register("semantic")
def _detector_semantic(config: TrainConfig) -> PlanningDetector:
    from retrain.planning.semantic import SemanticPlanningDetector
    return SemanticPlanningDetector(
        model_name=config.planning_model,
        threshold=config.planning_threshold,
    )


# -- data_source -----------------------------------------------------------

@data_source.register("math")
def _data_math(config: TrainConfig) -> DataSource:
    from retrain.data.math import MathDataSource
    return MathDataSource(config.max_examples)


# -- backpressure ----------------------------------------------------------

@backpressure.register("noop")
def _bp_noop(config: TrainConfig) -> BackPressure:
    from retrain.training.backpressure import NoOpBackPressure
    return NoOpBackPressure()


@backpressure.register("usl")
def _bp_usl(config: TrainConfig) -> BackPressure:
    from retrain.training.backpressure import USLBackPressure
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
    from retrain.training.runner import RetainRunner
    return RetainRunner()


@trainer.register("sft")
def _trainer_sft(config: TrainConfig) -> TrainingRunner:
    from retrain.training.runner import SftRunner
    return SftRunner()


@trainer.register("command")
def _trainer_command(config: TrainConfig) -> TrainingRunner:
    from retrain.training.runner import CommandRunner
    if not config.trainer_command:
        raise ValueError(
            "trainer='command' requires [training] trainer_command to be set."
        )
    return CommandRunner(config.trainer_command)


@trainer.register("ttt_discover")
def _trainer_ttt_discover(config: TrainConfig) -> TrainingRunner:
    from retrain.training.discover import TTTDiscoverRunner
    return TTTDiscoverRunner()


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------
