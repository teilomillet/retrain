"""Advantage pipeline context and spec types."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from retrain.advantages.credit import (
    apply_gtpo_weighting,
    apply_hicra,
    apply_sepa_amplification,
    apply_sepa_pooling,
)

# EntropyStats
# ---------------------------------------------------------------------------

@dataclass
class EntropyStats:
    """Summary stats for execution vs planning token-surprisal distributions.

    Note:
    Field names retain `entropy` for backward compatibility with existing logs
    and dashboards. Values are sampled-token surprisal (-logprob), not full
    Shannon entropy.
    """
    exec_mean: float = 0.0
    exec_var: float = 0.0
    exec_count: float = 0.0
    plan_mean: float = 0.0
    plan_var: float = 0.0
    plan_count: float = 0.0
    # Post-transform stats (after SEPA/entropy transform, before GTPO weighting)
    post_exec_mean: float = 0.0
    post_exec_var: float = 0.0
    post_exec_count: float = 0.0
    post_plan_mean: float = 0.0
    post_plan_var: float = 0.0
    post_plan_count: float = 0.0


@dataclass
class AdvantageResult:
    """Result of composable advantage computation."""
    token_advs: list[list[float]] = field(default_factory=list)
    has_stats: bool = False
    stats: EntropyStats = field(default_factory=EntropyStats)
    extra_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AdvantageContext:
    """Context passed to context-style advantage plugins."""

    rewards: list[float]
    params: Mapping[str, object] = field(default_factory=dict)
    step: int = 0


@dataclass
class AdvantageOutput:
    """Output container for context-style advantage plugins."""

    advantages: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class TransformContext:
    """Context passed to context-style transform plugins."""

    episode_advantages: list[float]
    logprobs_G: list[list[float]]
    planning_masks_G: list[list[int]]
    sepa_lambda: float = 0.0
    params: Mapping[str, object] = field(default_factory=dict)
    step: int = 0
    gtpo_beta: float = 0.1
    hicra_alpha: float = 0.2
    token_distributions_G: list[list[list[float]]] | None = None

    def gtpo(self, advantage: float, surprisals: list[float], beta: float | None = None) -> list[float]:
        """Utility helper exposed to transform plugins."""
        return apply_gtpo_weighting(
            advantage,
            surprisals,
            beta=self.gtpo_beta if beta is None else beta,
        )

    def hicra(self, token_advs: list[float], planning_mask: list[int], alpha: float | None = None) -> list[float]:
        """Utility helper exposed to transform plugins."""
        return apply_hicra(
            token_advs,
            planning_mask,
            alpha=self.hicra_alpha if alpha is None else alpha,
        )

    def sepa_pool(self, surprisals: list[float], planning_mask: list[int], lambda_t: float | None = None) -> list[float]:
        """Utility helper exposed to transform plugins."""
        return apply_sepa_pooling(
            surprisals,
            planning_mask,
            self.sepa_lambda if lambda_t is None else lambda_t,
        )

    def sepa_amp(self, surprisals: list[float], planning_mask: list[int], lambda_t: float | None = None) -> list[float]:
        """Utility helper exposed to transform plugins."""
        return apply_sepa_amplification(
            surprisals,
            planning_mask,
            self.sepa_lambda if lambda_t is None else lambda_t,
        )


@dataclass
class TransformOutput:
    """Output container for context-style transform plugins."""

    token_advs: list[list[float]] = field(default_factory=list)
    has_stats: bool = False
    stats: EntropyStats = field(default_factory=EntropyStats)
    extra_metrics: dict[str, float] = field(default_factory=dict)
    needs_planning: bool = False
    uses_sepa_controller: bool = False


@dataclass(frozen=True)
class AlgorithmContext:
    """Context passed to full algorithm plugins."""

    rewards_G: list[float]
    logprobs_G: list[list[float]]
    planning_masks_G: list[list[int]]
    params: Mapping[str, object] = field(default_factory=dict)
    step: int = 0
    sepa_lambda: float = 0.0
    gtpo_beta: float = 0.1
    hicra_alpha: float = 0.2
    token_distributions_G: list[list[list[float]]] | None = None
    precomputed_entropies_G: list[list[float]] | None = None


@dataclass
class AlgorithmOutput:
    """Output container for full algorithm plugins."""

    token_advs: list[list[float]] = field(default_factory=list)
    has_stats: bool = False
    stats: EntropyStats = field(default_factory=EntropyStats)
    extra_metrics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Uncertainty specs (pluggable token-uncertainty signals)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UncertaintyContext:
    """Input context for uncertainty compute functions."""

    logprobs: list[float]
    token_distributions: list[list[float]] | None = None
    precomputed_entropy: list[float] | None = None
    planning_mask: list[int] | None = None
    params: Mapping[str, object] = field(default_factory=dict)


UncertaintyComputeFn = Callable[[UncertaintyContext], list[float]]


@dataclass(frozen=True)
class UncertaintySpec:
    """Behavioral contract for uncertainty signals."""

    name: str
    compute: UncertaintyComputeFn
    needs_distributions: bool = False


# ---------------------------------------------------------------------------
# Transform specs (TOML-selectable, plugin-friendly)
# ---------------------------------------------------------------------------

EntropyTransformFn = Callable[[list[float], list[int], float], list[float]]
EpisodeAdvantageFn = Callable[[list[float]], list[float]]
EpisodeAdvantageFnWithParams = Callable[
    [list[float], Mapping[str, object]],
    list[float],
]
EpisodeAdvantageComputeFn = EpisodeAdvantageFn | EpisodeAdvantageFnWithParams
EpisodeAdvantageRunner = Callable[
    [list[float], Mapping[str, object]],
    list[float],
]
TransformContextFn = Callable[
    [TransformContext],
    TransformOutput | AdvantageResult | list[list[float]],
]
AlgorithmContextFn = Callable[
    [AlgorithmContext],
    AlgorithmOutput | AdvantageResult | list[list[float]],
]

PostProcessFn = Callable[
    [list[list[float]], list[list[float]], Mapping[str, object]],
    tuple[list[list[float]], dict[str, float]],
]


@dataclass(frozen=True)
class TransformSpec:
    """Behavioral contract for transform modes.

    `entropy_transform` can be overridden by custom plugins loaded from
    dotted-path names in TOML (e.g. `my_module.make_transform_spec`).
    `post_process` runs after GTPO weighting + HICRA, receives all token
    advantages and raw entropies, returns modified advantages and metrics.
    """

    name: str
    use_gtpo: bool = True
    needs_planning: bool = False
    uses_sepa_controller: bool = False
    apply_hicra: bool = False
    entropy_transform: EntropyTransformFn | None = None
    post_process: PostProcessFn | None = None
    compute_context: TransformContextFn | None = None


# ---------------------------------------------------------------------------
# Advantage specs (built-ins + dotted-path plugins)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdvantageSpec:
    """Behavioral contract for episode-level advantage modes."""

    name: str
    compute: EpisodeAdvantageRunner


@dataclass(frozen=True)
class AlgorithmSpec:
    """Behavioral contract for full algorithm plugins."""

    name: str
    compute: AlgorithmContextFn
    needs_planning: bool = False
    uses_sepa_controller: bool = False
