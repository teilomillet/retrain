"""Advantage computation: GRPO, MaxRL, GTPO, HICRA, SEPA + planning tokens.

Ports the core functions from src/advantages.mojo into pure Python.
"""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Callable

from retrain.plugin_resolver import get_plugin_runtime, resolve_dotted_attribute

# Cap token surprisal values to prevent inf from poisoning downstream math.
# Real per-token surprisal (-logprob of sampled token) rarely exceeds ~15;
# 50 is a safe upper bound.
MAX_SURPRISAL = 50.0
MAX_ENTROPY = MAX_SURPRISAL  # backward-compat alias

_UNCERTAINTY_KIND_ALIASES = {
    "surprisal": "surprisal",
    "token_surprisal": "surprisal",
    "neg_logprob": "surprisal",
    "negative_logprob": "surprisal",
    "nll": "surprisal",
    "shannon": "shannon_entropy",
    "entropy": "shannon_entropy",
    "token_entropy": "shannon_entropy",
    "shannon_entropy": "shannon_entropy",
    "varentropy": "varentropy",
    "predictive_variance": "predictive_variance",
    "pred_var": "predictive_variance",
    "bernoulli_variance": "predictive_variance",
}
_UNCERTAINTY_KIND_PARAM_KEYS = (
    "uncertainty_kind",
    "uncertainty_metric",
    "token_uncertainty",
)


# ---------------------------------------------------------------------------
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


def _normalize_advantage_compute(
    compute: EpisodeAdvantageComputeFn,
) -> EpisodeAdvantageRunner:
    """Adapt a plugin function to the internal `(rewards, params)` signature."""
    try:
        sig = inspect.signature(compute)
    except (TypeError, ValueError):
        return lambda rewards, _params: compute(rewards)

    params = sig.parameters
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) == 1 and positional[0].name in {"ctx", "context"}:
        def _run_ctx_style(
            rewards: list[float], params_map: Mapping[str, object]
        ) -> list[float]:
            out = compute(AdvantageContext(rewards=rewards, params=params_map))
            if isinstance(out, AdvantageOutput):
                return out.advantages
            return out  # type: ignore[return-value]

        return _run_ctx_style

    params_arg = params.get("params")
    accepts_varkw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    if params_arg is not None and params_arg.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        if params_arg.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            return lambda rewards, params_map: compute(rewards, params_map)
        return lambda rewards, params_map: compute(rewards, params=params_map)

    if accepts_varkw:
        return lambda rewards, params_map: compute(rewards, params=params_map)

    return lambda rewards, _params: compute(rewards)


def _coerce_advantages_output(
    raw_advantages: object, expected_len: int, mode_name: str
) -> list[float]:
    """Normalize and validate episode-level advantage output shape."""
    try:
        advantages = [float(v) for v in raw_advantages]  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"advantage_mode '{mode_name}' must return an iterable of floats."
        ) from exc

    if len(advantages) != expected_len:
        raise ValueError(
            f"advantage_mode '{mode_name}' returned {len(advantages)} values, "
            f"expected {expected_len}."
        )
    for idx, value in enumerate(advantages):
        if not math.isfinite(value):
            raise ValueError(
                f"advantage_mode '{mode_name}' returned non-finite value "
                f"at index {idx}: {value!r}"
            )
    return advantages


def _as_advantage_spec(
    name: str, compute: EpisodeAdvantageComputeFn
) -> AdvantageSpec:
    """Build an AdvantageSpec from a one-arg or two-arg callable."""
    return AdvantageSpec(name=name, compute=_normalize_advantage_compute(compute))


def _callable_takes_no_positional_args(obj: object) -> bool:
    """Return True when callable is likely a zero-arg factory."""
    if not callable(obj):
        return False
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return False
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    accepts_varargs = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )
    return not positional and not accepts_varargs


def _validate_token_advs(
    token_advs: list[list[float]],
    *,
    expected_lens: list[int],
    mode_name: str,
) -> None:
    """Validate shape and numeric sanity for token-level advantages."""
    if len(token_advs) != len(expected_lens):
        raise ValueError(
            f"{mode_name} returned {len(token_advs)} sequences, "
            f"expected {len(expected_lens)}."
        )
    for seq_idx, (seq, expected_len) in enumerate(zip(token_advs, expected_lens)):
        if len(seq) != expected_len:
            raise ValueError(
                f"{mode_name} returned {len(seq)} token advantages for sequence "
                f"{seq_idx}, expected {expected_len}."
            )
        for tok_idx, value in enumerate(seq):
            if not math.isfinite(value):
                raise ValueError(
                    f"{mode_name} returned non-finite value at sequence {seq_idx}, "
                    f"token {tok_idx}: {value!r}"
                )


def _coerce_transform_output(
    raw_output: object,
    *,
    expected_lens: list[int],
    mode_name: str,
) -> AdvantageResult:
    """Normalize transform plugin output to AdvantageResult."""
    if isinstance(raw_output, AdvantageResult):
        result = raw_output
    elif isinstance(raw_output, TransformOutput):
        result = AdvantageResult(
            token_advs=raw_output.token_advs,
            has_stats=raw_output.has_stats,
            stats=raw_output.stats,
            extra_metrics=raw_output.extra_metrics,
        )
    else:
        token_advs = raw_output  # type: ignore[assignment]
        result = AdvantageResult(
            token_advs=[[float(v) for v in seq] for seq in token_advs],  # type: ignore[arg-type]
            has_stats=False,
            stats=EntropyStats(),
            extra_metrics={},
        )

    _validate_token_advs(
        result.token_advs,
        expected_lens=expected_lens,
        mode_name=mode_name,
    )
    return result


def _coerce_algorithm_output(
    raw_output: object,
    *,
    expected_lens: list[int],
    mode_name: str,
) -> AdvantageResult:
    """Normalize algorithm plugin output to AdvantageResult."""
    if isinstance(raw_output, AlgorithmOutput):
        result = AdvantageResult(
            token_advs=raw_output.token_advs,
            has_stats=raw_output.has_stats,
            stats=raw_output.stats,
            extra_metrics=raw_output.extra_metrics,
        )
    else:
        result = _coerce_transform_output(
            raw_output,
            expected_lens=expected_lens,
            mode_name=mode_name,
        )
        return result

    _validate_token_advs(
        result.token_advs,
        expected_lens=expected_lens,
        mode_name=mode_name,
    )
    return result


# ---------------------------------------------------------------------------
# 0. GRPO advantages (simple reward centering)
# ---------------------------------------------------------------------------


def compute_grpo_advantages(rewards: list[float]) -> list[float]:
    """A_i = r_i - mean(r)."""
    n = len(rewards)
    if n == 0:
        return []
    mean_r = sum(rewards) / n
    return [r - mean_r for r in rewards]


# ---------------------------------------------------------------------------
# 1. MaxRL advantages (inverse success-rate reweighting)
# ---------------------------------------------------------------------------


def compute_maxrl_advantages(
    rewards: list[float], eps: float = 1e-6
) -> list[float]:
    """A_i = (r_i - mean(r)) / (mean(r) + eps). Zero if mean(r) ~ 0."""
    n = len(rewards)
    if n == 0:
        return []
    mean_r = sum(rewards) / n
    if mean_r <= eps:
        return [0.0] * n
    denom = mean_r + eps
    return [(r - mean_r) / denom for r in rewards]


# ---------------------------------------------------------------------------
# 1b. Advantage mode registry (built-ins + dotted-path plugins)
# ---------------------------------------------------------------------------

_BUILTIN_ADVANTAGE_SPECS: dict[str, AdvantageSpec] = {
    "grpo": _as_advantage_spec("grpo", compute_grpo_advantages),
    "maxrl": _as_advantage_spec("maxrl", compute_maxrl_advantages),
}

_ADVANTAGE_SPEC_CACHE: dict[str, AdvantageSpec] = {}


def register_advantage_mode(
    name: str, compute: EpisodeAdvantageComputeFn
) -> None:
    """Register or replace a short-name episode advantage mode at runtime."""
    if not name or "." in name:
        raise ValueError(
            "Advantage mode name must be non-empty and cannot contain '.'. "
            "Use dotted paths directly in TOML for external plugins."
        )
    _BUILTIN_ADVANTAGE_SPECS[name] = _as_advantage_spec(name, compute)
    _ADVANTAGE_SPEC_CACHE.pop(name, None)


def get_builtin_advantage_modes() -> list[str]:
    """Return sorted built-in advantage mode names."""
    return sorted(_BUILTIN_ADVANTAGE_SPECS)


def is_valid_advantage_mode_name(advantage_mode: str) -> bool:
    """True for built-ins or dotted plugin paths (`module.attr`)."""
    if advantage_mode in _BUILTIN_ADVANTAGE_SPECS:
        return True
    module_path, _, attr_name = advantage_mode.rpartition(".")
    return bool(module_path and attr_name)


def _load_custom_advantage_spec(dotted_path: str) -> AdvantageSpec:
    resolved = resolve_dotted_attribute(
        dotted_path,
        selector="advantage_mode",
        expected="a callable (rewards[, params]) or AdvantageSpec",
    )
    obj = resolved.obj

    if isinstance(obj, AdvantageSpec):
        return obj
    if callable(obj):
        if _callable_takes_no_positional_args(obj):
            built = obj()
            if isinstance(built, AdvantageSpec):
                return built
            if callable(built):
                return _as_advantage_spec(dotted_path, built)
            raise TypeError(
                f"advantage_mode '{dotted_path}' factory returned "
                f"{type(built).__name__}, expected AdvantageSpec or callable."
            )
        return _as_advantage_spec(dotted_path, obj)

    raise TypeError(
        f"advantage_mode '{dotted_path}' must resolve to AdvantageSpec "
        f"or a callable, got {type(obj).__name__}."
    )


def get_advantage_spec(advantage_mode: str) -> AdvantageSpec:
    """Resolve an advantage mode to a behavior spec."""
    cached = _ADVANTAGE_SPEC_CACHE.get(advantage_mode)
    if cached is not None:
        return cached

    spec = _BUILTIN_ADVANTAGE_SPECS.get(advantage_mode)
    if spec is None:
        if "." in advantage_mode:
            spec = _load_custom_advantage_spec(advantage_mode)
        else:
            raise ValueError(
                f"Unknown advantage_mode '{advantage_mode}'. "
                f"Built-in options: {get_builtin_advantage_modes()}. "
                "For custom advantages use dotted path format "
                "(e.g. 'my_module.my_advantage')."
            )
    _ADVANTAGE_SPEC_CACHE[advantage_mode] = spec
    return spec


# ---------------------------------------------------------------------------
# 1c. Algorithm mode registry (built-ins + dotted-path plugins)
# ---------------------------------------------------------------------------


def _make_builtin_algorithm_spec(
    name: str,
    *,
    advantage_mode: str,
    transform_mode: str,
    needs_planning: bool,
    uses_sepa_controller: bool,
) -> AlgorithmSpec:
    """Create a built-in AlgorithmSpec by delegating to composable pipeline."""

    def _compute(ctx: AlgorithmContext) -> AdvantageResult:
        post_process = dict(ctx.params.get("transform_params", {}))
        if "entropy_mask_rho" in ctx.params:
            post_process.setdefault("entropy_mask_rho", ctx.params["entropy_mask_rho"])
        return compute_composable_advantages(
            rewards_G=ctx.rewards_G,
            logprobs_G=ctx.logprobs_G,
            planning_masks_G=ctx.planning_masks_G,
            # Built-in algorithm modes are fixed recipes by design.
            advantage_mode=advantage_mode,
            transform_mode=transform_mode,
            gtpo_beta=ctx.gtpo_beta,
            hicra_alpha=ctx.hicra_alpha,
            sepa_lambda=ctx.sepa_lambda,
            advantage_params=ctx.params.get("advantage_params")
            if isinstance(ctx.params.get("advantage_params"), Mapping)
            else {},
            transform_params=ctx.params.get("transform_params")
            if isinstance(ctx.params.get("transform_params"), Mapping)
            else {},
            step=ctx.step,
            post_process_params=post_process,
        )

    return AlgorithmSpec(
        name=name,
        compute=_compute,
        needs_planning=needs_planning,
        uses_sepa_controller=uses_sepa_controller,
    )


_BUILTIN_ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "grpo_none": _make_builtin_algorithm_spec(
        "grpo_none",
        advantage_mode="grpo",
        transform_mode="none",
        needs_planning=False,
        uses_sepa_controller=False,
    ),
    "maxrl_none": _make_builtin_algorithm_spec(
        "maxrl_none",
        advantage_mode="maxrl",
        transform_mode="none",
        needs_planning=False,
        uses_sepa_controller=False,
    ),
    "maxrl_gtpo": _make_builtin_algorithm_spec(
        "maxrl_gtpo",
        advantage_mode="maxrl",
        transform_mode="gtpo",
        needs_planning=False,
        uses_sepa_controller=False,
    ),
    "maxrl_gtpo_hicra": _make_builtin_algorithm_spec(
        "maxrl_gtpo_hicra",
        advantage_mode="maxrl",
        transform_mode="gtpo_hicra",
        needs_planning=True,
        uses_sepa_controller=False,
    ),
    "maxrl_gtpo_sepa": _make_builtin_algorithm_spec(
        "maxrl_gtpo_sepa",
        advantage_mode="maxrl",
        transform_mode="gtpo_sepa",
        needs_planning=True,
        uses_sepa_controller=True,
    ),
}

_ALGORITHM_SPEC_CACHE: dict[str, AlgorithmSpec] = {}


def register_algorithm_mode(name: str, spec_or_fn: AlgorithmSpec | AlgorithmContextFn) -> None:
    """Register or replace a short-name algorithm mode at runtime."""
    if not name or "." in name:
        raise ValueError(
            "Algorithm mode name must be non-empty and cannot contain '.'. "
            "Use dotted paths directly in TOML for external plugins."
        )
    if isinstance(spec_or_fn, AlgorithmSpec):
        spec = spec_or_fn
    elif callable(spec_or_fn):
        spec = AlgorithmSpec(name=name, compute=spec_or_fn)
    else:
        raise TypeError(
            f"Algorithm registration for '{name}' must be AlgorithmSpec "
            f"or callable, got {type(spec_or_fn).__name__}."
        )
    _BUILTIN_ALGORITHM_SPECS[name] = spec
    _ALGORITHM_SPEC_CACHE.pop(name, None)


def get_builtin_algorithm_modes() -> list[str]:
    """Return sorted built-in algorithm mode names."""
    return sorted(_BUILTIN_ALGORITHM_SPECS)


def is_valid_algorithm_mode_name(algorithm_mode: str) -> bool:
    """True for built-ins or dotted plugin paths (`module.attr`)."""
    if algorithm_mode in _BUILTIN_ALGORITHM_SPECS:
        return True
    module_path, _, attr_name = algorithm_mode.rpartition(".")
    return bool(module_path and attr_name)


def _load_custom_algorithm_spec(dotted_path: str) -> AlgorithmSpec:
    resolved = resolve_dotted_attribute(
        dotted_path,
        selector="algorithm_mode",
        expected="AlgorithmSpec, callable(ctx), or zero-arg factory",
    )
    obj = resolved.obj
    if isinstance(obj, AlgorithmSpec):
        return obj
    if callable(obj):
        if _callable_takes_no_positional_args(obj):
            built = obj()
            if isinstance(built, AlgorithmSpec):
                return built
            if callable(built):
                return AlgorithmSpec(
                    name=dotted_path,
                    compute=built,
                    needs_planning=bool(getattr(built, "needs_planning", False)),
                    uses_sepa_controller=bool(
                        getattr(built, "uses_sepa_controller", False)
                    ),
                )
            raise TypeError(
                f"algorithm_mode '{dotted_path}' factory returned "
                f"{type(built).__name__}, expected AlgorithmSpec or callable."
            )
        return AlgorithmSpec(
            name=dotted_path,
            compute=obj,
            needs_planning=bool(getattr(obj, "needs_planning", False)),
            uses_sepa_controller=bool(
                getattr(obj, "uses_sepa_controller", False)
            ),
        )
    raise TypeError(
        f"algorithm_mode '{dotted_path}' must resolve to AlgorithmSpec "
        f"or callable, got {type(obj).__name__}."
    )


def get_algorithm_spec(algorithm_mode: str) -> AlgorithmSpec:
    """Resolve an algorithm mode to a behavior spec."""
    cached = _ALGORITHM_SPEC_CACHE.get(algorithm_mode)
    if cached is not None:
        return cached

    spec = _BUILTIN_ALGORITHM_SPECS.get(algorithm_mode)
    if spec is None:
        if "." in algorithm_mode:
            spec = _load_custom_algorithm_spec(algorithm_mode)
        else:
            raise ValueError(
                f"Unknown algorithm_mode '{algorithm_mode}'. "
                f"Built-in options: {get_builtin_algorithm_modes()}. "
                "For custom algorithms use dotted path format "
                "(e.g. 'my_module.my_algorithm')."
            )
    _ALGORITHM_SPEC_CACHE[algorithm_mode] = spec
    return spec


# ---------------------------------------------------------------------------
# 2. GTPO entropy-weighted credit assignment
# ---------------------------------------------------------------------------


def apply_gtpo_weighting(
    advantage: float, surprisals: list[float], beta: float = 0.1
) -> list[float]:
    """Surprisal-weighted token-level advantages."""
    n = len(surprisals)
    if n == 0:
        return []
    if beta == 0.0:
        return [advantage] * n

    surprisals = [min(e, MAX_SURPRISAL) for e in surprisals]
    mean_h = sum(surprisals) / n
    if mean_h < 1e-7:
        return [advantage] * n

    result = []
    for h in surprisals:
        h_norm = h / (mean_h + 1e-8)
        weight = max(0.0, 1.0 + beta * (h_norm - 1.0))
        result.append(advantage * weight)
    return result


# ---------------------------------------------------------------------------
# 3. HICRA planning token amplification
# ---------------------------------------------------------------------------


def apply_hicra(
    token_advs: list[float], planning_mask: list[int], alpha: float = 0.2
) -> list[float]:
    """A_HICRA(t) = A(t) + alpha * |A(t)| * mask(t)."""
    if len(token_advs) != len(planning_mask):
        raise ValueError(
            f"Length mismatch: token_advs ({len(token_advs)}) "
            f"vs planning_mask ({len(planning_mask)})"
        )
    if alpha == 0.0:
        return list(token_advs)
    return [
        a + alpha * abs(a) if m else a
        for a, m in zip(token_advs, planning_mask)
    ]


# ---------------------------------------------------------------------------
# 3b. Entropy masking (Yue et al. proxy replication)
# ---------------------------------------------------------------------------


def compute_entropy_mask_threshold(
    all_entropies: list[float], rho: float
) -> float:
    """Compute the threshold for top-ρ entropy masking.

    Returns the entropy value at the ρ-percentile boundary (descending).
    Tokens with entropy >= threshold are kept; the rest are zeroed.
    """
    if rho >= 1.0:
        return float("-inf")
    if rho <= 0.0:
        return float("inf")
    n = len(all_entropies)
    if n == 0:
        return 0.0
    sorted_desc = sorted(all_entropies, reverse=True)
    idx = max(1, int(n * rho)) - 1
    return sorted_desc[idx]


def apply_entropy_mask(
    token_advs: list[float], entropies: list[float], threshold: float
) -> list[float]:
    """Zero out advantages for tokens below the entropy threshold."""
    return [
        a if e >= threshold else 0.0
        for a, e in zip(token_advs, entropies)
    ]


def surprisal_mask_post_process(
    all_token_advs: list[list[float]],
    all_raw_surprisals: list[list[float]],
    params: Mapping[str, object],
) -> tuple[list[list[float]], dict[str, float]]:
    """Post-process hook: Yue et al. surprisal masking."""
    raw_rho = params.get("surprisal_mask_rho", params.get("entropy_mask_rho", 0.0))
    rho = float(raw_rho) if isinstance(raw_rho, int | float) else 0.0
    if rho <= 0.0:
        return all_token_advs, {}

    flat_surprisals = [e for seq in all_raw_surprisals for e in seq]
    threshold = compute_entropy_mask_threshold(flat_surprisals, rho)

    total_tokens = 0
    masked_tokens = 0
    for idx in range(len(all_token_advs)):
        all_token_advs[idx] = apply_entropy_mask(
            all_token_advs[idx], all_raw_surprisals[idx], threshold
        )
        for e in all_raw_surprisals[idx]:
            total_tokens += 1
            if e < threshold:
                masked_tokens += 1

    fraction = masked_tokens / total_tokens if total_tokens > 0 else 0.0
    return all_token_advs, {
        "entropy_mask_threshold": threshold,
        "entropy_mask_fraction": fraction,
    }


entropy_mask_post_process = surprisal_mask_post_process  # backward-compat alias


# ---------------------------------------------------------------------------
# 4. SEPA selective entropy pooling
# ---------------------------------------------------------------------------


def apply_sepa_pooling(
    surprisals: list[float], planning_mask: list[int], lambda_t: float
) -> list[float]:
    """Pull execution token surprisals toward their mean."""
    if len(surprisals) != len(planning_mask):
        raise ValueError(
            f"Length mismatch: surprisals ({len(surprisals)}) "
            f"vs planning_mask ({len(planning_mask)})"
        )
    lam = max(0.0, min(1.0, lambda_t))
    if lam == 0.0:
        return list(surprisals)

    surprisals = [min(e, MAX_SURPRISAL) for e in surprisals]
    exec_vals = [e for e, m in zip(surprisals, planning_mask) if m == 0]
    if not exec_vals:
        return list(surprisals)
    mean_h_exec = sum(exec_vals) / len(exec_vals)

    return [
        e if m else lam * mean_h_exec + (1.0 - lam) * e
        for e, m in zip(surprisals, planning_mask)
    ]


def apply_sepa_amplification(
    surprisals: list[float], planning_mask: list[int], lambda_t: float
) -> list[float]:
    """Push execution token surprisals away from their mean.

    h'_t = h_t + λ·(h_t - μ_exec) = (1+λ)·h_t - λ·μ_exec

    High-surprisal execution tokens get pushed higher (more GTPO gradient
    weight), low-surprisal ones get pushed lower. Planning tokens are
    left untouched.
    """
    if len(surprisals) != len(planning_mask):
        raise ValueError(
            f"Length mismatch: surprisals ({len(surprisals)}) "
            f"vs planning_mask ({len(planning_mask)})"
        )
    lam = max(0.0, min(1.0, lambda_t))
    if lam == 0.0:
        return list(surprisals)

    surprisals = [min(e, MAX_SURPRISAL) for e in surprisals]
    exec_vals = [e for e, m in zip(surprisals, planning_mask) if m == 0]
    if not exec_vals:
        return list(surprisals)
    mean_h_exec = sum(exec_vals) / len(exec_vals)

    return [
        e if m else (1.0 + lam) * e - lam * mean_h_exec
        for e, m in zip(surprisals, planning_mask)
    ]


def apply_sepa_amplification_clamped(
    surprisals: list[float], planning_mask: list[int], lambda_t: float
) -> list[float]:
    """Push execution token surprisals away from their mean, clamped to >= 0.

    Same as apply_sepa_amplification but floors results at zero so no token
    gets a negative surprisal value.  Keeps amplification purely soft —
    low-surprisal tokens shrink toward zero but never flip sign.
    """
    if len(surprisals) != len(planning_mask):
        raise ValueError(
            f"Length mismatch: surprisals ({len(surprisals)}) "
            f"vs planning_mask ({len(planning_mask)})"
        )
    lam = max(0.0, min(1.0, lambda_t))
    if lam == 0.0:
        return list(surprisals)

    surprisals = [min(e, MAX_SURPRISAL) for e in surprisals]
    exec_vals = [e for e, m in zip(surprisals, planning_mask) if m == 0]
    if not exec_vals:
        return list(surprisals)
    mean_h_exec = sum(exec_vals) / len(exec_vals)

    return [
        e if m else max(0.0, (1.0 + lam) * e - lam * mean_h_exec)
        for e, m in zip(surprisals, planning_mask)
    ]


# ---------------------------------------------------------------------------
# 4b. Transform mode registry (built-ins + dotted-path plugins)
# ---------------------------------------------------------------------------

_BUILTIN_TRANSFORM_SPECS: dict[str, TransformSpec] = {
    "none": TransformSpec(name="none", use_gtpo=False),
    "gtpo": TransformSpec(name="gtpo"),
    "entropy_mask": TransformSpec(name="entropy_mask", use_gtpo=True, post_process=surprisal_mask_post_process),
    "gtpo_hicra": TransformSpec(
        name="gtpo_hicra", needs_planning=True, apply_hicra=True
    ),
    "gtpo_sepa": TransformSpec(
        name="gtpo_sepa",
        needs_planning=True,
        uses_sepa_controller=True,
        entropy_transform=apply_sepa_pooling,
    ),
    "gtpo_sepa_amp": TransformSpec(
        name="gtpo_sepa_amp",
        needs_planning=True,
        uses_sepa_controller=True,
        entropy_transform=apply_sepa_amplification,
    ),
    "gtpo_sepa_amp_c": TransformSpec(
        name="gtpo_sepa_amp_c",
        needs_planning=True,
        uses_sepa_controller=True,
        entropy_transform=apply_sepa_amplification_clamped,
    ),
}

_TRANSFORM_SPEC_CACHE: dict[str, TransformSpec] = {}


def register_transform_mode(name: str, spec_or_fn: TransformSpec | TransformContextFn) -> None:
    """Register or replace a short-name transform mode at runtime."""
    if not name or "." in name:
        raise ValueError(
            "Transform mode name must be non-empty and cannot contain '.'. "
            "Use dotted paths directly in TOML for external plugins."
        )
    if isinstance(spec_or_fn, TransformSpec):
        spec = spec_or_fn
    elif callable(spec_or_fn):
        spec = TransformSpec(name=name, compute_context=spec_or_fn)
    else:
        raise TypeError(
            f"Transform registration for '{name}' must be TransformSpec "
            f"or callable, got {type(spec_or_fn).__name__}."
        )
    _BUILTIN_TRANSFORM_SPECS[name] = spec
    _TRANSFORM_SPEC_CACHE.pop(name, None)


def get_builtin_transform_modes() -> list[str]:
    """Return sorted built-in transform mode names."""
    return sorted(_BUILTIN_TRANSFORM_SPECS)


def is_valid_transform_mode_name(transform_mode: str) -> bool:
    """True for built-ins or dotted plugin paths (`module.attr`)."""
    if transform_mode in _BUILTIN_TRANSFORM_SPECS:
        return True
    module_path, _, attr_name = transform_mode.rpartition(".")
    return bool(module_path and attr_name)


def _load_custom_transform_spec(dotted_path: str) -> TransformSpec:
    resolved = resolve_dotted_attribute(
        dotted_path,
        selector="transform_mode",
        expected="TransformSpec, context callable, or zero-arg factory",
    )
    obj = resolved.obj

    if isinstance(obj, TransformSpec):
        return obj
    if callable(obj):
        if _callable_takes_no_positional_args(obj):
            built = obj()
            if isinstance(built, TransformSpec):
                return built
            if callable(built):
                return TransformSpec(
                    name=dotted_path,
                    needs_planning=bool(getattr(built, "needs_planning", False)),
                    uses_sepa_controller=bool(
                        getattr(built, "uses_sepa_controller", False)
                    ),
                    compute_context=built,
                )
            raise TypeError(
                f"transform_mode '{dotted_path}' factory returned "
                f"{type(built).__name__}, expected TransformSpec or callable."
            )
        return TransformSpec(
            name=dotted_path,
            needs_planning=bool(getattr(obj, "needs_planning", False)),
            uses_sepa_controller=bool(
                getattr(obj, "uses_sepa_controller", False)
            ),
            compute_context=obj,
        )

    raise TypeError(
        f"transform_mode '{dotted_path}' must resolve to TransformSpec "
        f"or callable, got {type(obj).__name__}."
    )


def get_transform_spec(transform_mode: str) -> TransformSpec:
    """Resolve a transform mode to a behavior spec."""
    cached = _TRANSFORM_SPEC_CACHE.get(transform_mode)
    if cached is not None:
        return cached

    spec = _BUILTIN_TRANSFORM_SPECS.get(transform_mode)
    if spec is None:
        if "." in transform_mode:
            spec = _load_custom_transform_spec(transform_mode)
        else:
            raise ValueError(
                f"Unknown transform_mode '{transform_mode}'. "
                f"Built-in options: {get_builtin_transform_modes()}. "
                "For custom transforms use dotted path format "
                "(e.g. 'my_module.make_transform_spec')."
            )
    _TRANSFORM_SPEC_CACHE[transform_mode] = spec
    return spec


# ---------------------------------------------------------------------------
# 4c. Uncertainty kind registry (built-ins + dotted-path plugins)
# ---------------------------------------------------------------------------


def _compute_surprisal(ctx: UncertaintyContext) -> list[float]:
    """Sampled-token surprisal: -logprob (clamped)."""
    return [min(-lp, MAX_SURPRISAL) for lp in ctx.logprobs]


def _compute_shannon_entropy(ctx: UncertaintyContext) -> list[float]:
    """Full-distribution Shannon entropy: -sum(p * log(p)).

    Fast path: if precomputed_entropy is available (GPU-computed scalars
    from PyTorchEngine), use those directly. Otherwise fall back to
    computing from full token_distributions.
    """
    if ctx.precomputed_entropy is not None:
        return [min(h, MAX_SURPRISAL) for h in ctx.precomputed_entropy]
    if ctx.token_distributions is not None:
        result: list[float] = []
        for dist in ctx.token_distributions:
            h = -sum(p * math.log(p) for p in dist if p > 0)
            result.append(min(h, MAX_SURPRISAL))
        return result
    raise ValueError(
        "shannon_entropy requires either precomputed per-token entropy "
        "(from PyTorch engine with compute_entropy=True) or full "
        "per-position token distributions. Use inference_engine = "
        "\"pytorch\" to enable GPU-side entropy computation."
    )


def _compute_predictive_variance(ctx: UncertaintyContext) -> list[float]:
    """Bernoulli predictive variance: p * (1 - p) where p = exp(logprob)."""
    result: list[float] = []
    for lp in ctx.logprobs:
        p = math.exp(max(lp, -MAX_SURPRISAL))
        p = max(0.0, min(1.0, p))
        result.append(p * (1.0 - p))
    return result


_BUILTIN_UNCERTAINTY_SPECS: dict[str, UncertaintySpec] = {
    "surprisal": UncertaintySpec(
        name="surprisal",
        compute=_compute_surprisal,
    ),
    "shannon_entropy": UncertaintySpec(
        name="shannon_entropy",
        compute=_compute_shannon_entropy,
        needs_distributions=True,
    ),
    "predictive_variance": UncertaintySpec(
        name="predictive_variance",
        compute=_compute_predictive_variance,
    ),
}

_UNCERTAINTY_SPEC_CACHE: dict[str, UncertaintySpec] = {}


def register_uncertainty_kind(
    name: str, spec_or_fn: UncertaintySpec | UncertaintyComputeFn
) -> None:
    """Register or replace a short-name uncertainty kind at runtime."""
    if not name or "." in name:
        raise ValueError(
            "Uncertainty kind name must be non-empty and cannot contain '.'. "
            "Use dotted paths directly in TOML for external plugins."
        )
    if isinstance(spec_or_fn, UncertaintySpec):
        spec = spec_or_fn
    elif callable(spec_or_fn):
        spec = UncertaintySpec(name=name, compute=spec_or_fn)
    else:
        raise TypeError(
            f"Uncertainty registration for '{name}' must be UncertaintySpec "
            f"or callable, got {type(spec_or_fn).__name__}."
        )
    _BUILTIN_UNCERTAINTY_SPECS[name] = spec
    _UNCERTAINTY_SPEC_CACHE.pop(name, None)


def get_builtin_uncertainty_kinds() -> list[str]:
    """Return sorted built-in uncertainty kind names."""
    return sorted(_BUILTIN_UNCERTAINTY_SPECS)


def is_valid_uncertainty_kind_name(kind: str) -> bool:
    """True for aliases, builtins, or dotted plugin paths (`module.attr`)."""
    if kind in _UNCERTAINTY_KIND_ALIASES:
        return True
    if kind in _BUILTIN_UNCERTAINTY_SPECS:
        return True
    module_path, _, attr_name = kind.rpartition(".")
    return bool(module_path and attr_name)


def _load_custom_uncertainty_spec(dotted_path: str) -> UncertaintySpec:
    resolved = resolve_dotted_attribute(
        dotted_path,
        selector="uncertainty_kind",
        expected="UncertaintySpec, compute callable, or zero-arg factory",
    )
    obj = resolved.obj

    if isinstance(obj, UncertaintySpec):
        return obj
    if callable(obj):
        if _callable_takes_no_positional_args(obj):
            built = obj()
            if isinstance(built, UncertaintySpec):
                return built
            if callable(built):
                return UncertaintySpec(name=dotted_path, compute=built)
            raise TypeError(
                f"uncertainty_kind '{dotted_path}' factory returned "
                f"{type(built).__name__}, expected UncertaintySpec or callable."
            )
        return UncertaintySpec(name=dotted_path, compute=obj)

    raise TypeError(
        f"uncertainty_kind '{dotted_path}' must resolve to UncertaintySpec "
        f"or callable, got {type(obj).__name__}."
    )


def get_uncertainty_spec(kind: str) -> UncertaintySpec:
    """Resolve an uncertainty kind to a behavior spec."""
    cached = _UNCERTAINTY_SPEC_CACHE.get(kind)
    if cached is not None:
        return cached

    spec = _BUILTIN_UNCERTAINTY_SPECS.get(kind)
    if spec is None:
        if "." in kind:
            spec = _load_custom_uncertainty_spec(kind)
        else:
            raise ValueError(
                f"Unknown uncertainty_kind '{kind}'. "
                f"Built-in options: {get_builtin_uncertainty_kinds()}. "
                "For custom uncertainty signals use dotted path format "
                "(e.g. 'my_module.my_uncertainty')."
            )
    _UNCERTAINTY_SPEC_CACHE[kind] = spec
    return spec


def _resolve_uncertainty_kind(params: Mapping[str, object]) -> str:
    """Resolve uncertainty metric for GTPO-family transforms.

    Default is sampled-token surprisal (`-logprob`).
    """
    raw_value: object = "surprisal"
    for key in _UNCERTAINTY_KIND_PARAM_KEYS:
        if key in params:
            raw_value = params[key]
            break
    return canonicalize_uncertainty_kind(raw_value)


def canonicalize_uncertainty_kind(raw_value: object) -> str:
    """Normalize a user-facing uncertainty-kind setting."""
    if not isinstance(raw_value, str):
        raise ValueError(
            f"{_UNCERTAINTY_KIND_PARAM_KEYS[0]} must be a string; "
            f"got {type(raw_value).__name__}."
        )
    # Check aliases first.
    canonical = _UNCERTAINTY_KIND_ALIASES.get(raw_value.strip().lower())
    if canonical is not None:
        return canonical
    # Accept runtime-registered builtins.
    stripped = raw_value.strip()
    if stripped in _BUILTIN_UNCERTAINTY_SPECS:
        return stripped
    # Accept dotted plugin paths.
    if "." in stripped:
        return stripped
    allowed = sorted(set(_UNCERTAINTY_KIND_ALIASES) | set(_BUILTIN_UNCERTAINTY_SPECS))
    raise ValueError(
        f"Unknown uncertainty kind '{raw_value}'. "
        f"Supported values: {allowed}."
    )


# ---------------------------------------------------------------------------
# 5. Entropy statistics
# ---------------------------------------------------------------------------


def compute_surprisal_stats(
    exec_surprisals: list[float], plan_surprisals: list[float]
) -> EntropyStats:
    """Compute summary stats for execution vs planning token surprisal."""
    stats = EntropyStats()

    exec_surprisals = [min(e, MAX_SURPRISAL) for e in exec_surprisals]
    plan_surprisals = [min(e, MAX_SURPRISAL) for e in plan_surprisals]

    if exec_surprisals:
        n = len(exec_surprisals)
        mean_e = sum(exec_surprisals) / n
        var_e = sum((e - mean_e) ** 2 for e in exec_surprisals) / n
        stats.exec_mean = mean_e
        stats.exec_var = var_e
        stats.exec_count = float(n)

    if plan_surprisals:
        n = len(plan_surprisals)
        mean_p = sum(plan_surprisals) / n
        var_p = sum((p - mean_p) ** 2 for p in plan_surprisals) / n
        stats.plan_mean = mean_p
        stats.plan_var = var_p
        stats.plan_count = float(n)

    return stats


compute_entropy_stats = compute_surprisal_stats  # backward-compat alias


# ---------------------------------------------------------------------------
# 6. Planning token identification (regex-based)
# ---------------------------------------------------------------------------

DEFAULT_STRATEGIC_GRAMS = [
    "wait let me",
    "let me think",
    "on second thought",
    "let me check",
    "let me verify",
    "is this right",
    "double check",
    "try another approach",
    "go back and",
    "start over",
    "that's not right",
    "that doesn't work",
    "another way to",
    "or we could",
    "what if we",
    "notice that",
    "the key is",
    "the key insight",
]


def _clean_token_fragment(fragment: str) -> str:
    """Clean a tokenizer fragment: replace subword markers with space."""
    # sentencepiece: \u2581, GPT-2/BPE: \u0120
    return fragment.replace("\u2581", " ").replace("\u0120", " ").strip()


# Cache compiled regex patterns keyed by the tuple of grams
_pattern_cache: dict[tuple[str, ...], list[re.Pattern[str]]] = {}


def _get_gram_patterns(strategic_grams: list[str]) -> list[re.Pattern[str]]:
    """Return compiled regex patterns for strategic grams (cached)."""
    key = tuple(strategic_grams)
    if key not in _pattern_cache:
        _pattern_cache[key] = [
            re.compile(r"\b" + re.escape(gram) + r"\b", re.IGNORECASE)
            for gram in strategic_grams
        ]
    return _pattern_cache[key]


def identify_planning_tokens(
    token_strs: list[str],
    strategic_grams: list[str],
    max_window: int = 5,
) -> list[int]:
    """Identify planning tokens via strategic gram matching.

    Sliding window over token fragments, checking for word-boundary matches.
    """
    n_tokens = len(token_strs)
    if n_tokens == 0 or not strategic_grams:
        return [0] * n_tokens

    # Effective window covers longest gram by word count
    effective_window = max(max_window, max(len(g.split()) for g in strategic_grams))

    # Pre-clean all fragments
    cleaned = [_clean_token_fragment(t) for t in token_strs]

    patterns = _get_gram_patterns(strategic_grams)

    mask = [0] * n_tokens

    for start in range(n_tokens):
        window_text = ""
        window_end = min(start + effective_window, n_tokens)

        for end in range(start, window_end):
            if cleaned[end]:
                if window_text:
                    window_text += " " + cleaned[end]
                else:
                    window_text = cleaned[end]

            matched = False
            for pat in patterns:
                if pat.search(window_text):
                    for idx in range(start, end + 1):
                        mask[idx] = 1
                    matched = True
                    break
            if matched:
                break

    return mask


def _raise_missing_data(
    uncertainty_kind: str,
    *,
    has_logprobs: bool,
    has_distributions: bool,
    n_episodes: int = 0,
) -> None:
    """Raise a diagnostic ValueError when required data is absent."""
    logprobs_status = f"provided ({n_episodes} episodes)" if has_logprobs else "absent"
    distributions_status = (
        f"provided ({n_episodes} episodes)" if has_distributions else "absent"
    )
    raise ValueError(
        f"uncertainty_kind='{uncertainty_kind}' requires per-position token distributions.\n"
        f"Data received: logprobs_G={logprobs_status}, "
        f"token_distributions_G={distributions_status}.\n"
        f"Use uncertainty_kind='surprisal' (requires only logprobs) or use a backend "
        f"that returns full token distributions."
    )


# ---------------------------------------------------------------------------
# Composable advantage pipeline
# ---------------------------------------------------------------------------


def compute_composable_advantages(
    rewards_G: list[float],
    logprobs_G: list[list[float]],
    planning_masks_G: list[list[int]],
    *,
    advantage_mode: str = "grpo",
    transform_mode: str = "none",
    gtpo_beta: float = 0.1,
    hicra_alpha: float = 0.2,
    sepa_lambda: float = 0.0,
    advantage_params: Mapping[str, object] | None = None,
    transform_params: Mapping[str, object] | None = None,
    step: int = 0,
    post_process_params: Mapping[str, object] | None = None,
    token_distributions_G: list[list[list[float]]] | None = None,
    precomputed_entropies_G: list[list[float]] | None = None,
) -> AdvantageResult:
    """Compute token-level advantages with composable transforms."""
    advantage_spec = get_advantage_spec(advantage_mode)
    transform_spec = get_transform_spec(transform_mode)

    # Step 1: Episode-level advantages
    raw_advantages = advantage_spec.compute(rewards_G, advantage_params or {})
    advantages_G = _coerce_advantages_output(
        raw_advantages,
        expected_len=len(rewards_G),
        mode_name=advantage_spec.name,
    )
    expected_lens = [len(seq) for seq in logprobs_G]

    merged_transform_params: dict[str, object] = {}
    if post_process_params:
        merged_transform_params.update(post_process_params)
    if transform_params:
        merged_transform_params.update(transform_params)

    if transform_spec.compute_context is not None:
        ctx = TransformContext(
            episode_advantages=advantages_G,
            logprobs_G=logprobs_G,
            planning_masks_G=planning_masks_G,
            sepa_lambda=sepa_lambda,
            params=merged_transform_params,
            step=step,
            gtpo_beta=gtpo_beta,
            hicra_alpha=hicra_alpha,
        )
        raw_output = transform_spec.compute_context(ctx)
        return _coerce_transform_output(
            raw_output,
            expected_lens=expected_lens,
            mode_name=f"transform_mode '{transform_spec.name}'",
        )

    # Step 2: Token-level expansion
    if not transform_spec.use_gtpo:
        all_token_advs = [
            [advantages_G[i]] * len(logprobs_G[i])
            for i in range(len(logprobs_G))
        ]
        _validate_token_advs(
            all_token_advs,
            expected_lens=expected_lens,
            mode_name=f"transform_mode '{transform_spec.name}'",
        )
        return AdvantageResult(all_token_advs, False, EntropyStats())

    uncertainty_kind = _resolve_uncertainty_kind(merged_transform_params)
    uncertainty_spec = get_uncertainty_spec(uncertainty_kind)

    if (
        uncertainty_spec.needs_distributions
        and token_distributions_G is None
        and precomputed_entropies_G is None
    ):
        _raise_missing_data(
            uncertainty_kind,
            has_logprobs=True,
            has_distributions=False,
            n_episodes=len(logprobs_G),
        )

    # GTPO-based transforms need per-token uncertainty values.
    # The pipeline is agnostic to the kind — the backend determines what's
    # available, and the user picks via uncertainty_kind.
    all_token_advs = []
    all_exec_surprisals: list[float] = []
    all_plan_surprisals: list[float] = []
    all_raw_surprisals: list[list[float]] = []  # for surprisal_mask compatibility

    for idx in range(len(logprobs_G)):
        logprobs = logprobs_G[idx]
        advantage = advantages_G[idx]
        planning_mask = planning_masks_G[idx]

        ctx = UncertaintyContext(
            logprobs=logprobs,
            token_distributions=token_distributions_G[idx] if token_distributions_G else None,
            precomputed_entropy=precomputed_entropies_G[idx] if precomputed_entropies_G else None,
            planning_mask=planning_mask,
            params=merged_transform_params,
        )
        surprisals = uncertainty_spec.compute(ctx)

        # Store raw surprisals before any transform (for surprisal masking)
        all_raw_surprisals.append(list(surprisals))

        # Collect surprisal stats
        for j, e in enumerate(surprisals):
            if planning_mask[j]:
                all_plan_surprisals.append(e)
            else:
                all_exec_surprisals.append(e)

        # Optional surprisal transform (SEPA variants or custom plugin)
        if transform_spec.entropy_transform is not None:
            surprisals = transform_spec.entropy_transform(
                surprisals, planning_mask, sepa_lambda
            )

        # GTPO weighting
        token_advs = apply_gtpo_weighting(advantage, surprisals, beta=gtpo_beta)

        # HICRA amplification
        if transform_spec.apply_hicra:
            token_advs = apply_hicra(token_advs, planning_mask, alpha=hicra_alpha)

        all_token_advs.append(token_advs)

    # Post-process hook (e.g. surprisal masking)
    extra_metrics: dict[str, float] = {}
    n_seqs = len(all_token_advs)
    seq_lens = [len(seq) for seq in all_token_advs]
    if transform_spec.post_process is not None:
        all_token_advs, extra_metrics = transform_spec.post_process(
            all_token_advs, all_raw_surprisals, merged_transform_params
        )
        # Validate hook output shape
        if len(all_token_advs) != n_seqs:
            raise ValueError(
                f"post_process hook '{transform_spec.name}' returned "
                f"{len(all_token_advs)} sequences, expected {n_seqs}"
            )
        for i, seq in enumerate(all_token_advs):
            if len(seq) != seq_lens[i]:
                raise ValueError(
                    f"post_process hook '{transform_spec.name}' returned "
                    f"{len(seq)} tokens for sequence {i}, expected {seq_lens[i]}"
                )

    _validate_token_advs(
        all_token_advs,
        expected_lens=expected_lens,
        mode_name=f"transform_mode '{transform_spec.name}'",
    )
    stats = compute_surprisal_stats(all_exec_surprisals, all_plan_surprisals)
    return AdvantageResult(all_token_advs, True, stats, extra_metrics=extra_metrics)


def compute_algorithm_advantages(
    rewards_G: list[float],
    logprobs_G: list[list[float]],
    planning_masks_G: list[list[int]],
    *,
    algorithm_mode: str,
    params: Mapping[str, object] | None = None,
    gtpo_beta: float = 0.1,
    hicra_alpha: float = 0.2,
    sepa_lambda: float = 0.0,
    step: int = 0,
    token_distributions_G: list[list[list[float]]] | None = None,
    precomputed_entropies_G: list[list[float]] | None = None,
) -> AdvantageResult:
    """Compute token-level advantages through a full algorithm plugin."""
    spec = get_algorithm_spec(algorithm_mode)
    ctx = AlgorithmContext(
        rewards_G=rewards_G,
        logprobs_G=logprobs_G,
        planning_masks_G=planning_masks_G,
        params=params or {},
        step=step,
        sepa_lambda=sepa_lambda,
        gtpo_beta=gtpo_beta,
        hicra_alpha=hicra_alpha,
        token_distributions_G=token_distributions_G,
        precomputed_entropies_G=precomputed_entropies_G,
    )
    try:
        raw_output = spec.compute(ctx)
        return _coerce_algorithm_output(
            raw_output,
            expected_lens=[len(seq) for seq in logprobs_G],
            mode_name=f"algorithm_mode '{spec.name}'",
        )
    except Exception:
        if get_plugin_runtime().strict:
            raise
        # Non-strict mode still surfaces errors by default to avoid silent bad runs.
        raise
