"""Token uncertainty signals and registry."""

from __future__ import annotations

import math
from collections.abc import Mapping

from retrain.advantages.constants import MAX_SURPRISAL
from retrain.advantages.plugin import (
    _callable_takes_no_positional_args,
    _is_dotted_plugin_path,
    _resolve_registry_spec,
    _validate_short_registry_name,
)
from retrain.advantages.types import (
    UncertaintyComputeFn,
    UncertaintyContext,
    UncertaintySpec,
)
from retrain.plugins.resolve import resolve_dotted_attribute

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
        '"pytorch" to enable GPU-side entropy computation.'
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
    _validate_short_registry_name(name, label="Uncertainty kind")
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
    return _is_dotted_plugin_path(kind)


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
            built = obj()  # type: ignore[call-top-callable]
            if isinstance(built, UncertaintySpec):
                return built
            if callable(built):
                return UncertaintySpec(name=dotted_path, compute=built)  # type: ignore[invalid-argument-type]
            raise TypeError(
                f"uncertainty_kind '{dotted_path}' factory returned "
                f"{type(built).__name__}, expected UncertaintySpec or callable."
            )
        return UncertaintySpec(name=dotted_path, compute=obj)  # type: ignore[invalid-argument-type]

    raise TypeError(
        f"uncertainty_kind '{dotted_path}' must resolve to UncertaintySpec "
        f"or callable, got {type(obj).__name__}."
    )


def get_uncertainty_spec(kind: str) -> UncertaintySpec:
    """Resolve an uncertainty kind to a behavior spec."""
    return _resolve_registry_spec(
        kind,
        builtins=_BUILTIN_UNCERTAINTY_SPECS,
        cache=_UNCERTAINTY_SPEC_CACHE,
        load_custom=_load_custom_uncertainty_spec,
        builtin_names=get_builtin_uncertainty_kinds,
        config_key="uncertainty_kind",
        custom_label="uncertainty signals",
        example="my_module.my_uncertainty",
    )


def _resolve_uncertainty_kind(params: Mapping[str, object]) -> str:
    """Resolve uncertainty metric for GTPO-family transforms.

    Default is sampled-token surprisal (`-logprob`).
    """
    override = get_uncertainty_kind_param(params)
    if override is not None:
        return override[1]
    return canonicalize_uncertainty_kind("surprisal")


def get_uncertainty_kind_param(params: Mapping[str, object]) -> tuple[str, str] | None:
    """Return the first configured uncertainty selector as ``(key, canonical_value)``."""
    selected: tuple[str, str] | None = None
    for key in _UNCERTAINTY_KIND_PARAM_KEYS:
        if key in params:
            value = canonicalize_uncertainty_kind(params[key])
            if selected is not None and value != selected[1]:
                raise ValueError(
                    "Conflicting uncertainty kind selector params: "
                    f"{selected[0]}={selected[1]!r} but {key}={value!r}."
                )
            if selected is None:
                selected = key, value
    return selected


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
        f"Unknown uncertainty kind '{raw_value}'. Supported values: {allowed}."
    )


# ---------------------------------------------------------------------------
