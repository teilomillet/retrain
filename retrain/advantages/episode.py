"""Episode-level advantage modes and registry."""

from __future__ import annotations

import math

from retrain.advantages.plugin import (
    _as_advantage_spec,
    _callable_takes_no_positional_args,
    _is_dotted_plugin_path,
    _resolve_registry_spec,
    _validate_short_registry_name,
)
from retrain.advantages.types import AdvantageSpec, EpisodeAdvantageComputeFn
from retrain.plugins.resolve import resolve_dotted_attribute

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
# 0b. REINFORCE++ advantages (group-mean subtraction, same as GRPO step 1)
# ---------------------------------------------------------------------------
# REINFORCE++ (Hu 2025, arxiv:2501.03262) uses the same per-group mean
# subtraction as GRPO but follows it with a global batch normalization pass.
# The episode-level computation is identical — the batch normalization is
# applied separately in the trainer via apply_batch_advantage_normalization().


def compute_reinforce_pp_advantages(rewards: list[float]) -> list[float]:
    """A_i = r_i - mean(r).

    Identical to GRPO at the per-group level. The key REINFORCE++ innovation
    is the batch-level normalization pass applied afterward — see
    ``apply_batch_advantage_normalization``.
    """
    return compute_grpo_advantages(rewards)


# ---------------------------------------------------------------------------
# 0c. Batch-level advantage normalization (REINFORCE++ step 2)
# ---------------------------------------------------------------------------


def apply_batch_advantage_normalization(
    all_advantages: list[list[float]],
    eps: float = 1e-8,
) -> tuple[list[list[float]], dict[str, float]]:
    """Normalize all non-zero advantages across the full batch.

    REINFORCE++ (Hu 2025) fixes GRPO's biased local normalization by
    normalizing advantages across the entire batch rather than within each
    prompt's group.  This makes the estimator effectively unbiased as the
    batch size grows (the bias vanishes as N→∞).

    Two-step process (per the paper):
      Step 1 (already done by advantage_mode): A' = r_i - mean_group(r)
      Step 2 (this function):
        A_norm = (A' - mean_batch(A')) / (std_batch(A') + eps)

    Only normalizes tokens with non-zero advantages (prompt tokens are padded
    with 0.0 and should not participate in batch statistics).

    Note: uses ``a != 0.0`` as a sentinel to distinguish prompt padding from
    response tokens.  This is safe because the only way a response advantage
    is exactly 0.0 is when all group rewards are identical, in which case
    exclusion from batch statistics is the correct behavior (no gradient
    signal).  If a future transform produces legitimate 0.0 advantages for
    individual response tokens, this function should be updated to accept a
    separate mask.

    Returns:
        (normalized_advantages, metrics_dict)
        metrics_dict contains batch_adv_mean, batch_adv_std, batch_adv_count.
    """
    # Collect all non-zero advantage values across the batch
    all_values: list[float] = []
    for seq in all_advantages:
        for a in seq:
            if a != 0.0:
                all_values.append(a)

    n = len(all_values)
    metrics: dict[str, float] = {
        "batch_adv_count": float(n),
        "batch_adv_mean": 0.0,
        "batch_adv_std": 0.0,
    }
    if n == 0:
        return all_advantages, metrics

    mean_val = sum(all_values) / n
    var_val = sum((v - mean_val) ** 2 for v in all_values) / n
    std_val = math.sqrt(var_val)
    metrics["batch_adv_mean"] = mean_val
    metrics["batch_adv_std"] = std_val

    # If std is near zero, all advantages are ~equal — just center them
    if std_val < eps:
        result: list[list[float]] = []
        for seq in all_advantages:
            result.append([(a - mean_val) if a != 0.0 else 0.0 for a in seq])
        return result, metrics

    denom = std_val + eps
    result = []
    for seq in all_advantages:
        result.append([(a - mean_val) / denom if a != 0.0 else 0.0 for a in seq])
    return result, metrics


# ---------------------------------------------------------------------------
# 1. MaxRL advantages (inverse success-rate reweighting)
# ---------------------------------------------------------------------------


def compute_maxrl_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
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
    "reinforce_pp": _as_advantage_spec("reinforce_pp", compute_reinforce_pp_advantages),
    "maxrl": _as_advantage_spec("maxrl", compute_maxrl_advantages),
}

_ADVANTAGE_SPEC_CACHE: dict[str, AdvantageSpec] = {}


def register_advantage_mode(name: str, compute: EpisodeAdvantageComputeFn) -> None:
    """Register or replace a short-name episode advantage mode at runtime."""
    _validate_short_registry_name(name, label="Advantage mode")
    _BUILTIN_ADVANTAGE_SPECS[name] = _as_advantage_spec(name, compute)
    _ADVANTAGE_SPEC_CACHE.pop(name, None)


def get_builtin_advantage_modes() -> list[str]:
    """Return sorted built-in advantage mode names."""
    return sorted(_BUILTIN_ADVANTAGE_SPECS)


def is_valid_advantage_mode_name(advantage_mode: str) -> bool:
    """True for built-ins or dotted plugin paths (`module.attr`)."""
    if advantage_mode in _BUILTIN_ADVANTAGE_SPECS:
        return True
    return _is_dotted_plugin_path(advantage_mode)


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
            built = obj()  # type: ignore[call-top-callable]
            if isinstance(built, AdvantageSpec):
                return built
            if callable(built):
                return _as_advantage_spec(dotted_path, built)  # type: ignore[invalid-argument-type]
            raise TypeError(
                f"advantage_mode '{dotted_path}' factory returned "
                f"{type(built).__name__}, expected AdvantageSpec or callable."
            )
        return _as_advantage_spec(dotted_path, obj)  # type: ignore[invalid-argument-type]

    raise TypeError(
        f"advantage_mode '{dotted_path}' must resolve to AdvantageSpec "
        f"or a callable, got {type(obj).__name__}."
    )


def get_advantage_spec(advantage_mode: str) -> AdvantageSpec:
    """Resolve an advantage mode to a behavior spec."""
    return _resolve_registry_spec(
        advantage_mode,
        builtins=_BUILTIN_ADVANTAGE_SPECS,
        cache=_ADVANTAGE_SPEC_CACHE,
        load_custom=_load_custom_advantage_spec,
        builtin_names=get_builtin_advantage_modes,
        config_key="advantage_mode",
        custom_label="advantages",
        example="my_module.my_advantage",
    )


# ---------------------------------------------------------------------------
