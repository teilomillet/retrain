"""Full algorithm modes and registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from retrain.advantages.discover import _compute_discover_entropic
from retrain.advantages.plugin import (
    _callable_takes_no_positional_args,
    _coerce_algorithm_output,
    _is_dotted_plugin_path,
    _resolve_registry_spec,
    _validate_short_registry_name,
)
from retrain.advantages.types import (
    AdvantageResult,
    AlgorithmContext,
    AlgorithmContextFn,
    AlgorithmSpec,
)
from retrain.plugins.resolve import get_plugin_runtime, resolve_dotted_attribute

# 1c. Algorithm mode registry (built-ins + dotted-path plugins)
# ---------------------------------------------------------------------------


def _mapping_param(params: Mapping[str, object], key: str) -> dict[str, object]:
    raw = params.get(key)
    if isinstance(raw, Mapping):
        return cast(dict[str, object], dict(raw))
    return {}


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
        advantage_params = _mapping_param(ctx.params, "advantage_params")
        transform_params = _mapping_param(ctx.params, "transform_params")
        post_process = dict(transform_params)
        if "entropy_mask_rho" in ctx.params:
            post_process.setdefault("entropy_mask_rho", ctx.params["entropy_mask_rho"])
        from retrain.advantages.pipeline import compute_composable_advantages

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
            advantage_params=advantage_params,
            transform_params=transform_params,
            step=ctx.step,
            post_process_params=post_process,
            token_distributions_G=ctx.token_distributions_G,
            precomputed_entropies_G=ctx.precomputed_entropies_G,
        )

    return AlgorithmSpec(
        name=name,
        compute=_compute,
        needs_planning=needs_planning,
        uses_sepa_controller=uses_sepa_controller,
    )


_BUILTIN_ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "discover_entropic": AlgorithmSpec(
        name="discover_entropic",
        compute=_compute_discover_entropic,
        needs_planning=False,
        uses_sepa_controller=False,
    ),
    "ttt_discover_entropic": AlgorithmSpec(
        name="ttt_discover_entropic",
        compute=_compute_discover_entropic,
        needs_planning=False,
        uses_sepa_controller=False,
    ),
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
    "reinforce_pp_none": _make_builtin_algorithm_spec(
        "reinforce_pp_none",
        advantage_mode="reinforce_pp",
        transform_mode="none",
        needs_planning=False,
        uses_sepa_controller=False,
    ),
    "reinforce_pp_gtpo": _make_builtin_algorithm_spec(
        "reinforce_pp_gtpo",
        advantage_mode="reinforce_pp",
        transform_mode="gtpo",
        needs_planning=False,
        uses_sepa_controller=False,
    ),
    "reinforce_pp_gtpo_sepa": _make_builtin_algorithm_spec(
        "reinforce_pp_gtpo_sepa",
        advantage_mode="reinforce_pp",
        transform_mode="gtpo_sepa",
        needs_planning=True,
        uses_sepa_controller=True,
    ),
    "reinforce_pp_delight": _make_builtin_algorithm_spec(
        "reinforce_pp_delight",
        advantage_mode="reinforce_pp",
        transform_mode="delight",
        needs_planning=False,
        uses_sepa_controller=False,
    ),
    "reinforce_pp_delight_sepa": _make_builtin_algorithm_spec(
        "reinforce_pp_delight_sepa",
        advantage_mode="reinforce_pp",
        transform_mode="delight_sepa",
        needs_planning=False,
        uses_sepa_controller=True,
    ),
}

_ALGORITHM_SPEC_CACHE: dict[str, AlgorithmSpec] = {}


def register_algorithm_mode(name: str, spec_or_fn: AlgorithmSpec | AlgorithmContextFn) -> None:
    """Register or replace a short-name algorithm mode at runtime."""
    _validate_short_registry_name(name, label="Algorithm mode")
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
    return _is_dotted_plugin_path(algorithm_mode)


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
            built = obj()  # type: ignore[call-top-callable]
            if isinstance(built, AlgorithmSpec):
                return built
            if callable(built):
                return AlgorithmSpec(
                    name=dotted_path,
                    compute=built,  # type: ignore[invalid-argument-type]
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
            compute=obj,  # type: ignore[invalid-argument-type]
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
    return _resolve_registry_spec(
        algorithm_mode,
        builtins=_BUILTIN_ALGORITHM_SPECS,
        cache=_ALGORITHM_SPEC_CACHE,
        load_custom=_load_custom_algorithm_spec,
        builtin_names=get_builtin_algorithm_modes,
        config_key="algorithm_mode",
        custom_label="algorithms",
        example="my_module.my_algorithm",
    )


# ---------------------------------------------------------------------------

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
