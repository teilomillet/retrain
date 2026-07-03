"""Advantage computation public API."""

from retrain.advantages.algorithm import (
    _ALGORITHM_SPEC_CACHE,
    _BUILTIN_ALGORITHM_SPECS,
    _compute_discover_entropic,
    compute_algorithm_advantages,
    get_algorithm_spec,
    get_builtin_algorithm_modes,
    is_valid_algorithm_mode_name,
    register_algorithm_mode,
)
from retrain.advantages.constants import MAX_ENTROPY, MAX_SURPRISAL
from retrain.advantages.credit import (
    apply_entropy_mask,
    apply_gtpo_weighting,
    apply_hicra,
    apply_sepa_amplification,
    apply_sepa_amplification_clamped,
    apply_sepa_pooling,
    compute_entropy_mask_threshold,
    entropy_mask_post_process,
    surprisal_mask_post_process,
)
from retrain.advantages.delight import (
    _compute_delight_gate_metrics,
    _compute_delight_sepa_transform,
    _compute_delight_transform,
    _compute_hard_delight_transform,
    apply_delight_gating,
    apply_delight_sepa_gating,
    apply_hard_delight_gating,
    apply_hard_delight_sepa_gating,
)
from retrain.advantages.discover import (
    _discover_kl_to_uniform,
    _discover_softmax_probs,
    _resolve_discover_beta,
)
from retrain.advantages.episode import (
    _ADVANTAGE_SPEC_CACHE,
    _BUILTIN_ADVANTAGE_SPECS,
    apply_batch_advantage_normalization,
    compute_grpo_advantages,
    compute_maxrl_advantages,
    compute_reinforce_pp_advantages,
    get_advantage_spec,
    get_builtin_advantage_modes,
    is_valid_advantage_mode_name,
    register_advantage_mode,
)
from retrain.advantages.pipeline import compute_composable_advantages
from retrain.advantages.planning import DEFAULT_STRATEGIC_GRAMS, identify_planning_tokens
from retrain.advantages.stats import compute_entropy_stats, compute_surprisal_stats
from retrain.advantages.transform import (
    _BUILTIN_TRANSFORM_SPECS,
    _TRANSFORM_SPEC_CACHE,
    get_builtin_transform_modes,
    get_transform_spec,
    is_valid_transform_mode_name,
    register_transform_mode,
)
from retrain.advantages.types import (
    AdvantageContext,
    AdvantageOutput,
    AdvantageResult,
    AdvantageSpec,
    AlgorithmContext,
    AlgorithmContextFn,
    AlgorithmOutput,
    AlgorithmSpec,
    EntropyStats,
    EntropyTransformFn,
    EpisodeAdvantageComputeFn,
    EpisodeAdvantageFn,
    EpisodeAdvantageFnWithParams,
    EpisodeAdvantageRunner,
    PostProcessFn,
    TransformContext,
    TransformContextFn,
    TransformOutput,
    TransformSpec,
    UncertaintyComputeFn,
    UncertaintyContext,
    UncertaintySpec,
)
from retrain.advantages.uncertainty import (
    _BUILTIN_UNCERTAINTY_SPECS,
    _UNCERTAINTY_KIND_ALIASES,
    _UNCERTAINTY_KIND_PARAM_KEYS,
    _UNCERTAINTY_SPEC_CACHE,
    _compute_predictive_variance,
    _compute_shannon_entropy,
    _compute_surprisal,
    canonicalize_uncertainty_kind,
    get_builtin_uncertainty_kinds,
    get_uncertainty_kind_param,
    get_uncertainty_spec,
    is_valid_uncertainty_kind_name,
    register_uncertainty_kind,
)
