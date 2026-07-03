"""Delight token-gating public surface."""

from retrain.advantages.delight.eta import (
    _compute_delight_corrected_ordering_gaps,
    _iter_delight_rollouts,
    _resolve_delight_eta,
)
from retrain.advantages.delight.gate import (
    _sigmoid,
    apply_delight_gating,
    apply_delight_sepa_gating,
    apply_hard_delight_gating,
    apply_hard_delight_sepa_gating,
)
from retrain.advantages.delight.metric import _compute_delight_gate_metrics
from retrain.advantages.delight.scale import (
    _apply_delight_norm_mode,
    _coerce_delight_norm_mode,
    _mad_scale_surprisals,
    _median,
    _normalize_surprisals,
    _quantile,
    _resolve_delight_eta_mode,
    _resolve_delight_norm_mode,
)
from retrain.advantages.delight.transform import (
    _compute_delight_sepa_transform,
    _compute_delight_transform,
    _compute_hard_delight_transform,
)
