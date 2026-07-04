"""retrain — TOML-first RLVR training framework for LLMs.

Public plugin surface: advantage/transform/algorithm/uncertainty specs
and their registration decorators, plus the TrainingFlow used by
``retrain trace``.
"""

# Enable hf_transfer for fast Hub downloads before anything imports
# huggingface_hub (which reads the flag once, at its import). Guarded: a no-op
# when the package is absent, and never overrides an explicit user setting.
from retrain._hf_transfer import enable_if_available as _enable_hf_transfer

_enable_hf_transfer()

from retrain.advantages import (
    AlgorithmContext as AlgorithmContext,
    AlgorithmOutput as AlgorithmOutput,
    AlgorithmSpec as AlgorithmSpec,
    AdvantageContext as AdvantageContext,
    AdvantageOutput as AdvantageOutput,
    AdvantageSpec as AdvantageSpec,
    PostProcessFn as PostProcessFn,
    TransformContext as TransformContext,
    TransformOutput as TransformOutput,
    TransformSpec as TransformSpec,
    UncertaintyContext as UncertaintyContext,
    UncertaintySpec as UncertaintySpec,
    entropy_mask_post_process as entropy_mask_post_process,
    surprisal_mask_post_process as surprisal_mask_post_process,
    register_advantage_mode as register_advantage_mode,
    register_algorithm_mode as register_algorithm_mode,
    register_transform_mode as register_transform_mode,
    register_uncertainty_kind as register_uncertainty_kind,
)
from retrain.training.flow import (
    TraceIssue as TraceIssue,
    TraceResult as TraceResult,
    TrainingFlow as TrainingFlow,
    build_flow as build_flow,
)
