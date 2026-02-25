"""JAX-style construct-and-trace for the training flow.

Eagerly resolve all training components, then probe with synthetic data
to catch incompatibilities before committing GPU time.

    flow = build_flow(config, gpu=False)
    result = flow.trace()
    if not result.ok:
        ...  # fix config
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retrain.advantages import (
    AlgorithmSpec,
    AdvantageSpec,
    TransformSpec,
    compute_algorithm_advantages,
    compute_composable_advantages,
    get_advantage_spec,
    get_algorithm_spec,
    get_transform_spec,
)
from retrain.backend_definitions import (
    BackendCapabilities,
    backend_capability_source,
    resolve_backend_capabilities,
)
from retrain.config import TrainConfig
from retrain.sepa import SEPAController

# ── Constants moved from trainer.py ──────────────────────────────────────

_SCALAR_BACKEND_DISALLOWED_BUILTIN_TRANSFORM_MODES = frozenset(
    {
        "gtpo",
        "entropy_mask",
        "gtpo_hicra",
        "gtpo_sepa",
        "gtpo_sepa_amp",
        "gtpo_sepa_amp_c",
    }
)
_SCALAR_BACKEND_DISALLOWED_BUILTIN_ALGORITHM_MODES = frozenset(
    {
        "maxrl_gtpo",
        "maxrl_gtpo_hicra",
        "maxrl_gtpo_sepa",
    }
)

_FLOW_PROBE_CASES = (
    {
        "rewards_G": [1.0, 0.0],
        "logprobs_G": [[-0.2, -1.1, -0.4], [-0.3, -0.6, -1.3]],
        "planning_masks_G": [[0, 1, 0], [1, 0, 1]],
        "sepa_lambda": 0.7,
        "step": 7,
    },
    {
        "rewards_G": [0.9, 0.1],
        "logprobs_G": [[-0.9, -0.1, -0.7], [-0.2, -1.4, -0.3]],
        "planning_masks_G": [[1, 0, 0], [0, 1, 0]],
        "sepa_lambda": 0.35,
        "step": 29,
    },
)
_UNIFORMITY_EPS = 1e-6


# ── Data types ───────────────────────────────────────────────────────────

@dataclass
class TraceIssue:
    severity: str   # "error" | "warning"
    category: str   # "compat" | "probe" | "dep" | "config"
    message: str


@dataclass
class TraceResult:
    issues: list[TraceIssue] = field(default_factory=list)
    probe_cases_run: int = 0
    probe_cases_passed: int = 0

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)


# ── TrainingFlow ─────────────────────────────────────────────────────────

@dataclass
class TrainingFlow:
    config: TrainConfig

    # Tier 1 — resolved without GPU (always set)
    algorithm_spec: AlgorithmSpec | None
    advantage_spec: AdvantageSpec | None
    transform_spec: TransformSpec | None
    backend_capabilities: BackendCapabilities
    backend_capability_source: str
    needs_planning: bool
    uses_sepa_controller: bool
    condition_label: str

    # Tier 2 — resolved with GPU (None when gpu=False)
    backend: object | None = None
    planning_detector: object | None = None
    backpressure: object | None = None
    sepa_controller: SEPAController | None = None

    def trace(self) -> TraceResult:
        """Probe the flow with synthetic data, collecting all issues at once."""
        issues: list[TraceIssue] = []
        probe_run = 0
        probe_passed = 0

        # Check 1 — disallowed built-in modes for scalar backends
        if not self.backend_capabilities.preserves_token_advantages:
            if self.algorithm_spec is not None:
                if (
                    self.config.algorithm_mode
                    in _SCALAR_BACKEND_DISALLOWED_BUILTIN_ALGORITHM_MODES
                ):
                    issues.append(TraceIssue(
                        severity="error",
                        category="compat",
                        message=(
                            f"backend='{self.config.backend}' with built-in "
                            f"algorithm_mode='{self.config.algorithm_mode}' "
                            "is disallowed: backend accepts one scalar advantage "
                            "per sample, so built-in token-varying credit assignment "
                            "would be reduced/lossy. Use a scalar-safe built-in mode "
                            "(grpo_none|maxrl_none), switch backend, or use a custom "
                            "dotted algorithm_mode with explicit scalar semantics."
                        ),
                    ))
            elif (
                self.config.transform_mode
                in _SCALAR_BACKEND_DISALLOWED_BUILTIN_TRANSFORM_MODES
            ):
                issues.append(TraceIssue(
                    severity="error",
                    category="compat",
                    message=(
                        f"backend='{self.config.backend}' with built-in "
                        f"transform_mode='{self.config.transform_mode}' "
                        "is disallowed: backend accepts one scalar advantage "
                        "per sample, so built-in token-varying transforms "
                        "would be reduced/lossy. Use transform_mode='none', "
                        "switch backend, or use a custom dotted transform_mode."
                    ),
                ))

        # Check 2 — synthetic probe
        for case in _FLOW_PROBE_CASES:
            probe_run += 1
            try:
                result = _run_probe(self, case)
            except Exception as exc:
                issues.append(TraceIssue(
                    severity="error",
                    category="probe",
                    message=(
                        f"Advantage-flow probe failed: {exc}"
                    ),
                ))
                continue

            if (
                not self.backend_capabilities.preserves_token_advantages
                and not _token_advs_are_uniform(result.token_advs)
            ):
                issues.append(TraceIssue(
                    severity="error",
                    category="probe",
                    message=(
                        f"backend='{self.config.backend}' does not preserve "
                        "token-level advantages, but probe detected "
                        f"token-varying advantages for "
                        f"'{self.condition_label}'."
                    ),
                ))
                continue

            probe_passed += 1

        # Check 3 — planning dependency
        if self.needs_planning and self.config.planning_detector in ("none", ""):
            issues.append(TraceIssue(
                severity="warning",
                category="dep",
                message=(
                    "Algorithm/transform needs planning masks but "
                    f"planning_detector='{self.config.planning_detector}'. "
                    "Planning masks will be all-zeros."
                ),
            ))

        # Check 4 — SEPA consistency
        if self.uses_sepa_controller and self.config.sepa_steps <= 0:
            issues.append(TraceIssue(
                severity="warning",
                category="config",
                message=(
                    "Transform uses SEPA controller but sepa_steps <= 0. "
                    "SEPA lambda will stay at 0."
                ),
            ))

        # Check 5 — backend loss semantics
        if not self.backend_capabilities.reports_sync_loss:
            issues.append(TraceIssue(
                severity="warning",
                category="config",
                message=(
                    f"backend='{self.config.backend}' reports placeholder "
                    "loss values (async design). Loss metrics will not "
                    "reflect true training loss."
                ),
            ))

        return TraceResult(
            issues=issues,
            probe_cases_run=probe_run,
            probe_cases_passed=probe_passed,
        )


# ── Helpers ──────────────────────────────────────────────────────────────

def _token_advs_are_uniform(
    token_advs: list[list[float]],
    *,
    eps: float = _UNIFORMITY_EPS,
) -> bool:
    """True if each sequence has effectively one scalar value."""
    for seq in token_advs:
        if len(seq) <= 1:
            continue
        lo = min(seq)
        hi = max(seq)
        if (hi - lo) > eps:
            return False
    return True


def _run_probe(
    flow: TrainingFlow,
    case: dict[str, object],
) -> object:
    """Run one synthetic probe case through the advantage pipeline."""
    config = flow.config
    rewards = list(case["rewards_G"])  # type: ignore[arg-type]
    logprobs = [list(seq) for seq in case["logprobs_G"]]  # type: ignore[union-attr]
    masks = [list(seq) for seq in case["planning_masks_G"]]  # type: ignore[union-attr]
    sepa_lambda = float(case["sepa_lambda"])  # type: ignore[arg-type]
    step = int(case["step"])  # type: ignore[arg-type]

    if config.algorithm_mode:
        return compute_algorithm_advantages(
            rewards_G=rewards,
            logprobs_G=logprobs,
            planning_masks_G=masks,
            algorithm_mode=config.algorithm_mode,
            params=config.effective_algorithm_params,
            gtpo_beta=config.gtpo_beta,
            hicra_alpha=config.hicra_alpha,
            sepa_lambda=sepa_lambda,
            step=step,
            token_distributions_G=None,
        )
    return compute_composable_advantages(
        rewards_G=rewards,
        logprobs_G=logprobs,
        planning_masks_G=masks,
        advantage_mode=config.advantage_mode,
        transform_mode=config.transform_mode,
        gtpo_beta=config.gtpo_beta,
        hicra_alpha=config.hicra_alpha,
        sepa_lambda=sepa_lambda,
        advantage_params=config.effective_advantage_params,
        transform_params=config.transform_params,
        step=step,
        post_process_params=config.post_process_params,
        token_distributions_G=None,
    )


def _condition_label(config: TrainConfig) -> str:
    """Human-readable algorithm condition label."""
    if config.algorithm_mode:
        return config.algorithm_mode
    return f"{config.advantage_mode}+{config.transform_mode}"


# ── build_flow ───────────────────────────────────────────────────────────

def build_flow(config: TrainConfig, *, gpu: bool = False) -> TrainingFlow:
    """Eagerly resolve all training-flow components.

    Tier 1 (always): specs, capabilities, flags — no hardware needed.
    Tier 2 (gpu=True): backend, detector, SEPA controller, backpressure.
    """
    # ── Tier 1 ────────────────────────────────────────────────────────
    algorithm_spec: AlgorithmSpec | None = None
    advantage_spec: AdvantageSpec | None = None
    transform_spec: TransformSpec | None = None

    if config.algorithm_mode:
        algorithm_spec = get_algorithm_spec(config.algorithm_mode)
        needs_planning = algorithm_spec.needs_planning
        uses_sepa = algorithm_spec.uses_sepa_controller
    else:
        advantage_spec = get_advantage_spec(config.advantage_mode)
        transform_spec = get_transform_spec(config.transform_mode)
        needs_planning = transform_spec.needs_planning
        uses_sepa = transform_spec.uses_sepa_controller

    backend_caps = resolve_backend_capabilities(
        config.backend, config.backend_options,
    )
    cap_source = backend_capability_source(
        config.backend, config.backend_options,
    )
    label = _condition_label(config)

    # ── Tier 2 (GPU) ─────────────────────────────────────────────────
    backend = None
    planning_detector = None
    backpressure_obj = None
    sepa_ctrl = None

    if gpu:
        from retrain.registry import get_registry

        backend = get_registry("backend").create(config.backend, config)
        planning_detector = get_registry("planning_detector").create(
            config.planning_detector, config,
        )
        sepa_ctrl = SEPAController(
            sepa_steps=config.sepa_steps,
            sepa_schedule=config.sepa_schedule,
            sepa_delay_steps=config.sepa_delay_steps,
            sepa_correct_rate_gate=config.sepa_correct_rate_gate,
        )
        bp_name = "usl" if config.bp_enabled else "noop"
        backpressure_obj = get_registry("backpressure").create(bp_name, config)

    return TrainingFlow(
        config=config,
        algorithm_spec=algorithm_spec,
        advantage_spec=advantage_spec,
        transform_spec=transform_spec,
        backend_capabilities=backend_caps,
        backend_capability_source=cap_source,
        needs_planning=needs_planning,
        uses_sepa_controller=uses_sepa,
        condition_label=label,
        backend=backend,
        planning_detector=planning_detector,
        backpressure=backpressure_obj,
        sepa_controller=sepa_ctrl,
    )
