"""Back pressure: USL+Roofline adaptive batch sizing.

Universal Scalability Law (Gunther 2008) models throughput as:
    C(p) = p / (1 + sigma(p-1) + kappa*p*(p-1))

Follows typing.Protocol for the two implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol


# ---------------------------------------------------------------------------
# Free functions: USL math
# ---------------------------------------------------------------------------


def usl_throughput(p: float, sigma: float, kappa: float) -> float:
    """USL throughput: C(p) = p / (1 + sigma(p-1) + kappa*p*(p-1))."""
    denom = 1.0 + sigma * (p - 1.0) + kappa * p * (p - 1.0)
    if denom <= 0.0:
        return 0.0
    return p / denom


def usl_optimal_p(sigma: float, kappa: float) -> float:
    """Optimal concurrency: p* = sqrt((1-sigma)/kappa)."""
    if kappa <= 0.0:
        return 1.0
    num = 1.0 - sigma
    if num <= 0.0:
        return 1.0
    return math.sqrt(num / kappa)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StepObservation:
    """Metrics from one training step."""
    step_time_s: float = 0.0
    sample_time_s: float = 0.0
    train_time_s: float = 0.0
    num_datums: int = 0
    batch_size: int = 0
    group_size: int = 0
    total_tokens: int = 0
    loss: float = 0.0
    skipped: bool = False


@dataclass
class BackPressureDecision:
    """Controller output: recommended action and diagnostics."""
    action: str = "hold"
    recommended_batch_size: int = 0
    recommended_group_size: int = 0
    utilization: float = 0.0
    regime: str = "warmup"
    p_star: float = 1.0
    sigma: float = 0.0
    kappa: float = 0.0
    throughput: float = 0.0


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class BackPressure(Protocol):
    """Adaptive concurrency controller."""

    def observe(self, obs: StepObservation) -> None: ...
    def recommend(self) -> BackPressureDecision: ...
    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# NoOpBackPressure
# ---------------------------------------------------------------------------


class NoOpBackPressure:
    """Zero-cost opt-out: always returns 'hold'."""

    def observe(self, obs: StepObservation) -> None:
        pass

    def recommend(self) -> BackPressureDecision:
        return BackPressureDecision()

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# USLBackPressure
# ---------------------------------------------------------------------------


class USLBackPressure:
    """Full USL+Roofline adaptive controller.

    Fits sigma/kappa from observed throughput via linearized least-squares,
    classifies compute-bound vs memory-bound regimes, and recommends
    batch_size adjustments.
    """

    def __init__(
        self,
        warmup_steps: int = 10,
        ema_decay: float = 0.9,
        throttle_margin: float = 0.85,
        increase_margin: float = 0.5,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
        min_group_size: int = 2,
        max_group_size: int = 64,
        peak_gflops: float = 0.0,
        peak_bw_gb_s: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("ema_decay must be in [0, 1]")
        if not 0.0 < throttle_margin <= 1.0:
            raise ValueError("throttle_margin must be in (0, 1]")
        if not 0.0 < increase_margin <= 1.0:
            raise ValueError("increase_margin must be in (0, 1]")
        if min_batch_size < 1:
            raise ValueError("min_batch_size must be >= 1")
        if max_batch_size < min_batch_size:
            raise ValueError("max_batch_size must be >= min_batch_size")
        if min_group_size < 1:
            raise ValueError("min_group_size must be >= 1")
        if max_group_size < min_group_size:
            raise ValueError("max_group_size must be >= min_group_size")

        self.warmup_steps = warmup_steps
        self.ema_decay = ema_decay
        self.throttle_margin = throttle_margin
        self.increase_margin = increase_margin
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.peak_gflops = peak_gflops
        self.peak_bw_gb_s = peak_bw_gb_s
        self.eps = eps

        self._max_obs = 100  # sliding window for USL fit

        self.step_count = 0
        self._ema_initialized = False
        self.obs_p: list[float] = []
        self.obs_x: list[float] = []
        self.sigma = 0.0
        self.kappa = 0.0
        self.p_star = 1.0
        self.fitted_lambda = 1.0  # recovered serial throughput coefficient
        self.ema_throughput = 0.0
        self.prev_ema_throughput = 0.0
        self.regime = "warmup"
        self.last_batch_size = min_batch_size
        self.last_group_size = min_group_size

    def observe(self, obs: StepObservation) -> None:
        """Record a step observation and refit USL parameters."""
        self.step_count += 1
        self.last_batch_size = obs.batch_size
        self.last_group_size = obs.group_size

        if obs.skipped:
            return

        p = float(obs.batch_size * obs.group_size)
        throughput = (
            float(obs.total_tokens) / obs.step_time_s
            if obs.step_time_s > self.eps
            else 0.0
        )

        # Update EMA — seed on first real (non-skipped) observation
        self.prev_ema_throughput = self.ema_throughput
        if not self._ema_initialized:
            self.ema_throughput = throughput
            self._ema_initialized = True
        else:
            self.ema_throughput = (
                self.ema_decay * self.ema_throughput
                + (1.0 - self.ema_decay) * throughput
            )

        # Sliding window: keep only the last _max_obs observations
        self.obs_p.append(p)
        self.obs_x.append(throughput)
        if len(self.obs_p) > self._max_obs:
            self.obs_p = self.obs_p[-self._max_obs:]
            self.obs_x = self.obs_x[-self._max_obs:]

        if len(self.obs_p) >= 3:
            self._fit_usl()

        self._classify_regime(p)

    def _fit_usl(self) -> None:
        """3-parameter polynomial fit via Cramer's rule."""
        s0 = s1 = s2 = s3 = s4 = sy = spy = sp2y = 0.0

        for p, x in zip(self.obs_p, self.obs_x):
            if x < self.eps:
                continue
            y = p / x
            p2 = p * p
            s0 += 1.0
            s1 += p
            s2 += p2
            s3 += p2 * p
            s4 += p2 * p2
            sy += y
            spy += p * y
            sp2y += p2 * y

        if s0 < 3.0:
            return

        det = (
            s0 * (s2 * s4 - s3 * s3)
            - s1 * (s1 * s4 - s3 * s2)
            + s2 * (s1 * s3 - s2 * s2)
        )
        if abs(det) < self.eps:
            return

        det_a = (
            sy * (s2 * s4 - s3 * s3)
            - s1 * (spy * s4 - sp2y * s3)
            + s2 * (spy * s3 - sp2y * s2)
        )
        det_b = (
            s0 * (spy * s4 - sp2y * s3)
            - sy * (s1 * s4 - s3 * s2)
            + s2 * (s1 * sp2y - spy * s2)
        )
        det_c = (
            s0 * (s2 * sp2y - s3 * spy)
            - s1 * (s1 * sp2y - spy * s2)
            + sy * (s1 * s3 - s2 * s2)
        )

        a_val = det_a / det
        b_val = det_b / det
        c_val = det_c / det

        abc = a_val + b_val + c_val
        if abc < self.eps:
            return

        sigma_raw = (b_val + c_val) / abc
        kappa_raw = c_val / abc

        self.sigma = max(0.0, min(1.0, sigma_raw))
        self.kappa = max(0.0, kappa_raw)
        self.fitted_lambda = 1.0 / abc  # recover serial throughput coefficient
        self.p_star = usl_optimal_p(self.sigma, self.kappa)

    def _classify_regime(self, current_p: float) -> None:
        """Classify the current operating regime."""
        if self.step_count <= self.warmup_steps:
            self.regime = "warmup"
            return

        if self.kappa > self.eps:
            threshold = self.p_star * self.throttle_margin
            if current_p > threshold and self.p_star > 1.0:
                self.regime = "retrograde"
                return

        if self.peak_gflops > 0.0 and self.peak_bw_gb_s > 0.0:
            predicted = usl_throughput(current_p, self.sigma, self.kappa)
            ratio = predicted / (self.peak_gflops * 1e9 + self.eps)
            self.regime = "compute_bound" if ratio > 0.7 else "memory_bound"
            return

        if self.ema_throughput > self.prev_ema_throughput * 1.05:
            self.regime = "memory_bound"
        elif self.ema_throughput < self.prev_ema_throughput * 0.95:
            self.regime = "retrograde"
        else:
            self.regime = "optimal"

    def recommend(self) -> BackPressureDecision:
        """Produce a back pressure decision based on current state."""
        decision = BackPressureDecision(
            sigma=self.sigma,
            kappa=self.kappa,
            p_star=self.p_star,
            throughput=self.ema_throughput,
            regime=self.regime,
            recommended_batch_size=self.last_batch_size,
            recommended_group_size=self.last_group_size,
        )

        if self.step_count <= self.warmup_steps:
            decision.action = "hold"
            decision.utilization = 0.0
            return decision

        # Utilization: actual throughput / predicted peak (using recovered lambda)
        peak_throughput = self.fitted_lambda * usl_throughput(
            self.p_star, self.sigma, self.kappa
        )
        if peak_throughput > self.eps:
            decision.utilization = self.ema_throughput / peak_throughput
        else:
            decision.utilization = 1.0

        current_p = float(self.last_batch_size * self.last_group_size)

        if self.regime == "retrograde":
            decision.action = "throttle"
            target_p = self.p_star * self.throttle_margin
            new_batch = int(target_p / self.last_group_size)
            decision.recommended_batch_size = max(
                self.min_batch_size, min(self.max_batch_size, new_batch)
            )
        elif current_p < self.p_star * self.increase_margin:
            decision.action = "increase"
            target_p = self.p_star * self.increase_margin
            new_batch = int(target_p / self.last_group_size)
            decision.recommended_batch_size = max(
                self.min_batch_size, min(self.max_batch_size, new_batch)
            )
        else:
            decision.action = "hold"

        return decision

    def reset(self) -> None:
        """Clear all state, keeping config."""
        self.step_count = 0
        self._ema_initialized = False
        self.obs_p = []
        self.obs_x = []
        self.sigma = 0.0
        self.kappa = 0.0
        self.p_star = 1.0
        self.fitted_lambda = 1.0
        self.ema_throughput = 0.0
        self.prev_ema_throughput = 0.0
        self.regime = "warmup"
        self.last_batch_size = self.min_batch_size
        self.last_group_size = self.min_group_size
