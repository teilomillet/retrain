"""Back pressure: USL+Roofline adaptive batch sizing.

Universal Scalability Law (Gunther 2008) models throughput as:
    C(p) = p / (1 + sigma(p-1) + kappa*p*(p-1))

Follows typing.Protocol for the two implementations.
"""

from __future__ import annotations

import math
from collections import deque
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

    Optimizations over naive implementation:
    - O(1) incremental Cramer sums (no per-step re-scan of observations)
    - deque sliding window (no list slice-copy allocation)
    - Model-based regime classification (compares EMA against USL prediction)
    - Converges toward throttle_margin * p* (not 0.5 * p*)
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
        self.obs_p: deque[float] = deque(maxlen=self._max_obs)
        self.obs_x: deque[float] = deque(maxlen=self._max_obs)

        # Incremental sums for O(1) Cramer's rule updates.
        # y_i = p_i / x_i  (linearized USL: p/X(p) = A + Bp + Cp²)
        self._s0 = 0.0   # count of valid (x >= eps) observations
        self._s1 = 0.0   # Σ p
        self._s2 = 0.0   # Σ p²
        self._s3 = 0.0   # Σ p³
        self._s4 = 0.0   # Σ p⁴
        self._sy = 0.0   # Σ y
        self._spy = 0.0  # Σ p·y
        self._sp2y = 0.0 # Σ p²·y

        self.sigma = 0.0
        self.kappa = 0.0
        self.p_star = 1.0
        self.fitted_lambda = 1.0  # recovered serial throughput coefficient
        self.ema_throughput = 0.0
        self.prev_ema_throughput = 0.0
        self.regime = "warmup"
        self.last_batch_size = min_batch_size
        self.last_group_size = min_group_size

    # -- Incremental sum helpers ------------------------------------------

    def _update_sums(self, p: float, x: float, sign: float) -> None:
        """Add (+1) or subtract (-1) one observation's contributions."""
        if x < self.eps:
            return
        y = p / x
        p2 = p * p
        self._s0 += sign
        self._s1 += sign * p
        self._s2 += sign * p2
        self._s3 += sign * p2 * p
        self._s4 += sign * p2 * p2
        self._sy += sign * y
        self._spy += sign * p * y
        self._sp2y += sign * p2 * y

    # -- Core methods -----------------------------------------------------

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

        # Sliding window: evict oldest if full, then append
        if len(self.obs_p) == self._max_obs:
            self._update_sums(self.obs_p[0], self.obs_x[0], -1.0)
        self.obs_p.append(p)
        self.obs_x.append(throughput)
        self._update_sums(p, throughput, +1.0)

        if self._s0 >= 3.0:
            self._fit_usl()

        self._classify_regime(p)

    def _fit_usl(self) -> None:
        """Solve 3-parameter polynomial fit via Cramer's rule.

        O(1) — reads from incrementally maintained sums.
        Linearized model: p/X(p) = A + B·p + C·p²
        where A = 1/λ, B = σ/λ - 1/λ, C = κ/λ.
        """
        s0 = self._s0
        s1 = self._s1
        s2 = self._s2
        s3 = self._s3
        s4 = self._s4
        sy = self._sy
        spy = self._spy
        sp2y = self._sp2y

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

        # Primary: kappa-based retrograde detection
        if self.kappa > self.eps:
            threshold = self.p_star * self.throttle_margin
            if current_p > threshold and self.p_star > 1.0:
                self.regime = "retrograde"
                return

        # Roofline hints (if hardware specs provided)
        if self.peak_gflops > 0.0 and self.peak_bw_gb_s > 0.0:
            predicted = usl_throughput(current_p, self.sigma, self.kappa)
            ratio = predicted / (self.peak_gflops * 1e9 + self.eps)
            self.regime = "compute_bound" if ratio > 0.7 else "memory_bound"
            return

        # Model-based: compare EMA throughput against fitted USL prediction.
        # The old code compared consecutive EMA values (ema vs prev_ema * 1.05),
        # which is dead code at ema_decay=0.9 — a 50%+ throughput spike needed
        # to move the EMA by just 5%.
        if self._s0 >= 3.0 and self.fitted_lambda > self.eps:
            predicted = self.fitted_lambda * usl_throughput(
                current_p, self.sigma, self.kappa
            )
            if predicted > self.eps:
                ratio = self.ema_throughput / predicted
                if ratio > 1.1:
                    # Outperforming model → headroom exists
                    self.regime = "memory_bound"
                elif ratio < 0.8:
                    # Underperforming model → degradation
                    self.regime = "retrograde"
                else:
                    self.regime = "optimal"
                return

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

        # Both throttle and increase converge toward throttle_margin * p*.
        # This is the highest safe operating point — just below the
        # retrograde cliff.  The old design targeted increase_margin * p*
        # (default 0.5 * p*), leaving ~13-15% throughput on the table.
        safe_target = self.p_star * self.throttle_margin

        if self.regime == "retrograde":
            decision.action = "throttle"
            new_batch = int(safe_target / self.last_group_size)
            decision.recommended_batch_size = max(
                self.min_batch_size, min(self.max_batch_size, new_batch)
            )
        elif current_p < safe_target * self.increase_margin:
            # Trigger increase when well below the safe target.
            # increase_margin acts as hysteresis gap to prevent oscillation.
            decision.action = "increase"
            new_batch = int(safe_target / self.last_group_size)
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
        self.obs_p.clear()
        self.obs_x.clear()
        self._s0 = self._s1 = self._s2 = self._s3 = self._s4 = 0.0
        self._sy = self._spy = self._sp2y = 0.0
        self.sigma = 0.0
        self.kappa = 0.0
        self.p_star = 1.0
        self.fitted_lambda = 1.0
        self.ema_throughput = 0.0
        self.prev_ema_throughput = 0.0
        self.regime = "warmup"
        self.last_batch_size = self.min_batch_size
        self.last_group_size = self.min_group_size
