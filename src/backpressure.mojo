"""Back pressure trait and USL+Roofline built-in implementation.

Universal Scalability Law (Gunther 2008) models throughput as:
    C(p) = p / (1 + σ(p-1) + κp(p-1))
where σ = contention, κ = coherency, p = concurrency.

Roofline model (Williams et al. 2009) classifies compute-bound vs
memory-bound regimes based on operational intensity.

Follows the same composable trait pattern as RewardFn, DataSource, etc.
"""

from math import sqrt, abs


# ---------------------------------------------------------------------------
# Free functions: USL math
# ---------------------------------------------------------------------------


fn usl_throughput(p: Float64, sigma: Float64, kappa: Float64) -> Float64:
    """USL throughput: C(p) = p / (1 + σ(p-1) + κp(p-1))."""
    var denom = 1.0 + sigma * (p - 1.0) + kappa * p * (p - 1.0)
    if denom <= 0.0:
        return 0.0
    return p / denom


fn usl_optimal_p(sigma: Float64, kappa: Float64) -> Float64:
    """Optimal concurrency: p* = sqrt((1-σ)/κ).

    Returns 1.0 if κ ≤ 0 (no coherency penalty → linear scaling).
    """
    if kappa <= 0.0:
        return 1.0
    var num = 1.0 - sigma
    if num <= 0.0:
        return 1.0
    return sqrt(num / kappa)


# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------


@fieldwise_init
struct StepObservation(Copyable, Movable):
    """Metrics from one training step."""

    var step_time_s: Float64
    var sample_time_s: Float64
    var train_time_s: Float64
    var num_datums: Int
    var batch_size: Int
    var group_size: Int
    var total_tokens: Int
    var loss: Float64
    var skipped: Bool

    fn __init__(out self):
        self.step_time_s = 0.0
        self.sample_time_s = 0.0
        self.train_time_s = 0.0
        self.num_datums = 0
        self.batch_size = 0
        self.group_size = 0
        self.total_tokens = 0
        self.loss = 0.0
        self.skipped = False


@fieldwise_init
struct BackPressureDecision(Copyable, Movable, Writable):
    """Controller output: recommended action and diagnostics."""

    var action: String  # "hold", "throttle", "increase"
    var recommended_batch_size: Int
    var recommended_group_size: Int
    var utilization: Float64
    var regime: String  # "warmup", "compute_bound", "memory_bound", "optimal", "retrograde"
    var p_star: Float64
    var sigma: Float64
    var kappa: Float64
    var throughput: Float64

    fn __init__(out self):
        self.action = "hold"
        self.recommended_batch_size = 0
        self.recommended_group_size = 0
        self.utilization = 0.0
        self.regime = "warmup"
        self.p_star = 1.0
        self.sigma = 0.0
        self.kappa = 0.0
        self.throughput = 0.0

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "BackPressureDecision(",
            "action=", self.action,
            ", regime=", self.regime,
            ", p_star=", self.p_star,
            ", sigma=", self.sigma,
            ", kappa=", self.kappa,
            ", utilization=", self.utilization,
            ", throughput=", self.throughput,
            ")",
        )


# ---------------------------------------------------------------------------
# Trait
# ---------------------------------------------------------------------------


trait BackPressure(Movable):
    """Adaptive concurrency controller.

    observe() feeds step metrics; recommend() returns a decision.
    """

    fn observe(mut self, obs: StepObservation) raises:
        ...

    fn recommend(self) -> BackPressureDecision:
        ...

    fn reset(mut self):
        ...


# ---------------------------------------------------------------------------
# NoOpBackPressure
# ---------------------------------------------------------------------------


struct NoOpBackPressure(BackPressure):
    """Zero-cost opt-out: always returns 'hold'."""

    fn __init__(out self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass

    fn observe(mut self, obs: StepObservation) raises:
        pass

    fn recommend(self) -> BackPressureDecision:
        return BackPressureDecision()

    fn reset(mut self):
        pass


# ---------------------------------------------------------------------------
# USLBackPressure
# ---------------------------------------------------------------------------


struct USLBackPressure(BackPressure):
    """Full USL+Roofline adaptive controller.

    Fits σ/κ from observed throughput via linearized least-squares,
    classifies compute-bound vs memory-bound regimes, and recommends
    batch_size adjustments.
    """

    # Config
    var warmup_steps: Int
    var ema_decay: Float64
    var throttle_margin: Float64
    var increase_margin: Float64
    var min_batch_size: Int
    var max_batch_size: Int
    var min_group_size: Int
    var max_group_size: Int
    var peak_gflops: Float64
    var peak_bw_gb_s: Float64
    var eps: Float64

    # State
    var step_count: Int
    var obs_p: List[Float64]       # concurrency (batch_size * group_size)
    var obs_x: List[Float64]       # throughput (tokens / step_time)
    var sigma: Float64
    var kappa: Float64
    var p_star: Float64
    var ema_throughput: Float64
    var prev_ema_throughput: Float64
    var regime: String
    var last_batch_size: Int
    var last_group_size: Int

    fn __init__(
        out self,
        warmup_steps: Int = 10,
        ema_decay: Float64 = 0.9,
        throttle_margin: Float64 = 0.85,
        increase_margin: Float64 = 0.5,
        min_batch_size: Int = 1,
        max_batch_size: Int = 64,
        min_group_size: Int = 2,
        max_group_size: Int = 64,
        peak_gflops: Float64 = 0.0,
        peak_bw_gb_s: Float64 = 0.0,
        eps: Float64 = 1e-8,
    ):
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

        self.step_count = 0
        self.obs_p = List[Float64]()
        self.obs_x = List[Float64]()
        self.sigma = 0.0
        self.kappa = 0.0
        self.p_star = 1.0
        self.ema_throughput = 0.0
        self.prev_ema_throughput = 0.0
        self.regime = "warmup"
        self.last_batch_size = min_batch_size
        self.last_group_size = min_group_size

    fn __moveinit__(out self, deinit take: Self):
        self.warmup_steps = take.warmup_steps
        self.ema_decay = take.ema_decay
        self.throttle_margin = take.throttle_margin
        self.increase_margin = take.increase_margin
        self.min_batch_size = take.min_batch_size
        self.max_batch_size = take.max_batch_size
        self.min_group_size = take.min_group_size
        self.max_group_size = take.max_group_size
        self.peak_gflops = take.peak_gflops
        self.peak_bw_gb_s = take.peak_bw_gb_s
        self.eps = take.eps
        self.step_count = take.step_count
        self.obs_p = take.obs_p^
        self.obs_x = take.obs_x^
        self.sigma = take.sigma
        self.kappa = take.kappa
        self.p_star = take.p_star
        self.ema_throughput = take.ema_throughput
        self.prev_ema_throughput = take.prev_ema_throughput
        self.regime = take.regime^
        self.last_batch_size = take.last_batch_size
        self.last_group_size = take.last_group_size

    fn observe(mut self, obs: StepObservation) raises:
        """Record a step observation and refit USL parameters."""
        self.step_count += 1
        self.last_batch_size = obs.batch_size
        self.last_group_size = obs.group_size

        if obs.skipped:
            return

        var p = Float64(obs.batch_size * obs.group_size)
        var throughput: Float64 = 0.0
        if obs.step_time_s > self.eps:
            throughput = Float64(obs.total_tokens) / obs.step_time_s

        # Update EMA
        self.prev_ema_throughput = self.ema_throughput
        if self.step_count == 1:
            self.ema_throughput = throughput
        else:
            self.ema_throughput = (
                self.ema_decay * self.ema_throughput
                + (1.0 - self.ema_decay) * throughput
            )

        # Record observation
        self.obs_p.append(p)
        self.obs_x.append(throughput)

        # Refit USL if we have enough data
        if len(self.obs_p) >= 3:
            self._fit_usl()

        # Update regime
        self._classify_regime(p)

    fn _fit_usl(mut self):
        """3-parameter polynomial fit for USL.

        From X(p) = λp / (1 + σ(p-1) + κp(p-1)), expanding gives:
            p/X(p) = A + Bp + Cp²
        where A = (1-σ)/λ, B = (σ-κ)/λ, C = κ/λ.
        Solve 3x3 normal equations via Cramer's rule, then recover σ, κ.
        """
        var n = len(self.obs_p)
        # Sums for normal equations
        var s0: Float64 = 0.0  # count
        var s1: Float64 = 0.0  # Σp
        var s2: Float64 = 0.0  # Σp²
        var s3: Float64 = 0.0  # Σp³
        var s4: Float64 = 0.0  # Σp⁴
        var sy: Float64 = 0.0  # Σy
        var spy: Float64 = 0.0  # Σpy
        var sp2y: Float64 = 0.0  # Σp²y

        for i in range(n):
            var p = self.obs_p[i]
            var x = self.obs_x[i]
            if x < self.eps:
                continue
            var y = p / x
            var p2 = p * p
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

        # Cramer's rule for 3x3:
        # |s0  s1  s2| |A|   |sy  |
        # |s1  s2  s3| |B| = |spy |
        # |s2  s3  s4| |C|   |sp2y|
        var det = (s0 * (s2 * s4 - s3 * s3)
                 - s1 * (s1 * s4 - s3 * s2)
                 + s2 * (s1 * s3 - s2 * s2))
        if abs(det) < self.eps:
            return

        var det_a = (sy * (s2 * s4 - s3 * s3)
                   - s1 * (spy * s4 - sp2y * s3)
                   + s2 * (spy * s3 - sp2y * s2))
        var det_b = (s0 * (spy * s4 - sp2y * s3)
                   - sy * (s1 * s4 - s3 * s2)
                   + s2 * (s1 * sp2y - spy * s2))
        var det_c = (s0 * (s2 * sp2y - s3 * spy)
                   - s1 * (s1 * sp2y - spy * s2)
                   + sy * (s1 * s3 - s2 * s2))

        var a_val = det_a / det
        var b_val = det_b / det
        var c_val = det_c / det

        # Recover σ, κ: λ = 1/(A+B+C), σ = (B+C)·λ, κ = C·λ
        var abc = a_val + b_val + c_val
        if abc < self.eps:
            return

        var sigma_raw = (b_val + c_val) / abc
        var kappa_raw = c_val / abc

        # Clamp: σ ∈ [0,1], κ ∈ [0,∞)
        self.sigma = max(0.0, min(1.0, sigma_raw))
        self.kappa = max(0.0, kappa_raw)

        # Update p*
        self.p_star = usl_optimal_p(self.sigma, self.kappa)

    fn _classify_regime(mut self, current_p: Float64):
        """Classify the current operating regime."""
        if self.step_count <= self.warmup_steps:
            self.regime = "warmup"
            return

        # Retrograde: operating beyond p* × throttle_margin
        if self.kappa > self.eps:
            var threshold = self.p_star * self.throttle_margin
            if current_p > threshold and self.p_star > 1.0:
                self.regime = "retrograde"
                return

        # Roofline classification if hardware params configured
        if self.peak_gflops > 0.0 and self.peak_bw_gb_s > 0.0:
            var predicted = usl_throughput(current_p, self.sigma, self.kappa)
            var ratio = predicted / (self.peak_gflops * 1e9 + self.eps)
            if ratio > 0.7:
                self.regime = "compute_bound"
            else:
                self.regime = "memory_bound"
            return

        # Fallback: EMA trend
        if self.ema_throughput > self.prev_ema_throughput * 1.05:
            self.regime = "memory_bound"  # still scaling up
        elif self.ema_throughput < self.prev_ema_throughput * 0.95:
            self.regime = "retrograde"
        else:
            self.regime = "optimal"

    fn recommend(self) -> BackPressureDecision:
        """Produce a back pressure decision based on current state."""
        var decision = BackPressureDecision()
        decision.sigma = self.sigma
        decision.kappa = self.kappa
        decision.p_star = self.p_star
        decision.throughput = self.ema_throughput
        decision.regime = self.regime
        decision.recommended_batch_size = self.last_batch_size
        decision.recommended_group_size = self.last_group_size

        if self.step_count <= self.warmup_steps:
            decision.action = "hold"
            decision.utilization = 0.0
            return decision^

        # Compute utilization: current throughput / predicted peak
        var peak_throughput = usl_throughput(self.p_star, self.sigma, self.kappa)
        if peak_throughput > self.eps:
            decision.utilization = self.ema_throughput / peak_throughput
        else:
            decision.utilization = 1.0

        var current_p = Float64(self.last_batch_size * self.last_group_size)

        if self.regime == "retrograde":
            # Throttle: reduce toward p*
            decision.action = "throttle"
            var target_p = self.p_star * self.throttle_margin
            var new_batch = Int(target_p / Float64(self.last_group_size))
            new_batch = max(self.min_batch_size, min(self.max_batch_size, new_batch))
            decision.recommended_batch_size = new_batch
        elif current_p < self.p_star * self.increase_margin:
            # Room to grow
            decision.action = "increase"
            var target_p = self.p_star * self.increase_margin
            var new_batch = Int(target_p / Float64(self.last_group_size))
            new_batch = max(self.min_batch_size, min(self.max_batch_size, new_batch))
            decision.recommended_batch_size = new_batch
        else:
            decision.action = "hold"

        return decision^

    fn reset(mut self):
        """Clear all state, keeping config."""
        self.step_count = 0
        self.obs_p = List[Float64]()
        self.obs_x = List[Float64]()
        self.sigma = 0.0
        self.kappa = 0.0
        self.p_star = 1.0
        self.ema_throughput = 0.0
        self.prev_ema_throughput = 0.0
        self.regime = "warmup"
        self.last_batch_size = self.min_batch_size
        self.last_group_size = self.min_group_size

    fn validate(self) raises:
        """Reject invalid config."""
        if self.warmup_steps < 0:
            raise Error("warmup_steps must be >= 0")
        if self.ema_decay < 0.0 or self.ema_decay > 1.0:
            raise Error("ema_decay must be in [0, 1]")
        if self.throttle_margin <= 0.0 or self.throttle_margin > 1.0:
            raise Error("throttle_margin must be in (0, 1]")
        if self.increase_margin <= 0.0 or self.increase_margin > 1.0:
            raise Error("increase_margin must be in (0, 1]")
        if self.min_batch_size < 1:
            raise Error("min_batch_size must be >= 1")
        if self.max_batch_size < self.min_batch_size:
            raise Error("max_batch_size must be >= min_batch_size")
        if self.min_group_size < 1:
            raise Error("min_group_size must be >= 1")
        if self.max_group_size < self.min_group_size:
            raise Error("max_group_size must be >= min_group_size")
