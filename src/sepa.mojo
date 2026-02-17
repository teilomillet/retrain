"""SEPAController for Tinker GPU training.

Ports the scheduling and state-tracking logic from
textpolicy/tinker/sepa.py into native Mojo.

The actual entropy pooling is done by advantages.apply_sepa_pooling();
this struct handles when and how much to pool.
"""

from math import isfinite
from collections import Optional


struct SEPAController:
    """SEPA scheduler: controls pooling strength over training.

    Two scheduling modes:
        linear — lambda ramps from 0 to 1 over sepa_steps (after optional delay).
        auto   — lambda adapts based on execution-token entropy variance decay,
                 with linear as a fallback floor.

    Optional correctness gate: SEPA stays disabled (lambda=0) until the model
    achieves a minimum correct rate, then becomes sticky-open.
    """

    var sepa_steps: Int
    var sepa_schedule: String
    var sepa_delay_steps: Int
    var sepa_correct_rate_gate: Float64
    var sepa_ema_decay: Float64
    var sepa_var_threshold: Float64
    var sepa_warmup: Int
    var eps: Float64

    # Auto schedule state
    var _var_ema: Optional[Float64]
    var _var_0: Optional[Float64]
    var _warmup_seen: Int
    var _gate_open: Bool

    fn __init__(
        out self,
        *,
        sepa_steps: Int = 0,
        sepa_schedule: String = "linear",
        sepa_delay_steps: Int = 0,
        sepa_correct_rate_gate: Float64 = 0.0,
        sepa_ema_decay: Float64 = 0.99,
        sepa_var_threshold: Float64 = 0.2,
        sepa_warmup: Int = 50,
        eps: Float64 = 1e-8,
    ) raises:
        if sepa_steps < 0:
            raise Error("sepa_steps must be >= 0, got " + String(sepa_steps))
        if sepa_delay_steps < 0:
            raise Error("sepa_delay_steps must be >= 0, got " + String(sepa_delay_steps))
        if sepa_correct_rate_gate < 0.0 or sepa_correct_rate_gate > 1.0:
            raise Error(
                "sepa_correct_rate_gate must be in [0, 1], got "
                + String(sepa_correct_rate_gate)
            )
        if sepa_ema_decay < 0.0 or sepa_ema_decay > 1.0:
            raise Error("sepa_ema_decay must be in [0, 1], got " + String(sepa_ema_decay))
        if sepa_var_threshold <= 0.0:
            raise Error("sepa_var_threshold must be > 0, got " + String(sepa_var_threshold))
        if sepa_warmup < 1:
            raise Error("sepa_warmup must be >= 1, got " + String(sepa_warmup))
        if eps <= 0.0:
            raise Error("eps must be > 0, got " + String(eps))

        # Normalize schedule
        var sched = sepa_schedule
        if sched != "linear" and sched != "auto":
            raise Error("sepa_schedule must be 'linear' or 'auto', got " + sched)

        self.sepa_steps = sepa_steps
        self.sepa_schedule = sched
        self.sepa_delay_steps = sepa_delay_steps
        self.sepa_correct_rate_gate = sepa_correct_rate_gate
        self.sepa_ema_decay = sepa_ema_decay
        self.sepa_var_threshold = sepa_var_threshold
        self.sepa_warmup = sepa_warmup
        self.eps = eps

        self._var_ema = Optional[Float64](None)
        self._var_0 = Optional[Float64](None)
        self._warmup_seen = 0
        self._gate_open = sepa_correct_rate_gate <= 0.0

    fn enabled(self) -> Bool:
        """Whether SEPA pooling is active."""
        return self.sepa_steps > 0 or self.sepa_schedule == "auto"

    fn gate_open(self) -> Bool:
        """Whether correctness gate currently allows non-zero lambda."""
        return self._gate_open

    fn observe_correct_rate(mut self, correct_rate: Optional[Float64]):
        """Update correctness gate from observed correct rate.

        Gate is sticky-open: once threshold is met, SEPA stays enabled.
        """
        if self._gate_open or self.sepa_correct_rate_gate <= 0.0:
            return
        if correct_rate is None:
            return
        var rate = correct_rate.value()
        if not isfinite(rate):
            return
        if rate >= self.sepa_correct_rate_gate:
            self._gate_open = True

    fn _linear_lambda(self, step: Float64) -> Float64:
        if self.sepa_steps <= 0:
            return 0.0
        var shifted = step - Float64(self.sepa_delay_steps)
        if shifted <= 0.0:
            return 0.0
        var result = shifted / Float64(self.sepa_steps)
        if result > 1.0:
            return 1.0
        return result

    fn _auto_lambda(self) -> Float64:
        if self.sepa_schedule != "auto":
            return 0.0
        if self._var_ema is None or self._var_0 is None:
            return 0.0

        var var_ema = self._var_ema.value()
        var var_0 = self._var_0.value()

        var threshold = self.sepa_var_threshold
        if threshold < self.eps:
            threshold = self.eps

        var denom = var_0
        if denom < self.eps:
            denom = self.eps

        var ratio = var_ema / denom
        if ratio < 0.0:
            ratio = 0.0

        var scaled = ratio / threshold
        if scaled > 1.0:
            scaled = 1.0

        return 1.0 - scaled

    fn resolve_lambda(self, step: Float64) -> Float64:
        """Resolve current pooling strength lambda.

        Args:
            step: Current training step.

        Returns:
            Lambda in [0, 1]. 0 = no pooling, 1 = full pooling.
        """
        var linear_val = self._linear_lambda(step)
        if not self._gate_open:
            return 0.0
        if self.sepa_schedule == "auto":
            var auto_val = self._auto_lambda()
            if auto_val > linear_val:
                return auto_val
            return linear_val
        return linear_val

    fn update_auto_state(mut self, exec_entropies: List[Float64]):
        """Update auto-schedule variance tracking from execution entropies.

        Args:
            exec_entropies: Entropy values for execution tokens.
        """
        if self.sepa_schedule != "auto":
            return
        if len(exec_entropies) == 0:
            return

        var n = len(exec_entropies)
        var total: Float64 = 0.0
        for i in range(n):
            total += exec_entropies[i]
        var mean_h = total / Float64(n)

        var var_sum: Float64 = 0.0
        for i in range(n):
            var diff = exec_entropies[i] - mean_h
            var_sum += diff * diff
        var var_batch = var_sum / Float64(n)

        if not isfinite(var_batch):
            return

        if self._var_ema is None:
            self._var_ema = var_batch
        else:
            var d = self.sepa_ema_decay
            self._var_ema = d * self._var_ema.value() + (1.0 - d) * var_batch

        if self._var_0 is None:
            self._warmup_seen += 1
            if self._warmup_seen >= self.sepa_warmup:
                var ema_val = self._var_ema.value()
                if ema_val < self.eps:
                    ema_val = self.eps
                self._var_0 = ema_val
