"""SEPAController for SEPA scheduling.

Ports the scheduling and state-tracking logic from src/sepa.mojo.
The actual entropy pooling is done by advantages.apply_sepa_pooling();
this class handles when and how much to pool.
"""

from __future__ import annotations

import math
from typing import TypeAlias


SEPAStateValue: TypeAlias = str | int | float | bool | None
SEPAStateDict: TypeAlias = dict[str, SEPAStateValue]


class SEPAController:
    """SEPA scheduler: controls pooling strength over training.

    Two scheduling modes:
        linear -- lambda ramps from 0 to 1 over sepa_steps (after optional delay).
        auto   -- lambda adapts based on execution-token entropy variance decay,
                  with linear as a fallback floor.

    Optional correctness gate: SEPA stays disabled (lambda=0) until the model
    achieves a minimum correct rate, then becomes sticky-open.
    """

    def __init__(
        self,
        *,
        sepa_steps: int = 0,
        sepa_schedule: str = "linear",
        sepa_delay_steps: int = 0,
        sepa_correct_rate_gate: float = 0.0,
        sepa_ema_decay: float = 0.99,
        sepa_var_threshold: float = 0.2,
        sepa_warmup: int = 50,
        eps: float = 1e-8,
    ) -> None:
        if sepa_steps < 0:
            raise ValueError(f"sepa_steps must be >= 0, got {sepa_steps}")
        if sepa_delay_steps < 0:
            raise ValueError(f"sepa_delay_steps must be >= 0, got {sepa_delay_steps}")
        if not 0.0 <= sepa_correct_rate_gate <= 1.0:
            raise ValueError(
                f"sepa_correct_rate_gate must be in [0, 1], got {sepa_correct_rate_gate}"
            )
        if not 0.0 <= sepa_ema_decay <= 1.0:
            raise ValueError(f"sepa_ema_decay must be in [0, 1], got {sepa_ema_decay}")
        if sepa_var_threshold <= 0.0:
            raise ValueError(f"sepa_var_threshold must be > 0, got {sepa_var_threshold}")
        if sepa_warmup < 1:
            raise ValueError(f"sepa_warmup must be >= 1, got {sepa_warmup}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if sepa_schedule not in ("linear", "auto"):
            raise ValueError(f"sepa_schedule must be 'linear' or 'auto', got {sepa_schedule}")

        self.sepa_steps = sepa_steps
        self.sepa_schedule = sepa_schedule
        self.sepa_delay_steps = sepa_delay_steps
        self.sepa_correct_rate_gate = sepa_correct_rate_gate
        self.sepa_ema_decay = sepa_ema_decay
        self.sepa_var_threshold = sepa_var_threshold
        self.sepa_warmup = sepa_warmup
        self.eps = eps

        # Auto schedule state
        self._var_ema: float | None = None
        self._var_0: float | None = None
        self._warmup_seen = 0
        self._gate_open = sepa_correct_rate_gate <= 0.0

    def enabled(self) -> bool:
        """Whether SEPA pooling is active."""
        return self.sepa_steps > 0 or self.sepa_schedule == "auto"

    def gate_open(self) -> bool:
        """Whether correctness gate currently allows non-zero lambda."""
        return self._gate_open

    def observe_correct_rate(self, correct_rate: float | None) -> None:
        """Update correctness gate. Sticky-open once threshold met."""
        if self._gate_open or self.sepa_correct_rate_gate <= 0.0:
            return
        if correct_rate is None or not math.isfinite(correct_rate):
            return
        if correct_rate >= self.sepa_correct_rate_gate:
            self._gate_open = True

    def _linear_lambda(self, step: float) -> float:
        if self.sepa_steps <= 0:
            return 0.0
        shifted = step - self.sepa_delay_steps
        if shifted <= 0.0:
            return 0.0
        return min(shifted / self.sepa_steps, 1.0)

    def _auto_lambda(self) -> float:
        if self.sepa_schedule != "auto":
            return 0.0
        if self._var_ema is None or self._var_0 is None:
            return 0.0

        threshold = max(self.sepa_var_threshold, self.eps)
        denom = max(self._var_0, self.eps)

        ratio = max(0.0, self._var_ema / denom)
        scaled = min(ratio / threshold, 1.0)
        return 1.0 - scaled

    def resolve_lambda(self, step: float) -> float:
        """Resolve current pooling strength lambda in [0, 1]."""
        linear_val = self._linear_lambda(step)
        if not self._gate_open:
            return 0.0
        if self.sepa_schedule == "auto":
            return max(self._auto_lambda(), linear_val)
        return linear_val

    def update_auto_state(self, exec_entropies: list[float]) -> None:
        """Update auto-schedule variance tracking from execution entropies."""
        if self.sepa_schedule != "auto" or not exec_entropies:
            return

        n = len(exec_entropies)
        mean_h = sum(exec_entropies) / n
        var_batch = sum((e - mean_h) ** 2 for e in exec_entropies) / n

        if not math.isfinite(var_batch):
            return

        if self._var_ema is None:
            self._var_ema = var_batch
        else:
            d = self.sepa_ema_decay
            self._var_ema = d * self._var_ema + (1.0 - d) * var_batch

        if self._var_0 is None:
            self._warmup_seen += 1
            if self._warmup_seen >= self.sepa_warmup:
                self._var_0 = max(self._var_ema, self.eps)

    def state_dict(self) -> SEPAStateDict:
        """Serialize scheduler state for checkpointing."""
        return {
            "sepa_steps": self.sepa_steps,
            "sepa_schedule": self.sepa_schedule,
            "sepa_delay_steps": self.sepa_delay_steps,
            "sepa_correct_rate_gate": self.sepa_correct_rate_gate,
            "sepa_ema_decay": self.sepa_ema_decay,
            "sepa_var_threshold": self.sepa_var_threshold,
            "sepa_warmup": self.sepa_warmup,
            "eps": self.eps,
            "var_ema": self._var_ema,
            "var_0": self._var_0,
            "warmup_seen": self._warmup_seen,
            "gate_open": self._gate_open,
        }

    def load_state_dict(self, state: SEPAStateDict) -> None:
        """Restore scheduler state from checkpoint."""
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dict, got {type(state)!r}.")

        def _maybe_float(key: str) -> float | None:
            value = state.get(key)
            if value is None:
                return None
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"state[{key!r}] must be finite, got {parsed}.")
            return parsed

        self._var_ema = _maybe_float("var_ema")
        self._var_0 = _maybe_float("var_0")

        warmup_seen = state.get("warmup_seen", self._warmup_seen)
        self._warmup_seen = int(warmup_seen)
        if self._warmup_seen < 0:
            raise ValueError(
                f"state['warmup_seen'] must be >= 0, got {self._warmup_seen}."
            )

        gate_open = state.get("gate_open", self._gate_open)
        if not isinstance(gate_open, bool):
            raise ValueError("state['gate_open'] must be a boolean.")
        self._gate_open = gate_open
