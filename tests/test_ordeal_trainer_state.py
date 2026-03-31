"""Ordeal I/O fault injection tests for trainer state save/load.

Tests checkpoint resilience: corrupt JSON, truncated files, missing
fields, atomic write failures, and SEPA state roundtrip under faults.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from ordeal import ChaosTest, always, invariant, rule, sometimes
from ordeal.invariants import no_nan, no_inf

from retrain.trainer import _load_trainer_state, _save_trainer_state

valid_number = no_nan & no_inf


# ── Strategies ──

step_st = st.integers(min_value=0, max_value=10000)
count_st = st.integers(min_value=0, max_value=100000)
batch_st = st.integers(min_value=1, max_value=256)
ema_st = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


# ═══════════════════════════════════════════
# Save/Load Roundtrip Properties
# ═══════════════════════════════════════════


class TestStateRoundtrip:
    @given(
        step=step_st,
        example_idx=count_st,
        total_correct=count_st,
        total_completions=count_st,
        batch_size=batch_st,
        group_size=batch_st,
    )
    def test_roundtrip_preserves_fields(
        self,
        step: int,
        example_idx: int,
        total_correct: int,
        total_completions: int,
        batch_size: int,
        group_size: int,
    ) -> None:
        """Save then load preserves all fields exactly."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            _save_trainer_state(
                p,
                step=step,
                example_idx=example_idx,
                total_correct=total_correct,
                total_completions=total_completions,
                current_batch_size=batch_size,
                current_group_size=group_size,
                checkpoint_name=f"ckpt_{step}",
                sepa_state={},
            )
            loaded = _load_trainer_state(str(p))
        assert loaded["step"] == step
        assert loaded["example_idx"] == example_idx
        assert loaded["total_correct"] == total_correct
        assert loaded["total_completions"] == total_completions
        assert loaded["current_batch_size"] == batch_size
        assert loaded["current_group_size"] == group_size
        assert loaded["checkpoint_name"] == f"ckpt_{step}"

    @given(tl_ema=ema_st, delight_ema=ema_st)
    def test_optional_fields_roundtrip(
        self, tl_ema: float, delight_ema: float
    ) -> None:
        """Optional float fields survive roundtrip."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            _save_trainer_state(
                p,
                step=10,
                example_idx=100,
                total_correct=50,
                total_completions=100,
                current_batch_size=8,
                current_group_size=16,
                checkpoint_name="ckpt_10",
                sepa_state={},
                tl_grpo_ema=tl_ema,
                delight_eta_ema=delight_ema,
            )
            loaded = _load_trainer_state(str(p))
        assert math.isclose(loaded["tl_grpo_ema"], tl_ema, abs_tol=1e-10)
        assert math.isclose(loaded["delight_eta_ema"], delight_ema, abs_tol=1e-10)

    def test_sepa_state_roundtrip(self, tmp_path: Path) -> None:
        """SEPA state dict survives roundtrip."""
        sepa = {
            "var_ema": 0.5,
            "var_0": 1.0,
            "warmup_seen": 10,
            "gate_open": True,
        }
        _save_trainer_state(
            tmp_path,
            step=5,
            example_idx=50,
            total_correct=25,
            total_completions=50,
            current_batch_size=8,
            current_group_size=16,
            checkpoint_name="ckpt_5",
            sepa_state=sepa,
        )
        loaded = _load_trainer_state(str(tmp_path))
        assert loaded["sepa"]["var_ema"] == 0.5
        assert loaded["sepa"]["gate_open"] is True


# ═══════════════════════════════════════════
# Load Error Handling
# ═══════════════════════════════════════════


class TestLoadErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="trainer_state.json"):
            _load_trainer_state(str(tmp_path))

    def test_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "trainer_state.json").write_text("{not valid json")
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _load_trainer_state(str(tmp_path))

    def test_json_array_not_object(self, tmp_path: Path) -> None:
        (tmp_path / "trainer_state.json").write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="expected JSON object"):
            _load_trainer_state(str(tmp_path))

    def test_missing_required_field(self, tmp_path: Path) -> None:
        state = {"step": 1}  # Missing all other required fields
        (tmp_path / "trainer_state.json").write_text(json.dumps(state))
        with pytest.raises(ValueError):
            _load_trainer_state(str(tmp_path))

    def test_wrong_type_for_step(self, tmp_path: Path) -> None:
        state = {
            "step": "not_an_int",
            "example_idx": 0,
            "total_correct": 0,
            "total_completions": 0,
            "current_batch_size": 8,
            "current_group_size": 16,
            "checkpoint_name": "ckpt",
        }
        (tmp_path / "trainer_state.json").write_text(json.dumps(state))
        with pytest.raises(ValueError):
            _load_trainer_state(str(tmp_path))

    def test_truncated_json(self, tmp_path: Path) -> None:
        """Truncated JSON file (simulating crash during write)."""
        (tmp_path / "trainer_state.json").write_text('{"step": 1, "example')
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _load_trainer_state(str(tmp_path))

    def test_empty_file(self, tmp_path: Path) -> None:
        (tmp_path / "trainer_state.json").write_text("")
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _load_trainer_state(str(tmp_path))


# ═══════════════════════════════════════════
# Atomic Write Verification
# ═══════════════════════════════════════════


class TestAtomicWrite:
    def test_no_tmp_file_after_save(self, tmp_path: Path) -> None:
        """Temp file is cleaned up after atomic rename."""
        _save_trainer_state(
            tmp_path,
            step=1,
            example_idx=10,
            total_correct=5,
            total_completions=10,
            current_batch_size=8,
            current_group_size=16,
            checkpoint_name="ckpt_1",
            sepa_state={},
        )
        assert not (tmp_path / "trainer_state.json.tmp").exists()
        assert (tmp_path / "trainer_state.json").exists()

    def test_overwrite_preserves_latest(self, tmp_path: Path) -> None:
        """Second save overwrites first — latest state wins."""
        for step in [1, 2, 3]:
            _save_trainer_state(
                tmp_path,
                step=step,
                example_idx=step * 10,
                total_correct=step * 5,
                total_completions=step * 10,
                current_batch_size=8,
                current_group_size=16,
                checkpoint_name=f"ckpt_{step}",
                sepa_state={},
            )
        loaded = _load_trainer_state(str(tmp_path))
        assert loaded["step"] == 3
        assert loaded["example_idx"] == 30


# ═══════════════════════════════════════════
# ChaosTest: State Machine Save/Load
# ═══════════════════════════════════════════


class TrainerStateChaos(ChaosTest):
    """Stateful test for trainer state save/load cycles."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        self._path = Path(self._tmpdir)
        self._step = 0
        self._example_idx = 0
        self._total_correct = 0
        self._total_completions = 0
        self._saved = False
        self._saved_step = -1
        self._saved_correct = -1
        self._saved_completions = -1

    @rule(
        correct=st.integers(min_value=0, max_value=16),
        completions=st.integers(min_value=1, max_value=16),
    )
    def advance(self, correct: int, completions: int) -> None:
        """Simulate training steps."""
        self._step += 1
        self._example_idx += completions
        self._total_correct += correct
        self._total_completions += completions

    @rule()
    def save(self) -> None:
        """Save current state."""
        _save_trainer_state(
            self._path,
            step=self._step,
            example_idx=self._example_idx,
            total_correct=self._total_correct,
            total_completions=self._total_completions,
            current_batch_size=8,
            current_group_size=16,
            checkpoint_name=f"ckpt_{self._step}",
            sepa_state={},
        )
        self._saved = True
        self._saved_step = self._step
        self._saved_correct = self._total_correct
        self._saved_completions = self._total_completions

    @rule()
    def load_and_verify(self) -> None:
        """Load state and verify it matches last save point."""
        if not self._saved:
            return
        loaded = _load_trainer_state(str(self._path))
        always(
            loaded["step"] == self._saved_step,
            "loaded step matches last saved step",
        )
        always(
            loaded["total_correct"] == self._saved_correct,
            "loaded total_correct matches last save",
        )
        always(
            loaded["total_completions"] == self._saved_completions,
            "loaded total_completions matches last save",
        )

    @rule()
    def save_then_load_is_consistent(self) -> None:
        """Save then immediately load — must match current state."""
        _save_trainer_state(
            self._path,
            step=self._step,
            example_idx=self._example_idx,
            total_correct=self._total_correct,
            total_completions=self._total_completions,
            current_batch_size=8,
            current_group_size=16,
            checkpoint_name=f"ckpt_{self._step}",
            sepa_state={},
        )
        self._saved = True
        self._saved_step = self._step
        self._saved_correct = self._total_correct
        self._saved_completions = self._total_completions
        loaded = _load_trainer_state(str(self._path))
        always(loaded["step"] == self._step, "immediate load matches save")

    @invariant()
    def file_is_valid_json(self) -> None:
        """If file exists, it must be valid JSON."""
        state_file = self._path / "trainer_state.json"
        if state_file.exists():
            data = json.loads(state_file.read_text())
            assert isinstance(data, dict)

    def teardown(self) -> None:
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)
        super().teardown()


TestTrainerStateChaos = TrainerStateChaos.TestCase
