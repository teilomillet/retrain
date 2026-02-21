"""Tests for trainer state save/load (checkpoint resume).

Unit tests for the individual functions, plus integration tests that simulate
the full resume flow: SEPA state consistency, loop range correctness,
backpressure graceful degradation, and dataset cursor continuity.
"""

import json

import pytest

from retrain.trainer import _save_trainer_state, _load_trainer_state, _TRAINER_STATE_FILE


class TestSaveTrainerState:
    def test_writes_json(self, tmp_path):
        _save_trainer_state(
            tmp_path,
            step=39,
            example_idx=320,
            total_correct=95,
            total_completions=640,
            current_batch_size=8,
            current_group_size=16,
            checkpoint_name="checkpoint_step_40",
            sepa_state={"var_ema": 0.5, "gate_open": True},
        )
        state_file = tmp_path / _TRAINER_STATE_FILE
        assert state_file.is_file()
        state = json.loads(state_file.read_text())
        assert state["step"] == 39
        assert state["example_idx"] == 320
        assert state["total_correct"] == 95
        assert state["total_completions"] == 640
        assert state["current_batch_size"] == 8
        assert state["current_group_size"] == 16
        assert state["checkpoint_name"] == "checkpoint_step_40"
        assert state["sepa"]["var_ema"] == 0.5
        assert state["sepa"]["gate_open"] is True

    def test_atomic_overwrite(self, tmp_path):
        """Second save overwrites the first (always latest)."""
        _save_trainer_state(
            tmp_path, step=10, example_idx=80,
            total_correct=20, total_completions=160,
            current_batch_size=8, current_group_size=16,
            checkpoint_name="checkpoint_step_10",
            sepa_state={},
        )
        _save_trainer_state(
            tmp_path, step=20, example_idx=160,
            total_correct=45, total_completions=320,
            current_batch_size=8, current_group_size=16,
            checkpoint_name="checkpoint_step_20",
            sepa_state={},
        )
        state = json.loads((tmp_path / _TRAINER_STATE_FILE).read_text())
        assert state["step"] == 20
        assert state["checkpoint_name"] == "checkpoint_step_20"

    def test_no_tmp_file_left(self, tmp_path):
        _save_trainer_state(
            tmp_path, step=5, example_idx=40,
            total_correct=10, total_completions=80,
            current_batch_size=8, current_group_size=16,
            checkpoint_name="ckpt", sepa_state={},
        )
        tmp_file = tmp_path / f"{_TRAINER_STATE_FILE}.tmp"
        assert not tmp_file.exists()


class TestLoadTrainerState:
    def test_roundtrip(self, tmp_path):
        _save_trainer_state(
            tmp_path, step=39, example_idx=320,
            total_correct=95, total_completions=640,
            current_batch_size=8, current_group_size=16,
            checkpoint_name="checkpoint_step_40",
            sepa_state={"var_ema": 1.2, "gate_open": False},
        )
        state = _load_trainer_state(str(tmp_path))
        assert state["step"] == 39
        assert state["example_idx"] == 320
        assert state["checkpoint_name"] == "checkpoint_step_40"
        assert state["sepa"]["var_ema"] == 1.2
        assert state["sepa"]["gate_open"] is False

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="trainer_state.json"):
            _load_trainer_state(str(tmp_path))

    def test_resume_step_is_next(self, tmp_path):
        """Resume should start from saved step + 1."""
        _save_trainer_state(
            tmp_path, step=19, example_idx=160,
            total_correct=40, total_completions=320,
            current_batch_size=8, current_group_size=16,
            checkpoint_name="checkpoint_step_20",
            sepa_state={},
        )
        state = _load_trainer_state(str(tmp_path))
        start_step = state["step"] + 1
        assert start_step == 20


class TestSEPAStateRoundtripViaTrainer:
    """End-to-end: save SEPA state via trainer, reload into controller."""

    def test_sepa_state_survives_trainer_roundtrip(self, tmp_path):
        from retrain.sepa import SEPAController

        # Original controller with accumulated state
        ctrl = SEPAController(
            sepa_steps=500, sepa_schedule="auto",
            sepa_warmup=2, sepa_ema_decay=0.5,
            sepa_correct_rate_gate=0.1,
        )
        ctrl.observe_correct_rate(0.15)
        ctrl.update_auto_state([1.0, 3.0, 5.0])
        ctrl.update_auto_state([2.0, 4.0, 6.0])
        lam_before = ctrl.resolve_lambda(step=100.0)

        # Save via trainer
        _save_trainer_state(
            tmp_path, step=99, example_idx=800,
            total_correct=200, total_completions=1600,
            current_batch_size=8, current_group_size=16,
            checkpoint_name="checkpoint_step_100",
            sepa_state=ctrl.state_dict(),
        )

        # Load and restore
        saved = _load_trainer_state(str(tmp_path))
        ctrl2 = SEPAController(
            sepa_steps=500, sepa_schedule="auto",
            sepa_warmup=2, sepa_ema_decay=0.5,
            sepa_correct_rate_gate=0.1,
        )
        ctrl2.load_state_dict(saved["sepa"])

        assert ctrl2.gate_open() == ctrl.gate_open()
        assert ctrl2.resolve_lambda(step=100.0) == pytest.approx(lam_before)


# ---------------------------------------------------------------------------
# Integration: simulate a split run and verify resume consistency
# ---------------------------------------------------------------------------


class TestResumeIntegration:
    """Simulate the resume flow as the trainer would execute it.

    These tests verify the invariant: saving at step N and resuming
    produces the same state as a continuous run up to step N.
    """

    def _simulate_sepa_steps(self, ctrl, n_steps, *, start=0):
        """Feed synthetic observations to a SEPA controller for n steps."""
        for step in range(start, start + n_steps):
            # Synthetic: correct rate rises, variance drops
            correct_rate = min(0.05 + step * 0.01, 0.6)
            ctrl.observe_correct_rate(correct_rate)
            # Decreasing variance as training progresses
            spread = max(5.0 - step * 0.1, 0.1)
            exec_ent = [3.0 - spread, 3.0, 3.0 + spread]
            ctrl.update_auto_state(exec_ent)
        return ctrl

    def test_split_run_sepa_consistency(self, tmp_path):
        """A continuous 40-step run and a 20+20 split run produce the same SEPA lambda."""
        from retrain.sepa import SEPAController

        kwargs = dict(
            sepa_steps=100, sepa_schedule="auto",
            sepa_warmup=5, sepa_ema_decay=0.9,
            sepa_correct_rate_gate=0.1,
        )

        # Continuous: 40 steps straight
        continuous = SEPAController(**kwargs)
        self._simulate_sepa_steps(continuous, 40)
        lam_continuous = continuous.resolve_lambda(step=39.0)

        # Split: 20 steps → save → restore → 20 more steps
        first_half = SEPAController(**kwargs)
        self._simulate_sepa_steps(first_half, 20)
        _save_trainer_state(
            tmp_path, step=19, example_idx=160,
            total_correct=40, total_completions=320,
            current_batch_size=8, current_group_size=16,
            checkpoint_name="checkpoint_step_20",
            sepa_state=first_half.state_dict(),
        )

        # Resume
        saved = _load_trainer_state(str(tmp_path))
        second_half = SEPAController(**kwargs)
        second_half.load_state_dict(saved["sepa"])
        self._simulate_sepa_steps(second_half, 20, start=20)
        lam_split = second_half.resolve_lambda(step=39.0)

        assert lam_split == pytest.approx(lam_continuous)
        assert second_half.gate_open() == continuous.gate_open()

    def test_split_run_counters_consistent(self, tmp_path):
        """Verify example_idx, total_correct, total_completions survive resume."""
        batch_size = 8
        group_size = 16

        # Simulate 20 steps of training
        example_idx = 0
        total_correct = 0
        total_completions = 0
        for step in range(20):
            example_idx += batch_size
            batch_correct = 3  # synthetic
            total_correct += batch_correct
            total_completions += batch_size * group_size

        _save_trainer_state(
            tmp_path, step=19, example_idx=example_idx,
            total_correct=total_correct, total_completions=total_completions,
            current_batch_size=batch_size, current_group_size=group_size,
            checkpoint_name="checkpoint_step_20",
            sepa_state={},
        )

        # Resume and simulate 20 more steps
        saved = _load_trainer_state(str(tmp_path))
        r_example_idx = saved["example_idx"]
        r_total_correct = saved["total_correct"]
        r_total_completions = saved["total_completions"]
        start_step = saved["step"] + 1

        for step in range(start_step, 40):
            r_example_idx += batch_size
            r_total_correct += 3
            r_total_completions += batch_size * group_size

        # Compare with continuous 40-step run
        c_example_idx = 0
        c_total_correct = 0
        c_total_completions = 0
        for step in range(40):
            c_example_idx += batch_size
            c_total_correct += 3
            c_total_completions += batch_size * group_size

        assert r_example_idx == c_example_idx
        assert r_total_correct == c_total_correct
        assert r_total_completions == c_total_completions

    def test_loop_range_correctness(self, tmp_path):
        """Training loop range(start_step, max_steps) covers the right steps."""
        max_steps = 100
        save_every = 20

        # Simulate periodic saves
        for batch_idx in range(max_steps):
            if save_every > 0 and (batch_idx + 1) % save_every == 0:
                _save_trainer_state(
                    tmp_path, step=batch_idx, example_idx=batch_idx * 8,
                    total_correct=batch_idx, total_completions=batch_idx * 16,
                    current_batch_size=8, current_group_size=16,
                    checkpoint_name=f"checkpoint_step_{batch_idx + 1}",
                    sepa_state={},
                )

        # Resume from latest checkpoint
        saved = _load_trainer_state(str(tmp_path))
        start_step = saved["step"] + 1

        # Last checkpoint was at batch_idx=99 (step 100)
        assert saved["step"] == 99
        assert saved["checkpoint_name"] == "checkpoint_step_100"
        assert start_step == 100

        # Loop range should be empty (training already complete)
        remaining_steps = list(range(start_step, max_steps))
        assert remaining_steps == []

    def test_resume_mid_run(self, tmp_path):
        """Resume from mid-run checkpoint leaves correct remaining steps."""
        max_steps = 100
        # Simulate crash at step 45 — last checkpoint was step 39
        _save_trainer_state(
            tmp_path, step=39, example_idx=320,
            total_correct=80, total_completions=640,
            current_batch_size=8, current_group_size=16,
            checkpoint_name="checkpoint_step_40",
            sepa_state={},
        )

        saved = _load_trainer_state(str(tmp_path))
        start_step = saved["step"] + 1
        remaining_steps = list(range(start_step, max_steps))

        assert start_step == 40
        assert len(remaining_steps) == 60
        assert remaining_steps[0] == 40
        assert remaining_steps[-1] == 99

    def test_backpressure_graceful_on_resume(self):
        """USL controller starts cold on resume but batch_size is preserved."""
        from retrain.backpressure import USLBackPressure, StepObservation

        # Fresh USL controller (as happens on resume)
        bp = USLBackPressure(warmup_steps=5)

        # First recommendations should be "hold" (warmup)
        for i in range(5):
            bp.observe(StepObservation(
                step_time_s=1.0, sample_time_s=0.5,
                batch_size=8, group_size=16,
                total_tokens=1000, loss=0.5,
            ))
            decision = bp.recommend()
            assert decision.action == "hold"

        # After warmup, it starts making real decisions
        bp.observe(StepObservation(
            step_time_s=1.0, sample_time_s=0.5,
            batch_size=8, group_size=16,
            total_tokens=1000, loss=0.5,
        ))
        decision = bp.recommend()
        # Action could be hold/throttle/increase — point is it doesn't crash
        assert decision.action in ("hold", "throttle", "increase")

    def test_checkpoint_name_tracks_step(self, tmp_path):
        """Checkpoint name in saved state matches the step-based naming convention."""
        save_every = 20
        for batch_idx in range(60):
            if save_every > 0 and (batch_idx + 1) % save_every == 0:
                ckpt_name = f"checkpoint_step_{batch_idx + 1}"
                _save_trainer_state(
                    tmp_path, step=batch_idx, example_idx=batch_idx * 8,
                    total_correct=0, total_completions=0,
                    current_batch_size=8, current_group_size=16,
                    checkpoint_name=ckpt_name,
                    sepa_state={},
                )

        saved = _load_trainer_state(str(tmp_path))
        # Last save was at batch_idx=59 → checkpoint_step_60
        assert saved["step"] == 59
        assert saved["checkpoint_name"] == "checkpoint_step_60"
        # Resume would call helper.load_state("checkpoint_step_60")
        assert saved["checkpoint_name"] == f"checkpoint_step_{saved['step'] + 1}"
