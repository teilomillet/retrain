"""Characterization contracts for the effectful trainer step shell."""

from __future__ import annotations

from pathlib import Path
import pytest

from retrain.training import trainer as trainer_mod
from tests.support.trainer_case import (
    Case,
    make_config,
    capture_config,
    fail_train,
    setup,
    recommend,
    track_loggers,
    track_states,
    resume_config,
    mode,
)


def test_skip_step_has_no_training_or_step_log_and_updates_backpressure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = setup(
        monkeypatch,
        tmp_path,
        case=Case(rl=False, echo=False),
    )

    trainer_mod.train(make_config(tmp_path), flow=env.flow)

    assert "train" not in env.trace.events
    assert "log" not in env.trace.events
    assert env.trace.events.index("rollout") < env.trace.events.index(
        "backpressure:observe"
    )
    assert getattr(env.trace.pressure[0], "skipped") is True
    assert env.trace.logs == []
    assert env.backend.shutdown_called is True


@pytest.mark.parametrize(
    ("rl", "echo", "expected"),
    [(True, False, "rl"), (False, True, "echo"), (True, True, "joint")],
)
def test_modes_preserve_rows_and_effect_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rl: bool,
    echo: bool,
    expected: str,
) -> None:
    env = setup(
        monkeypatch,
        tmp_path,
        case=Case(
            rl=rl,
            echo=echo,
            loss=0.875,
        ),
    )
    config = make_config(
        tmp_path,
        echo_enabled=echo,
        echo_target_retention="all" if echo else "bounded",
        echo_entropy_floor=0.0 if echo else 0.5,
    )

    trainer_mod.train(config, flow=env.flow)

    assert len(env.trace.trains) == 1
    assert mode(env.trace.trains[0]) == expected
    events = env.trace.events
    assert events.index("checkpoint:step_0") < events.index("rollout")
    assert events.index("rollout") < events.index("train")
    assert events.index("train") < events.index("backpressure:observe")
    assert events.index("backpressure:recommend") < events.index("log")
    assert getattr(env.trace.pressure[0], "loss") == pytest.approx(0.875)
    assert getattr(env.trace.logs[0], "loss_value") == pytest.approx(0.875)


def test_capture_precedes_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = setup(
        monkeypatch,
        tmp_path,
        case=Case(rl=True, echo=False),
    )

    trainer_mod.train(
        capture_config(monkeypatch, env, tmp_path),
        flow=env.flow,
    )

    assert env.trace.events.index("capture") < env.trace.events.index("train")


def test_periodic_checkpoint_follows_logging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = setup(
        monkeypatch,
        tmp_path,
        case=Case(rl=True, echo=False),
    )

    trainer_mod.train(make_config(tmp_path, save_every=1), flow=env.flow)

    events = env.trace.events
    assert events.index("log") < events.index("save:checkpoint_step_1")
    assert events.index("save:checkpoint_step_1") < events.index("state:0")
    assert events[-1] == "shutdown"


def test_checkpoint_persists_post_step_counters_and_backpressure_batch_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = setup(
        monkeypatch,
        tmp_path,
        case=Case(rl=True, echo=False),
    )
    recommend(env, 2)
    saved_states = track_states(monkeypatch)
    config = make_config(
        tmp_path,
        save_every=1,
        bp_enabled=True,
        bp_warmup_steps=0,
        bp_min_batch_size=1,
        bp_max_batch_size=4,
    )

    trainer_mod.train(config, flow=env.flow)

    periodic = next(
        state for state in saved_states if state["checkpoint_name"] != "final"
    )
    assert periodic["step"] == 0
    assert periodic["example_idx"] == 1
    assert periodic["total_correct"] == 1
    assert periodic["total_completions"] == 1
    assert periodic["current_batch_size"] == 2
    assert periodic["current_group_size"] == 1


def test_backend_training_failure_still_shuts_down_and_closes_loggers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = setup(
        monkeypatch,
        tmp_path,
        case=Case(rl=True, echo=False),
    )
    logger_instances = track_loggers(monkeypatch)
    fail_train(monkeypatch, env)

    with pytest.raises(RuntimeError, match="backend exploded"):
        trainer_mod.train(make_config(tmp_path), flow=env.flow)

    assert env.backend.shutdown_called is True
    assert len(logger_instances) == 3
    assert all(getattr(logger, "closed") for logger in logger_instances)


@pytest.mark.parametrize(
    ("saved_step", "expect_warmup"),
    [(-1, True), (0, False)],
)
def test_resume_verifies_warmup_boundary_before_loading_backend_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    saved_step: int,
    expect_warmup: bool,
) -> None:
    env = setup(
        monkeypatch,
        tmp_path,
        case=Case(rl=True, echo=False),
    )
    config = resume_config(
        monkeypatch,
        env,
        tmp_path,
        saved_step=saved_step,
        warmup_steps=1,
    )

    trainer_mod.train(config, flow=env.flow)

    events = env.trace.events
    assert events.index("resume:verify") < events.index("load:/checkpoint/resume")
    if expect_warmup:
        assert "warmup" in events
        assert "rollout" not in events
    else:
        assert "warmup" not in events
        assert events.index("load:/checkpoint/resume") < events.index("rollout")
