"""Functional trainer cases and their minimal effect adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from retrain.backends.catalog import BackendCapabilities
from retrain.config import TrainConfig
from retrain.training import trainer as trainer_mod
from retrain.training.batch_digest import logical_optimizer_batch_sha256
from retrain.training.echo import EchoBuildStats

if TYPE_CHECKING:
    from pytest import MonkeyPatch


@dataclass(frozen=True)
class Case:
    rl: bool
    echo: bool
    loss: float = 0.75


@dataclass(frozen=True)
class Call:
    tokens: list[list[int]]
    logprobs: list[list[float]]
    advantages: list[list[float]]
    echo_advantages: list[list[float]]
    echo_counts: list[int]
    kwargs: dict[str, object]


@dataclass
class Trace:
    events: list[str] = field(default_factory=list)
    trains: list[Call] = field(default_factory=list)
    logs: list[object] = field(default_factory=list)
    pressure: list[object] = field(default_factory=list)


@dataclass(frozen=True)
class TestEnv:
    trace: Trace
    backend: object
    backpressure: object
    flow: object


@dataclass(frozen=True)
class _RolloutValues:
    advantages: tuple[tuple[float, ...], ...]
    echo_advantages: tuple[tuple[float, ...], ...]


def mode(call: Call) -> str:
    if call.echo_advantages:
        return "joint" if any(any(row) for row in call.advantages) else "echo"
    return "rl"


def make_config(tmp_path: Path, **overrides) -> TrainConfig:
    values = {
        "backend": "local",
        "model": "fake/model",
        "max_steps": 1,
        "batch_size": 1,
        "group_size": 1,
        "max_tokens": 8,
        "save_every": 0,
        "log_dir": str(tmp_path / "logs"),
        "adapter_path": str(tmp_path / "adapter"),
    }
    values.update(overrides)
    return TrainConfig(**values)


def setup(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    case: Case,
) -> TestEnv:
    trace = Trace()
    backend = _RecordingBackend(trace, tmp_path / "adapter")
    backpressure = _Backpressure(trace)
    flow = _flow(backend, backpressure)
    _patch_common(monkeypatch, trace)
    monkeypatch.setattr(
        trainer_mod,
        "run_singleturn",
        _rollout_effect(trace, case),
    )
    monkeypatch.setattr(
        trainer_mod,
        "run_rl_echo_train_step",
        _train_effect(trace, case),
    )
    monkeypatch.setattr(
        trainer_mod,
        "record_training_step",
        _log_effect(trace),
    )
    return TestEnv(trace, backend, backpressure, flow)


def capture_config(
    monkeypatch: MonkeyPatch,
    env: TestEnv,
    tmp_path: Path,
) -> TrainConfig:
    monkeypatch.setattr(
        trainer_mod,
        "preflight_optimizer_batch_capture",
        lambda _config: SimpleNamespace(path="initial"),
    )

    def capture(_log_path, *, step, batch, config, initial_adapter):
        _ = step, config, initial_adapter
        env.trace.events.append("capture")
        digest = logical_optimizer_batch_sha256(
            batch.tokens,
            batch.old_logprobs,
            batch.advantages,
        )
        return SimpleNamespace(
            logical_batch_sha256=digest,
            manifest_path=tmp_path / "capture.json",
            payload_sha256="payload",
            manifest_sha256="manifest",
            config_sha256="config",
            optimizer_contract_sha256="contract",
            initial_adapter_sha256="adapter",
        )

    monkeypatch.setattr(trainer_mod, "save_optimizer_batch_capture", capture)
    _set_resume_payload(monkeypatch, step=-1, checkpoint_path="/checkpoint/initial")
    env.flow.sepa_controller.load_state_dict = lambda _state: None
    return make_config(
        tmp_path,
        optimizer_batch_capture=True,
        resume_from=str(tmp_path / "resume"),
    )


def track_states(
    monkeypatch: MonkeyPatch,
) -> list[dict[str, object]]:
    states: list[dict[str, object]] = []
    monkeypatch.setattr(
        trainer_mod,
        "save_trainer_state",
        lambda *_a, **kwargs: states.append(dict(kwargs)),
    )
    return states


def recommend(env: TestEnv, size: int) -> None:
    env.backpressure.recommend = lambda: (
        env.trace.events.append("backpressure:recommend")
        or SimpleNamespace(action="increase", recommended_batch_size=size)
    )


def fail_train(monkeypatch: MonkeyPatch, env: TestEnv) -> None:
    def fail(*_args, **_kwargs):
        env.trace.events.append("train:failed")
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(trainer_mod, "run_rl_echo_train_step", fail)


def track_loggers(monkeypatch: MonkeyPatch) -> list[object]:
    loggers: list[object] = []

    class Logger:
        def __init__(self, *_args, **_kwargs) -> None:
            self.closed = False
            loggers.append(self)

        def log(self, _payload) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(trainer_mod, "JsonlLogger", Logger)
    return loggers


def resume_config(
    monkeypatch: MonkeyPatch,
    env: TestEnv,
    tmp_path: Path,
    *,
    saved_step: int,
    warmup_steps: int,
) -> TrainConfig:
    sft_data = SimpleNamespace(examples=[object()], schedule_contract={})
    monkeypatch.setattr(trainer_mod, "load_sft_warmup_data", lambda *_a: sft_data)
    _set_resume_payload(
        monkeypatch,
        step=saved_step,
        checkpoint_path="/checkpoint/resume",
        sft_schedule={},
    )
    monkeypatch.setattr(
        trainer_mod,
        "verify_sft_warmup_resume_schedule",
        lambda *_a, **_kw: env.trace.events.append("resume:verify"),
    )
    monkeypatch.setattr(
        trainer_mod,
        "run_sft_warmup_step",
        lambda *_a, **_kw: env.trace.events.append("warmup") or None,
    )
    env.flow.sepa_controller.load_state_dict = lambda _state: env.trace.events.append(
        "sepa:load"
    )
    return make_config(
        tmp_path,
        max_steps=saved_step + 2,
        sft_warmup_steps=warmup_steps,
        sft_data_path=str(tmp_path / "sft.jsonl"),
        resume_from=str(tmp_path / "resume"),
    )


def _set_resume_payload(
    monkeypatch: MonkeyPatch,
    *,
    step: int,
    checkpoint_path: str,
    sft_schedule: dict[str, object] | None = None,
) -> None:
    monkeypatch.setattr(
        trainer_mod,
        "load_trainer_state",
        lambda _path: {
            "step": step,
            "example_idx": 0,
            "total_correct": 0,
            "total_completions": 0,
            "current_batch_size": 1,
            "current_group_size": 1,
            "checkpoint_name": "resume",
            "checkpoint_path": checkpoint_path,
            "sepa": {},
            "sft_schedule": sft_schedule,
        },
    )


def _rollout_values(case: Case) -> _RolloutValues:
    return _RolloutValues(
        advantages=((0.0, 1.0 if case.rl else 0.0),),
        echo_advantages=((0.0, 0.5),) if case.echo else (),
    )


class _Tokenizer:
    vocab_size = 32
    added_tokens_encoder: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = text, add_special_tokens
        return [7]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [str(value) for value in ids]


class _RecordingBackend:
    """The smallest effectful adapter satisfying the trainer protocol."""

    def __init__(self, trace: Trace, adapter_path: Path) -> None:
        self.trace = trace
        self.adapter_path = adapter_path
        self.loaded: list[str] = []
        self.shutdown_called = False

    def checkpoint(self, name: str) -> None:
        self.trace.events.append(f"checkpoint:{name}")

    def save_adapter(self, path: str, name: str) -> str:
        self.trace.events.append(f"save:{name}")
        save_dir = Path(path) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        return str(save_dir)

    def load_state(self, name: str) -> None:
        self.trace.events.append(f"load:{name}")
        self.loaded.append(name)

    def shutdown(self) -> None:
        self.trace.events.append("shutdown")
        self.shutdown_called = True


class _Backpressure:
    def __init__(self, trace: Trace) -> None:
        self.trace = trace

    def observe(self, observation: object) -> None:
        self.trace.events.append("backpressure:observe")
        self.trace.pressure.append(observation)

    def recommend(self) -> object:
        self.trace.events.append("backpressure:recommend")
        return SimpleNamespace(action="hold", recommended_batch_size=1)


def _flow(backend: object, backpressure: object) -> object:
    return SimpleNamespace(
        backend=backend,
        backend_capabilities=BackendCapabilities(
            reports_sync_loss=True,
            preserves_token_advantages=True,
            supports_checkpoint_resume=True,
            resume_runtime_dependent=False,
            checkpoint_resume_mode="adapter_only",
            supports_echo_shared_forward=True,
        ),
        backend_capability_source="test",
        planning_detector=None,
        sepa_controller=SimpleNamespace(state_dict=lambda: {}),
        backpressure=backpressure,
        needs_planning=False,
        uses_sepa_controller=False,
    )


def _patch_common(monkeypatch: MonkeyPatch, trace: Trace) -> None:
    example = SimpleNamespace(prompt="prompt", reference="answer", task="task", info={})
    prompts = SimpleNamespace(
        ids=[[7]],
        objs=["prompt"],
        answers=["answer"],
        tasks=["task"],
        infos=[{}],
    )
    monkeypatch.setattr(
        trainer_mod, "assert_readiness_runtime_matches_file", lambda _c: None
    )
    monkeypatch.setattr(
        trainer_mod,
        "load_training_examples",
        lambda *_a, **_kw: SimpleNamespace(examples=[example], environment=None),
    )
    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *_a, **_kw: _Tokenizer()),
    )
    monkeypatch.setattr(
        trainer_mod,
        "get_registry",
        lambda _kind: SimpleNamespace(
            create=lambda *_a, **_kw: SimpleNamespace(score=lambda *_args: 0.0)
        ),
    )
    monkeypatch.setattr(
        trainer_mod,
        "select_prompt_batch",
        lambda *_a, **_kw: (prompts, 1),
    )
    monkeypatch.setattr(trainer_mod, "load_sft_warmup_data", lambda *_a: None)
    monkeypatch.setattr(trainer_mod, "init_wandb", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        trainer_mod, "announce_checkpoint_recoverability", lambda *_a: None
    )
    monkeypatch.setattr(
        trainer_mod,
        "upload_checkpoint_artifact",
        lambda *_a, **_kw: trace.events.append("upload"),
    )
    monkeypatch.setattr(
        trainer_mod,
        "save_trainer_state",
        lambda *_a, **kw: trace.events.append(f"state:{kw['step']}"),
    )


def _rollout_effect(trace: Trace, case: Case):
    values = _rollout_values(case)

    def apply(*args, **kwargs) -> None:
        trace.events.append("rollout")
        acc = kwargs.get("acc", args[6])
        acc.rewards = [1.0]
        acc.correct = 1
        acc.datum_tokens = [[7, 8]]
        acc.datum_logprobs = [[0.0, -0.1]]
        acc.datum_advantages = [list(row) for row in values.advantages]
        acc.sample_time_s = 0.25
        acc.sampled_completion_token_count = 1
        if values.echo_advantages:
            acc.datum_echo_advantages = [list(row) for row in values.echo_advantages]
            acc.datum_echo_terminal_masks = [[0, 0]]
            acc.datum_echo_full_observation_counts = [1]
            acc.echo_eligible_rollout_count = 1
            acc.echo_build = EchoBuildStats(candidate_datums=1, candidate_tokens=1)

    return apply


def _train_effect(trace: Trace, case: Case):
    def apply(
        _backend,
        tokens,
        logprobs,
        advantages,
        echo_advantages,
        echo_counts,
        **kwargs,
    ):
        trace.events.append("train")
        trace.trains.append(
            Call(
                tokens,
                logprobs,
                advantages,
                echo_advantages,
                echo_counts,
                kwargs,
            )
        )
        return case.loss, 0.125 if echo_advantages else 0.0, bool(echo_advantages)

    return apply


def _log_effect(trace: Trace):
    def apply(context, **_kwargs):
        trace.events.append("log")
        trace.logs.append(context)
        return SimpleNamespace(delight_eta_ema=None)

    return apply
