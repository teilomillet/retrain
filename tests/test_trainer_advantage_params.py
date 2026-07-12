"""Trainer wiring tests for composable-advantage parameters."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from retrain.training import trainer as trainer_mod
from retrain.training import signals as signal_mod
from retrain.advantages import AdvantageResult
from retrain.training.backpressure import NoOpBackPressure
from retrain.training.batch_digest import logical_optimizer_batch_sha256
from retrain.training.optimizer_batch import load_optimizer_batch_capture
from retrain.config import TrainConfig
from retrain.data.source import Example
from retrain.training.flow import build_flow
from retrain.environments.verifiers import VerifiersRolloutTiming, VerifiersTurnSample
from retrain.planning.types import PlanningDetector


class _FakeTokenizer:
    vocab_size = 256
    added_tokens_encoder: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [ord(char) for char in text]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [str(i) for i in ids]

    def batch_decode(
        self,
        token_seqs: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        _ = skip_special_tokens
        texts: list[str] = []
        for seq in token_seqs:
            if seq and seq[0] == 42:
                texts.append("\\boxed{42}")
            else:
                texts.append("\\boxed{0}")
        return texts


class _FakeHelper:
    def __init__(self, adapter_path: str) -> None:
        self.adapter_path = Path(adapter_path)

    def checkpoint(self, name: str) -> None:
        _ = name

    def sample(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[tuple[list[int], list[float]]]]:
        _ = num_samples, max_tokens, temperature, top_p
        groups: list[list[tuple[list[int], list[float]]]] = []
        for _prompt_ids in prompt_ids_list:
            groups.append(
                [
                    ([42], [-0.1]),
                    ([13], [-0.2]),
                ]
            )
        return groups

    def train_step(
        self,
        all_tokens: list[list[int]],
        all_logprobs: list[list[float]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float:
        _ = all_tokens, all_logprobs, all_advantages, lr, weight_decay
        return 0.123

    def save_adapter(self, path: str, name: str) -> str:
        save_dir = Path(path) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        return str(save_dir)

    def load_state(self, name: str) -> None:
        _ = name


class _RewardOrderedFakeHelper(_FakeHelper):
    def sample(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[tuple[list[int], list[float]]]]:
        _ = num_samples, max_tokens, temperature, top_p
        groups: list[list[tuple[list[int], list[float]]]] = []
        for _prompt_ids in prompt_ids_list:
            groups.append(
                [
                    ([13], [-0.2]),
                    ([42], [-0.1]),
                ]
            )
        return groups


class _EchoRecordingFakeHelper(_FakeHelper):
    def __init__(self, adapter_path: str) -> None:
        super().__init__(adapter_path)
        self.train_calls: list[dict[str, object]] = []
        self.sft_calls: list[dict[str, object]] = []
        self.hybrid_calls: list[dict[str, object]] = []

    def train_step(
        self,
        all_tokens: list[list[int]],
        all_logprobs: list[list[float]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float:
        self.train_calls.append(
            {
                "tokens": all_tokens,
                "logprobs": all_logprobs,
                "advantages": all_advantages,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
        return 0.123

    def sft_train_step(
        self,
        all_tokens: list[list[int]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float:
        self.sft_calls.append(
            {
                "tokens": all_tokens,
                "advantages": all_advantages,
                "lr": lr,
                "weight_decay": weight_decay,
                "loss_fn": getattr(self, "sft_loss_fn", ""),
            }
        )
        return 0.031

    def train_step_with_echo_masks(
        self,
        all_tokens: list[list[int]],
        all_logprobs: list[list[float]],
        all_advantages: list[list[float]],
        echo_advantages: list[list[float]],
        echo_full_observation_counts: list[int],
        echo_loss_fn: str,
        lr: float,
        weight_decay: float,
        echo_rollout_denominator: int = 0,
    ) -> tuple[float, float]:
        self.hybrid_calls.append(
            {
                "tokens": all_tokens,
                "logprobs": all_logprobs,
                "advantages": all_advantages,
                "echo_advantages": echo_advantages,
                "echo_full_observation_counts": echo_full_observation_counts,
                "echo_loss_fn": echo_loss_fn,
                "lr": lr,
                "weight_decay": weight_decay,
                "echo_rollout_denominator": echo_rollout_denominator,
            }
        )
        return 0.123, 0.031


def _make_fake_flow(cfg, helper):
    """Build a Tier-1 flow, then inject fake Tier-2 objects."""
    from retrain.training.sepa import SEPAController

    flow = build_flow(cfg, gpu=False)
    flow.backend = helper
    flow.planning_detector = cast(
        PlanningDetector,
        SimpleNamespace(detect=lambda token_strs: [0] * len(token_strs)),
    )
    flow.sepa_controller = SEPAController(
        sepa_steps=cfg.sepa_steps,
        sepa_schedule=cfg.sepa_schedule,
        sepa_delay_steps=cfg.sepa_delay_steps,
        sepa_correct_rate_gate=cfg.sepa_correct_rate_gate,
    )
    flow.backpressure = NoOpBackPressure()
    return flow


def test_train_persists_sft_warmup_schedule(monkeypatch, tmp_path):
    data_path = tmp_path / "sft.jsonl"
    data_path.write_text(json.dumps({"text": "supervised target"}) + "\n")
    helper = _EchoRecordingFakeHelper(adapter_path=str(tmp_path / "adapter"))

    def fake_get_registry(kind: str):
        if kind == "data_source":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    load=lambda: [
                        Example(
                            prompt="unused during warmup",
                            reference="42",
                            task="math",
                            info={"id": 1},
                        )
                    ]
                )
            )
        if kind == "reward":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    score=lambda _response, _reference: 0.0
                )
            )
        raise AssertionError(f"Unexpected registry kind: {kind}")

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model, **_kw: _FakeTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "get_registry", fake_get_registry)
    saved_schedules: list[object] = []
    real_save_trainer_state = trainer_mod.save_trainer_state

    def capture_save(*args, **kwargs):
        saved_schedules.append(kwargs.get("sft_schedule"))
        return real_save_trainer_state(*args, **kwargs)

    monkeypatch.setattr(trainer_mod, "save_trainer_state", capture_save)
    config = TrainConfig(
        backend="local",
        model="fake/model",
        max_steps=1,
        batch_size=1,
        group_size=2,
        max_tokens=8,
        save_every=1,
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapter"),
        sft_warmup_steps=1,
        sft_data_path=str(data_path),
        sft_batch_size=1,
        sft_reshuffle_each_epoch=True,
        seed=19,
    )

    final_path = trainer_mod.train(
        config,
        flow=_make_fake_flow(config, helper),
    )

    assert final_path == str(Path(config.adapter_path) / "final")
    assert len(helper.sft_calls) == 1
    assert len(saved_schedules) == 2
    for schedule in saved_schedules:
        assert isinstance(schedule, dict)
        assert schedule["data_sha256"] == hashlib.sha256(
            data_path.read_bytes()
        ).hexdigest()
        assert schedule["seed"] == 19
        assert schedule["reshuffle_each_epoch"] is True

    state = json.loads((Path(config.log_dir) / "trainer_state.json").read_text())
    assert state["sft_schedule"] == saved_schedules[-1]


def test_train_forwards_effective_advantage_params(monkeypatch, tmp_path):
    captured_kwargs: list[dict[str, object]] = []

    def fake_compute_composable_advantages(
        rewards_G: list[float],
        logprobs_G: list[list[float]],
        planning_masks_G: list[list[int]],
        **kwargs: object,
    ) -> AdvantageResult:
        _ = rewards_G, planning_masks_G
        captured_kwargs.append(dict(kwargs))
        return AdvantageResult(
            token_advs=[[0.5] * len(seq) for seq in logprobs_G],
            has_stats=False,
        )

    helper = _FakeHelper(adapter_path=str(tmp_path / "adapter"))

    def fake_get_registry(kind: str):
        if kind == "data_source":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    load=lambda: [
                        Example(
                            prompt="2+2?",
                            reference="42",
                            task="math",
                            info={"id": 1},
                        )
                    ]
                )
            )
        if kind == "reward":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    score=lambda response, reference: (
                        1.0
                        if ("\\boxed{42}" in response and reference == "42")
                        else 0.0
                    )
                )
            )
        raise AssertionError(f"Unexpected registry kind: {kind}")

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model, **_kw: _FakeTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "get_registry", fake_get_registry)
    monkeypatch.setattr(
        signal_mod,
        "compute_composable_advantages",
        fake_compute_composable_advantages,
    )
    monkeypatch.setattr(
        trainer_mod,
        "encode_prompt_for_sampling",
        lambda _tokenizer, _prompt: [101],
    )
    monkeypatch.setattr(trainer_mod, "prompt_preview", lambda prompt: str(prompt))

    initial_adapter = tmp_path / "initial-adapter"
    initial_adapter.mkdir()
    (initial_adapter / "adapter_model.safetensors").write_bytes(b"initial")
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    (resume_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "step": -1,
                "example_idx": 0,
                "total_correct": 0,
                "total_completions": 0,
                "current_batch_size": 1,
                "current_group_size": 2,
                "checkpoint_name": "init",
                "checkpoint_path": str(initial_adapter),
                "sepa": {},
            }
        )
    )
    cfg = TrainConfig(
        backend="local",
        model="fake/model",
        max_steps=1,
        batch_size=1,
        group_size=2,
        max_tokens=8,
        save_every=0,
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapter"),
        advantage_mode="grpo",
        transform_mode="none",
        advantage_params={"scale": 3.0},
        adv_clip_max=0.25,
        resume_from=str(resume_dir),
        optimizer_batch_capture=True,
    )

    flow = _make_fake_flow(cfg, helper)
    final_path = trainer_mod.train(cfg, flow=flow)

    assert captured_kwargs, "compute_composable_advantages should be called"
    assert captured_kwargs[0]["advantage_params"] == cfg.effective_advantage_params
    assert final_path == str(Path(cfg.adapter_path) / "final")
    metrics = json.loads((Path(cfg.log_dir) / "metrics.jsonl").read_text().splitlines()[0])
    assert metrics[
        "optimizer/logical_batch_sha256"
    ] == logical_optimizer_batch_sha256(
        [[101, 42], [101, 13]],
        [[0.0, -0.1], [0.0, -0.2]],
        [[0.0, 0.25], [0.0, 0.25]],
    )
    manifest_path = Path(cfg.log_dir) / "optimizer_batch_step_000000.manifest.json"
    captured = load_optimizer_batch_capture(manifest_path)
    assert captured.batch.advantages == [[0.0, 0.25], [0.0, 0.25]]
    assert metrics["optimizer_batch/mode"] == "capture"
    assert metrics["optimizer_batch/payload_sha256"] == captured.payload_sha256


def test_train_feeds_previous_delight_eta_into_next_step(monkeypatch, tmp_path):
    captured_kwargs: list[dict[str, object]] = []
    eta_by_call = [2.0, 1.5]

    def fake_compute_composable_advantages(
        rewards_G: list[float],
        logprobs_G: list[list[float]],
        planning_masks_G: list[list[int]],
        **kwargs: object,
    ) -> AdvantageResult:
        _ = rewards_G, planning_masks_G
        captured_kwargs.append(dict(kwargs))
        idx = min(len(captured_kwargs) - 1, len(eta_by_call) - 1)
        return AdvantageResult(
            token_advs=[[0.5] * len(seq) for seq in logprobs_G],
            has_stats=False,
            extra_metrics={"dg_eta": eta_by_call[idx]},
        )

    helper = _FakeHelper(adapter_path=str(tmp_path / "adapter"))

    def fake_get_registry(kind: str):
        if kind == "data_source":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    load=lambda: [
                        Example(
                            prompt="2+2?",
                            reference="42",
                            task="math",
                            info={"id": 1},
                        )
                    ]
                )
            )
        if kind == "reward":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    score=lambda response, reference: (
                        1.0
                        if ("\\boxed{42}" in response and reference == "42")
                        else 0.0
                    )
                )
            )
        raise AssertionError(f"Unexpected registry kind: {kind}")

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model, **_kw: _FakeTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "get_registry", fake_get_registry)
    monkeypatch.setattr(
        signal_mod,
        "compute_composable_advantages",
        fake_compute_composable_advantages,
    )
    monkeypatch.setattr(
        trainer_mod,
        "encode_prompt_for_sampling",
        lambda _tokenizer, _prompt: [101],
    )
    monkeypatch.setattr(trainer_mod, "prompt_preview", lambda prompt: str(prompt))

    cfg = TrainConfig(
        backend="local",
        model="fake/model",
        max_steps=2,
        batch_size=1,
        group_size=2,
        max_tokens=8,
        save_every=0,
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapter"),
        advantage_mode="grpo",
        transform_mode="delight",
        transform_params={
            "delight_eta_mode": "adaptive",
            "delight_eta_ema_decay": 0.8,
        },
    )

    flow = _make_fake_flow(cfg, helper)
    trainer_mod.train(cfg, flow=flow)

    assert len(captured_kwargs) >= 2
    first_params = cast(dict[str, object], captured_kwargs[0]["transform_params"])
    second_params = cast(dict[str, object], captured_kwargs[1]["transform_params"])
    assert "delight_eta_prev" not in first_params
    assert second_params["delight_eta_prev"] == pytest.approx(2.0)


def test_train_prefers_algorithm_mode_over_composable(monkeypatch, tmp_path):
    called = {"algorithm": 0, "composable": 0}
    captured_algorithm_params: list[dict[str, object]] = []

    def fake_compute_algorithm_advantages(
        rewards_G: list[float],
        logprobs_G: list[list[float]],
        planning_masks_G: list[list[int]],
        **kwargs: object,
    ) -> AdvantageResult:
        _ = rewards_G, planning_masks_G
        called["algorithm"] += 1
        captured_algorithm_params.append(dict(kwargs))
        return AdvantageResult(
            token_advs=[[0.25] * len(seq) for seq in logprobs_G],
            has_stats=False,
        )

    def fake_compute_composable_advantages(
        *args: object, **kwargs: object
    ) -> AdvantageResult:
        called["composable"] += 1
        raise AssertionError(
            "Composable path should be bypassed when algorithm_mode is set."
        )

    helper = _FakeHelper(adapter_path=str(tmp_path / "adapter"))

    # Create a custom algorithm plugin for build_flow to resolve
    module_name = "custom_algorithms"
    plugin_file = tmp_path / f"{module_name}.py"
    plugin_file.write_text(
        "def my_algo(ctx):\n    return [[0.25] * len(seq) for seq in ctx.logprobs_G]\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    def fake_get_registry(kind: str):
        if kind == "data_source":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    load=lambda: [
                        Example(
                            prompt="2+2?",
                            reference="42",
                            task="math",
                            info={"id": 1},
                        )
                    ]
                )
            )
        if kind == "reward":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    score=lambda response, reference: (
                        1.0
                        if ("\\boxed{42}" in response and reference == "42")
                        else 0.0
                    )
                )
            )
        raise AssertionError(f"Unexpected registry kind: {kind}")

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model, **_kw: _FakeTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "get_registry", fake_get_registry)
    monkeypatch.setattr(
        signal_mod,
        "compute_algorithm_advantages",
        fake_compute_algorithm_advantages,
    )
    monkeypatch.setattr(
        signal_mod,
        "compute_composable_advantages",
        fake_compute_composable_advantages,
    )
    monkeypatch.setattr(
        trainer_mod,
        "encode_prompt_for_sampling",
        lambda _tokenizer, _prompt: [101],
    )
    monkeypatch.setattr(trainer_mod, "prompt_preview", lambda prompt: str(prompt))

    cfg = TrainConfig(
        backend="local",
        model="fake/model",
        max_steps=1,
        batch_size=1,
        group_size=2,
        max_tokens=8,
        save_every=0,
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapter"),
        algorithm_mode=f"{module_name}.my_algo",
        algorithm_params={"alpha": 0.3},
        advantage_mode="grpo",
        transform_mode="none",
    )

    flow = _make_fake_flow(cfg, helper)
    _ = trainer_mod.train(cfg, flow=flow)

    assert called["algorithm"] > 0
    assert called["composable"] == 0
    assert captured_algorithm_params[0]["algorithm_mode"] == cfg.algorithm_mode
    assert captured_algorithm_params[0]["params"] == cfg.effective_algorithm_params


def test_tinker_entropy_stats_pipeline_does_not_break(monkeypatch, tmp_path):
    """Entropy-weighted transforms should run cleanly on token-preserving tinker backend."""

    helper = _FakeHelper(adapter_path=str(tmp_path / "adapter"))

    def fake_get_registry(kind: str):
        if kind == "data_source":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    load=lambda: [
                        Example(
                            prompt="2+2?",
                            reference="42",
                            task="math",
                            info={"id": 1},
                        )
                    ]
                )
            )
        if kind == "reward":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    score=lambda response, reference: (
                        1.0
                        if ("\\boxed{42}" in response and reference == "42")
                        else 0.0
                    )
                )
            )
        raise AssertionError(f"Unexpected registry kind: {kind}")

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model, **_kw: _FakeTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "get_registry", fake_get_registry)
    monkeypatch.setattr(
        trainer_mod,
        "encode_prompt_for_sampling",
        lambda _tokenizer, _prompt: [101],
    )
    monkeypatch.setattr(trainer_mod, "prompt_preview", lambda prompt: str(prompt))

    cfg = TrainConfig(
        backend="tinker",
        model="fake/model",
        max_steps=1,
        batch_size=1,
        group_size=2,
        max_tokens=8,
        save_every=0,
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapter"),
        advantage_mode="maxrl",
        transform_mode="gtpo",
    )

    flow = _make_fake_flow(cfg, helper)
    trace = flow.trace()
    assert trace.ok, "tinker flow should accept entropy-weighted transforms"

    final_path = trainer_mod.train(cfg, flow=flow)
    assert final_path == str(Path(cfg.adapter_path) / "final")

    metrics_path = Path(cfg.log_dir) / "metrics.jsonl"
    entries = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    assert entries
    step_metrics = entries[-1]

    assert step_metrics["backend_preserves_token_advantages"] is True
    assert "exec_entropy_mean" in step_metrics
    assert "exec_entropy_var" in step_metrics
    assert math.isfinite(float(step_metrics["exec_entropy_mean"]))
    assert math.isfinite(float(step_metrics["exec_entropy_var"]))


def test_train_generation_logging_defaults_skip_surprisal_tokens(
    monkeypatch,
    tmp_path,
):
    helper = _FakeHelper(adapter_path=str(tmp_path / "adapter"))

    def fake_get_registry(kind: str):
        if kind == "data_source":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    load=lambda: [
                        Example(
                            prompt="2+2?",
                            reference="42",
                            task="math",
                            info={"id": 1},
                        )
                    ]
                )
            )
        if kind == "reward":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    score=lambda response, reference: (
                        1.0
                        if ("\\boxed{42}" in response and reference == "42")
                        else 0.0
                    )
                )
            )
        raise AssertionError(f"Unexpected registry kind: {kind}")

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model, **_kw: _FakeTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "get_registry", fake_get_registry)
    monkeypatch.setattr(
        trainer_mod,
        "encode_prompt_for_sampling",
        lambda _tokenizer, _prompt: [101],
    )
    monkeypatch.setattr(trainer_mod, "prompt_preview", lambda prompt: str(prompt))

    cfg = TrainConfig(
        backend="local",
        model="fake/model",
        max_steps=1,
        batch_size=1,
        group_size=2,
        max_tokens=8,
        save_every=0,
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapter"),
        advantage_mode="grpo",
        transform_mode="none",
    )

    flow = _make_fake_flow(cfg, helper)
    trainer_mod.train(cfg, flow=flow)

    generations_path = Path(cfg.log_dir) / "emergence" / "generations.jsonl"
    entries = [
        json.loads(line)
        for line in generations_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(entries) == 1
    assert entries[0]["completion"] == "\\boxed{42}"
    assert all("top_surprisal_tokens" not in entry for entry in entries)


def test_train_generation_logging_can_limit_density_and_enable_surprisal(
    monkeypatch,
    tmp_path,
):
    helper = _FakeHelper(adapter_path=str(tmp_path / "adapter"))

    def fake_get_registry(kind: str):
        if kind == "data_source":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    load=lambda: [
                        Example(
                            prompt="2+2?",
                            reference="42",
                            task="math",
                            info={"id": 1},
                        )
                    ]
                )
            )
        if kind == "reward":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    score=lambda response, reference: (
                        1.0
                        if ("\\boxed{42}" in response and reference == "42")
                        else 0.0
                    )
                )
            )
        raise AssertionError(f"Unexpected registry kind: {kind}")

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model, **_kw: _FakeTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "get_registry", fake_get_registry)
    monkeypatch.setattr(
        trainer_mod,
        "encode_prompt_for_sampling",
        lambda _tokenizer, _prompt: [101],
    )
    monkeypatch.setattr(trainer_mod, "prompt_preview", lambda prompt: str(prompt))

    cfg = TrainConfig(
        backend="local",
        model="fake/model",
        max_steps=1,
        batch_size=1,
        group_size=2,
        max_tokens=8,
        save_every=0,
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapter"),
        advantage_mode="grpo",
        transform_mode="none",
        generation_log_samples_per_prompt=1,
        generation_top_surprisal_limit=1,
    )

    flow = _make_fake_flow(cfg, helper)
    trainer_mod.train(cfg, flow=flow)

    generations_path = Path(cfg.log_dir) / "emergence" / "generations.jsonl"
    entries = [
        json.loads(line)
        for line in generations_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(entries) == 1
    assert entries[0]["completion"] == "\\boxed{42}"
    assert entries[0]["top_surprisal_tokens"] == [
        {"pos": 0, "surprisal": 0.1, "token": "42"},
    ]


def test_train_generation_logging_prefers_high_reward_samples(
    monkeypatch,
    tmp_path,
):
    helper = _RewardOrderedFakeHelper(adapter_path=str(tmp_path / "adapter"))

    def fake_get_registry(kind: str):
        if kind == "data_source":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    load=lambda: [
                        Example(
                            prompt="2+2?",
                            reference="42",
                            task="math",
                            info={"id": 1},
                        )
                    ]
                )
            )
        if kind == "reward":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    score=lambda response, reference: (
                        1.0
                        if ("\\boxed{42}" in response and reference == "42")
                        else 0.0
                    )
                )
            )
        raise AssertionError(f"Unexpected registry kind: {kind}")

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model, **_kw: _FakeTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "get_registry", fake_get_registry)
    monkeypatch.setattr(
        trainer_mod,
        "encode_prompt_for_sampling",
        lambda _tokenizer, _prompt: [101],
    )
    monkeypatch.setattr(trainer_mod, "prompt_preview", lambda prompt: str(prompt))

    cfg = TrainConfig(
        backend="local",
        model="fake/model",
        max_steps=1,
        batch_size=1,
        group_size=2,
        max_tokens=8,
        save_every=0,
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapter"),
        advantage_mode="grpo",
        transform_mode="none",
        generation_log_samples_per_prompt=1,
    )

    flow = _make_fake_flow(cfg, helper)
    trainer_mod.train(cfg, flow=flow)

    generations_path = Path(cfg.log_dir) / "emergence" / "generations.jsonl"
    entries = [
        json.loads(line)
        for line in generations_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(entries) == 1
    assert entries[0]["completion"] == "\\boxed{42}"


def test_train_can_disable_generation_logging(monkeypatch, tmp_path):
    helper = _FakeHelper(adapter_path=str(tmp_path / "adapter"))

    def fake_get_registry(kind: str):
        if kind == "data_source":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    load=lambda: [
                        Example(
                            prompt="2+2?",
                            reference="42",
                            task="math",
                            info={"id": 1},
                        )
                    ]
                )
            )
        if kind == "reward":
            return SimpleNamespace(
                create=lambda _name, _cfg: SimpleNamespace(
                    score=lambda response, reference: (
                        1.0
                        if ("\\boxed{42}" in response and reference == "42")
                        else 0.0
                    )
                )
            )
        raise AssertionError(f"Unexpected registry kind: {kind}")

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model, **_kw: _FakeTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "get_registry", fake_get_registry)
    monkeypatch.setattr(
        trainer_mod,
        "encode_prompt_for_sampling",
        lambda _tokenizer, _prompt: [101],
    )
    monkeypatch.setattr(trainer_mod, "prompt_preview", lambda prompt: str(prompt))

    cfg = TrainConfig(
        backend="local",
        model="fake/model",
        max_steps=1,
        batch_size=1,
        group_size=2,
        max_tokens=8,
        save_every=0,
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapter"),
        advantage_mode="grpo",
        transform_mode="none",
        log_generations=False,
    )

    flow = _make_fake_flow(cfg, helper)
    trainer_mod.train(cfg, flow=flow)

    generations_path = Path(cfg.log_dir) / "emergence" / "generations.jsonl"
    assert not generations_path.exists()


def test_train_echo_multiturn_algorithm_mode_trains_prompt_suffix_and_logs_metrics(
    monkeypatch,
    tmp_path,
):
    helper = _EchoRecordingFakeHelper(adapter_path=str(tmp_path / "adapter"))
    env = SimpleNamespace(env_id="fake/multiturn")

    def fake_run_multiturn_group(*args: object, **kwargs: object):
        _ = args, kwargs
        turns = [
            [
                VerifiersTurnSample(
                    prompt_ids=[1, 2],
                    completion_ids=[3],
                    completion_logprobs=[-0.1],
                    completion_text="a",
                ),
                VerifiersTurnSample(
                    prompt_ids=[1, 2, 3, 50, 51],
                    completion_ids=[4],
                    completion_logprobs=[-0.2],
                    completion_text="b",
                    observation_mask=[0, 0, 0, 1, 1],
                ),
                VerifiersTurnSample(
                    prompt_ids=[9, 8],
                    completion_ids=[7],
                    completion_logprobs=[-0.3],
                    completion_text="e",
                ),
            ],
            [
                VerifiersTurnSample(
                    prompt_ids=[1, 2],
                    completion_ids=[5],
                    completion_logprobs=[-0.1],
                    completion_text="c",
                ),
                VerifiersTurnSample(
                    prompt_ids=[1, 2, 5, 60],
                    completion_ids=[6],
                    completion_logprobs=[-0.2],
                    completion_text="d",
                    observation_mask=[0, 0, 0, 1],
                ),
            ],
        ]
        return (
            [1.0, 0.0],
            turns,
            ["abe", "cd"],
            [],
            [],
            [],
            [],
            VerifiersRolloutTiming(model_tokens=5, turns=5),
        )

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model, **_kw: _FakeTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "load_verifiers_environment", lambda _cfg: env)
    monkeypatch.setattr(
        trainer_mod,
        "load_examples_from_environment",
        lambda _env, _cfg: [
            Example(
                prompt=[{"role": "user", "content": "task"}],
                reference="42",
                task="fake",
                info={"id": 1},
            )
        ],
    )
    monkeypatch.setattr(trainer_mod, "is_multiturn_environment", lambda _env: True)
    monkeypatch.setattr(trainer_mod, "run_multiturn_group", fake_run_multiturn_group)
    monkeypatch.setattr(
        trainer_mod,
        "encode_prompt_for_sampling",
        lambda _tokenizer, _prompt: [101],
    )
    monkeypatch.setattr(trainer_mod, "prompt_preview", lambda prompt: str(prompt))

    cfg = TrainConfig(
        backend="local",
        model="fake/model",
        max_steps=1,
        batch_size=1,
        group_size=2,
        max_tokens=8,
        save_every=0,
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapter"),
        environment_provider="verifiers",
        environment_id="fake/multiturn",
        algorithm_mode="reinforce_pp_gtpo",
        echo_enabled=True,
        echo_weight=0.2,
        echo_max_tokens_per_step=10,
        echo_max_token_ratio=10.0,
    )

    flow = _make_fake_flow(cfg, helper)
    trainer_mod.train(cfg, flow=flow)

    assert helper.train_calls == []
    assert helper.sft_calls == []
    assert len(helper.hybrid_calls) == 1
    assert helper.hybrid_calls[0]["echo_loss_fn"] == "cross_entropy"
    assert helper.hybrid_calls[0]["tokens"] == [
        [1, 2, 3, 50, 51, 4],
        [9, 8, 7],
        [1, 2, 5, 60, 6],
    ]
    echo_advantages = helper.hybrid_calls[0]["echo_advantages"]
    assert echo_advantages == [
        [0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.2, 0.0],
    ]
    assert helper.hybrid_calls[0]["echo_full_observation_counts"] == [2, 2, 1]
    assert helper.hybrid_calls[0]["echo_rollout_denominator"] == 2

    metrics_path = Path(cfg.log_dir) / "metrics.jsonl"
    entries = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    step_metrics = entries[-1]
    assert step_metrics["condition"] == "reinforce_pp_gtpo+echo"
    assert step_metrics["echo/enabled"] == 1
    assert step_metrics["echo/candidate_tokens"] == 3
    assert step_metrics["echo/observation_mask_datums"] == 2
    assert step_metrics["echo/kept_tokens"] == 3
    assert step_metrics["echo/token_ratio"] == pytest.approx(0.6)
    assert step_metrics["echo/split_non_prefix"] == 1
    assert step_metrics["rl/sampled_action_tokens"] == 5
    assert step_metrics["rl/eligible_action_tokens"] == 5
    assert step_metrics["rl/datumized_action_tokens"] == 5
    assert step_metrics["rl/action_token_datumization_ratio"] == pytest.approx(1.0)
    assert step_metrics["echo/skipped_entropy_floor"] == 0
    assert step_metrics["echo/mode_collapse_guard"] == 0
    assert step_metrics["echo/joint_optimizer_step"] == 1


@pytest.mark.parametrize("strict_live_bridge", [False, True])
def test_train_echo_entropy_floor_skips_or_fails_closed(
    monkeypatch,
    tmp_path,
    strict_live_bridge,
):
    helper = _EchoRecordingFakeHelper(adapter_path=str(tmp_path / "adapter"))
    env = SimpleNamespace(env_id="fake/multiturn")

    def fake_run_multiturn_group(*args: object, **kwargs: object):
        _ = args, kwargs
        turns = [
            [
                VerifiersTurnSample(
                    prompt_ids=[1, 2],
                    completion_ids=[3],
                    completion_logprobs=[-0.01],
                    completion_text="a",
                ),
                VerifiersTurnSample(
                    prompt_ids=[1, 2, 3, 50],
                    completion_ids=[4],
                    completion_logprobs=[-0.01],
                    completion_text="b",
                ),
            ],
            [
                VerifiersTurnSample(
                    prompt_ids=[1, 2],
                    completion_ids=[5],
                    completion_logprobs=[-0.01],
                    completion_text="c",
                ),
                VerifiersTurnSample(
                    prompt_ids=[1, 2, 5, 60],
                    completion_ids=[6],
                    completion_logprobs=[-0.01],
                    completion_text="d",
                ),
            ],
        ]
        return (
            [1.0, 0.0],
            turns,
            ["ab", "cd"],
            [],
            [],
            [],
            [],
            VerifiersRolloutTiming(model_tokens=4, turns=4),
        )

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model, **_kw: _FakeTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "load_verifiers_environment", lambda _cfg: env)
    monkeypatch.setattr(
        trainer_mod,
        "load_examples_from_environment",
        lambda _env, _cfg: [
            Example(
                prompt=[{"role": "user", "content": "task"}],
                reference="42",
                task="fake",
                info={"id": 1},
            )
        ],
    )
    monkeypatch.setattr(trainer_mod, "is_multiturn_environment", lambda _env: True)
    monkeypatch.setattr(trainer_mod, "run_multiturn_group", fake_run_multiturn_group)
    monkeypatch.setattr(
        trainer_mod,
        "encode_prompt_for_sampling",
        lambda _tokenizer, _prompt: [101],
    )
    monkeypatch.setattr(trainer_mod, "prompt_preview", lambda prompt: str(prompt))

    cfg = TrainConfig(
        backend="local",
        model="fake/model",
        max_steps=1,
        batch_size=1,
        group_size=2,
        max_tokens=8,
        save_every=0,
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapter"),
        environment_provider="verifiers",
        environment_id="fake/multiturn",
        advantage_mode="grpo",
        transform_mode="none",
        echo_enabled=True,
        echo_weight=0.2,
        echo_entropy_floor=0.5,
        echo_require_live_observation_bridge=strict_live_bridge,
    )

    flow = _make_fake_flow(cfg, helper)
    if strict_live_bridge:
        with pytest.raises(
            RuntimeError,
            match="ECHO live-observation contract failed before optimizer",
        ):
            trainer_mod.train(cfg, flow=flow)
        assert helper.train_calls == []
        assert helper.sft_calls == []
        assert helper.hybrid_calls == []
        return

    trainer_mod.train(cfg, flow=flow)

    assert len(helper.train_calls) == 1
    assert helper.sft_calls == []
    assert helper.hybrid_calls == []

    metrics_path = Path(cfg.log_dir) / "metrics.jsonl"
    entries = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    step_metrics = entries[-1]
    assert step_metrics["echo/candidate_tokens"] == 2
    assert step_metrics["echo/kept_tokens"] == 0
    assert step_metrics["echo/skipped_entropy_floor"] == 1
    assert step_metrics["echo/joint_optimizer_step"] == 0
    assert step_metrics["echo/mode_collapse_guard"] == 1


def test_train_echo_keeps_uniform_failed_rollout_observations(monkeypatch, tmp_path):
    helper = _EchoRecordingFakeHelper(adapter_path=str(tmp_path / "adapter"))
    env = SimpleNamespace(env_id="fake/multiturn")

    def fake_run_multiturn_group(*args: object, **kwargs: object):
        _ = args, kwargs
        turns = [
            [
                VerifiersTurnSample(
                    prompt_ids=[1, 2],
                    completion_ids=[3],
                    completion_logprobs=[-0.1],
                    completion_text="a",
                ),
                VerifiersTurnSample(
                    prompt_ids=[1, 2, 3, 50, 51],
                    completion_ids=[4],
                    completion_logprobs=[-0.2],
                    completion_text="b",
                    observation_mask=[0, 0, 0, 1, 1],
                ),
            ],
            [
                VerifiersTurnSample(
                    prompt_ids=[1, 2],
                    completion_ids=[5],
                    completion_logprobs=[-0.1],
                    completion_text="c",
                ),
                VerifiersTurnSample(
                    prompt_ids=[1, 2, 5, 60],
                    completion_ids=[6],
                    completion_logprobs=[-0.2],
                    completion_text="d",
                    observation_mask=[0, 0, 0, 1],
                ),
            ],
        ]
        return (
            [0.0, 0.0],
            turns,
            ["ab", "cd"],
            [],
            [],
            [],
            [],
            VerifiersRolloutTiming(model_tokens=4, turns=4),
        )

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model, **_kw: _FakeTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "load_verifiers_environment", lambda _cfg: env)
    monkeypatch.setattr(
        trainer_mod,
        "load_examples_from_environment",
        lambda _env, _cfg: [
            Example(
                prompt=[{"role": "user", "content": "task"}],
                reference="42",
                task="fake",
                info={"id": 1},
            )
        ],
    )
    monkeypatch.setattr(trainer_mod, "is_multiturn_environment", lambda _env: True)
    monkeypatch.setattr(trainer_mod, "run_multiturn_group", fake_run_multiturn_group)
    monkeypatch.setattr(
        trainer_mod,
        "encode_prompt_for_sampling",
        lambda _tokenizer, _prompt: [101],
    )
    monkeypatch.setattr(trainer_mod, "prompt_preview", lambda prompt: str(prompt))

    cfg = TrainConfig(
        backend="local",
        model="fake/model",
        max_steps=1,
        batch_size=1,
        group_size=2,
        max_tokens=8,
        save_every=0,
        log_dir=str(tmp_path / "logs"),
        adapter_path=str(tmp_path / "adapter"),
        environment_provider="verifiers",
        environment_id="fake/multiturn",
        advantage_mode="grpo",
        transform_mode="none",
        echo_enabled=True,
        echo_weight=0.2,
        echo_max_token_ratio=10.0,
    )

    flow = _make_fake_flow(cfg, helper)
    trainer_mod.train(cfg, flow=flow)

    assert helper.train_calls == []
    assert helper.sft_calls == []
    assert len(helper.hybrid_calls) == 1
    assert helper.hybrid_calls[0]["tokens"] == [
        [1, 2, 3, 50, 51, 4],
        [1, 2, 5, 60, 6],
    ]
    assert helper.hybrid_calls[0]["advantages"] == [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert helper.hybrid_calls[0]["echo_advantages"] == [
        [0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.2, 0.0],
    ]
    assert helper.hybrid_calls[0]["echo_full_observation_counts"] == [2, 1]

    metrics_path = Path(cfg.log_dir) / "metrics.jsonl"
    entries = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    step_metrics = entries[-1]
    assert step_metrics["rl/completion_tokens"] == 4
    assert step_metrics["echo/candidate_tokens"] == 3
    assert step_metrics["echo/kept_tokens"] == 3
    assert step_metrics["echo/reference_completion_tokens"] == 4
    assert step_metrics["echo/token_ratio"] == pytest.approx(0.75)
    assert step_metrics["echo/skipped_entropy_floor"] == 0
    assert step_metrics["echo/joint_optimizer_step"] == 1
