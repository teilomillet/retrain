"""Trainer wiring tests for composable-advantage parameters."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

from retrain import trainer as trainer_mod
from retrain.advantages import AdvantageResult
from retrain.backpressure import NoOpBackPressure
from retrain.config import TrainConfig
from retrain.data import Example
from retrain.flow import build_flow


class _FakeTokenizer:
    vocab_size = 256
    added_tokens_encoder: dict[str, int] = {}

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


def _make_fake_flow(cfg, helper):
    """Build a Tier-1 flow, then inject fake Tier-2 objects."""
    from retrain.sepa import SEPAController

    flow = build_flow(cfg, gpu=False)
    flow.backend = helper
    flow.planning_detector = SimpleNamespace(
        detect=lambda token_strs: [0] * len(token_strs)
    )
    flow.sepa_controller = SEPAController(
        sepa_steps=cfg.sepa_steps,
        sepa_schedule=cfg.sepa_schedule,
        sepa_delay_steps=cfg.sepa_delay_steps,
        sepa_correct_rate_gate=cfg.sepa_correct_rate_gate,
    )
    flow.backpressure = NoOpBackPressure()
    return flow


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
                        1.0 if ("\\boxed{42}" in response and reference == "42") else 0.0
                    )
                )
            )
        raise AssertionError(f"Unexpected registry kind: {kind}")

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model: _FakeTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "get_registry", fake_get_registry)
    monkeypatch.setattr(
        trainer_mod,
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
        advantage_mode="grpo",
        transform_mode="none",
        advantage_params={"scale": 3.0},
    )

    flow = _make_fake_flow(cfg, helper)
    final_path = trainer_mod.train(cfg, flow=flow)

    assert captured_kwargs, "compute_composable_advantages should be called"
    assert captured_kwargs[0]["advantage_params"] == cfg.effective_advantage_params
    assert final_path == str(Path(cfg.adapter_path) / "final")


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

    def fake_compute_composable_advantages(*args: object, **kwargs: object) -> AdvantageResult:
        called["composable"] += 1
        raise AssertionError("Composable path should be bypassed when algorithm_mode is set.")

    helper = _FakeHelper(adapter_path=str(tmp_path / "adapter"))

    # Create a custom algorithm plugin for build_flow to resolve
    module_name = "custom_algorithms"
    plugin_file = tmp_path / f"{module_name}.py"
    plugin_file.write_text(
        "def my_algo(ctx):\n"
        "    return [[0.25] * len(seq) for seq in ctx.logprobs_G]\n"
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
                        1.0 if ("\\boxed{42}" in response and reference == "42") else 0.0
                    )
                )
            )
        raise AssertionError(f"Unexpected registry kind: {kind}")

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model: _FakeTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "get_registry", fake_get_registry)
    monkeypatch.setattr(
        trainer_mod,
        "compute_algorithm_advantages",
        fake_compute_algorithm_advantages,
    )
    monkeypatch.setattr(
        trainer_mod,
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
                        1.0 if ("\\boxed{42}" in response and reference == "42") else 0.0
                    )
                )
            )
        raise AssertionError(f"Unexpected registry kind: {kind}")

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda _model: _FakeTokenizer()),
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
