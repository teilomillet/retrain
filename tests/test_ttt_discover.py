"""Tests for the TTT-Discover runner and archive."""

from __future__ import annotations

import json
from dataclasses import MISSING, fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from retrain.advantages import compute_algorithm_advantages
from retrain.backend_definitions import BackendCapabilities
from retrain.config import TrainConfig
from retrain.data import Example
from retrain.ttt_discover import (
    DiscoverArchive,
    TTTDiscoverRunner,
    build_discovery_prompt,
)
from retrain.training_runner import TrainingRunResult


def _bare_config(**overrides: object) -> TrainConfig:
    """Build a TrainConfig with defaults, skipping __post_init__ validation."""
    config = TrainConfig.__new__(TrainConfig)
    for f in fields(TrainConfig):
        if f.default is not MISSING:
            setattr(config, f.name, f.default)
        elif f.default_factory is not MISSING:
            setattr(config, f.name, f.default_factory())
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestDiscoverEntropicAlgorithm:
    def test_favors_max_reward_with_zero_sum_episode_advantages(self):
        result = compute_algorithm_advantages(
            rewards_G=[0.0, 1.0, 3.0],
            logprobs_G=[[-0.1], [-0.2], [-0.3]],
            planning_masks_G=[[0], [0], [0]],
            algorithm_mode="discover_entropic",
            params={"target_kl": 0.2},
        )

        values = [seq[0] for seq in result.token_advs]
        assert values[2] > values[1] > values[0]
        assert sum(values) == pytest.approx(0.0, abs=1e-6)
        assert result.extra_metrics["discover_kl"] <= 0.201

    def test_uniform_rewards_produce_zero_advantages(self):
        result = compute_algorithm_advantages(
            rewards_G=[2.0, 2.0, 2.0],
            logprobs_G=[[-0.1, -0.2], [-0.3], [-0.4, -0.5, -0.6]],
            planning_masks_G=[[0, 0], [0], [0, 0, 0]],
            algorithm_mode="discover_entropic",
            params={"target_kl": 0.2},
        )

        assert result.token_advs == [
            [0.0, 0.0],
            [0.0],
            [0.0, 0.0, 0.0],
        ]
        assert result.extra_metrics["discover_beta"] == 0.0
        assert result.extra_metrics["discover_kl"] == 0.0


class TestDiscoverArchive:
    def test_select_prefers_high_reward_when_exploration_zero(self):
        archive = DiscoverArchive(empty_reward=0.0)
        low = archive.add_attempt(parent_id=0, text="low", reward=0.5, step=0)
        high = archive.add_attempt(parent_id=0, text="high", reward=2.0, step=0)

        selected = archive.select(batch_size=1, exploration=0.0)

        assert selected[0].entry_id == high.entry_id
        context = archive.context_entries(high.entry_id, limit=4)
        assert all(entry.entry_id != high.entry_id for entry in context)
        assert context[0].entry_id == low.entry_id

    def test_prune_drops_low_reward_leaf(self):
        archive = DiscoverArchive(empty_reward=0.0)
        archive.add_attempt(parent_id=0, text="one", reward=1.0, step=0)
        worst = archive.add_attempt(parent_id=0, text="worst", reward=-1.0, step=1)
        archive.add_attempt(parent_id=0, text="two", reward=2.0, step=2)

        archive.prune(max_entries=3)

        assert worst.entry_id not in archive.entries


class TestBuildDiscoveryPrompt:
    def test_string_prompt_appends_memory(self):
        archive = DiscoverArchive(empty_reward=0.0)
        ctx = archive.add_attempt(parent_id=0, text="beta", reward=1.0, step=0)

        prompt = build_discovery_prompt(
            "Solve the task.",
            start_text="alpha",
            context_entries=[ctx],
            candidate_char_limit=100,
            context_char_limit=100,
        )

        assert isinstance(prompt, str)
        assert "Current candidate to improve" in prompt
        assert "Other promising attempts" in prompt

    def test_chat_prompt_appends_user_message(self):
        archive = DiscoverArchive(empty_reward=0.0)
        ctx = archive.add_attempt(parent_id=0, text="beta", reward=1.0, step=0)

        prompt = build_discovery_prompt(
            [{"role": "user", "content": "Solve the task."}],
            start_text="alpha",
            context_entries=[ctx],
        )

        assert isinstance(prompt, list)
        assert prompt[-1]["role"] == "user"
        assert "Discovery memory" in prompt[-1]["content"]


class TestTTTDiscoverRunner:
    def test_smoke_run_writes_metrics_and_summary(self, tmp_path, monkeypatch):
        class FakeHelper:
            def __init__(self) -> None:
                self.checkpoints: list[str] = []
                self.train_calls: list[tuple] = []

            def checkpoint(self, name: str) -> None:
                self.checkpoints.append(name)

            def sample(
                self,
                prompt_ids_list: list[list[int]],
                num_samples: int,
                max_tokens: int,
                temperature: float,
                top_p: float,
            ) -> list[list[tuple[list[int], list[float]]]]:
                return [[([111], [-0.1]), ([112, 113], [-0.2, -0.3])]]

            def train_step(
                self,
                all_tokens: list[list[int]],
                all_logprobs: list[list[float]],
                all_advantages: list[list[float]],
                lr: float,
                weight_decay: float,
            ) -> float:
                self.train_calls.append(
                    (all_tokens, all_logprobs, all_advantages, lr, weight_decay)
                )
                return 0.123

            def save_adapter(self, path: str, name: str) -> str:
                out_dir = Path(path)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{name}.txt").write_text("ok")
                return str(out_dir)

        class FakeTokenizer:
            vocab_size = 128
            added_tokens_encoder: dict[str, int] = {}

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [ord(ch) for ch in text]

            def batch_decode(
                self,
                token_ids: list[list[int]],
                *,
                skip_special_tokens: bool = True,
            ) -> list[str]:
                return ["".join(chr(tok) for tok in seq) for seq in token_ids]

            def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
                return [chr(tok) for tok in ids]

        class FakeReward:
            def score(self, text: str, answer: str) -> float:
                return float(len(text))

        helper = FakeHelper()
        flow = SimpleNamespace(
            backend=helper,
            backend_capabilities=BackendCapabilities(
                reports_sync_loss=True,
                preserves_token_advantages=True,
                supports_checkpoint_resume=True,
                resume_runtime_dependent=False,
            ),
            backend_capability_source="builtin",
            planning_detector=None,
            backpressure=SimpleNamespace(
                observe=lambda obs: None,
                recommend=lambda: SimpleNamespace(
                    action="hold",
                    recommended_batch_size=0,
                ),
            ),
            sepa_controller=SimpleNamespace(
                resolve_lambda=lambda step: 0.0,
                observe_correct_rate=lambda value: None,
                enabled=lambda: False,
                sepa_schedule="linear",
                update_auto_state=lambda values: None,
                state_dict=lambda: {},
            ),
            needs_planning=False,
            uses_sepa_controller=False,
            condition_label="discover_entropic",
        )

        monkeypatch.setattr("retrain.ttt_discover.build_flow", lambda config, gpu=True: flow)
        monkeypatch.setattr(
            "retrain.ttt_discover._load_discovery_source",
            lambda config: (
                Example(prompt="solve", reference="unused"),
                None,
                FakeReward(),
            ),
        )
        monkeypatch.setattr(
            "retrain.ttt_discover.AutoTokenizer.from_pretrained",
            lambda model, trust_remote_code=True: FakeTokenizer(),
        )

        config = _bare_config(
            trainer="ttt_discover",
            algorithm_mode="discover_entropic",
            max_steps=1,
            batch_size=1,
            group_size=2,
            model="fake-model",
            backend="local",
            adapter_path=str(tmp_path / "adapter"),
            log_dir=str(tmp_path / "logs"),
        )

        runner = TTTDiscoverRunner()
        result = runner.run(config)

        assert isinstance(result, TrainingRunResult)
        assert result.ok
        assert result.policy_ref == str(tmp_path / "adapter")
        assert helper.checkpoints == ["step_0"]
        assert len(helper.train_calls) == 1

        metrics_path = tmp_path / "logs" / "metrics.jsonl"
        summary_path = tmp_path / "logs" / "ttt_discover.json"
        assert metrics_path.exists()
        assert summary_path.exists()

        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        assert metrics[0]["archive_size"] >= 3
        summary = json.loads(summary_path.read_text())
        assert summary["best_reward"] == pytest.approx(2.0)
