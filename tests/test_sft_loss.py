"""Tests for SFT loss policy resolution."""

from __future__ import annotations

from retrain.config import TrainConfig
from retrain.training.sft import effective_sft_loss_fn


def test_effective_sft_loss_preserves_legacy_warmup_default():
    assert effective_sft_loss_fn(TrainConfig(trainer="retrain")) == "importance_sampling"


def test_effective_sft_loss_uses_cross_entropy_for_standalone_sft_auto():
    config = TrainConfig(trainer="sft", sft_data_path="/tmp/data.jsonl")

    assert effective_sft_loss_fn(config) == "cross_entropy"


def test_effective_sft_loss_preserves_explicit_choice():
    config = TrainConfig(
        trainer="sft",
        sft_data_path="/tmp/data.jsonl",
        sft_loss_fn="importance_sampling",
    )

    assert effective_sft_loss_fn(config) == "importance_sampling"
