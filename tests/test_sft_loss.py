"""Tests for SFT loss policy resolution."""

from __future__ import annotations

from retrain.config import TrainConfig
import pytest

from retrain.training.sft import (
    SftTokenizedBatch,
    build_sft_batch_metrics,
    effective_sft_loss_fn,
)


def test_sft_batch_metrics_describe_logical_padding_and_supervision():
    metrics = build_sft_batch_metrics(
        SftTokenizedBatch(
            tokens=[[1, 2, 3], [4, 5]],
            advantages=[[0.0, 1.0, 1.0], [0.0, 1.0]],
            total_tokens=5,
            supervised_tokens=3,
        )
    )

    assert metrics["sft_sequence_length_min"] == 2
    assert metrics["sft_sequence_length_mean"] == pytest.approx(2.5)
    assert metrics["sft_sequence_length_p50"] == 2
    assert metrics["sft_sequence_length_p95"] == 3
    assert metrics["sft_sequence_length_max"] == 3
    assert metrics["sft_logical_padded_tokens"] == 6
    assert metrics["sft_logical_padding_tokens"] == 1
    assert metrics["sft_logical_padding_fraction"] == pytest.approx(1 / 6)
    assert metrics["sft_supervised_token_fraction"] == pytest.approx(0.6)


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
