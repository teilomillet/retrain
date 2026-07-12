"""Ownership and compatibility checks for the split SFT modules."""

from __future__ import annotations

import pytest

import retrain.training.sft as sft
import retrain.training.sft_audit as audit
import retrain.training.sft_artifacts as artifacts
import retrain.training.sft_data as data
import retrain.training.sft_schedule as schedule
import retrain.training.sft_tokenization as tokenization
from retrain.config import TrainConfig
from retrain.training.sft_telemetry import (
    build_sft_step_metrics,
    build_warmup_sft_wandb_metrics,
)


_SFT_EXPORT_OWNERS = {
    "SFT_AUDIT_SCHEMA": audit,
    "SFT_DATA_SNAPSHOT_MAX_BYTES": artifacts,
    "build_sft_artifact_manifest": artifacts,
    "write_sft_artifact_manifest": artifacts,
    "write_sft_run_snapshot_artifacts": artifacts,
    "SftDataProvenance": data,
    "SftDataset": data,
    "SftExample": data,
    "load_sft_dataset": data,
    "load_sft_jsonl": data,
    "sft_tokenizer_load_kwargs": data,
    "verify_sft_data_contract": data,
    "SFT_RESUME_SCHEDULE_ALGORITHM": schedule,
    "SFT_RESUME_SCHEDULE_CONTRACT_VERSION": schedule,
    "build_sft_epoch_order": schedule,
    "build_sft_example_order": schedule,
    "build_sft_resume_schedule_contract": schedule,
    "build_sft_schedule_metrics": schedule,
    "describe_sft_batch_position": schedule,
    "select_sft_batch_indices": schedule,
    "sft_indices_sha256": schedule,
    "verify_sft_resume_schedule_contract": schedule,
    "SftTokenizedBatch": tokenization,
    "SftTokenizedExample": tokenization,
    "build_sft_batch_metrics": tokenization,
    "build_sft_tokenized_batch": tokenization,
    "tokenize_sft_batch": tokenization,
    "tokenize_sft_dataset": tokenization,
    "tokenize_sft_example": tokenization,
}


@pytest.mark.parametrize("name,owner", _SFT_EXPORT_OWNERS.items())
def test_sft_facade_exports_canonical_objects(name: str, owner: object) -> None:
    assert getattr(sft, name) is getattr(owner, name)


def test_shared_sft_metrics_project_the_warmup_schedule() -> None:
    config = TrainConfig(
        sft_batch_order="length_bucket",
        sft_length_bucket_size=8,
        sft_reshuffle_each_epoch=True,
    )
    metrics = build_sft_step_metrics(
        config,
        step=2,
        loss=0.25,
        datums=3,
        total_tokens=21,
        supervised_tokens=9,
        batch_size=3,
        example_count=8,
        elapsed=1.234,
        lr=1e-4,
    )
    metrics.update(
        {
            "sft_signal": 0.5,
            "sft_epoch": 0,
            "sft_epoch_end": 1,
            "sft_epoch_seed": 42,
            "sft_epoch_end_seed": 43,
            "sft_epoch_sample_offset": 6,
            "sft_batch_indices_sha256": "batch",
            "sft_epoch_start_order_sha256": "start",
            "sft_epoch_end_order_sha256": "end",
        }
    )

    projected = build_warmup_sft_wandb_metrics(
        metrics,
        recoverability_metrics={"train/recoverable": 1},
    )

    assert metrics["sft_dataset_coverage"] == 1.0
    assert projected["train/sft/batch_indices_sha256"] == "batch"
    assert projected["train/sft/epoch_end_order_sha256"] == "end"
    assert projected["train/recoverable"] == 1
