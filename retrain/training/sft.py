"""Stable import surface for supervised fine-tuning helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from retrain.training.sft_audit import SFT_AUDIT_SCHEMA as SFT_AUDIT_SCHEMA
from retrain.training.sft_artifacts import (
    SFT_DATA_SNAPSHOT_MAX_BYTES as SFT_DATA_SNAPSHOT_MAX_BYTES,
    build_sft_artifact_manifest as build_sft_artifact_manifest,
    write_sft_artifact_manifest as write_sft_artifact_manifest,
    write_sft_run_snapshot_artifacts as write_sft_run_snapshot_artifacts,
)
from retrain.training.sft_data import (
    SftDataProvenance as SftDataProvenance,
    SftDataset as SftDataset,
    SftExample as SftExample,
    load_sft_dataset as load_sft_dataset,
    load_sft_jsonl as load_sft_jsonl,
    sft_tokenizer_load_kwargs as sft_tokenizer_load_kwargs,
    verify_sft_data_contract as verify_sft_data_contract,
)
from retrain.training.sft_schedule import (
    SFT_RESUME_SCHEDULE_ALGORITHM as SFT_RESUME_SCHEDULE_ALGORITHM,
    SFT_RESUME_SCHEDULE_CONTRACT_VERSION as SFT_RESUME_SCHEDULE_CONTRACT_VERSION,
    build_sft_epoch_order as build_sft_epoch_order,
    build_sft_example_order as build_sft_example_order,
    build_sft_resume_schedule_contract as build_sft_resume_schedule_contract,
    build_sft_schedule_metrics as build_sft_schedule_metrics,
    describe_sft_batch_position as describe_sft_batch_position,
    select_sft_batch_indices as select_sft_batch_indices,
    sft_indices_sha256 as sft_indices_sha256,
    verify_sft_resume_schedule_contract as verify_sft_resume_schedule_contract,
)
from retrain.training.sft_tokenization import (
    SftTokenizedBatch as SftTokenizedBatch,
    SftTokenizedExample as SftTokenizedExample,
    build_sft_batch_metrics as build_sft_batch_metrics,
    build_sft_tokenized_batch as build_sft_tokenized_batch,
    tokenize_sft_batch as tokenize_sft_batch,
    tokenize_sft_dataset as tokenize_sft_dataset,
    tokenize_sft_example as tokenize_sft_example,
)

if TYPE_CHECKING:
    from retrain.config import TrainConfig


def effective_sft_loss_fn(config: "TrainConfig") -> str:
    """Resolve SFT loss without changing legacy warmup defaults."""

    if config.sft_loss_fn != "auto":
        return config.sft_loss_fn
    if config.trainer == "sft":
        return "cross_entropy"
    return "importance_sampling"
