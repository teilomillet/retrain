"""Fail-closed tests for local backend OScaR sampling options."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from retrain.backends.local.train import LocalTrainHelper


def test_local_helper_rejects_oscar_with_non_pytorch_engine() -> None:
    with pytest.raises(ValueError, match="only supported with inference.engine='pytorch'"):
        LocalTrainHelper(
            "Qwen/Qwen3-4B-Instruct-2507",
            "/tmp/retrain-oscar-test",
            ["cpu"],
            4,
            "vllm",
            "",
            sample_kv_quantization="oscar",
        )


def test_local_helper_rejects_oscar_without_cache() -> None:
    with pytest.raises(ValueError, match="requires sample_use_cache=true"):
        LocalTrainHelper(
            "Qwen/Qwen3-4B-Instruct-2507",
            "/tmp/retrain-oscar-test",
            ["cpu"],
            4,
            "pytorch",
            "",
            sample_use_cache=False,
            sample_kv_quantization="oscar",
        )


def test_local_helper_rejects_oscar_without_split_mode() -> None:
    with pytest.raises(ValueError, match="sampling-only and requires local split mode"):
        LocalTrainHelper(
            "Qwen/Qwen3-4B-Instruct-2507",
            "/tmp/retrain-oscar-test",
            "cpu",
            4,
            "pytorch",
            "",
            sample_kv_quantization="oscar",
        )
