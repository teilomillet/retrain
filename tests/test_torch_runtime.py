"""Tests for shared PyTorch runtime support."""

from __future__ import annotations

import torch

from retrain.backends.torch import (
    is_cuda_device,
    pad_to_width,
    parse_device_spec,
    timer_start,
    timer_stop,
)


def test_parse_device_spec_preserves_existing_device_mapping() -> None:
    assert parse_device_spec("gpu:2") == "cuda:2"
    assert parse_device_spec(" cpu ") == "cpu"
    assert parse_device_spec("cuda:1") == "cuda:0"


def test_pad_to_width_returns_original_tensor_when_wide_enough() -> None:
    tensor = torch.tensor([[1, 2, 3]])

    assert pad_to_width(tensor, 3, 0) is tensor
    assert pad_to_width(tensor, 2, 0) is tensor


def test_pad_to_width_right_pads_batch_major_tensor() -> None:
    tensor = torch.tensor([[1, 2], [3, 4]])

    padded = pad_to_width(tensor, 4, -1)

    assert padded.tolist() == [[1, 2, -1, -1], [3, 4, -1, -1]]


def test_is_cuda_device_matches_torch_cuda_availability() -> None:
    assert is_cuda_device("cpu") is False
    assert is_cuda_device(torch.device("cuda:0")) is False
    assert is_cuda_device("cuda:0") is torch.cuda.is_available()


def test_cpu_timer_reports_nonnegative_elapsed_seconds() -> None:
    elapsed = timer_stop(timer_start("cpu"))

    assert elapsed >= 0.0
