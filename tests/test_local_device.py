from __future__ import annotations

import torch

from retrain.backends.local import device as local_device


def test_resolve_single_cuda_device_when_available(monkeypatch) -> None:
    monkeypatch.setattr(local_device.torch.cuda, "is_available", lambda: True)

    plan = local_device.resolve("cuda:0", "pytorch")

    assert plan.infer_device == "cuda:0"
    assert plan.train_device == "cuda:0"
    assert plan.split_mode is False
    assert plan.external_engine is False
    assert plan.server_engine is False
    assert plan.use_amp is True
    assert plan.dtype is torch.bfloat16


def test_resolve_split_cuda_devices_when_available(monkeypatch) -> None:
    monkeypatch.setattr(local_device.torch.cuda, "is_available", lambda: True)

    plan = local_device.resolve("gpu:0,gpu:2", "pytorch")

    assert plan.infer_device == "cuda:0"
    assert plan.train_device == "cuda:2"
    assert plan.split_mode is True
    assert plan.use_amp is True


def test_resolve_cuda_falls_back_to_cpu_when_unavailable(monkeypatch, capsys) -> None:
    monkeypatch.setattr(local_device.torch.cuda, "is_available", lambda: False)

    plan = local_device.resolve("cuda:1", "pytorch")

    assert plan.infer_device == "cpu"
    assert plan.train_device == "cpu"
    assert plan.split_mode is False
    assert plan.use_amp is False
    assert plan.dtype is torch.float32
    assert "CUDA not available, falling back to CPU" in capsys.readouterr().out


def test_resolve_external_engine_uses_last_device_without_split(monkeypatch) -> None:
    monkeypatch.setattr(local_device.torch.cuda, "is_available", lambda: True)

    plan = local_device.resolve("gpu:0,gpu:3", "vllm")

    assert plan.infer_device == "cuda:3"
    assert plan.train_device == "cuda:3"
    assert plan.split_mode is False
    assert plan.external_engine is True
    assert plan.server_engine is True


def test_resolve_max_engine_is_external_but_not_server(monkeypatch) -> None:
    monkeypatch.setattr(local_device.torch.cuda, "is_available", lambda: True)

    plan = local_device.resolve("cpu", "max")

    assert plan.infer_device == "cpu"
    assert plan.train_device == "cpu"
    assert plan.external_engine is True
    assert plan.server_engine is False
