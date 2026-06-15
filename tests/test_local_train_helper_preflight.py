"""Tests for local backend model-load preflight checks."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from retrain import local_train_helper as helper


@pytest.mark.parametrize(
    ("raw", "parsed"),
    [
        ("gpu", "cuda:0"),
        ("gpu:0", "cuda:0"),
        ("cuda", "cuda:0"),
        ("cuda:2", "cuda:2"),
        ("cpu", "cpu"),
        ("mps", "mps"),
        ("mps:0", "mps"),
    ],
)
def test_parse_device_accepts_explicit_local_devices(raw, parsed):
    assert helper._parse_device(raw) == parsed


@pytest.mark.parametrize("raw", ["metal:0", "gpu:", "gpu:x", "cuda:", "cuda:x", "mps:1"])
def test_parse_device_rejects_unknown_device(raw):
    with pytest.raises(ValueError, match="Unsupported local backend device"):
        helper._parse_device(raw)


def test_resolve_mps_requires_available_runtime(monkeypatch):
    monkeypatch.setattr(helper, "_mps_is_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS was requested"):
        helper._resolve_available_device("mps")


def test_resolve_mps_keeps_available_runtime(monkeypatch):
    monkeypatch.setattr(helper, "_mps_is_available", lambda: True)

    assert helper._resolve_available_device("mps") == "mps"


def test_mps_dtype_amp_and_scaler_policy():
    assert helper._model_dtype_for_device("mps") is torch.float16
    assert helper._use_amp_for_device("mps") is True
    assert helper._use_grad_scaler_for_device("mps") is False

    assert helper._model_dtype_for_device("cuda:0") is torch.bfloat16
    assert helper._use_amp_for_device("cuda:0") is True
    assert helper._use_grad_scaler_for_device("cuda:0") is True

    assert helper._model_dtype_for_device("cpu") is torch.float32
    assert helper._use_amp_for_device("cpu") is False
    assert helper._use_grad_scaler_for_device("cpu") is False


def test_empty_accelerator_cache_uses_mps_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(helper, "_mps_is_available", lambda: True)
    monkeypatch.setattr(
        helper.torch,
        "mps",
        SimpleNamespace(empty_cache=lambda: calls.append("mps")),
        raising=False,
    )

    helper._empty_accelerator_cache("mps")

    assert calls == ["mps"]


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_local_train_helper_tiny_lora_step_and_sample_on_device(device, tmp_path):
    if device == "mps" and not helper._mps_is_available():
        pytest.skip("PyTorch MPS is not available on this host")

    model_dir = tmp_path / f"tiny-llama-{device}"
    adapter_dir = tmp_path / f"adapter-{device}"
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    LlamaForCausalLM(config).save_pretrained(model_dir)

    train = helper.LocalTrainHelper(
        str(model_dir),
        str(adapter_dir),
        device,
        lora_rank=2,
        engine_type="pytorch",
        sample_use_cache=False,
        train_microbatch_size=1,
    )

    before = {
        name: param.detach().float().cpu().clone()
        for name, param in train.train_model.named_parameters()
        if "lora_" in name
    }
    loss = train.train_step(
        [[1, 3, 4, 5], [1, 6, 7, 2]],
        [[0.0, 0.0, -0.1, -0.2], [0.0, 0.0, -0.3, -0.4]],
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, -1.0, -1.0]],
        lr=1e-4,
        weight_decay=0.0,
    )
    samples = train.sample(
        [[1, 3, 4]],
        num_samples=1,
        max_tokens=2,
        temperature=0.8,
        top_p=0.95,
    )
    after = {
        name: param.detach().float().cpu().clone()
        for name, param in train.train_model.named_parameters()
        if "lora_" in name
    }
    lora_changed = any(
        (after[name] - before[name]).abs().max().item() > 0
        for name in before
    )

    assert math.isfinite(loss)
    assert lora_changed
    assert len(samples) == 1
    assert len(samples[0]) == 1
    assert len(samples[0][0][0]) == 2
    assert len(samples[0][0][1]) == 2
    assert train.train_device == device
    if device == "mps":
        assert train.autocast_dtype is torch.float16
        assert train.use_amp is True
        assert train.scaler.is_enabled() is False
    else:
        assert train.autocast_dtype is torch.float32
        assert train.use_amp is False
        assert train.scaler.is_enabled() is False


def test_nemotron_h_requires_trust_remote_code(monkeypatch):
    monkeypatch.setattr(helper, "_preflight_model_type", lambda *_args: "nemotron_h")

    with pytest.raises(RuntimeError, match="trust_remote_code = true"):
        helper._preflight_local_model_prerequisites(
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            trust_remote_code=False,
            require_causal_conv1d=False,
            devices=("cpu",),
        )


def test_nemotron_h_requires_mamba_ssm(monkeypatch):
    def fake_import_error(module_name):
        if module_name == "mamba_ssm":
            return ImportError("missing")
        return None

    monkeypatch.setattr(helper, "_preflight_model_type", lambda *_args: "nemotron_h")
    monkeypatch.setattr(helper, "_import_error", fake_import_error)

    with pytest.raises(RuntimeError, match="requires mamba-ssm"):
        helper._preflight_local_model_prerequisites(
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            trust_remote_code=True,
            require_causal_conv1d=False,
            devices=("cpu",),
        )


def test_nemotron_h_warns_when_cuda_fast_path_missing(monkeypatch):
    def fake_import_error(module_name):
        if module_name == "causal_conv1d":
            return ImportError("missing")
        return None

    monkeypatch.setattr(helper, "_preflight_model_type", lambda *_args: "nemotron_h")
    monkeypatch.setattr(helper, "_import_error", fake_import_error)
    monkeypatch.setattr(helper.torch.cuda, "is_available", lambda: True)

    with pytest.warns(RuntimeWarning, match="fast Mamba path"):
        helper._preflight_local_model_prerequisites(
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            trust_remote_code=True,
            require_causal_conv1d=False,
            devices=("cuda:0",),
        )


def test_nemotron_h_can_require_cuda_fast_path(monkeypatch):
    def fake_import_error(module_name):
        if module_name == "causal_conv1d":
            return ImportError("missing")
        return None

    monkeypatch.setattr(helper, "_preflight_model_type", lambda *_args: "nemotron_h")
    monkeypatch.setattr(helper, "_import_error", fake_import_error)
    monkeypatch.setattr(helper.torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="fast Mamba path"):
        helper._preflight_local_model_prerequisites(
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            trust_remote_code=True,
            require_causal_conv1d=True,
            devices=("cuda:0",),
        )


def test_train_microbatch_accumulates_one_optimizer_step(monkeypatch):
    class FakeTrainModel:
        def __init__(self):
            self.train_calls = 0

        def train(self):
            self.train_calls += 1

    class FakeOptimizer:
        def __init__(self):
            self.zero_grad_calls = 0
            self.step_calls = 0

        def zero_grad(self):
            self.zero_grad_calls += 1

    class FakeScaler:
        def __init__(self):
            self.scaled_losses = []
            self.update_calls = 0

        def scale(self, loss):
            self.scaled_losses.append(float(loss.detach()))
            return loss

        def step(self, optimizer):
            optimizer.step_calls += 1

        def update(self):
            self.update_calls += 1

    train = object.__new__(helper.LocalTrainHelper)
    train.train_microbatch_size = 1
    train.train_model = FakeTrainModel()
    train.optimizer = FakeOptimizer()
    train.scaler = FakeScaler()
    train._clip_fraction = 0.0
    train.split_mode = False
    train._external_engine = False
    train._snapshot_called = 0

    def fake_loss_for_batch(_input_ids, _old_logprobs, _advantages, attention_mask):
        token_count = attention_mask[:, 1:].sum().clamp(min=1).item()
        value = torch.tensor(float(token_count), requires_grad=True)
        return value, 0.25, token_count

    def fake_snapshot():
        train._snapshot_called += 1

    monkeypatch.setattr(train, "_loss_for_batch", fake_loss_for_batch)
    monkeypatch.setattr(train, "_snapshot_lora_weights_after_step", fake_snapshot)

    input_ids = torch.tensor([[1, 2, 0], [1, 2, 3]])
    old_logprobs = torch.zeros_like(input_ids, dtype=torch.float32)
    advantages = torch.ones_like(input_ids, dtype=torch.float32)
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)

    loss = train._do_train_microbatched_impl(
        input_ids,
        old_logprobs,
        advantages,
        attention_mask,
    )

    assert loss == pytest.approx((1.0 * 1.0 + 2.0 * 2.0) / 3.0)
    assert train.scaler.scaled_losses == pytest.approx([1.0 / 3.0, 4.0 / 3.0])
    assert train.optimizer.step_calls == 1
    assert train.optimizer.zero_grad_calls == 2
    assert train.scaler.update_calls == 1
    assert train._snapshot_called == 1
