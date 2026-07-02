import sys
import types

import pytest
import torch
import torch.nn.functional as F

import retrain.accelerators as accelerators
from retrain.accelerators import (
    install_cudnn_causal_conv1d_shim,
    patch_qwen35_gated_delta_kernel,
)


def torch_chunk_gated_delta_rule(*args, **kwargs):
    _ = args, kwargs
    return "torch-rule"


class Qwen3_5GatedDeltaNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def _chunk_gated_delta_rule(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": dict(kwargs)})
        return "default-rule"

    chunk_gated_delta_rule = _chunk_gated_delta_rule


class _TinyQwen35(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = Qwen3_5GatedDeltaNet()


def _clear_causal_conv1d(monkeypatch):
    monkeypatch.delitem(sys.modules, "causal_conv1d", raising=False)


def test_cudnn_causal_conv1d_shim_installs_and_matches_reference(monkeypatch):
    _clear_causal_conv1d(monkeypatch)
    original_module_available = accelerators.module_available
    monkeypatch.setattr(
        accelerators,
        "module_available",
        lambda name: False
        if name == "causal_conv1d"
        else original_module_available(name),
    )

    class _Ops:
        @staticmethod
        def causal_conv1d(x, weight, bias=None, activation="identity"):
            out = F.conv1d(
                x,
                weight.unsqueeze(1),
                bias,
                padding=weight.shape[1] - 1,
                groups=weight.shape[0],
            )[..., : x.shape[-1]]
            if activation == "silu":
                out = F.silu(out)
            return out

    monkeypatch.setitem(
        sys.modules,
        "cudnn",
        types.SimpleNamespace(__version__="test-cudnn", ops=_Ops),
    )

    report = install_cudnn_causal_conv1d_shim(enabled=True)

    assert report["cudnn_causal_conv1d_shim_installed"] == 1
    assert report["cudnn_causal_conv1d_frontend_version"] == "test-cudnn"
    shim = sys.modules["causal_conv1d"]
    assert shim.__version__ == "cudnn-shim"

    x = torch.randn(2, 3, 5, requires_grad=True)
    weight = torch.randn(3, 4, requires_grad=True)
    bias = torch.randn(3, requires_grad=True)

    out = shim.causal_conv1d_fn(x, weight, bias=bias, activation="silu")
    expected = F.silu(
        F.conv1d(
            x,
            weight.unsqueeze(1),
            bias,
            padding=3,
            groups=3,
        )[..., :5]
    )

    torch.testing.assert_close(out, expected)
    out.sum().backward()
    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None


def test_cudnn_causal_conv1d_shim_does_not_override_real_package(monkeypatch):
    existing = types.ModuleType("causal_conv1d")
    existing.__version__ = "real"
    monkeypatch.setitem(sys.modules, "causal_conv1d", existing)

    report = install_cudnn_causal_conv1d_shim(enabled=True)

    assert report["cudnn_causal_conv1d_shim_installed"] == 0
    assert report["cudnn_causal_conv1d_shim_error"] == "causal_conv1d_already_available"
    assert sys.modules["causal_conv1d"] is existing


def test_qwen35_gated_delta_auto_keeps_default_rule():
    model = _TinyQwen35()

    report = patch_qwen35_gated_delta_kernel(model, mode="auto", device="cuda:0")

    assert report["qwen35_gated_delta_kernel_active"] == "default"
    assert report["qwen35_gated_delta_kernel_patched_modules"] == 0
    assert model.linear_attn.chunk_gated_delta_rule(marker=True) == "default-rule"


def test_qwen35_gated_delta_torch_patch_uses_modeling_fallback():
    model = _TinyQwen35()

    report = patch_qwen35_gated_delta_kernel(model, mode="torch", device="cuda:0")

    assert report["qwen35_gated_delta_kernel_active"] == "torch"
    assert report["qwen35_gated_delta_kernel_patched_modules"] == 1
    assert report["qwen35_gated_delta_torch_fallback"] == 1
    assert model.linear_attn.chunk_gated_delta_rule(marker=True) == "torch-rule"


def test_qwen35_gated_delta_flash_qla_patch_normalizes_qk(monkeypatch):
    model = _TinyQwen35()
    captured: dict[str, object] = {}

    def fake_l2norm(tensor):
        return tensor + 10

    def fake_chunk_gated_delta_rule(**kwargs):
        captured.update(kwargs)
        return "flash-rule", "state"

    monkeypatch.setitem(
        sys.modules,
        "flash_qla",
        types.SimpleNamespace(chunk_gated_delta_rule=fake_chunk_gated_delta_rule),
    )
    monkeypatch.setitem(
        sys.modules,
        "flash_qla.utils",
        types.SimpleNamespace(l2norm=fake_l2norm),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))

    report = patch_qwen35_gated_delta_kernel(
        model,
        mode="flash_qla",
        device="cuda:0",
    )

    assert report["qwen35_gated_delta_kernel_active"] == "flash_qla"
    assert report["qwen35_gated_delta_kernel_patched_modules"] == 1
    q = torch.ones(1, 2, 1, 4)
    k = torch.full_like(q, 2)
    v = torch.full_like(q, 3)
    result = model.linear_attn.chunk_gated_delta_rule(
        q,
        k,
        v,
        g=torch.zeros(1, 2, 1),
        beta=torch.ones(1, 2, 1),
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    assert result == ("flash-rule", "state")
    torch.testing.assert_close(captured["q"], q + 10)
    torch.testing.assert_close(captured["k"], k + 10)
    torch.testing.assert_close(captured["v"], v)
    assert captured["output_final_state"] is False


def test_qwen35_gated_delta_flash_qla_rejects_sm89(monkeypatch):
    model = _TinyQwen35()
    monkeypatch.setitem(
        sys.modules,
        "flash_qla",
        types.SimpleNamespace(chunk_gated_delta_rule=lambda **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "flash_qla.utils",
        types.SimpleNamespace(l2norm=lambda tensor: tensor),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))

    with pytest.raises(RuntimeError, match="SM90 or SM100"):
        patch_qwen35_gated_delta_kernel(
            model,
            mode="flash_qla",
            device="cuda:0",
        )
