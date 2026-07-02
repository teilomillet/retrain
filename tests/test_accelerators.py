import sys
import types

import torch
import torch.nn.functional as F

import retrain.accelerators as accelerators
from retrain.accelerators import install_cudnn_causal_conv1d_shim


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
