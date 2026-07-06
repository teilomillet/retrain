"""Unit tests for optional OScaR Qwen3 integration helpers."""

from __future__ import annotations

import builtins
import importlib
from types import ModuleType, SimpleNamespace
import sys

import pytest

from retrain.models.oscar_qwen3 import (
    OscarQwen3Options,
    _patch_qwen3_module,
    apply_oscar_config,
    normalize_sample_kv_quantization,
    oscar_options_from_mapping,
    prepare_oscar_runtime,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("off", "off"),
        ("oscar", "oscar"),
        ("", "off"),
        (None, "off"),
        ("false", "off"),
    ],
)
def test_normalize_sample_kv_quantization(raw, expected) -> None:
    assert normalize_sample_kv_quantization(raw) == expected


def test_normalize_sample_kv_quantization_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="sample_kv_quantization"):
        normalize_sample_kv_quantization("vericache")


def test_oscar_options_validate_ranges() -> None:
    assert oscar_options_from_mapping({"sample_oscar_bits": "4"}).bits == 4

    with pytest.raises(ValueError, match="sample_oscar_bits"):
        oscar_options_from_mapping({"sample_oscar_bits": 3})

    with pytest.raises(ValueError, match="sample_oscar_group_size"):
        oscar_options_from_mapping({"sample_oscar_group_size": -1})

    with pytest.raises(ValueError, match="sample_oscar_residual_block_size"):
        oscar_options_from_mapping({"sample_oscar_residual_block_size": 0})


def test_apply_oscar_config_sets_upstream_qwen3_fields() -> None:
    config = SimpleNamespace()
    apply_oscar_config(config, OscarQwen3Options(bits=2, group_size=0))

    assert config._attn_implementation == "sdpa"
    assert config.attn_backend == "oscar"
    assert config.num_bits == 2
    assert config.quant_mode == "k-channel"
    assert config.group_size == 32
    assert config.kv_rotation == "hadamard"
    assert config.kv_norm == "1"
    assert config.residual_block_size == 128


def test_apply_oscar_config_uses_4bit_default_group_size() -> None:
    config = SimpleNamespace()
    apply_oscar_config(config, OscarQwen3Options(bits=4, group_size=0))
    assert config.group_size == 128


def test_prepare_oscar_runtime_reports_missing_package(monkeypatch) -> None:
    real_import = builtins.__import__
    fake_flash_attn = ModuleType("flash_attn")
    fake_flash_attn.flash_attn_with_kvcache = object

    def fake_import(name, *args, **kwargs):
        if name == "oscar" or name.startswith("oscar."):
            raise ModuleNotFoundError("No module named 'oscar'", name="oscar")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(sys.modules, "flash_attn", fake_flash_attn)

    with pytest.raises(RuntimeError, match="requires the upstream OScaR package"):
        prepare_oscar_runtime(OscarQwen3Options(repo="/missing/oscar"))


def test_prepare_oscar_runtime_reports_missing_flash_attn(monkeypatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "flash_attn":
            raise ModuleNotFoundError("No module named 'flash_attn'", name="flash_attn")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)

    with pytest.raises(RuntimeError, match="requires a working flash-attn"):
        prepare_oscar_runtime(OscarQwen3Options())


def test_patch_qwen3_module_adds_default_rope_initializer() -> None:
    fake_model_cls = type(
        "Qwen3ForCausalLM",
        (),
        {"_tied_weights_keys": ["lm_head.weight"]},
    )
    qwen3 = SimpleNamespace(
        ROPE_INIT_FUNCTIONS={},
        Qwen3ForCausalLM=fake_model_cls,
        SlidingWindowCache=object,
    )

    _patch_qwen3_module(qwen3)

    assert "default" in qwen3.ROPE_INIT_FUNCTIONS
    config = SimpleNamespace(
        rope_scaling={"rope_theta": 5_000_000, "rope_type": "default"},
        hidden_size=256,
        num_attention_heads=8,
    )
    inv_freq, attention_factor = qwen3.ROPE_INIT_FUNCTIONS["default"](config)
    assert inv_freq.shape[0] == 16
    assert attention_factor == 1.0
    assert fake_model_cls._tied_weights_keys == {
        "lm_head.weight": "model.embed_tokens.weight",
    }
