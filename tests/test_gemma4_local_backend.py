"""Focused tests for Gemma4 local-backend integration helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from peft import get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from retrain import local_train_helper as local_mod
from retrain.gemma4_text import (
    DEFAULT_LORA_TARGET_MODULES,
    eos_token_ids,
    forward_logits,
    is_gemma4_text_model,
    parse_lora_target_module_suffixes,
    resolve_lora_target_modules,
)
from retrain.inference_engine.pytorch_engine import (
    PyTorchEngine,
    _sample_next_token,
    _shannon_entropy_from_probs_logprobs,
)
from retrain.local_train_helper import LocalTrainHelper
from retrain.local_train_helper import _FastLoRALinearFunction
from retrain.local_train_helper import _parse_lora_layers_to_transform


class _ExistingModel:
    def __init__(self) -> None:
        self.to_calls: list[str] = []

    def to(self, device: str):
        self.to_calls.append(device)
        return self


class _FakeLanguageModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        self.calls.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "kwargs": kwargs,
        })
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeGemma4TextModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(model_type="gemma4", eos_token_id=[7, 8])
        self.generation_config = SimpleNamespace(eos_token_id=9)
        self.language_model = _FakeLanguageModel()
        self.model = SimpleNamespace(language_model=self.language_model)
        self.lm_head = lambda hidden: hidden + 1.0

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("Gemma4 multimodal wrapper should not be called")

    def eval(self):
        return self

    def to(self, _device: str):
        return self

    def named_modules(self):
        names = [
            "",
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.self_attn.k_proj",
            "model.language_model.layers.0.mlp.down_proj",
            "model.vision_tower.layers.0.self_attn.q_proj",
        ]
        return [(name, object()) for name in names]


class _WrappedGemma4TextModel:
    def __init__(self, model: _FakeGemma4TextModel) -> None:
        self.base_model = SimpleNamespace(model=model)

    def named_modules(self):
        return [("base_model.model.model.language_model.layers.0.self_attn.q_proj", object())]


class _FailingEngine:
    def generate(self, *_args, **_kwargs):
        raise RuntimeError("sample failed")


class _CheckpointToggleModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=False)
        self.checkpointing_enabled = True
        self.disable_calls = 0
        self.enable_calls = 0
        self.enable_kwargs: list[dict] = []

    def gradient_checkpointing_disable(self) -> None:
        self.checkpointing_enabled = False
        self.disable_calls += 1

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.checkpointing_enabled = True
        self.enable_calls += 1
        self.enable_kwargs.append(dict(kwargs))


class _CheckpointLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gradient_checkpointing = False


class _LayerCheckpointModel(torch.nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            _CheckpointLayer() for _ in range(layer_count)
        )
        self.enable_kwargs: list[dict] = []

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.enable_kwargs.append(dict(kwargs))
        for layer in self.layers:
            layer.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        for layer in self.layers:
            layer.gradient_checkpointing = False


class _ConstructorFakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(use_cache=False)
        self.gradient_checkpointing_enable_calls = 0
        self.gradient_checkpointing_enable_kwargs: list[dict] = []

    def print_trainable_parameters(self) -> None:
        pass

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.gradient_checkpointing_enable_calls += 1
        self.gradient_checkpointing_enable_kwargs.append(dict(kwargs))


class _AssertingCacheEngine:
    def __init__(self, model: _CheckpointToggleModel) -> None:
        self.model = model
        self.saw_cache_enabled = False

    def generate(self, *_args, **_kwargs):
        self.saw_cache_enabled = (
            self.model.checkpointing_enabled is False
            and self.model.config.use_cache is True
        )
        return [[SimpleNamespace(token_ids=[1, 2], logprobs=[0.0, 0.0])]]


class _FakeGenericCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(eos_token_id=7)
        self.generation_config = SimpleNamespace(eos_token_id=7)
        self.calls: list[dict] = []

    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        self.calls.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
                "past_key_values": past_key_values,
            }
        )
        logits = torch.full((*input_ids.shape, 8), -100.0, device=input_ids.device)
        logits[..., 3] = 100.0
        cache = ("cache",) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


class _PrefixCacheCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(eos_token_id=7)
        self.generation_config = SimpleNamespace(eos_token_id=7)
        self.calls: list[dict] = []

    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        _ = attention_mask
        prefix_len = (
            int(past_key_values[0])
            if isinstance(past_key_values, tuple) and past_key_values
            else 0
        )
        self.calls.append(
            {
                "input_len": input_ids.shape[1],
                "prefix_len": prefix_len,
                "use_cache": use_cache,
            }
        )
        logits = torch.full((*input_ids.shape, 8), -100.0, device=input_ids.device)
        logits[..., 3] = 100.0
        next_len = prefix_len + int(input_ids.shape[1])
        return SimpleNamespace(logits=logits, past_key_values=(next_len,))


def test_gemma4_lora_targets_are_language_tower_exact_names():
    model = _FakeGemma4TextModel()

    targets = resolve_lora_target_modules(model, DEFAULT_LORA_TARGET_MODULES)

    assert targets == [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.mlp.down_proj",
    ]


def test_gemma4_lora_targets_scan_unwrapped_peft_model():
    model = _FakeGemma4TextModel()
    wrapped = _WrappedGemma4TextModel(model)

    targets = resolve_lora_target_modules(wrapped, DEFAULT_LORA_TARGET_MODULES)

    assert targets == [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.mlp.down_proj",
    ]


def test_lora_layers_to_transform_parser_supports_gate_specs():
    assert _parse_lora_layers_to_transform("", 4) is None
    assert _parse_lora_layers_to_transform("all", 4) is None
    assert _parse_lora_layers_to_transform("last:2", 4) == [2, 3]
    assert _parse_lora_layers_to_transform("first:2", 4) == [0, 1]
    assert _parse_lora_layers_to_transform("0,2-3", 4) == [0, 2, 3]

    with pytest.raises(ValueError, match="duplicate"):
        _parse_lora_layers_to_transform("1,1", 4)
    with pytest.raises(ValueError, match="exceeds 3"):
        _parse_lora_layers_to_transform("4", 4)


def test_local_peft_config_can_select_qwen_layer_subset():
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    helper = object.__new__(LocalTrainHelper)
    helper.lora_layers_to_transform_spec = "last:2"
    helper.lora_layers_pattern = "layers"

    peft_config = helper._build_peft_config(
        model,
        lora_rank=2,
        lora_alpha=0,
        lora_dropout=0.0,
    )
    wrapped = get_peft_model(model, peft_config)
    lora_names = [name for name, _ in wrapped.named_parameters() if "lora_" in name]

    assert peft_config.layers_to_transform == [2, 3]
    assert peft_config.layers_pattern == "layers"
    assert lora_names
    assert all(".layers.0." not in name and ".layers.1." not in name for name in lora_names)
    assert any(".layers.2." in name for name in lora_names)
    assert any(".layers.3." in name for name in lora_names)


def test_local_peft_config_can_select_target_modules():
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    helper = object.__new__(LocalTrainHelper)
    helper.lora_target_module_suffixes = parse_lora_target_module_suffixes("o_proj")
    helper.lora_layers_to_transform_spec = ""
    helper.lora_layers_pattern = "layers"

    peft_config = helper._build_peft_config(
        model,
        lora_rank=2,
        lora_alpha=0,
        lora_dropout=0.0,
    )
    wrapped = get_peft_model(model, peft_config)
    lora_names = [name for name, _ in wrapped.named_parameters() if "lora_" in name]

    assert peft_config.target_modules == {"o_proj"}
    assert lora_names
    assert all(".o_proj." in name for name in lora_names)
    assert not any(".q_proj." in name for name in lora_names)
    assert not any(".up_proj." in name for name in lora_names)


def test_lora_detached_input_hook_preserves_weight_grad_but_stops_input_grad():
    root = torch.nn.Module()
    branch = torch.nn.Module()
    branch.lora_A = torch.nn.ModuleDict(
        {"default": torch.nn.Linear(4, 2, bias=False)}
    )
    root.branch = branch

    helper = object.__new__(LocalTrainHelper)
    helper.train_model = root
    helper.lora_detach_input = True

    helper._configure_lora_detached_input()

    x = torch.randn(3, 4, requires_grad=True)
    y = root.branch.lora_A["default"](x).sum()
    y.backward()

    assert helper._lora_detach_input_hook_count == 1
    assert root.branch.lora_A["default"].weight.grad is not None
    assert x.grad is None


def test_fast_lora_linear_matches_dense_lora_forward_and_grad():
    torch.manual_seed(123)
    x = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4)
    bias = torch.randn(5)
    lora_a = torch.randn(2, 4, requires_grad=True)
    lora_b = torch.randn(5, 2, requires_grad=True)
    grad = torch.randn(2, 3, 5)
    scaling = 0.75

    baseline = x @ weight.T + bias + (x @ lora_a.T @ lora_b.T) * scaling
    baseline.backward(grad)
    expected_x_grad = x.grad.clone()
    expected_a_grad = lora_a.grad.clone()
    expected_b_grad = lora_b.grad.clone()

    x.grad = None
    lora_a.grad = None
    lora_b.grad = None

    actual = _FastLoRALinearFunction.apply(
        x,
        weight,
        bias,
        lora_a,
        lora_b,
        scaling,
        False,
        False,
    )
    actual.backward(grad)

    assert torch.allclose(actual, baseline)
    assert torch.allclose(x.grad, expected_x_grad)
    assert torch.allclose(lora_a.grad, expected_a_grad)
    assert torch.allclose(lora_b.grad, expected_b_grad)


def test_fast_lora_linear_detach_input_matches_dense_detached_lora_grad():
    torch.manual_seed(123)
    x = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4)
    bias = torch.randn(5)
    lora_a = torch.randn(2, 4, requires_grad=True)
    lora_b = torch.randn(5, 2, requires_grad=True)
    grad = torch.randn(2, 3, 5)
    scaling = 0.75

    baseline = x @ weight.T + bias + (x.detach() @ lora_a.T @ lora_b.T) * scaling
    baseline.backward(grad)
    expected_x_grad = x.grad.clone()
    expected_a_grad = lora_a.grad.clone()
    expected_b_grad = lora_b.grad.clone()

    x.grad = None
    lora_a.grad = None
    lora_b.grad = None

    actual = _FastLoRALinearFunction.apply(
        x,
        weight,
        bias,
        lora_a,
        lora_b,
        scaling,
        True,
        False,
    )
    actual.backward(grad)

    assert torch.allclose(actual, baseline)
    assert torch.allclose(x.grad, expected_x_grad)
    assert torch.allclose(lora_a.grad, expected_a_grad)
    assert torch.allclose(lora_b.grad, expected_b_grad)


def test_fast_lora_linear_freeze_a_matches_dense_lora_frozen_a_grad():
    torch.manual_seed(123)
    x = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4)
    bias = torch.randn(5)
    lora_a = torch.randn(2, 4, requires_grad=True)
    lora_b = torch.randn(5, 2, requires_grad=True)
    grad = torch.randn(2, 3, 5)
    scaling = 0.75

    baseline = x @ weight.T + bias + (x @ lora_a.T @ lora_b.T) * scaling
    baseline.backward(grad)
    expected_x_grad = x.grad.clone()
    expected_b_grad = lora_b.grad.clone()

    x.grad = None
    lora_a.grad = None
    lora_b.grad = None

    actual = _FastLoRALinearFunction.apply(
        x,
        weight,
        bias,
        lora_a,
        lora_b,
        scaling,
        False,
        True,
    )
    actual.backward(grad)

    assert torch.allclose(actual, baseline)
    assert torch.allclose(x.grad, expected_x_grad)
    assert lora_a.grad is None
    assert torch.allclose(lora_b.grad, expected_b_grad)


def test_fast_lora_linear_patch_wraps_eligible_peft_module():
    class FakeLoraLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(4, 5)
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(4, 2, bias=False)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 5, bias=False)}
            )
            self.lora_dropout = {"default": torch.nn.Identity()}
            self.scaling = {"default": 0.5}
            self.use_dora = {"default": False}
            self.active_adapters = ["default"]
            self.disable_adapters = False
            self.merged = False

        def _check_forward_args(self, *_args, **_kwargs):
            return None

        def forward(self, x):
            return self.base_layer(x) + (
                self.lora_B["default"](self.lora_A["default"](x)) * 0.5
            )

    root = torch.nn.Module()
    root.proj = FakeLoraLinear()
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = root
    helper.lora_fast_linear = True
    helper.lora_detach_input = False
    helper.lora_freeze_a = False

    x = torch.randn(2, 3, 4)
    expected = root.proj(x)

    helper._configure_lora_fast_linear()

    assert helper._lora_fast_linear_patch_count == 1
    assert torch.allclose(root.proj(x), expected)


def test_lora_freeze_a_marks_only_lora_a_not_trainable():
    root = torch.nn.Module()
    root.branch = torch.nn.Module()
    root.branch.lora_A = torch.nn.ModuleDict(
        {"default": torch.nn.Linear(4, 2, bias=False)}
    )
    root.branch.lora_B = torch.nn.ModuleDict(
        {"default": torch.nn.Linear(2, 5, bias=False)}
    )

    helper = object.__new__(LocalTrainHelper)
    helper.train_model = root
    helper.lora_freeze_a = True

    helper._configure_lora_frozen_a()

    assert helper._lora_frozen_a_tensor_count == 1
    assert root.branch.lora_A["default"].weight.requires_grad is False
    assert root.branch.lora_B["default"].weight.requires_grad is True


def test_fast_lora_linear_patch_keeps_detached_lora_input_grad_boundary():
    class FakeLoraLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(4, 5)
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(4, 2, bias=False)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 5, bias=False)}
            )
            self.lora_dropout = {"default": torch.nn.Identity()}
            self.scaling = {"default": 0.5}
            self.use_dora = {"default": False}
            self.active_adapters = ["default"]
            self.disable_adapters = False
            self.merged = False

        def _check_forward_args(self, *_args, **_kwargs):
            return None

        def forward(self, x):
            return self.base_layer(x) + (
                self.lora_B["default"](self.lora_A["default"](x)) * 0.5
            )

    root = torch.nn.Module()
    root.proj = FakeLoraLinear()
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = root
    helper.lora_fast_linear = True
    helper.lora_detach_input = True
    helper.lora_freeze_a = False

    x = torch.randn(2, 3, 4, requires_grad=True)
    expected = root.proj.base_layer(x) + (
        root.proj.lora_B["default"](root.proj.lora_A["default"](x.detach())) * 0.5
    )
    expected.backward(torch.ones_like(expected))
    expected_x_grad = x.grad.clone()

    root.proj.zero_grad(set_to_none=True)
    x.grad = None
    helper._configure_lora_fast_linear()
    actual = root.proj(x)
    actual.backward(torch.ones_like(actual))

    assert helper._lora_fast_linear_patch_count == 1
    assert root.proj._retrain_fast_lora_detach_input is True
    assert torch.allclose(actual, expected.detach())
    assert torch.allclose(x.grad, expected_x_grad)
    assert root.proj.lora_A["default"].weight.grad is not None
    assert root.proj.lora_B["default"].weight.grad is not None


def test_fast_lora_linear_patch_keeps_frozen_a_grad_boundary():
    class FakeLoraLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(4, 5)
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(4, 2, bias=False)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 5, bias=False)}
            )
            self.lora_dropout = {"default": torch.nn.Identity()}
            self.scaling = {"default": 0.5}
            self.use_dora = {"default": False}
            self.active_adapters = ["default"]
            self.disable_adapters = False
            self.merged = False

        def _check_forward_args(self, *_args, **_kwargs):
            return None

        def forward(self, x):
            return self.base_layer(x) + (
                self.lora_B["default"](self.lora_A["default"](x)) * 0.5
            )

    root = torch.nn.Module()
    root.proj = FakeLoraLinear()
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = root
    helper.lora_freeze_a = True
    helper.lora_fast_linear = True
    helper.lora_detach_input = False

    helper._configure_lora_frozen_a()
    x = torch.randn(2, 3, 4, requires_grad=True)
    expected = root.proj(x)

    helper._configure_lora_fast_linear()
    actual = root.proj(x)
    actual.sum().backward()

    assert helper._lora_frozen_a_tensor_count == 1
    assert helper._lora_fast_linear_patch_count == 1
    assert root.proj._retrain_fast_lora_freeze_a is True
    assert torch.allclose(actual, expected)
    assert root.proj.lora_A["default"].weight.grad is None
    assert root.proj.lora_B["default"].weight.grad is not None


def test_gemma4_forward_logits_bypasses_multimodal_wrapper():
    model = _FakeGemma4TextModel()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[True, True, True]])

    logits = forward_logits(model, input_ids, attention_mask)

    assert is_gemma4_text_model(model)
    assert logits.shape == (1, 3, 3)
    assert model.language_model.calls[0]["attention_mask"] is attention_mask
    assert model.language_model.calls[0]["use_cache"] is False
    assert torch.equal(logits[0, :, 0], torch.tensor([2.0, 3.0, 4.0]))


def test_gemma4_eos_ids_prefer_generation_config():
    model = _FakeGemma4TextModel()

    assert eos_token_ids(model) == {9}


def test_pytorch_engine_reuses_existing_model_without_moving_it():
    model = _ExistingModel()

    engine = PyTorchEngine(
        model_name="unused",
        device="cuda:7",
        peft_config=None,
        dtype=None,
        existing_model=model,
    )

    assert engine.model is model
    assert model.to_calls == []


def test_pytorch_engine_rejects_peft_config_with_existing_model():
    with pytest.raises(ValueError, match="peft_config must be None"):
        PyTorchEngine(
            model_name="unused",
            device="cpu",
            peft_config=object(),
            dtype=None,
            existing_model=_ExistingModel(),
        )


def test_gemma4_text_sampler_can_disable_kv_cache():
    torch.manual_seed(123)
    model = _FakeGemma4TextModel()
    engine = PyTorchEngine(
        model_name="unused",
        device="cpu",
        peft_config=None,
        dtype=None,
        existing_model=model,
        sample_use_cache=False,
    )

    results = engine.generate([[1, 2]], 1, 2, temperature=1.0, top_p=1.0)

    assert len(results) == 1
    assert len(results[0]) == 1
    assert len(model.language_model.calls) == 2
    assert [call["input_ids"].shape[1] for call in model.language_model.calls] == [
        2,
        3,
    ]
    assert [call["use_cache"] for call in model.language_model.calls] == [
        False,
        False,
    ]
    assert [call["kwargs"] for call in model.language_model.calls] == [{}, {}]


def test_local_helper_gradient_checkpointing_can_use_non_reentrant_mode():
    model = _CheckpointToggleModel()
    helper = LocalTrainHelper.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.gradient_checkpointing = True
    helper.gradient_checkpointing_use_reentrant = "false"

    helper._configure_gradient_checkpointing()

    assert model.checkpointing_enabled is True
    assert model.enable_calls == 1
    assert model.enable_kwargs == [
        {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    ]


def test_local_helper_gradient_checkpointing_auto_preserves_legacy_call():
    model = _CheckpointToggleModel()
    helper = LocalTrainHelper.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.gradient_checkpointing = True
    helper.gradient_checkpointing_use_reentrant = "auto"

    helper._configure_gradient_checkpointing()

    assert model.enable_kwargs == [{}]


def test_local_helper_gradient_checkpointing_can_skip_last_layers():
    model = _LayerCheckpointModel(layer_count=4)
    helper = LocalTrainHelper.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.gradient_checkpointing = True
    helper.gradient_checkpointing_use_reentrant = "auto"
    helper.gradient_checkpointing_skip_last_n = 2

    helper._configure_gradient_checkpointing()

    assert [layer.gradient_checkpointing for layer in model.layers] == [
        True,
        True,
        False,
        False,
    ]
    assert helper._gradient_checkpointing_layer_metrics == {
        "total": 4,
        "enabled": 2,
        "skipped_last_n": 2,
    }


def test_local_helper_enables_cache_during_shared_model_sampling():
    model = _CheckpointToggleModel()
    engine = _AssertingCacheEngine(model)
    helper = LocalTrainHelper.__new__(LocalTrainHelper)
    helper.gradient_checkpointing = True
    helper.sample_use_cache = True
    helper._external_engine = False
    helper.split_mode = False
    helper.train_model = model
    helper.engine = engine
    helper.cuda_empty_cache = False
    helper.infer_device = "cpu"
    helper.train_device = "cpu"
    helper._last_sample_metrics = {}

    result = helper.sample([[1, 2]], 1, 2, temperature=1.0, top_p=1.0)

    assert result == [[([1, 2], [0.0, 0.0])]]
    assert engine.saw_cache_enabled is True
    assert model.disable_calls == 1
    assert model.enable_calls == 1
    assert model.enable_kwargs == [{}]
    assert model.checkpointing_enabled is True
    assert model.config.use_cache is False
    assert helper._last_sample_metrics["local_sample_gc_disabled_for_cache"] == 1


def test_shared_model_sampling_restores_checkpointing_mode_kwargs():
    model = _CheckpointToggleModel()
    engine = _AssertingCacheEngine(model)
    helper = LocalTrainHelper.__new__(LocalTrainHelper)
    helper.gradient_checkpointing = True
    helper.gradient_checkpointing_use_reentrant = "false"
    helper.sample_use_cache = True
    helper._external_engine = False
    helper.split_mode = False
    helper.train_model = model
    helper.engine = engine
    helper.cuda_empty_cache = False
    helper.infer_device = "cpu"
    helper.train_device = "cpu"
    helper._last_sample_metrics = {}

    helper.sample([[1, 2]], 1, 2, temperature=1.0, top_p=1.0)

    assert model.enable_kwargs == [
        {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    ]


def test_local_helper_routes_trtllm_as_external_server(monkeypatch, tmp_path):
    train_model = _ConstructorFakeModel()
    from_pretrained_calls = []
    create_engine_calls = []
    fake_engine = object()

    def fake_from_pretrained(model_name, torch_dtype=None, **kwargs):  # noqa: ANN001
        from_pretrained_calls.append(
            {
                "model_name": model_name,
                "torch_dtype": torch_dtype,
                "kwargs": kwargs,
            }
        )
        return _ConstructorFakeModel()

    def fake_get_peft_model(_model, _peft_config):  # noqa: ANN001
        return train_model

    def fake_create_engine(**kwargs):  # noqa: ANN001
        create_engine_calls.append(kwargs)
        return fake_engine

    monkeypatch.setattr(
        local_mod.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )
    monkeypatch.setattr(local_mod, "get_peft_model", fake_get_peft_model)
    monkeypatch.setattr(
        local_mod,
        "resolve_lora_target_modules",
        lambda _model, _defaults: ["q_proj"],
    )
    monkeypatch.setattr(local_mod, "create_engine", fake_create_engine)

    helper = LocalTrainHelper(
        model_name="Qwen/Qwen3.5-2B",
        adapter_path=str(tmp_path / "adapter"),
        devices="cpu",
        engine_type="trtllm",
        inference_url="http://localhost:31000",
        lora_rank=8,
        gradient_checkpointing=True,
        liger_kernel=False,
    )

    assert helper._server_engine is True
    assert helper._external_engine is True
    assert helper.engine is fake_engine
    assert len(from_pretrained_calls) == 1
    assert create_engine_calls[0]["engine_type"] == "trtllm"
    assert create_engine_calls[0]["model_name"] == "Qwen/Qwen3.5-2B"
    assert create_engine_calls[0]["inference_url"] == "http://localhost:31000"
    assert "existing_model" not in create_engine_calls[0]
    assert train_model.gradient_checkpointing_enable_calls == 1


def test_pytorch_engine_reports_prefill_decode_metrics_for_generic_model():
    torch.manual_seed(123)
    model = _FakeGenericCausalModel()
    engine = PyTorchEngine(
        model_name="unused",
        device="cpu",
        peft_config=None,
        dtype=None,
        existing_model=model,
        sample_use_cache=True,
        prefix_caching=False,
    )

    results = engine.generate([[1, 2]], 1, 2, temperature=1.0, top_p=1.0)
    counters = engine.performance_counters()

    assert [sample.token_ids for sample in results[0]] == [[3, 3]]
    assert [call["input_ids"].shape[1] for call in model.calls] == [2, 1]
    assert counters["engine_prompt_tokens"] == 2
    assert counters["engine_generated_tokens"] == 2
    assert counters["engine_prompt_prefill_s"] >= 0.0
    assert counters["engine_decode_s"] >= 0.0
    assert counters["engine_generation_tokens_per_s"] > 0.0


def test_pytorch_engine_reuses_exact_prefix_kv_for_generic_model():
    torch.manual_seed(123)
    model = _PrefixCacheCausalModel()
    engine = PyTorchEngine(
        model_name="unused",
        device="cpu",
        peft_config=None,
        dtype=None,
        existing_model=model,
        sample_use_cache=True,
        prefix_caching=True,
    )

    first = engine.generate([[1, 2]], 1, 1, temperature=1.0, top_p=1.0)
    second = engine.generate([[1, 2, 3, 4]], 1, 1, temperature=1.0, top_p=1.0)
    counters = engine.performance_counters()

    assert first[0][0].token_ids == [3]
    assert second[0][0].token_ids == [3]
    assert model.calls[0]["input_len"] == 2
    assert model.calls[0]["prefix_len"] == 0
    assert model.calls[1]["input_len"] == 1
    assert model.calls[1]["prefix_len"] == 2
    assert model.calls[2]["input_len"] == 1
    assert model.calls[2]["prefix_len"] == 3
    assert counters["engine_prefix_cache_hits"] == 1
    assert counters["engine_prefix_cache_misses"] == 1
    assert counters["engine_prefix_cache_reused_tokens"] == 3
    assert counters["engine_prefix_cache_entries"] > 0

    engine.clear_prefix_cache()
    assert engine.performance_counters()["engine_prefix_cache_entries"] == 0


def test_top_p_sampling_entropy_uses_full_distribution():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    probs = F.softmax(logits, dim=-1)
    expected_entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

    token, logprob, entropy = _sample_next_token(
        logits,
        temperature=1.0,
        top_p=0.4,
        compute_entropy=True,
    )

    assert token.tolist() == [[0]]
    assert logprob.tolist() == pytest.approx([0.0])
    assert entropy.tolist() == pytest.approx(expected_entropy.tolist())


def test_sampling_skips_entropy_when_not_requested():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])

    _token, _logprob, entropy = _sample_next_token(
        logits,
        temperature=1.0,
        top_p=1.0,
        compute_entropy=False,
    )

    assert entropy is None


def test_entropy_helper_treats_zero_probability_as_zero_contribution():
    probs = torch.tensor([[1.0, 0.0, 0.0]])
    log_probs = torch.tensor([[0.0, float("-inf"), float("-inf")]])

    entropy = _shannon_entropy_from_probs_logprobs(probs, log_probs)

    assert torch.isfinite(entropy).all()
    assert entropy.tolist() == pytest.approx([0.0])


def test_local_sample_empty_cache_runs_after_engine_error(monkeypatch):
    calls: list[str] = []
    helper = object.__new__(LocalTrainHelper)
    helper.engine = _FailingEngine()
    helper.cuda_empty_cache = True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    with pytest.raises(RuntimeError, match="sample failed"):
        helper.sample([[1, 2]], 1, 1, 1.0, 1.0)

    assert calls == ["empty"]


class _TinyCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.proj = torch.nn.Linear(4, 8)

    def forward(self, input_ids, attention_mask=None):
        _ = attention_mask
        return SimpleNamespace(logits=self.proj(self.embedding(input_ids)))


def _helper_for_model(model: torch.nn.Module, *, train_microbatch_size: int):
    helper = object.__new__(LocalTrainHelper)
    helper.train_model = model
    helper.optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    helper.scaler = torch.amp.GradScaler(enabled=False)
    helper.train_device = "cpu"
    helper.use_amp = False
    helper.clip_eps = 0.0
    helper.clip_eps_high = 0.0
    helper._clip_fraction = 0.0
    helper.split_mode = False
    helper._external_engine = False
    helper._train_future = None
    helper.train_microbatch_size = train_microbatch_size
    helper.train_sft_microbatch_token_budget = 0
    helper.cuda_empty_cache = False
    return helper


def test_local_train_step_microbatch_matches_full_batch_update():
    torch.manual_seed(123)
    base = _TinyCausalModel()
    full_model = _TinyCausalModel()
    micro_model = _TinyCausalModel()
    full_model.load_state_dict(base.state_dict())
    micro_model.load_state_dict(base.state_dict())

    tokens = [
        [1, 2, 3, 4],
        [1, 3, 4],
        [2, 4, 5, 6],
    ]
    logprobs = [[0.0] * len(row) for row in tokens]
    advantages = [
        [0.0, 1.0, -0.5, 0.25],
        [0.0, -1.0, 0.5],
        [0.0, 0.25, 0.75, -0.25],
    ]

    full = _helper_for_model(full_model, train_microbatch_size=0)
    micro = _helper_for_model(micro_model, train_microbatch_size=1)

    full_loss = full.train_step(tokens, logprobs, advantages, lr=0.05, weight_decay=0.0)
    micro_loss = micro.train_step(tokens, logprobs, advantages, lr=0.05, weight_decay=0.0)

    assert micro_loss == pytest.approx(full_loss, rel=1e-6, abs=1e-6)
    assert micro._clip_fraction == pytest.approx(full._clip_fraction)
    metrics = micro.runtime_metrics()
    assert metrics["local_train_microbatches"] == 3
    assert metrics["local_train_forward_s"] >= 0.0
    assert metrics["local_train_backward_s"] >= 0.0
    assert metrics["local_train_optimizer_s"] >= 0.0
    for full_param, micro_param in zip(full_model.parameters(), micro_model.parameters()):
        assert torch.allclose(full_param, micro_param, atol=1e-6)


def test_sft_train_step_microbatch_local_padding_matches_full_batch_update():
    torch.manual_seed(321)
    base = _TinyCausalModel()
    full_model = _TinyCausalModel()
    micro_model = _TinyCausalModel()
    full_model.load_state_dict(base.state_dict())
    micro_model.load_state_dict(base.state_dict())

    tokens = [
        [1, 2, 3, 4, 5, 6],
        [1, 3, 4],
        [2, 4, 5, 6, 7],
    ]
    advantages = [
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
    ]

    full = _helper_for_model(full_model, train_microbatch_size=0)
    micro = _helper_for_model(micro_model, train_microbatch_size=1)
    full.sft_loss_fn = "cross_entropy"
    micro.sft_loss_fn = "cross_entropy"

    full_loss = full.sft_train_step(tokens, advantages, lr=0.05, weight_decay=0.0)
    micro_loss = micro.sft_train_step(tokens, advantages, lr=0.05, weight_decay=0.0)

    assert micro_loss == pytest.approx(full_loss, rel=1e-6, abs=1e-6)
    metrics = micro.runtime_metrics()
    assert metrics["local_train_microbatches"] == 3
    assert metrics["local_train_microbatch_local_padding"] == 1
    assert metrics["local_train_global_padded_tokens"] == 18
    assert metrics["local_train_microbatch_padded_tokens"] == 14
    assert metrics["local_train_padding_tokens_avoided"] == 4
    for full_param, micro_param in zip(full_model.parameters(), micro_model.parameters()):
        assert torch.allclose(full_param, micro_param, atol=1e-6)


def test_sft_train_step_token_budget_matches_full_batch_update():
    torch.manual_seed(432)
    base = _TinyCausalModel()
    full_model = _TinyCausalModel()
    budget_model = _TinyCausalModel()
    full_model.load_state_dict(base.state_dict())
    budget_model.load_state_dict(base.state_dict())

    tokens = [
        [1, 2, 3, 4, 5, 6],
        [1, 3, 4],
        [2, 4, 5, 6, 7],
        [1, 2],
    ]
    advantages = [
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0],
    ]

    full = _helper_for_model(full_model, train_microbatch_size=0)
    budgeted = _helper_for_model(budget_model, train_microbatch_size=0)
    budgeted.train_sft_microbatch_token_budget = 8
    full.sft_loss_fn = "cross_entropy"
    budgeted.sft_loss_fn = "cross_entropy"

    full_loss = full.sft_train_step(tokens, advantages, lr=0.05, weight_decay=0.0)
    budget_loss = budgeted.sft_train_step(tokens, advantages, lr=0.05, weight_decay=0.0)

    assert budget_loss == pytest.approx(full_loss, rel=1e-6, abs=1e-6)
    metrics = budgeted.runtime_metrics()
    assert metrics["local_train_microbatches"] == 4
    assert metrics["local_train_global_padded_tokens"] == 24
    assert metrics["local_train_microbatch_padded_tokens"] == 16
    assert metrics["local_train_sft_microbatch_token_budget"] == 8
    for full_param, budget_param in zip(full_model.parameters(), budget_model.parameters()):
        assert torch.allclose(full_param, budget_param, atol=1e-6)


def test_microbatch_padding_stats_preserve_full_batch_when_unsplit():
    assert LocalTrainHelper._microbatch_padding_stats([6, 3, 5], 0) == (18, 18)
    assert LocalTrainHelper._microbatch_padding_stats([6, 3, 5], 8) == (18, 18)
    assert LocalTrainHelper._microbatch_padding_stats([6, 3, 5], 1) == (18, 14)
    assert LocalTrainHelper._microbatch_padding_stats([6, 3, 5, 2], 0, 8) == (
        24,
        16,
    )
