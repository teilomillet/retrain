"""Backend contract tests for built-in backends."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from retrain.backends import TrainHelper
from retrain.config import TrainConfig
from retrain.registry import backend


class _BaseFakeHelper:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def checkpoint(self, name: str) -> None:
        self.calls.append(f"checkpoint:{name}")

    def sample(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[tuple[list[int], list[float]]]]:
        self.calls.append("sample")
        _ = max_tokens, temperature, top_p
        return [
            [([101, 102], [-0.1, -0.2]) for _ in range(num_samples)]
            for _ in prompt_ids_list
        ]

    def sample_with_entropy(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[tuple[list[int], list[float], list[float] | None]]]:
        self.calls.append("sample_with_entropy")
        _ = max_tokens, temperature, top_p
        return [
            [([101, 102], [-0.1, -0.2], None) for _ in range(num_samples)]
            for _ in prompt_ids_list
        ]

    def train_step(
        self,
        all_tokens: list[list[int]],
        all_logprobs: list[list[float]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float:
        self.calls.append("train_step")
        _ = all_tokens, all_logprobs, all_advantages, lr, weight_decay
        return 0.0

    def save_adapter(self, path: str, name: str) -> str:
        self.calls.append(f"save_adapter:{name}")
        return f"{path.rstrip('/')}/{name}"

    def load_state(self, name: str) -> None:
        self.calls.append(f"load_state:{name}")


class _FakeLocalTrainHelper(_BaseFakeHelper):
    init_calls: list[dict] = []

    def __init__(self, *args, **kwargs):
        self.init_calls.append({"args": args, "kwargs": kwargs})
        super().__init__()


class _FakeTinkerTrainHelper(_BaseFakeHelper):
    def __init__(self, *args, **kwargs):
        _ = args, kwargs
        super().__init__()


class _FakePrimeRLTrainHelper(_BaseFakeHelper):
    def __init__(self, *args, **kwargs):
        _ = args, kwargs
        super().__init__()


class _FakeUnslothTrainHelper(_BaseFakeHelper):
    init_calls: list[dict] = []

    def __init__(self, *args, **kwargs):
        self.init_calls.append({"args": args, "kwargs": kwargs})
        super().__init__()


def _exercise_lifecycle_step(helper: TrainHelper, step_name: str) -> None:
    helper.checkpoint(step_name)
    out = helper.sample(
        prompt_ids_list=[[1, 2, 3]],
        num_samples=1,
        max_tokens=32,
        temperature=0.7,
        top_p=0.95,
    )
    assert out == [[([101, 102], [-0.1, -0.2])]]

    loss = helper.train_step(
        all_tokens=[[1, 2, 3]],
        all_logprobs=[[0.0, -0.1, -0.2]],
        all_advantages=[[0.0, 1.0, 1.0]],
        lr=1e-4,
        weight_decay=0.0,
    )
    assert isinstance(loss, float)

    save_path = helper.save_adapter("/tmp/backend-contract", step_name)
    assert save_path.endswith(f"/{step_name}")

    helper.load_state(step_name)


def _assert_lifecycle_calls(helper: _BaseFakeHelper) -> None:
    assert helper.calls == [
        "checkpoint:step_0",
        "sample",
        "train_step",
        "save_adapter:step_0",
        "load_state:step_0",
        "checkpoint:step_1",
        "sample",
        "train_step",
        "save_adapter:step_1",
        "load_state:step_1",
    ]


def test_local_backend_contract(monkeypatch):
    _FakeLocalTrainHelper.init_calls.clear()
    fake_mod = SimpleNamespace(LocalTrainHelper=_FakeLocalTrainHelper)
    monkeypatch.setitem(sys.modules, "retrain.local_train_helper", fake_mod)

    cfg = TrainConfig(backend="local")
    helper = backend.create("local", cfg)
    assert isinstance(helper, TrainHelper)
    assert isinstance(helper, _BaseFakeHelper)
    _exercise_lifecycle_step(helper, "step_0")
    _exercise_lifecycle_step(helper, "step_1")
    _assert_lifecycle_calls(helper)
    assert _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["train_microbatch_size"] == 0
    assert _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["cuda_empty_cache"] is True
    assert _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["sample_use_cache"] is True
    assert _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["gradient_checkpointing"] is True
    assert _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["prefix_caching"] is True
    assert _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["policy_loss_mode"] == "standard"


def test_local_backend_passes_memory_control_options(monkeypatch):
    _FakeLocalTrainHelper.init_calls.clear()
    fake_mod = SimpleNamespace(LocalTrainHelper=_FakeLocalTrainHelper)
    monkeypatch.setitem(sys.modules, "retrain.local_train_helper", fake_mod)

    cfg = TrainConfig(
        backend="local",
        backend_options={
            "train_microbatch_size": 2,
            "cuda_empty_cache": True,
            "sample_use_cache": False,
            "gradient_checkpointing": False,
            "train_save_on_cpu_pin_memory": False,
            "train_save_on_cpu_min_numel": 65536,
            "train_supervised_context_tokens": 512,
            "train_unsloth_fused_ce": "require",
            "train_unsloth_fused_ce_target_gb": 1.25,
            "train_unsloth_fused_ce_torch_compile": False,
        },
        policy_loss_mode="kl_cov",
        kl_cov_percent=0.4,
        kl_cov_coef=0.5,
        prefix_caching=False,
    )
    backend.create("local", cfg)

    assert _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["train_microbatch_size"] == 2
    assert _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["cuda_empty_cache"] is True
    assert _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["sample_use_cache"] is False
    assert _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["gradient_checkpointing"] is False
    assert (
        _FakeLocalTrainHelper.init_calls[-1]["kwargs"][
            "train_save_on_cpu_pin_memory"
        ]
        is False
    )
    assert (
        _FakeLocalTrainHelper.init_calls[-1]["kwargs"][
            "train_supervised_context_tokens"
        ]
        == 512
    )
    assert (
        _FakeLocalTrainHelper.init_calls[-1]["kwargs"][
            "train_save_on_cpu_min_numel"
        ]
        == 65536
    )
    assert (
        _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["train_unsloth_fused_ce"]
        == "require"
    )
    assert (
        _FakeLocalTrainHelper.init_calls[-1]["kwargs"][
            "train_unsloth_fused_ce_target_gb"
        ]
        == 1.25
    )
    assert (
        _FakeLocalTrainHelper.init_calls[-1]["kwargs"][
            "train_unsloth_fused_ce_torch_compile"
        ]
        is False
    )
    assert _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["prefix_caching"] is False
    assert _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["policy_loss_mode"] == "kl_cov"
    assert _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["kl_cov_percent"] == 0.4
    assert _FakeLocalTrainHelper.init_calls[-1]["kwargs"]["kl_cov_coef"] == 0.5


def test_unsloth_backend_contract(monkeypatch):
    _FakeUnslothTrainHelper.init_calls.clear()
    fake_mod = SimpleNamespace(UnslothTrainHelper=_FakeUnslothTrainHelper)
    monkeypatch.setitem(sys.modules, "retrain.unsloth_backend", fake_mod)

    cfg = TrainConfig(
        backend="unsloth",
        backend_options={
            "max_seq_length": 32768,
            "load_in_4bit": True,
            "gpu_memory_utilization": 0.9,
            "train_microbatch_size": 1,
            "offload_embedding": True,
            "unsloth_tiled_mlp": True,
            "unsloth_tiled_mlp_mode": "target:0.25",
            "train_selective_suffix_logits": True,
            "train_save_on_cpu": True,
            "train_save_on_cpu_pin_memory": True,
            "train_save_on_cpu_min_numel": 65536,
            "train_supervised_context_tokens": 4096,
            "train_unsloth_fused_ce": "require",
            "train_unsloth_fused_ce_target_gb": 1.5,
            "train_unsloth_fused_ce_torch_compile": False,
        },
        model="Qwen/Qwen3.5-2B",
    )
    helper = backend.create("unsloth", cfg)
    assert isinstance(helper, TrainHelper)
    assert isinstance(helper, _BaseFakeHelper)
    _exercise_lifecycle_step(helper, "step_0")
    _exercise_lifecycle_step(helper, "step_1")
    _assert_lifecycle_calls(helper)

    kwargs = _FakeUnslothTrainHelper.init_calls[-1]["kwargs"]
    assert kwargs["max_seq_length"] == 32768
    assert kwargs["load_in_4bit"] is True
    assert kwargs["load_in_8bit"] is False
    assert kwargs["load_in_16bit"] is False
    assert kwargs["gpu_memory_utilization"] == 0.9
    assert kwargs["device_map"] == "retrain"
    assert kwargs["offload_embedding"] is True
    assert kwargs["unsloth_tiled_mlp"] is True
    assert kwargs["unsloth_tiled_mlp_mode"] == "target:0.25"
    assert kwargs["train_selective_suffix_logits"] is True
    assert kwargs["train_save_on_cpu"] is True
    assert kwargs["train_save_on_cpu_pin_memory"] is True
    assert kwargs["train_save_on_cpu_min_numel"] == 65536
    assert kwargs["train_supervised_context_tokens"] == 4096
    assert kwargs["train_microbatch_size"] == 1
    assert kwargs["train_unsloth_fused_ce"] == "require"
    assert kwargs["train_unsloth_fused_ce_target_gb"] == 1.5
    assert kwargs["train_unsloth_fused_ce_torch_compile"] is False
    assert kwargs["liger_kernel"] is False
    assert kwargs["liger_fused_linear_ce"] is True
    assert kwargs["sample_use_cache"] is True
    assert kwargs["gradient_checkpointing"] is True
    assert kwargs["qwen35_gated_delta_chunk_size"] == "auto"


def test_unsloth_backend_defaults_max_seq_length_from_max_tokens(monkeypatch):
    _FakeUnslothTrainHelper.init_calls.clear()
    fake_mod = SimpleNamespace(UnslothTrainHelper=_FakeUnslothTrainHelper)
    monkeypatch.setitem(sys.modules, "retrain.unsloth_backend", fake_mod)

    cfg = TrainConfig(
        backend="unsloth",
        max_tokens=4096,
        backend_options={"max_seq_length": 0},
    )
    backend.create("unsloth", cfg)

    kwargs = _FakeUnslothTrainHelper.init_calls[-1]["kwargs"]
    assert kwargs["max_seq_length"] == 4096


def test_tinker_backend_contract(monkeypatch):
    fake_mod = SimpleNamespace(TinkerTrainHelper=_FakeTinkerTrainHelper)
    monkeypatch.setitem(sys.modules, "retrain.tinker_backend", fake_mod)

    cfg = TrainConfig(backend="tinker")
    helper = backend.create("tinker", cfg)
    assert isinstance(helper, TrainHelper)
    assert isinstance(helper, _BaseFakeHelper)
    _exercise_lifecycle_step(helper, "step_0")
    _exercise_lifecycle_step(helper, "step_1")
    _assert_lifecycle_calls(helper)


def test_prime_rl_backend_contract(monkeypatch):
    fake_mod = SimpleNamespace(PrimeRLTrainHelper=_FakePrimeRLTrainHelper)
    monkeypatch.setitem(sys.modules, "retrain.prime_rl_backend", fake_mod)

    cfg = TrainConfig(backend="prime_rl")
    helper = backend.create("prime_rl", cfg)
    assert isinstance(helper, TrainHelper)
    assert isinstance(helper, _BaseFakeHelper)
    _exercise_lifecycle_step(helper, "step_0")
    _exercise_lifecycle_step(helper, "step_1")
    _assert_lifecycle_calls(helper)
