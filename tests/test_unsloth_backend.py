from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest
import torch

from retrain.unsloth_backend import UnslothTrainHelper


class Qwen3_5GatedDeltaNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.chunk_kwargs: list[dict[str, object]] = []

    def _chunk_gated_delta_rule(self, *args, **kwargs):
        _ = args
        self.chunk_kwargs.append(dict(kwargs))
        return "ok"

    chunk_gated_delta_rule = _chunk_gated_delta_rule


class _TinyCausalModel(torch.nn.Module):
    def __init__(self, *, raise_on_to=False):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.lm_head = torch.nn.Linear(8, 16, bias=False)
        self.linear_attn = Qwen3_5GatedDeltaNet()
        self.config = SimpleNamespace(use_cache=False, eos_token_id=None)
        self.generation_config = SimpleNamespace(eos_token_id=None)
        self.mode = "training"
        self.raise_on_to = raise_on_to
        self.to_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.last_logits_to_keep = 0
        self.last_input_shape = None

    def forward(self, input_ids, attention_mask=None, **kwargs):
        _ = attention_mask, kwargs
        self.last_input_shape = tuple(input_ids.shape)
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        logits_to_keep = int(kwargs.get("logits_to_keep") or 0)
        self.last_logits_to_keep = logits_to_keep
        if logits_to_keep > 0:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)

    def to(self, *args, **kwargs):
        self.to_calls.append((args, dict(kwargs)))
        if self.raise_on_to:
            raise AssertionError("quantized Unsloth models must not be moved with .to()")
        return super().to(*args, **kwargs)

    def print_trainable_parameters(self):
        return None

    def save_pretrained(self, path):
        _ = path


class _FakeFastLanguageModel:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        max_seq_length,
        dtype,
        load_in_4bit,
        load_in_8bit,
        load_in_16bit,
        full_finetuning,
        device_map,
        fast_inference,
        gpu_memory_utilization,
        float8_kv_cache,
        max_lora_rank,
        trust_remote_code,
        use_exact_model_name,
        offload_embedding,
        use_gradient_checkpointing,
        random_state,
        unsloth_tiled_mlp,
        text_only,
    ):
        kwargs = {
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "dtype": dtype,
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": load_in_8bit,
            "load_in_16bit": load_in_16bit,
            "full_finetuning": full_finetuning,
            "device_map": device_map,
            "fast_inference": fast_inference,
            "gpu_memory_utilization": gpu_memory_utilization,
            "float8_kv_cache": float8_kv_cache,
            "max_lora_rank": max_lora_rank,
            "trust_remote_code": trust_remote_code,
            "use_exact_model_name": use_exact_model_name,
            "offload_embedding": offload_embedding,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "random_state": random_state,
            "unsloth_tiled_mlp": unsloth_tiled_mlp,
            "text_only": text_only,
        }
        cls.calls.append(("from_pretrained", dict(kwargs)))
        return _TinyCausalModel(
            raise_on_to=bool(kwargs.get("load_in_4bit") or kwargs.get("load_in_8bit"))
        ), object()

    @classmethod
    def get_peft_model(
        cls,
        model,
        r,
        target_modules,
        lora_alpha,
        lora_dropout,
        bias,
        use_gradient_checkpointing,
        random_state,
        max_seq_length,
        use_rslora,
    ):
        kwargs = {
            "r": r,
            "target_modules": target_modules,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": bias,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "random_state": random_state,
            "max_seq_length": max_seq_length,
            "use_rslora": use_rslora,
        }
        cls.calls.append(("get_peft_model", dict(kwargs)))
        model.peft_kwargs = dict(kwargs)
        return model

    @classmethod
    def for_inference(cls, model):
        cls.calls.append(("for_inference", {}))
        model.mode = "inference"

    @classmethod
    def for_training(cls, model):
        cls.calls.append(("for_training", {}))
        model.mode = "training"


def test_unsloth_helper_loads_model_samples_and_runs_echo(monkeypatch, tmp_path):
    _FakeFastLanguageModel.calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(FastLanguageModel=_FakeFastLanguageModel),
    )

    helper = UnslothTrainHelper(
        "Qwen/Qwen3.5-2B",
        str(tmp_path),
        "cpu",
        lora_rank=2,
        max_seq_length=32768,
        load_in_4bit=True,
        fast_inference=False,
        gpu_memory_utilization=0.9,
        train_microbatch_size=1,
        liger_kernel=False,
        liger_fused_linear_ce=False,
        gradient_checkpointing=True,
        sample_use_cache=True,
    )
    try:
        from_call = _FakeFastLanguageModel.calls[0]
        assert from_call[0] == "from_pretrained"
        assert from_call[1]["model_name"] == "Qwen/Qwen3.5-2B"
        assert from_call[1]["max_seq_length"] == 32768
        assert from_call[1]["load_in_4bit"] is True
        assert from_call[1]["fast_inference"] is False
        assert from_call[1]["gpu_memory_utilization"] == 0.9
        assert from_call[1]["offload_embedding"] is False
        assert from_call[1]["use_gradient_checkpointing"] == "unsloth"
        assert from_call[1]["device_map"] == {"": "cpu"}

        peft_call = _FakeFastLanguageModel.calls[1]
        assert peft_call[0] == "get_peft_model"
        assert peft_call[1]["r"] == 2
        assert peft_call[1]["use_gradient_checkpointing"] == "unsloth"
        assert helper.train_model.to_calls == []

        samples = helper.sample([[1, 2]], 1, 1, 1.0, 1.0)
        assert len(samples) == 1
        assert len(samples[0]) == 1
        assert len(samples[0][0][0]) == 1
        call_names = [name for name, _ in _FakeFastLanguageModel.calls]
        assert "for_inference" in call_names
        assert "for_training" in call_names

        rl_loss, echo_loss = helper.train_step_with_echo_masks(
            all_tokens=[[1, 2, 3, 4]],
            all_logprobs=[[0.0, 0.0, 0.0, 0.0]],
            all_advantages=[[0.0, 1.0, 1.0, 1.0]],
            echo_advantages=[[0.0, 0.0, 0.2, 0.0]],
            echo_full_observation_counts=[1],
            echo_loss_fn="cross_entropy",
            lr=1e-3,
            weight_decay=0.0,
        )
        assert math.isfinite(float(rl_loss))
        assert math.isfinite(float(echo_loss))
        assert float(echo_loss) > 0.0

        metrics = helper.runtime_metrics()
        assert metrics["unsloth_backend_enabled"] == 1
        assert metrics["unsloth_max_seq_length"] == 32768
        assert metrics["unsloth_load_in_4bit"] == 1
        assert metrics["unsloth_device_map_retrain"] == 1
        assert metrics["unsloth_offload_embedding"] == 0
    finally:
        helper.shutdown()


def test_unsloth_helper_can_train_with_selective_suffix_logits(monkeypatch, tmp_path):
    _FakeFastLanguageModel.calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(FastLanguageModel=_FakeFastLanguageModel),
    )

    helper = UnslothTrainHelper(
        "Qwen/Qwen3.5-2B",
        str(tmp_path),
        "cpu",
        lora_rank=2,
        load_in_4bit=True,
        train_microbatch_size=1,
        train_selective_suffix_logits=True,
        train_save_on_cpu_min_numel=7,
        liger_kernel=False,
        liger_fused_linear_ce=False,
        sample_use_cache=True,
    )
    try:
        rl_loss, echo_loss = helper.train_step_with_echo_masks(
            all_tokens=[[1, 2, 3, 4, 5]],
            all_logprobs=[[0.0, 0.0, 0.0, 0.0, 0.0]],
            all_advantages=[[0.0, 0.0, 0.0, 0.0, 1.0]],
            echo_advantages=[[0.0, 0.0, 0.0, 0.2, 0.0]],
            echo_full_observation_counts=[1],
            echo_loss_fn="cross_entropy",
            lr=1e-3,
            weight_decay=0.0,
        )
        assert math.isfinite(float(rl_loss))
        assert math.isfinite(float(echo_loss))
        assert helper.train_model.last_logits_to_keep == 3
        metrics = helper.runtime_metrics()
        assert metrics["local_train_selective_suffix_logits"] == 1
        assert metrics["local_train_save_on_cpu_min_numel"] == 7
    finally:
        helper.shutdown()


def test_unsloth_helper_can_train_on_supervised_context_window(monkeypatch, tmp_path):
    _FakeFastLanguageModel.calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(FastLanguageModel=_FakeFastLanguageModel),
    )

    helper = UnslothTrainHelper(
        "Qwen/Qwen3.5-2B",
        str(tmp_path),
        "cpu",
        lora_rank=2,
        load_in_4bit=True,
        train_microbatch_size=1,
        train_selective_suffix_logits=True,
        train_supervised_context_tokens=3,
        liger_kernel=False,
        liger_fused_linear_ce=False,
        sample_use_cache=True,
    )
    try:
        rl_loss, echo_loss = helper.train_step_with_echo_masks(
            all_tokens=[[1, 2, 3, 4, 5, 6, 7, 8]],
            all_logprobs=[[0.0] * 8],
            all_advantages=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            echo_advantages=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]],
            echo_full_observation_counts=[1],
            echo_loss_fn="cross_entropy",
            lr=1e-3,
            weight_decay=0.0,
        )
        assert math.isfinite(float(rl_loss))
        assert math.isfinite(float(echo_loss))
        assert helper.train_model.last_input_shape == (1, 6)
        metrics = helper.runtime_metrics()
        assert metrics["local_train_supervised_context_tokens"] == 3
        assert metrics["local_train_context_rows_cropped"] == 1
        assert metrics["local_train_context_tokens_removed"] == 2
        assert metrics["local_train_context_original_max_tokens"] == 8
        assert metrics["local_train_context_cropped_max_tokens"] == 6
    finally:
        helper.shutdown()


def test_unsloth_helper_sets_tiled_mlp_mode_env(monkeypatch, tmp_path):
    _FakeFastLanguageModel.calls.clear()
    monkeypatch.delenv("UNSLOTH_TILED_MLP", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(FastLanguageModel=_FakeFastLanguageModel),
    )

    class _EnvCheckingFastLanguageModel(_FakeFastLanguageModel):
        @classmethod
        def from_pretrained(
            cls,
            model_name,
            max_seq_length,
            dtype,
            load_in_4bit,
            load_in_8bit,
            load_in_16bit,
            full_finetuning,
            device_map,
            fast_inference,
            gpu_memory_utilization,
            float8_kv_cache,
            max_lora_rank,
            trust_remote_code,
            use_exact_model_name,
            offload_embedding,
            use_gradient_checkpointing,
            random_state,
            unsloth_tiled_mlp,
            text_only,
        ):
            cls.calls.append(
                (
                    "env",
                    {
                        "UNSLOTH_TILED_MLP": __import__("os").environ.get(
                            "UNSLOTH_TILED_MLP"
                        )
                    },
                )
            )
            return super().from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                load_in_16bit=load_in_16bit,
                full_finetuning=full_finetuning,
                device_map=device_map,
                fast_inference=fast_inference,
                gpu_memory_utilization=gpu_memory_utilization,
                float8_kv_cache=float8_kv_cache,
                max_lora_rank=max_lora_rank,
                trust_remote_code=trust_remote_code,
                use_exact_model_name=use_exact_model_name,
                offload_embedding=offload_embedding,
                use_gradient_checkpointing=use_gradient_checkpointing,
                random_state=random_state,
                unsloth_tiled_mlp=unsloth_tiled_mlp,
                text_only=text_only,
            )

    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(FastLanguageModel=_EnvCheckingFastLanguageModel),
    )

    helper = UnslothTrainHelper(
        "Qwen/Qwen3.5-2B",
        str(tmp_path),
        "cpu",
        load_in_4bit=True,
        train_microbatch_size=1,
        liger_kernel=False,
        liger_fused_linear_ce=False,
        unsloth_tiled_mlp=True,
        unsloth_tiled_mlp_mode="target:0.25",
    )
    try:
        env_call = _EnvCheckingFastLanguageModel.calls[0]
        assert env_call == ("env", {"UNSLOTH_TILED_MLP": "target:0.25"})
        assert __import__("os").environ.get("UNSLOTH_TILED_MLP") is None
        metrics = helper.runtime_metrics()
        assert metrics["unsloth_tiled_mlp"] == 1
        assert metrics["unsloth_tiled_mlp_mode"] == "target:0.25"
    finally:
        helper.shutdown()


def test_unsloth_helper_can_force_qwen35_gated_delta_chunk(monkeypatch, tmp_path):
    _FakeFastLanguageModel.calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(FastLanguageModel=_FakeFastLanguageModel),
    )

    helper = UnslothTrainHelper(
        "Qwen/Qwen3.5-2B",
        str(tmp_path),
        "cpu",
        load_in_4bit=True,
        train_microbatch_size=1,
        liger_kernel=False,
        liger_fused_linear_ce=False,
        qwen35_gated_delta_chunk_size=32,
    )
    try:
        result = helper.train_model.linear_attn.chunk_gated_delta_rule(marker=True)
        assert result == "ok"
        assert helper.train_model.linear_attn.chunk_kwargs[-1]["chunk_size"] == 32
        metrics = helper.runtime_metrics()
        assert metrics["unsloth_qwen35_gated_delta_chunk_size"] == 32
        assert metrics["unsloth_qwen35_gated_delta_patched_modules"] == 1
    finally:
        helper.shutdown()


def test_unsloth_qwen35_auto_keeps_installed_gated_delta_path(monkeypatch, tmp_path):
    _FakeFastLanguageModel.calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(FastLanguageModel=_FakeFastLanguageModel),
    )
    monkeypatch.setattr(
        UnslothTrainHelper,
        "_qwen35_gated_delta_shared_memory_limit",
        lambda self: 101376,
    )

    helper = UnslothTrainHelper(
        "Qwen/Qwen3.5-2B",
        str(tmp_path),
        "cpu",
        load_in_4bit=True,
        train_microbatch_size=1,
        liger_kernel=False,
        liger_fused_linear_ce=False,
        qwen35_gated_delta_chunk_size="auto",
    )
    try:
        result = helper.train_model.linear_attn.chunk_gated_delta_rule(marker=True)
        assert result == "ok"
        assert "chunk_size" not in helper.train_model.linear_attn.chunk_kwargs[-1]
        metrics = helper.runtime_metrics()
        assert metrics["unsloth_qwen35_gated_delta_chunk_size"] == 0
        assert metrics["unsloth_qwen35_gated_delta_patched_modules"] == 0
        assert metrics["unsloth_qwen35_gated_delta_torch_fallback"] == 0
        assert metrics["unsloth_qwen35_gated_delta_shared_memory_limit"] == 101376
    finally:
        helper.shutdown()


def test_unsloth_rejects_mixed_qwen35_gated_delta_overrides(tmp_path):
    with pytest.raises(ValueError, match="cannot be combined"):
        UnslothTrainHelper(
            "Qwen/Qwen3.5-2B",
            str(tmp_path),
            "cpu",
            load_in_4bit=True,
            train_microbatch_size=1,
            liger_kernel=False,
            liger_fused_linear_ce=False,
            qwen35_gated_delta_chunk_size=32,
            qwen35_gated_delta_kernel="flash_qla",
        )


def test_unsloth_helper_moves_non_quantized_models(monkeypatch, tmp_path):
    _FakeFastLanguageModel.calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(FastLanguageModel=_FakeFastLanguageModel),
    )

    helper = UnslothTrainHelper(
        "Qwen/Qwen3.5-2B",
        str(tmp_path),
        "cpu",
        load_in_4bit=False,
        load_in_16bit=True,
        train_microbatch_size=1,
        liger_kernel=False,
        liger_fused_linear_ce=False,
    )
    try:
        assert helper.train_model.to_calls
        assert helper.train_model.to_calls[0][0] == ("cpu",)
    finally:
        helper.shutdown()


def test_unsloth_helper_rejects_conflicting_precision_modes(tmp_path):
    try:
        UnslothTrainHelper(
            "Qwen/Qwen3.5-2B",
            str(tmp_path),
            "cpu",
            load_in_4bit=True,
            load_in_16bit=True,
        )
    except ValueError as exc:
        assert "only one active precision" in str(exc)
    else:
        raise AssertionError("expected conflicting Unsloth precision modes to fail")
