"""Tests for backend capability metadata resolution."""

import sys
from types import SimpleNamespace
from typing import cast

from retrain.backends.catalog import (
    BackendOptionSpec,
    backend_capability_source,
    describe_backends_catalog,
    normalize_backend_options,
    resolve_backend_capabilities,
)
from retrain.training.resume import (
    RESUME_ADAPTER_ONLY,
    RESUME_EXACT,
    RESUME_UNSUPPORTED,
    contract_for_capabilities,
    normalize_resume_mode,
)


def _as_tuple(backend_name: str) -> tuple[bool, bool, bool, bool, str]:
    caps = resolve_backend_capabilities(backend_name, {})
    return (
        caps.reports_sync_loss,
        caps.preserves_token_advantages,
        caps.supports_checkpoint_resume,
        caps.resume_runtime_dependent,
        caps.checkpoint_resume_mode,
    )


def test_builtin_capabilities_match_spec() -> None:
    assert _as_tuple("local") == (True, True, True, False, RESUME_ADAPTER_ONLY)
    assert _as_tuple("unsloth") == (True, True, True, False, RESUME_ADAPTER_ONLY)
    assert _as_tuple("tinker") == (True, True, True, True, RESUME_EXACT)
    assert _as_tuple("prime_rl") == (False, False, True, False, RESUME_ADAPTER_ONLY)
    assert backend_capability_source("local") == "builtin"
    assert resolve_backend_capabilities("unsloth", {}).supports_echo_shared_forward


def test_plugin_capabilities_use_conservative_defaults() -> None:
    assert _as_tuple("my_backend.Factory") == (
        True,
        False,
        True,
        False,
        RESUME_ADAPTER_ONLY,
    )
    assert backend_capability_source("my_backend.Factory") == "plugin/default"


def test_resume_contracts_warn_for_non_exact_modes() -> None:
    local_contract = contract_for_capabilities(resolve_backend_capabilities("local", {}))
    assert local_contract.mode == RESUME_ADAPTER_ONLY
    assert "optimizer/scaler/RNG" in local_contract.warning

    tinker_contract = contract_for_capabilities(resolve_backend_capabilities("tinker", {}))
    assert tinker_contract.mode == RESUME_EXACT
    assert "runtime-dependent" in tinker_contract.warning

    assert normalize_resume_mode("exact") == RESUME_EXACT
    assert normalize_resume_mode("not-a-mode") == RESUME_ADAPTER_ONLY
    assert normalize_resume_mode("exact", supports_resume=False) == RESUME_UNSUPPORTED


def test_resolver_accepts_backend_options_for_builtins() -> None:
    caps = resolve_backend_capabilities(
        "prime_rl",
        {
            "transport": "zmq",
            "strict_advantages": True,
        },
    )
    assert caps.reports_sync_loss is False
    assert caps.preserves_token_advantages is False


def test_local_backend_options_accept_memory_controls() -> None:
    assert normalize_backend_options(
        "local",
        {
            "train_microbatch_size": "2",
            "train_sft_microbatch_token_budget": "9500",
            "cuda_empty_cache": "true",
            "cuda_expandable_segments": "on",
            "strict_deterministic": "true",
            "sample_use_cache": "false",
            "gradient_checkpointing": "false",
            "gradient_checkpointing_use_reentrant": "false",
            "gradient_checkpointing_skip_last_n": "2",
            "cudnn_causal_conv1d_shim": "true",
            "lora_target_modules": "o_proj",
            "lora_layers_to_transform": "last:8",
            "lora_layers_pattern": "layers",
            "lora_detach_input": "true",
            "lora_fast_linear": "true",
            "lora_freeze_a": "true",
            "qwen35_gated_delta_kernel": "torch",
            "trust_remote_code": "true",
        },
    ) == {
        "train_microbatch_size": 2,
        "train_sft_microbatch_token_budget": 9500,
        "train_logprob_chunk_size": 0,
        "liger_kernel": True,
        "liger_fused_linear_ce": True,
        "cuda_empty_cache": True,
        "cuda_expandable_segments": "on",
        "strict_deterministic": True,
        "sample_use_cache": False,
        "sample_kv_quantization": "off",
        "sample_oscar_repo": "",
        "sample_oscar_bits": 2,
        "sample_oscar_quant_mode": "k-channel",
        "sample_oscar_group_size": 0,
        "sample_oscar_kv_rotation": "hadamard",
        "sample_oscar_kv_norm": "1",
        "sample_oscar_residual_block_size": 128,
        "sample_oscar_attn_implementation": "sdpa",
        "gradient_checkpointing": False,
        "gradient_checkpointing_use_reentrant": "false",
        "gradient_checkpointing_skip_last_n": 2,
        "cudnn_causal_conv1d_shim": True,
        "train_selective_suffix_logits": False,
        "train_save_on_cpu": False,
        "train_save_on_cpu_pin_memory": True,
        "train_save_on_cpu_min_numel": 0,
        "train_supervised_context_tokens": 0,
        "train_unsloth_fused_ce": "off",
        "train_unsloth_fused_ce_target_gb": 0.0,
        "train_unsloth_fused_ce_torch_compile": True,
        "train_compile_selective_ce": "off",
        "train_compile_selective_ce_min_tokens": 128,
        "lora_target_modules": "o_proj",
        "lora_layers_to_transform": "last:8",
        "lora_layers_pattern": "layers",
        "lora_detach_input": True,
        "lora_fast_linear": True,
        "lora_freeze_a": True,
        "qwen35_gated_delta_kernel": "torch",
        "trust_remote_code": True,
    }


def test_local_backend_options_accept_oscar_sampling_controls() -> None:
    assert normalize_backend_options(
        "local",
        {
            "sample_use_cache": "true",
            "sample_kv_quantization": "oscar",
            "sample_oscar_repo": "/opt/OScaR-KV-Quant",
            "sample_oscar_bits": "2",
            "sample_oscar_quant_mode": "k-channel",
            "sample_oscar_group_size": "32",
            "sample_oscar_kv_rotation": "hadamard",
            "sample_oscar_kv_norm": "1",
            "sample_oscar_residual_block_size": "128",
            "sample_oscar_attn_implementation": "sdpa",
        },
    ) == {
        "train_microbatch_size": 0,
        "train_sft_microbatch_token_budget": 0,
        "train_logprob_chunk_size": 0,
        "liger_kernel": True,
        "liger_fused_linear_ce": True,
        "cuda_empty_cache": True,
        "cuda_expandable_segments": "auto",
        "strict_deterministic": False,
        "sample_use_cache": True,
        "sample_kv_quantization": "oscar",
        "sample_oscar_repo": "/opt/OScaR-KV-Quant",
        "sample_oscar_bits": 2,
        "sample_oscar_quant_mode": "k-channel",
        "sample_oscar_group_size": 32,
        "sample_oscar_kv_rotation": "hadamard",
        "sample_oscar_kv_norm": "1",
        "sample_oscar_residual_block_size": 128,
        "sample_oscar_attn_implementation": "sdpa",
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": "auto",
        "gradient_checkpointing_skip_last_n": 0,
        "cudnn_causal_conv1d_shim": False,
        "train_selective_suffix_logits": False,
        "train_save_on_cpu": False,
        "train_save_on_cpu_pin_memory": True,
        "train_save_on_cpu_min_numel": 0,
        "train_supervised_context_tokens": 0,
        "train_unsloth_fused_ce": "off",
        "train_unsloth_fused_ce_target_gb": 0.0,
        "train_unsloth_fused_ce_torch_compile": True,
        "train_compile_selective_ce": "off",
        "train_compile_selective_ce_min_tokens": 128,
        "lora_target_modules": "",
        "lora_layers_to_transform": "",
        "lora_layers_pattern": "layers",
        "lora_detach_input": False,
        "lora_fast_linear": False,
        "lora_freeze_a": False,
        "qwen35_gated_delta_kernel": "auto",
        "trust_remote_code": False,
    }


def test_unsloth_backend_options_accept_long_context_controls() -> None:
    normalized = normalize_backend_options(
        "unsloth",
        {
            "max_seq_length": "262144",
            "load_in_4bit": "true",
            "fast_inference": "false",
            "gpu_memory_utilization": "0.9",
            "device_map": "retrain",
            "offload_embedding": "true",
            "unsloth_tiled_mlp": "true",
            "unsloth_tiled_mlp_mode": "target:0.25",
            "train_selective_suffix_logits": "true",
            "train_save_on_cpu": "true",
            "train_save_on_cpu_pin_memory": "false",
            "train_save_on_cpu_min_numel": "65536",
            "train_supervised_context_tokens": "4096",
            "train_microbatch_size": "1",
            "train_unsloth_fused_ce": "require",
            "train_unsloth_fused_ce_target_gb": "1.5",
            "train_unsloth_fused_ce_torch_compile": "false",
            "train_compile_selective_ce": "auto",
            "train_compile_selective_ce_min_tokens": "256",
            "lora_target_modules": "q_proj,o_proj",
            "qwen35_gated_delta_kernel": "flash_qla",
        },
    )
    assert normalized["max_seq_length"] == 262144
    assert normalized["load_in_4bit"] is True
    assert normalized["load_in_8bit"] is False
    assert normalized["load_in_16bit"] is False
    assert normalized["fast_inference"] is False
    assert normalized["gpu_memory_utilization"] == 0.9
    assert normalized["device_map"] == "retrain"
    assert normalized["offload_embedding"] is True
    assert normalized["unsloth_tiled_mlp"] is True
    assert normalized["unsloth_tiled_mlp_mode"] == "target:0.25"
    assert normalized["train_selective_suffix_logits"] is True
    assert normalized["train_save_on_cpu"] is True
    assert normalized["train_save_on_cpu_pin_memory"] is False
    assert normalized["train_save_on_cpu_min_numel"] == 65536
    assert normalized["train_supervised_context_tokens"] == 4096
    assert normalized["train_microbatch_size"] == 1
    assert normalized["train_unsloth_fused_ce"] == "require"
    assert normalized["train_unsloth_fused_ce_target_gb"] == 1.5
    assert normalized["train_unsloth_fused_ce_torch_compile"] is False
    assert normalized["train_compile_selective_ce"] == "auto"
    assert normalized["train_compile_selective_ce_min_tokens"] == 256
    assert normalized["lora_target_modules"] == "q_proj,o_proj"
    assert normalized["liger_kernel"] is False
    assert normalized["liger_fused_linear_ce"] is True
    assert normalized["qwen35_gated_delta_chunk_size"] == "auto"
    assert normalized["qwen35_gated_delta_kernel"] == "flash_qla"


def test_plugin_capability_hook_and_option_schema(monkeypatch) -> None:
    class _PluginFactory:
        retrain_backend_capabilities = {
            "reports_sync_loss": False,
            "preserves_token_advantages": True,
            "supports_checkpoint_resume": True,
            "resume_runtime_dependent": True,
            "checkpoint_resume_mode": "exact",
        }
        retrain_backend_option_schema = {
            "workers": BackendOptionSpec(value_type=int, default=2),
            "mode": {
                "type": "str",
                "default": "fast",
                "choices": ("fast", "safe"),
            },
        }

    monkeypatch.setitem(
        sys.modules,
        "plugin_mod",
        SimpleNamespace(PluginFactory=_PluginFactory),
    )

    caps = resolve_backend_capabilities(
        "plugin_mod.PluginFactory",
        {"workers": "4"},
    )
    assert caps.reports_sync_loss is False
    assert caps.resume_runtime_dependent is True
    assert caps.checkpoint_resume_mode == RESUME_EXACT
    assert (
        backend_capability_source("plugin_mod.PluginFactory", {"workers": "4"})
        == "plugin/hook"
    )

    normalized = normalize_backend_options(
        "plugin_mod.PluginFactory",
        {"workers": "4", "mode": "safe"},
    )
    assert normalized == {"workers": 4, "mode": "safe"}


def test_backends_catalog_payload_shape() -> None:
    payload = describe_backends_catalog()
    builtins = cast(list[dict[str, object]], payload["builtins"])
    plugin = cast(dict[str, object], payload["plugin"])
    names = {item["name"] for item in builtins}
    assert {"local", "unsloth", "tinker", "prime_rl"} <= names
    assert "capability_hooks" in plugin
    unsloth = next(item for item in builtins if item["name"] == "unsloth")
    capabilities = cast(dict[str, object], unsloth["capabilities"])
    assert capabilities["supports_echo_shared_forward"] is True
    assert capabilities["checkpoint_resume_mode"] == RESUME_ADAPTER_ONLY
