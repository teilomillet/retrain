"""Optional OScaR Qwen3 integration for local PyTorch sampling.

The public OScaR implementation ships its Qwen3 model fork in the repository's
``evaluation/`` directory, while the pip package exposes the custom cache and
CUDA extension as ``oscar``. Keep this integration dependency-optional and
default-off so normal retrain installs do not need OScaR.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class OscarQwen3Options:
    """Runtime knobs for the upstream OScaR Qwen3 fork."""

    repo: str = ""
    bits: int = 2
    quant_mode: str = "k-channel"
    group_size: int = 0
    kv_rotation: str = "hadamard"
    kv_norm: str = "1"
    residual_block_size: int = 128
    attn_implementation: str = "sdpa"


def normalize_sample_kv_quantization(raw: object) -> str:
    """Normalize the local sampling KV-cache mode."""
    mode = str(raw or "off").strip().lower()
    if mode in {"", "none", "false", "0"}:
        return "off"
    if mode not in {"off", "oscar"}:
        raise ValueError(
            "sample_kv_quantization must be 'off' or 'oscar'."
        )
    return mode


def oscar_options_from_mapping(options: Mapping[str, object]) -> OscarQwen3Options:
    """Extract OScaR-specific backend options from a config mapping."""
    bits = _int_option(options, "sample_oscar_bits", 2)
    if bits not in (2, 4):
        raise ValueError("sample_oscar_bits must be 2 or 4.")
    group_size = _int_option(options, "sample_oscar_group_size", 0)
    if group_size < 0:
        raise ValueError("sample_oscar_group_size must be non-negative.")
    residual_block_size = _int_option(
        options,
        "sample_oscar_residual_block_size",
        128,
    )
    if residual_block_size <= 0:
        raise ValueError("sample_oscar_residual_block_size must be positive.")
    return OscarQwen3Options(
        repo=_str_option(options, "sample_oscar_repo", ""),
        bits=bits,
        quant_mode=_str_option(options, "sample_oscar_quant_mode", "k-channel"),
        group_size=group_size,
        kv_rotation=_str_option(options, "sample_oscar_kv_rotation", "hadamard"),
        kv_norm=_str_option(options, "sample_oscar_kv_norm", "1"),
        residual_block_size=residual_block_size,
        attn_implementation=str(
            options.get("sample_oscar_attn_implementation", "sdpa") or "sdpa"
        ),
    )


def prepare_oscar_runtime(options: OscarQwen3Options) -> dict[str, int | str]:
    """Make upstream OScaR imports available and return setup telemetry."""
    metrics: dict[str, int | str] = {
        "oscar_repo_path_inserted": 0,
        "oscar_flash_attn_available": 0,
    }
    _insert_oscar_repo_paths(options.repo, metrics)
    _ensure_flash_attn_with_kvcache(metrics)
    try:
        _patch_transformers_cache_utils()
    except ModuleNotFoundError as exc:
        if exc.name == "oscar":
            raise RuntimeError(_missing_oscar_message(options)) from exc
        raise
    return metrics


def load_oscar_qwen3_causal_lm(
    model_name: str,
    *,
    dtype: object,
    options: OscarQwen3Options,
    trust_remote_code: bool = False,
    device_map: object | None = None,
):
    """Load the upstream OScaR Qwen3 CausalLM class with configured KV mode."""
    metrics = prepare_oscar_runtime(options)
    try:
        qwen3 = importlib.import_module("qwen3")
    except ModuleNotFoundError as exc:
        if exc.name == "qwen3":
            raise RuntimeError(_missing_qwen3_message(options)) from exc
        raise
    _patch_qwen3_module(qwen3)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    apply_oscar_config(config, options)
    kwargs: dict[str, object] = {
        "config": config,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = qwen3.Qwen3ForCausalLM.from_pretrained(model_name, **kwargs)
    metrics["oscar_qwen3_loaded"] = 1
    return model, metrics


def apply_oscar_config(config: object, options: OscarQwen3Options) -> None:
    """Apply OScaR fields expected by the upstream Qwen3 fork."""
    group_size = options.group_size or (32 if options.bits == 2 else 128)
    setattr(config, "_attn_implementation", options.attn_implementation)
    setattr(config, "attn_backend", "oscar")
    setattr(config, "num_bits", int(options.bits))
    setattr(config, "quant_mode", options.quant_mode)
    setattr(config, "group_size", int(group_size))
    setattr(config, "kv_rotation", options.kv_rotation)
    setattr(config, "kv_norm", options.kv_norm)
    setattr(config, "residual_block_size", int(options.residual_block_size))


def new_oscar_dynamic_cache(options: OscarQwen3Options):
    """Create a fresh OScaR DynamicCache for one prompt generation."""
    prepare_oscar_runtime(options)
    return oscar_dynamic_cache_class()()


def oscar_dynamic_cache_class():
    """Return OScaR's DynamicCache class after runtime setup."""
    oscar_module = importlib.import_module("oscar")
    return getattr(oscar_module, "DynamicCache")


def _missing_oscar_message(options: OscarQwen3Options) -> str:
    repo_hint = f" Current sample_oscar_repo={options.repo!r}." if options.repo else ""
    return (
        "sample_kv_quantization='oscar' requires the upstream OScaR package "
        "to be importable as 'oscar'. Install OScaR in the environment or set "
        "sample_oscar_repo to a checkout containing the package."
        f"{repo_hint}"
    )


def _missing_qwen3_message(options: OscarQwen3Options) -> str:
    repo_hint = f" Current sample_oscar_repo={options.repo!r}." if options.repo else ""
    return (
        "sample_kv_quantization='oscar' requires OScaR's evaluation/qwen3.py "
        "to be importable. Set sample_oscar_repo to an OScaR-KV-Quant checkout "
        "or add its evaluation directory to PYTHONPATH."
        f"{repo_hint}"
    )


def _insert_oscar_repo_paths(
    repo: str,
    metrics: dict[str, int | str],
) -> None:
    if not repo:
        return
    root = Path(repo).expanduser()
    evaluation = root / "evaluation"
    for path in (evaluation, root):
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)
            metrics["oscar_repo_path_inserted"] = 1


def _ensure_flash_attn_with_kvcache(metrics: dict[str, int | str]) -> None:
    try:
        flash_attn = importlib.import_module("flash_attn")
    except Exception as exc:
        raise RuntimeError(
            "sample_kv_quantization='oscar' requires a working flash-attn "
            "installation. The stubbed flash-attn path was measured and did "
            "not pass retrain sampling."
        ) from exc
    if not hasattr(flash_attn, "flash_attn_with_kvcache"):
        raise RuntimeError(
            "sample_kv_quantization='oscar' requires flash_attn_with_kvcache "
            "from flash-attn."
        )
    metrics["oscar_flash_attn_available"] = 1


def _patch_transformers_cache_utils() -> None:
    import transformers.cache_utils

    oscar_module = importlib.import_module("oscar")
    transformers.cache_utils.Cache = getattr(oscar_module, "Cache")
    transformers.cache_utils.DynamicCache = getattr(oscar_module, "DynamicCache")
    transformers.cache_utils.StaticCache = getattr(oscar_module, "StaticCache")


def _patch_qwen3_module(qwen3: object) -> None:
    rope_functions = getattr(qwen3, "ROPE_INIT_FUNCTIONS", None)
    if isinstance(rope_functions, dict) and "default" not in rope_functions:
        rope_functions["default"] = _compute_default_rope_parameters
    model_cls = getattr(qwen3, "Qwen3ForCausalLM", None)
    tied_keys = getattr(model_cls, "_tied_weights_keys", None)
    if model_cls is not None and tied_keys == ["lm_head.weight"]:
        setattr(
            model_cls,
            "_tied_weights_keys",
            {"lm_head.weight": "model.embed_tokens.weight"},
        )
    if hasattr(qwen3, "SlidingWindowCache"):
        return
    cache_utils = importlib.import_module("oscar.models.cache_utils")
    setattr(qwen3, "SlidingWindowCache", getattr(cache_utils, "SlidingWindowCache"))


def _compute_default_rope_parameters(
    config: object | None = None,
    device: "torch.device | str | int | None" = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple["torch.Tensor", float]:
    """Compatibility implementation for Transformers builds without default RoPE."""
    _ = seq_len
    import torch

    if config is None:
        raise ValueError("config is required for default Qwen3 RoPE parameters.")
    standardize_rope_params = getattr(config, "standardize_rope_params", None)
    if callable(standardize_rope_params):
        standardize_rope_params()
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and layer_type is not None:
        rope_parameters = rope_parameters.get(layer_type, rope_parameters)
    if not isinstance(rope_parameters, dict):
        rope_parameters = {}
    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict):
        rope_scaling = {}
    base = (
        rope_parameters.get("rope_theta")
        or rope_scaling.get("rope_theta")
        or getattr(config, "rope_theta", 10000.0)
    )
    partial_rotary_factor = (
        rope_parameters.get("partial_rotary_factor")
        or getattr(config, "partial_rotary_factor", 1.0)
    )
    head_dim = getattr(config, "head_dim", None) or (
        int(getattr(config, "hidden_size")) // int(getattr(config, "num_attention_heads"))
    )
    dim = int(head_dim * float(partial_rotary_factor))
    inv_freq = 1.0 / (
        float(base)
        ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


def _int_option(
    options: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    raw = options.get(key, default)
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int | str):
        return int(raw)
    raise ValueError(f"{key} must be an integer.")


def _str_option(
    options: Mapping[str, object],
    key: str,
    default: str,
) -> str:
    raw = options.get(key, default)
    return str(raw or default)
