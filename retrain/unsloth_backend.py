"""Unsloth-backed local TrainHelper.

This backend keeps retrain's environment, rollout, advantage, ECHO, and logging
contracts while swapping only the local model construction path to Unsloth's
FastLanguageModel. The training loss implementation is inherited from
LocalTrainHelper so token-level advantages and ECHO masks remain auditable.
"""

from __future__ import annotations

import inspect
import os
from contextlib import contextmanager
from functools import wraps

try:
    from unsloth import FastLanguageModel as _BOOTSTRAP_FAST_LANGUAGE_MODEL
except ImportError:
    _BOOTSTRAP_FAST_LANGUAGE_MODEL = None

from retrain.local_train_helper import LocalTrainHelper

REQUIRED_FROM_PRETRAINED_PARAMS = frozenset(
    {
        "model_name",
        "max_seq_length",
        "dtype",
        "load_in_4bit",
        "load_in_8bit",
        "load_in_16bit",
        "full_finetuning",
        "device_map",
        "fast_inference",
        "gpu_memory_utilization",
        "float8_kv_cache",
        "max_lora_rank",
        "use_gradient_checkpointing",
        "trust_remote_code",
        "use_exact_model_name",
        "offload_embedding",
        "random_state",
        "unsloth_tiled_mlp",
        "text_only",
    }
)

REQUIRED_GET_PEFT_MODEL_PARAMS = frozenset(
    {
        "r",
        "target_modules",
        "lora_alpha",
        "lora_dropout",
        "bias",
        "use_gradient_checkpointing",
        "random_state",
        "max_seq_length",
        "use_rslora",
    }
)


def validate_fast_language_model_api(fast_language_model) -> dict[str, object]:
    """Validate the installed Unsloth FastLanguageModel API used by retrain."""
    from_pretrained = inspect.signature(fast_language_model.from_pretrained)
    get_peft_model = inspect.signature(fast_language_model.get_peft_model)
    from_params = set(from_pretrained.parameters)
    peft_params = set(get_peft_model.parameters)
    missing_from = sorted(REQUIRED_FROM_PRETRAINED_PARAMS - from_params)
    missing_peft = sorted(REQUIRED_GET_PEFT_MODEL_PARAMS - peft_params)
    missing_modes = [
        name
        for name in ("for_inference", "for_training")
        if not callable(getattr(fast_language_model, name, None))
    ]
    if missing_from or missing_peft or missing_modes:
        raise RuntimeError(
            "Installed Unsloth FastLanguageModel API is incompatible: "
            f"from_pretrained missing={missing_from}, "
            f"get_peft_model missing={missing_peft}, "
            f"mode helpers missing={missing_modes}"
        )
    return {
        "from_pretrained_params": sorted(from_params),
        "get_peft_model_params": sorted(peft_params),
    }


def _validate_load_mode(
    *,
    load_in_4bit: bool,
    load_in_8bit: bool,
    load_in_16bit: bool,
    full_finetuning: bool,
) -> None:
    active = sum(
        int(value)
        for value in (
            load_in_4bit,
            load_in_8bit,
            load_in_16bit,
            full_finetuning,
        )
    )
    if active > 1:
        raise ValueError(
            "Unsloth backend accepts only one active precision/training mode: "
            "load_in_4bit, load_in_8bit, load_in_16bit, or full_finetuning."
        )
    if full_finetuning:
        raise ValueError(
            "Unsloth full_finetuning is not supported by retrain's adapter-based "
            "checkpoint/sync contract yet. Use LoRA or QLoRA."
        )


def _module_global(module, name: str):
    forward = getattr(type(module), "forward", None)
    globals_dict = getattr(forward, "__globals__", {})
    return globals_dict.get(name)


@contextmanager
def _temporary_env(name: str, value: str | None):
    if value is None:
        yield
        return
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


class UnslothTrainHelper(LocalTrainHelper):
    """Local retrain backend using Unsloth for model loading/patching."""

    def __init__(
        self,
        model_name,
        adapter_path,
        devices,
        lora_rank=32,
        engine_type="pytorch",
        inference_url="",
        *,
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        load_in_16bit=False,
        full_finetuning=False,
        fast_inference=False,
        gpu_memory_utilization=0.5,
        float8_kv_cache=False,
        max_lora_rank=64,
        use_gradient_checkpointing="unsloth",
        device_map="retrain",
        trust_remote_code=False,
        use_exact_model_name=False,
        offload_embedding=False,
        unsloth_tiled_mlp=False,
        unsloth_tiled_mlp_mode="",
        text_only=False,
        use_rslora=False,
        random_state=3407,
        qwen35_gated_delta_chunk_size="auto",
        **kwargs,
    ):
        _validate_load_mode(
            load_in_4bit=bool(load_in_4bit),
            load_in_8bit=bool(load_in_8bit),
            load_in_16bit=bool(load_in_16bit),
            full_finetuning=bool(full_finetuning),
        )
        self.unsloth_max_seq_length = max(1, int(max_seq_length))
        self.unsloth_load_in_4bit = bool(load_in_4bit)
        self.unsloth_load_in_8bit = bool(load_in_8bit)
        self.unsloth_load_in_16bit = bool(load_in_16bit)
        self.unsloth_full_finetuning = bool(full_finetuning)
        self.unsloth_fast_inference = bool(fast_inference)
        self.unsloth_gpu_memory_utilization = float(gpu_memory_utilization)
        self.unsloth_float8_kv_cache = bool(float8_kv_cache)
        self.unsloth_max_lora_rank = max(1, int(max_lora_rank))
        self.unsloth_use_gradient_checkpointing = use_gradient_checkpointing
        self.unsloth_device_map = str(device_map or "retrain")
        self.unsloth_trust_remote_code = bool(trust_remote_code)
        self.unsloth_use_exact_model_name = bool(use_exact_model_name)
        self.unsloth_offload_embedding = bool(offload_embedding)
        self.unsloth_tiled_mlp = bool(unsloth_tiled_mlp)
        self.unsloth_tiled_mlp_mode = str(unsloth_tiled_mlp_mode or "")
        self.unsloth_text_only = bool(text_only)
        self.unsloth_use_rslora = bool(use_rslora)
        self.unsloth_random_state = int(random_state)
        self.unsloth_qwen35_gated_delta_chunk_size = qwen35_gated_delta_chunk_size
        self._unsloth_qwen35_gated_delta_patch = {
            "mode": "off",
            "chunk_size": 0,
            "patched_modules": 0,
            "shared_memory_limit": 0,
        }
        self._unsloth_fast_language_model = None
        self._unsloth_last_mode = "training"
        super().__init__(
            model_name,
            adapter_path,
            devices,
            lora_rank,
            engine_type,
            inference_url,
            **kwargs,
        )

    def _load_train_model(self, model_name, dtype, lora_rank, lora_alpha, lora_dropout):
        fast_language_model = _BOOTSTRAP_FAST_LANGUAGE_MODEL
        if fast_language_model is None:
            try:
                from unsloth import FastLanguageModel as fast_language_model
            except ImportError as exc:
                raise RuntimeError(
                    "Backend 'unsloth' requires Unsloth Core. "
                    "Install it with: uv pip install unsloth --torch-backend=auto"
                ) from exc

        validate_fast_language_model_api(fast_language_model)
        self._unsloth_fast_language_model = fast_language_model
        max_lora_rank = max(int(lora_rank), int(self.unsloth_max_lora_rank))
        tiled_mode = self.unsloth_tiled_mlp_mode.strip() or None
        with _temporary_env("UNSLOTH_TILED_MLP", tiled_mode):
            model, _tokenizer = fast_language_model.from_pretrained(
                model_name=model_name,
                max_seq_length=self.unsloth_max_seq_length,
                dtype=dtype,
                load_in_4bit=self.unsloth_load_in_4bit,
                load_in_8bit=self.unsloth_load_in_8bit,
                load_in_16bit=self.unsloth_load_in_16bit,
                full_finetuning=self.unsloth_full_finetuning,
                device_map=self._resolve_device_map(),
                fast_inference=self.unsloth_fast_inference,
                gpu_memory_utilization=self.unsloth_gpu_memory_utilization,
                float8_kv_cache=self.unsloth_float8_kv_cache,
                max_lora_rank=max_lora_rank,
                trust_remote_code=self.unsloth_trust_remote_code,
                use_exact_model_name=self.unsloth_use_exact_model_name,
                offload_embedding=self.unsloth_offload_embedding,
                use_gradient_checkpointing=(
                    self.unsloth_use_gradient_checkpointing
                    if getattr(self, "gradient_checkpointing", True)
                    else False
                ),
                random_state=self.unsloth_random_state,
                unsloth_tiled_mlp=self.unsloth_tiled_mlp,
                text_only=self.unsloth_text_only,
            )
        peft_config = self._build_peft_config(
            model,
            lora_rank,
            lora_alpha,
            lora_dropout,
        )
        model = fast_language_model.get_peft_model(
            model,
            r=lora_rank,
            target_modules=list(peft_config.target_modules or []),
            lora_alpha=peft_config.lora_alpha,
            lora_dropout=peft_config.lora_dropout,
            bias=peft_config.bias,
            use_gradient_checkpointing=(
                self.unsloth_use_gradient_checkpointing
                if getattr(self, "gradient_checkpointing", True)
                else False
            ),
            random_state=self.unsloth_random_state,
            max_seq_length=self.unsloth_max_seq_length,
            use_rslora=self.unsloth_use_rslora,
        )
        self._patch_qwen35_gated_delta_rule_if_needed(model)
        return model, peft_config

    def _count_tiled_mlp_modules(self) -> int:
        model = getattr(self, "train_model", None)
        if model is None:
            return 0
        return sum(
            1
            for module in model.modules()
            if hasattr(module, "_original_forward")
            and hasattr(module, "_unsloth_forward")
        )

    def _qwen35_gated_delta_shared_memory_limit(self) -> int:
        try:
            import torch

            if not (self.train_device.startswith("cuda") and torch.cuda.is_available()):
                return 0
            props = torch.cuda.get_device_properties(torch.device(self.train_device))
            return int(
                getattr(
                    props,
                    "shared_memory_per_block_optin",
                    getattr(props, "shared_memory_per_block", 0),
                )
                or 0
            )
        except Exception:  # noqa: BLE001 - diagnostics only.
            return 0

    def _resolve_qwen35_gated_delta_patch(self) -> tuple[str, int]:
        raw = self.unsloth_qwen35_gated_delta_chunk_size
        text = str(raw).strip().lower()
        if text in {"", "none", "off", "false", "0"}:
            return "off", 0
        if text in {"torch", "torch_fallback", "safe"}:
            return "torch", 0
        if text == "auto":
            shared_limit = self._qwen35_gated_delta_shared_memory_limit()
            # The default FLA Qwen3.5 chunk kernel needed 131072 bytes in
            # remote smoke tests; Ada consumer cards such as 4070 Ti expose
            # 101376 bytes. FLA currently hardcodes chunk_size=64 internally,
            # so route these cards to the model's torch fallback.
            if shared_limit and shared_limit < 131072:
                return "torch", 0
            return "off", 0
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "qwen35_gated_delta_chunk_size must be 'auto', 'off', 'torch', "
                "or a non-negative integer."
            ) from exc
        if value < 0:
            raise ValueError("qwen35_gated_delta_chunk_size must be non-negative.")
        return "chunk_size", value

    def _patch_qwen35_gated_delta_rule_if_needed(self, model) -> None:
        mode, chunk_size = self._resolve_qwen35_gated_delta_patch()
        shared_limit = self._qwen35_gated_delta_shared_memory_limit()
        patch = {
            "mode": mode,
            "chunk_size": int(chunk_size),
            "patched_modules": 0,
            "shared_memory_limit": int(shared_limit),
        }
        if mode == "off":
            self._unsloth_qwen35_gated_delta_patch = patch
            return

        patched = 0
        for module in model.modules():
            rule = getattr(module, "chunk_gated_delta_rule", None)
            if not callable(rule):
                continue
            if not (
                "Qwen3_5GatedDeltaNet" in type(module).__name__
                or callable(_module_global(module, "torch_chunk_gated_delta_rule"))
            ):
                continue
            if getattr(rule, "_retrain_qwen35_chunk_size", None) == chunk_size:
                patched += 1
                continue

            if mode == "torch":
                torch_rule = _module_global(module, "torch_chunk_gated_delta_rule")
                if not callable(torch_rule):
                    continue
                torch_rule._retrain_qwen35_chunk_size = 0
                module.chunk_gated_delta_rule = torch_rule
            else:
                @wraps(rule)
                def chunked_rule(
                    *args,
                    __rule=rule,
                    __chunk_size=chunk_size,
                    **kwargs,
                ):
                    kwargs.setdefault("chunk_size", __chunk_size)
                    return __rule(*args, **kwargs)

                chunked_rule._retrain_qwen35_chunk_size = chunk_size
                module.chunk_gated_delta_rule = chunked_rule
            patched += 1

        patch["mode"] = mode if patched else "not_found"
        patch["patched_modules"] = patched
        self._unsloth_qwen35_gated_delta_patch = patch

    def _resolve_device_map(self):
        value = self.unsloth_device_map.strip()
        if value in ("", "retrain"):
            return {"": self.train_device}
        if value.lower() in ("none", "null"):
            return None
        return value

    def _move_train_model_to_device(self):
        if self.unsloth_load_in_4bit or self.unsloth_load_in_8bit:
            return
        super()._move_train_model_to_device()

    def _configure_gradient_checkpointing(self):
        # Unsloth applies its gradient-checkpointing policy in get_peft_model.
        # Running the generic HF toggle afterwards can replace the Unsloth policy.
        return

    @contextmanager
    def _shared_model_sampling_cache_context(self):
        fast_language_model = self._unsloth_fast_language_model
        for_inference = getattr(fast_language_model, "for_inference", None)
        for_training = getattr(fast_language_model, "for_training", None)
        if callable(for_inference) and not getattr(self, "_external_engine", False):
            for_inference(self.train_model)
            self._unsloth_last_mode = "inference"
        try:
            with super()._shared_model_sampling_cache_context():
                yield
        finally:
            if callable(for_training) and not getattr(self, "_external_engine", False):
                for_training(self.train_model)
                self._unsloth_last_mode = "training"

    def runtime_metrics(self):
        metrics = super().runtime_metrics()
        metrics.update(
            {
                "unsloth_backend_enabled": 1,
                "unsloth_max_seq_length": self.unsloth_max_seq_length,
                "unsloth_load_in_4bit": int(self.unsloth_load_in_4bit),
                "unsloth_load_in_8bit": int(self.unsloth_load_in_8bit),
                "unsloth_load_in_16bit": int(self.unsloth_load_in_16bit),
                "unsloth_full_finetuning": int(self.unsloth_full_finetuning),
                "unsloth_fast_inference": int(self.unsloth_fast_inference),
                "unsloth_gpu_memory_utilization": self.unsloth_gpu_memory_utilization,
                "unsloth_float8_kv_cache": int(self.unsloth_float8_kv_cache),
                "unsloth_offload_embedding": int(self.unsloth_offload_embedding),
                "unsloth_device_map_retrain": int(
                    self.unsloth_device_map.strip() in ("", "retrain")
                ),
                "unsloth_tiled_mlp": int(self.unsloth_tiled_mlp),
                "unsloth_tiled_mlp_patched_modules": self._count_tiled_mlp_modules(),
                "unsloth_tiled_mlp_mode": self.unsloth_tiled_mlp_mode,
                "unsloth_text_only": int(self.unsloth_text_only),
                "unsloth_last_mode_is_inference": int(
                    self._unsloth_last_mode == "inference"
                ),
                "unsloth_qwen35_gated_delta_chunk_size": int(
                    self._unsloth_qwen35_gated_delta_patch.get("chunk_size", 0)
                ),
                "unsloth_qwen35_gated_delta_patched_modules": int(
                    self._unsloth_qwen35_gated_delta_patch.get("patched_modules", 0)
                ),
                "unsloth_qwen35_gated_delta_torch_fallback": int(
                    self._unsloth_qwen35_gated_delta_patch.get("mode") == "torch"
                ),
                "unsloth_qwen35_gated_delta_shared_memory_limit": int(
                    self._unsloth_qwen35_gated_delta_patch.get(
                        "shared_memory_limit",
                        0,
                    )
                ),
            }
        )
        return metrics
