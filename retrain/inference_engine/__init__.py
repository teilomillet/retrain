"""Pluggable inference engines for retrain.

Engines handle sampling (completions + logprobs) independently from
the PyTorch/PEFT training model. Available engines:

- pytorch:   Local PyTorch model.generate (default, works everywhere incl. Mac)
- max:       MAX inference — auto-detects mode:
               with --inference-url: HTTP client to `max serve` (production, multi-GPU)
               without URL: in-process MAX pipeline (dev/testing)
- vllm:      vLLM OpenAI-compatible server
- sglang:    SGLang OpenAI-compatible server
- trtllm:    TensorRT-LLM OpenAI-compatible server
- mlx:       MLX-LM OpenAI-compatible server (Apple Silicon)
- openai:    Any OpenAI-compatible endpoint
"""

from retrain.inference_engine.base import InferenceEngine, SampleResult


def create_engine(
    engine_type,
    model_name,
    device,
    peft_config,
    dtype,
    inference_url="",
    existing_model=None,
    sample_use_cache=True,
    prefix_caching=True,
    attention_kernel="default",
    liger_kernel=True,
    sample_kv_quantization="off",
    sample_oscar_options=None,
):
    """Factory: create the right InferenceEngine based on engine_type.

    Args:
        engine_type: One of "pytorch", "max", "vllm", "sglang", "trtllm", "mlx", "openai".
        model_name: HuggingFace model ID.
        device: Torch device string for local engines.
        peft_config: LoraConfig for PyTorchEngine to wrap a freshly loaded
            model. Must be None when existing_model is provided.
        dtype: Model dtype (used by PyTorchEngine).
        inference_url: Server URL for server-based engines.
        existing_model: Already wrapped PyTorch/PEFT model to reuse for
            single-model local inference.
        prefix_caching: Whether local PyTorch should reuse exact-prefix KV
            cache entries within a rollout sampling phase.

    Returns:
        An InferenceEngine instance.
    """
    if engine_type == "pytorch":
        from retrain.inference_engine.pytorch_engine import PyTorchEngine

        return PyTorchEngine(
            model_name,
            device,
            peft_config,
            dtype,
            existing_model=existing_model,
            sample_use_cache=sample_use_cache,
            prefix_caching=prefix_caching,
            attention_kernel=attention_kernel,
            liger_kernel=liger_kernel,
            sample_kv_quantization=sample_kv_quantization,
            sample_oscar_options=sample_oscar_options,
        )

    elif engine_type == "max":
        from retrain.inference_engine.max_engine import create_max_engine

        return create_max_engine(model_name, inference_url)

    elif engine_type in ("vllm", "sglang", "trtllm", "mlx", "openai"):
        from retrain.inference_engine.openai_engine import OpenAIEngine

        if not inference_url:
            defaults = {
                "vllm": "http://localhost:8000",
                "sglang": "http://localhost:30000",
                "trtllm": "http://localhost:31000",
                "mlx": "http://localhost:8080",
                "openai": "http://localhost:8000",
            }
            inference_url = defaults[engine_type]

        return OpenAIEngine(
            base_url=inference_url,
            model_name=model_name,
            engine_type=engine_type,
        )

    else:
        raise ValueError(
            f"Unknown inference engine: {engine_type!r}. "
            f"Expected: pytorch, max, vllm, sglang, trtllm, mlx, openai"
        )


__all__ = ["InferenceEngine", "SampleResult", "create_engine"]
