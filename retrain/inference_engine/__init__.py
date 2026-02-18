"""Pluggable inference engines for retrain.

Engines handle sampling (completions + logprobs) independently from
the PyTorch/PEFT training model. Available engines:

- pytorch:   Local PyTorch model.generate (default, works everywhere incl. Mac)
- max:       MAX inference — auto-detects mode:
               with --inference-url: HTTP client to `max serve` (production, multi-GPU)
               without URL: in-process MAX pipeline (dev/testing)
- vllm:      vLLM OpenAI-compatible server
- sglang:    SGLang OpenAI-compatible server
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
):
    """Factory: create the right InferenceEngine based on engine_type.

    Args:
        engine_type: One of "pytorch", "max", "vllm", "sglang", "openai".
        model_name: HuggingFace model ID.
        device: Torch device string for local engines.
        peft_config: LoraConfig (used by PyTorchEngine).
        dtype: Model dtype (used by PyTorchEngine).
        inference_url: Server URL for server-based engines.

    Returns:
        An InferenceEngine instance.
    """
    if engine_type == "pytorch":
        from retrain.inference_engine.pytorch_engine import PyTorchEngine

        return PyTorchEngine(model_name, device, peft_config, dtype)

    elif engine_type == "max":
        from retrain.inference_engine.max_engine import create_max_engine

        return create_max_engine(model_name, inference_url)

    elif engine_type in ("vllm", "sglang", "openai"):
        from retrain.inference_engine.openai_engine import OpenAIEngine

        if not inference_url:
            defaults = {
                "vllm": "http://localhost:8000",
                "sglang": "http://localhost:30000",
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
            f"Expected: pytorch, max, vllm, sglang, openai"
        )


__all__ = ["InferenceEngine", "SampleResult", "create_engine"]
