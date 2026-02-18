"""MAX inference engine — smart auto-detect between in-process and server.

Two modes based on whether inference_url is provided:

1. Server mode (inference_url set): HTTP client to a `max serve` instance.
   This is the production path — max serve handles tensor parallelism,
   continuous batching, and GPU management across multiple GPUs.

2. In-process mode (no URL): Uses MAX's pipeline API directly for
   inference with logprobs. Good for single-GPU development/testing.

Both modes return full token IDs + per-token logprobs.
"""

import asyncio
import uuid

from retrain.inference_engine.base import InferenceEngine, SampleResult


def create_max_engine(model_name, inference_url=""):
    """Factory: create the right MAXEngine variant.

    Args:
        model_name: HuggingFace model ID.
        inference_url: If set, use HTTP client to max serve. If empty, in-process.

    Returns:
        MAXServeEngine or MAXLocalEngine.
    """
    if inference_url:
        return MAXServeEngine(model_name, inference_url)
    else:
        return MAXLocalEngine(model_name)


class MAXServeEngine(InferenceEngine):
    """HTTP client to a `max serve` instance (OpenAI-compatible API).

    Production path for multi-GPU: max serve handles tensor parallelism
    and continuous batching. Wraps OpenAIEngine with MAX-specific defaults.
    """

    def __init__(self, model_name, base_url):
        from retrain.inference_engine.openai_engine import OpenAIEngine

        self.model_name = model_name
        self._engine = OpenAIEngine(
            base_url=base_url,
            model_name=model_name,
            engine_type="max",
        )
        print(f"MAXServeEngine ready ({model_name} @ {base_url}).")

    def generate(self, prompt_ids_list, num_samples, max_tokens, temperature, top_p):
        return self._engine.generate(
            prompt_ids_list, num_samples, max_tokens, temperature, top_p
        )

    def reload_weights(self, adapter_path):
        self._engine.reload_weights(adapter_path)

    def shutdown(self):
        self._engine.shutdown()


class MAXLocalEngine(InferenceEngine):
    """In-process MAX inference via the pipeline API.

    Uses MAX's lower-level pipeline to get TextGenerationOutput with
    token IDs and per-token logprobs. The high-level LLM.generate()
    only returns strings, so we go one layer deeper.

    Good for single-GPU dev/testing. For multi-GPU production, use
    MAXServeEngine (provide --inference-url).
    """

    def __init__(self, model_name):
        from max.entrypoints.llm import LLM
        from max.pipelines import PipelineConfig
        from transformers import AutoTokenizer

        self.model_name = model_name

        print(f"Loading MAX inference engine (in-process): {model_name}...")
        config = PipelineConfig(model_path=model_name)
        self._llm = LLM(config)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"MAXLocalEngine ready ({model_name}).")

    def generate(self, prompt_ids_list, num_samples, max_tokens, temperature, top_p):
        """Generate completions with per-token logprobs via MAX pipeline."""
        from max.interfaces.pipeline_variants.text_generation import (
            TextGenerationRequest,
        )
        from max.interfaces.sampling_params import SamplingParams

        results = []

        for prompt_ids in prompt_ids_list:
            prompt_text = self._tokenizer.decode(prompt_ids, skip_special_tokens=False)

            requests = []
            for _ in range(num_samples):
                req = TextGenerationRequest(
                    model_name=self.model_name,
                    prompt=prompt_text,
                    messages=[],
                    images=[],
                    tools=None,
                    response_format=None,
                    logprobs=1,
                    echo=False,
                    sampling_params=SamplingParams(
                        max_new_tokens=max_tokens,
                        temperature=max(temperature, 1e-7),
                        top_p=top_p,
                    ),
                )
                requests.append(req)

            group = []
            for req in requests:
                output = self._generate_with_logprobs(req)
                token_ids = output.tokens if output.tokens else []

                logprobs = []
                if output.log_probabilities:
                    for lp in output.log_probabilities:
                        if lp and lp.token_log_probabilities:
                            logprobs.extend(lp.token_log_probabilities)

                min_len = min(len(token_ids), len(logprobs))
                group.append(SampleResult(
                    token_ids=list(token_ids[:min_len]),
                    logprobs=logprobs[:min_len],
                ))
            results.append(group)

        return results

    def _generate_with_logprobs(self, request):
        """Submit a request through the pipeline and return full output."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._async_generate(request))
        finally:
            loop.close()

    async def _async_generate(self, request):
        """Collect full TextGenerationOutput with tokens + logprobs."""
        from max.interfaces.pipeline_variants.text_generation import (
            TextGenerationOutput,
        )

        chunks = []
        async for chunk in self._llm._pipeline.generate_async(request):
            chunks.append(chunk)

        if chunks:
            return TextGenerationOutput.merge(chunks)

        return TextGenerationOutput(
            request_id=str(uuid.uuid4()),
            tokens=[],
            final_status=None,
            log_probabilities=None,
        )

    def reload_weights(self, adapter_path):
        """In-process MAX doesn't support dynamic LoRA reload yet."""
        print(f"Warning: in-process MAX adapter reload not yet supported. "
              f"Use --inference-url to point at max serve for dynamic LoRA.")

    def shutdown(self):
        del self._llm
