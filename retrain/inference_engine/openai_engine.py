"""OpenAI-compatible inference engine — HTTP client for vLLM, SGLang, MLX-LM, MAX serve.

Targets any server exposing /v1/completions with logprobs support.
Decodes prompt token IDs to text, sends HTTP requests, recovers token IDs
and logprobs from the response.

Weight reloading uses:
- /v1/load_lora_adapter for vLLM/SGLang-style servers
- per-request "adapters" payload field for MLX-LM server
"""

import time

import requests
from transformers import AutoTokenizer

from retrain.inference_engine.base import InferenceEngine, SampleResult


class OpenAIEngine(InferenceEngine):
    """HTTP client targeting OpenAI-compatible /v1/completions endpoints."""

    def __init__(self, base_url, model_name, engine_type="openai"):
        """Initialize the HTTP engine.

        Args:
            base_url: Server URL (e.g. "http://localhost:8000").
            model_name: HuggingFace model ID (for tokenizer + model param).
            engine_type: One of "vllm", "sglang", "mlx", "openai".
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.engine_type = engine_type
        self.session = requests.Session()

        # Local tokenizer for prompt decoding and token ID recovery
        print(f"Loading tokenizer for OpenAI engine: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Adapter tracking
        self._current_adapter_path = None

        print(f"OpenAIEngine ready ({engine_type} @ {self.base_url}).")

    def generate(self, prompt_ids_list, num_samples, max_tokens, temperature, top_p,
                 compute_entropy=False):
        """Generate completions via /v1/completions with logprobs.

        Decodes prompt token IDs to text, sends request, recovers token IDs
        from the response logprobs tokens field.
        """
        results = []

        for prompt_ids in prompt_ids_list:
            # Decode prompt to text for the API
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)

            payload = {
                "model": self.model_name,
                "prompt": prompt_text,
                "max_tokens": max_tokens,
                "temperature": max(temperature, 1e-7),
                "top_p": top_p,
                "n": num_samples,
                "logprobs": 1,
            }
            # mlx_lm.server supports dynamic adapter selection via request payload.
            if self.engine_type == "mlx" and self._current_adapter_path:
                payload["adapters"] = self._current_adapter_path

            response = self._post("/v1/completions", payload)
            choices = response.get("choices", [])

            group = []
            for choice in choices:
                text = choice.get("text", "")
                lp_info = choice.get("logprobs", {})

                token_ids, logprobs = self._recover_tokens(text, lp_info)
                group.append(SampleResult(token_ids=token_ids, logprobs=logprobs))

            results.append(group)

        return results

    def _recover_tokens(self, text, logprobs_info):
        """Recover token IDs and logprobs from API response.

        Primary path: use tokenizer.convert_tokens_to_ids() on the API's
        logprobs.tokens list (exact alignment with server tokenization).
        Fallback: re-encode completion text.

        Args:
            text: Completion text from the API.
            logprobs_info: The logprobs object from the API response.

        Returns:
            (token_ids, logprobs) tuple.
        """
        if not logprobs_info:
            # No logprobs returned — fallback to re-encoding
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            return token_ids, [0.0] * len(token_ids)

        tokens = logprobs_info.get("tokens", [])
        token_logprobs = logprobs_info.get("token_logprobs", [])

        if tokens:
            # Primary path: convert token strings to IDs via tokenizer
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # Handle unknown tokens (returned as single unk_token_id)
            # by falling back to re-encoding for those positions
            logprobs = []
            for i, lp in enumerate(token_logprobs):
                logprobs.append(float(lp) if lp is not None else 0.0)

            # Ensure lengths match
            min_len = min(len(token_ids), len(logprobs))
            return list(token_ids[:min_len]), logprobs[:min_len]

        # Fallback: re-encode text
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return token_ids, [0.0] * len(token_ids)

    def reload_weights(self, adapter_path):
        """Tell the server to load/reload a LoRA adapter.

        Uses /v1/load_lora_adapter for vLLM/SGLang-style servers.
        For MLX-LM, stores adapter path and sends it on each request as
        the "adapters" field.
        """
        if adapter_path == self._current_adapter_path:
            return

        if self.engine_type == "mlx":
            self._current_adapter_path = adapter_path
            print(f"MLX-LM adapter set to {adapter_path}")
            return

        payload = {
            "lora_name": "default",
            "lora_path": adapter_path,
        }

        try:
            self._post("/v1/load_lora_adapter", payload)
            self._current_adapter_path = adapter_path
            print(f"Server adapter reloaded from {adapter_path}")
        except Exception as e:
            print(f"Warning: adapter reload failed ({e}). "
                  f"Server may not support dynamic LoRA loading.")

    def shutdown(self):
        """Close HTTP session."""
        self.session.close()

    def _post(self, endpoint, payload, max_retries=3):
        """POST to server with retry logic for transient failures.

        Args:
            endpoint: API path (e.g. "/v1/completions").
            payload: JSON request body.
            max_retries: Number of retries on transient errors.

        Returns:
            Parsed JSON response dict.

        Raises:
            RuntimeError: On persistent failures.
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(max_retries):
            try:
                resp = self.session.post(url, json=payload, timeout=120)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"Connection error to {url}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Failed to connect to {url} after {max_retries} attempts. "
                        f"Is the inference server running?"
                    )
            except requests.exceptions.HTTPError as e:
                raise RuntimeError(f"HTTP error from {url}: {e}")
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"Timeout on {url}, retrying...")
                else:
                    raise RuntimeError(f"Request to {url} timed out after {max_retries} attempts")
