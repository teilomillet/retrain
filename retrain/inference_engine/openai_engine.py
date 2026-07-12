"""OpenAI-compatible inference engine — HTTP client for vLLM, SGLang, TensorRT-LLM, MLX-LM, MAX serve.

Targets any server exposing /v1/completions with logprobs support.
For vLLM/SGLang, sends prompt token IDs directly to avoid client-side
decode + server-side re-tokenization. Other OpenAI-compatible servers keep the
text prompt fallback path.

Weight reloading uses:
- /v1/load_lora_adapter for vLLM
- /load_lora_adapter for SGLang
- per-request "lora_request" payload field for TensorRT-LLM
- per-request "adapters" payload field for MLX-LM server
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol, cast

import requests
from transformers import AutoTokenizer

from retrain.inference_engine.base import InferenceEngine, SampleResult


class _Tokenizer(Protocol):
    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str: ...

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]: ...

    def convert_tokens_to_ids(
        self,
        tokens: Sequence[str],
    ) -> int | list[int | None]: ...


_TOKEN_NATIVE_ENGINES = frozenset({"vllm", "sglang"})
_LIVE_LORA_NAME = "default"
_LIVE_LORA_INT_ID = 0


class OpenAIEngine(InferenceEngine):
    """HTTP client targeting OpenAI-compatible /v1/completions endpoints."""

    def __init__(self, base_url: str, model_name: str, engine_type: str = "openai") -> None:
        """Initialize the HTTP engine.

        Args:
            base_url: Server URL (e.g. "http://localhost:8000").
            model_name: HuggingFace model ID (for tokenizer + model param).
            engine_type: One of "vllm", "sglang", "trtllm", "mlx", "openai".
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.engine_type = engine_type
        self.session = requests.Session()

        # Local tokenizer for prompt decoding and token ID recovery
        print(f"Loading tokenizer for OpenAI engine: {model_name}...")
        self.tokenizer = cast(_Tokenizer, AutoTokenizer.from_pretrained(model_name))

        # Adapter tracking
        self._current_adapter_path: str | None = None
        self._prompt_text_cache: dict[tuple[int, ...], str] = {}
        self._prompt_decode_calls = 0
        self._prompt_cache_hits = 0
        self._token_prompt_calls = 0
        self._token_prompt_fallbacks = 0
        self._adapter_reload_calls = 0
        self._adapter_reload_failures = 0
        self._adapter_reload_skips = 0

        print(f"OpenAIEngine ready ({engine_type} @ {self.base_url}).")

    def generate(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        compute_entropy: bool = False,
    ) -> list[list[SampleResult]]:
        """Generate completions via /v1/completions with logprobs.

        vLLM/SGLang receive prompt token IDs directly. Text-only servers receive
        decoded prompts with a local decode cache.
        """
        results: list[list[SampleResult]] = []

        for prompt_ids in prompt_ids_list:
            prompt_payload, token_native = self._prompt_payload(prompt_ids)

            payload: dict[str, object] = {
                "model": self._request_model_name(),
                "prompt": prompt_payload,
                "max_tokens": max_tokens,
                "temperature": max(temperature, 1e-7),
                "top_p": top_p,
                "n": num_samples,
                "logprobs": 1,
            }
            # mlx_lm.server supports dynamic adapter selection via request payload.
            if self.engine_type == "mlx" and self._current_adapter_path:
                payload["adapters"] = self._current_adapter_path
            if self.engine_type == "trtllm" and self._current_adapter_path:
                payload["lora_request"] = {
                    "lora_name": _LIVE_LORA_NAME,
                    "lora_int_id": _LIVE_LORA_INT_ID,
                    "lora_path": self._current_adapter_path,
                }

            try:
                response = self._post("/v1/completions", payload)
            except RuntimeError as exc:
                if not token_native or not self._is_token_prompt_rejection(exc):
                    raise
                self._token_prompt_fallbacks += 1
                payload["prompt"] = self._prompt_text(prompt_ids)
                response = self._post("/v1/completions", payload)
            group: list[SampleResult] = []
            for choice in _response_choices(response):
                raw_text = choice.get("text", "")
                text = raw_text if isinstance(raw_text, str) else ""
                raw_finish_reason = choice.get("finish_reason")
                finish_reason = raw_finish_reason if isinstance(raw_finish_reason, str) else None
                logprobs_info = choice.get("logprobs", {})

                token_ids, logprobs = self._recover_tokens(
                    prompt_ids,
                    text,
                    logprobs_info,
                    choice,
                )
                group.append(
                    SampleResult(
                        token_ids=token_ids,
                        logprobs=logprobs,
                        finish_reason=finish_reason,
                    )
                )

            results.append(group)

        return results

    def _request_model_name(self) -> str:
        if self.engine_type == "vllm" and self._current_adapter_path:
            return _LIVE_LORA_NAME
        if self.engine_type == "sglang" and self._current_adapter_path:
            return f"{self.model_name}:{_LIVE_LORA_NAME}"
        return self.model_name

    def _prompt_payload(self, prompt_ids: list[int]) -> tuple[list[int] | str, bool]:
        if self.engine_type in _TOKEN_NATIVE_ENGINES:
            self._token_prompt_calls += 1
            return list(prompt_ids), True
        return self._prompt_text(prompt_ids), False

    def _prompt_text(self, prompt_ids: Sequence[int]) -> str:
        """Decode prompt ids to text once per unique prompt."""
        key = tuple(prompt_ids)
        prompt_text = self._prompt_text_cache.get(key)
        if prompt_text is None:
            self._prompt_decode_calls += 1
            prompt_text = self.tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
            )
            self._prompt_text_cache[key] = prompt_text
        else:
            self._prompt_cache_hits += 1
        return prompt_text

    def performance_counters(self) -> dict[str, int]:
        """Return cumulative prompt-decode cache counters."""
        return {
            "engine_prompt_decode_calls": self._prompt_decode_calls,
            "engine_prompt_cache_hits": self._prompt_cache_hits,
            "engine_prompt_cache_size": len(self._prompt_text_cache),
            "engine_token_prompt_calls": self._token_prompt_calls,
            "engine_token_prompt_fallbacks": self._token_prompt_fallbacks,
            "engine_token_native_prompt_enabled": int(
                self.engine_type in _TOKEN_NATIVE_ENGINES
            ),
            "engine_adapter_reload_calls": self._adapter_reload_calls,
            "engine_adapter_reload_failures": self._adapter_reload_failures,
            "engine_adapter_reload_skips": self._adapter_reload_skips,
        }

    @staticmethod
    def _is_token_prompt_rejection(exc: RuntimeError) -> bool:
        text = str(exc)
        return "400" in text or "422" in text

    def _recover_tokens(
        self,
        prompt_ids: list[int],
        text: str,
        logprobs_info: object,
        choice: dict[str, object] | None = None,
    ) -> tuple[list[int], list[float]]:
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
        if not isinstance(logprobs_info, dict):
            # No logprobs returned — fallback to re-encoding
            direct_ids = None if choice is None else self._choice_token_ids(choice)
            if isinstance(direct_ids, list):
                return direct_ids, [0.0] * len(direct_ids)
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            return token_ids, [0.0] * len(token_ids)

        info = cast(dict[str, object], logprobs_info)
        content = info.get("content", [])
        if isinstance(content, list):
            ids: list[int] = []
            logprobs: list[float] = []
            for raw_item in content:
                if not isinstance(raw_item, dict):
                    continue
                item = cast(dict[str, object], raw_item)
                logprob = _float_value(item.get("logprob"))
                if logprob is None:
                    continue
                token_id = _int_value(item.get("token_id"))
                if token_id is None:
                    ids = []
                    break
                ids.append(token_id)
                logprobs.append(logprob)
            if ids and len(ids) == len(logprobs):
                return ids, logprobs

        tokens = info.get("tokens", [])
        token_logprobs = info.get("token_logprobs", [])
        direct_ids = self._choice_token_ids(info)
        if direct_ids is None and choice is not None:
            direct_ids = self._choice_token_ids(choice)
        if isinstance(direct_ids, list) and isinstance(token_logprobs, list):
            logprobs = [_float_value(lp) or 0.0 for lp in token_logprobs]
            return self._align_ids_and_logprobs(direct_ids, logprobs, prompt_ids)

        token_strings: list[str] | None = None
        if (
            isinstance(tokens, list)
            and tokens
            and all(isinstance(token, str) for token in tokens)
        ):
            token_strings = cast(list[str], tokens)
        if token_strings is not None:
            # Primary path: convert token strings to IDs via tokenizer
            token_ids = self.tokenizer.convert_tokens_to_ids(token_strings)

            # Handle unknown tokens (returned as single unk_token_id)
            # by falling back to re-encoding for those positions
            logprobs: list[float] = []
            if isinstance(token_logprobs, list):
                for lp in token_logprobs:
                    logprobs.append(_float_value(lp) or 0.0)

            if isinstance(token_ids, int):
                token_ids = [token_ids]
            converted_ids: list[int] | None = None
            if isinstance(token_ids, list) and len(token_ids) == len(token_strings):
                maybe_ids = [
                    token_id for token_id in token_ids if isinstance(token_id, int)
                ]
                if len(maybe_ids) == len(token_strings):
                    converted_ids = maybe_ids
            if converted_ids is None:
                encoded_ids = self.tokenizer.encode(text, add_special_tokens=False)
                min_len = min(len(encoded_ids), len(logprobs))
                return list(encoded_ids[:min_len]), logprobs[:min_len]

            # Ensure lengths match
            min_len = min(len(converted_ids), len(logprobs))
            return list(converted_ids[:min_len]), logprobs[:min_len]

        # Fallback: re-encode text
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return token_ids, [0.0] * len(token_ids)

    @staticmethod
    def _choice_token_ids(payload: dict[str, object]) -> list[int] | None:
        raw_ids = payload.get("completion_ids") or payload.get("token_ids")
        if isinstance(raw_ids, list):
            ids: list[int] = []
            for raw_id in raw_ids:
                token_id = _int_value(raw_id)
                if token_id is None:
                    return None
                ids.append(token_id)
            return ids
        return None

    @staticmethod
    def _align_ids_and_logprobs(
        ids: list[int],
        logprobs: list[float],
        prompt_ids: list[int],
    ) -> tuple[list[int], list[float]]:
        if len(ids) > len(logprobs) and len(logprobs) > 0:
            ids = ids[-len(logprobs):]
        if len(ids) >= len(prompt_ids) and ids[:len(prompt_ids)] == prompt_ids:
            ids = ids[len(prompt_ids):]
            if len(logprobs) > len(ids):
                logprobs = logprobs[-len(ids):]
        n = min(len(ids), len(logprobs))
        return list(ids[:n]), list(logprobs[:n])

    def reload_weights(self, adapter_path: str) -> None:
        """Tell the server to load/reload a LoRA adapter.

        Uses the backend-specific LoRA load route for vLLM/SGLang servers.
        For TensorRT-LLM and MLX-LM, stores adapter path and sends it on each
        request using the server-specific per-request adapter field.
        """
        if adapter_path == self._current_adapter_path and self.engine_type == "mlx":
            self._adapter_reload_skips += 1
            return

        if self.engine_type == "mlx":
            self._current_adapter_path = adapter_path
            print(f"MLX-LM adapter set to {adapter_path}")
            return

        if self.engine_type == "trtllm":
            self._adapter_reload_calls += 1
            self._current_adapter_path = adapter_path
            print(f"TensorRT-LLM adapter set for per-request LoRA: {adapter_path}")
            return

        payload: dict[str, object] = {
            "lora_name": _LIVE_LORA_NAME,
            "lora_path": adapter_path,
        }
        if self.engine_type == "vllm":
            payload["load_inplace"] = True

        self._adapter_reload_calls += 1
        try:
            if self.engine_type == "sglang" and self._current_adapter_path:
                self._unload_sglang_adapter()
            self._post(self._adapter_reload_endpoint(), payload, expect_json=False)
            self._current_adapter_path = adapter_path
            print(f"Server adapter reloaded from {adapter_path}")
        except Exception as e:
            self._adapter_reload_failures += 1
            print(f"Warning: adapter reload failed ({e}). "
                  f"Server may not support dynamic LoRA loading.")

    def _adapter_reload_endpoint(self) -> str:
        if self.engine_type == "sglang":
            return "/load_lora_adapter"
        return "/v1/load_lora_adapter"

    def _unload_sglang_adapter(self) -> None:
        payload: dict[str, object] = {"lora_name": _LIVE_LORA_NAME}
        self._post("/unload_lora_adapter", payload, expect_json=False)

    def shutdown(self) -> None:
        """Close HTTP session."""
        self.session.close()

    def _post(
        self,
        endpoint: str,
        payload: dict[str, object],
        max_retries: int = 3,
        expect_json: bool = True,
    ) -> dict[str, object]:
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
                if not expect_json:
                    try:
                        return cast(dict[str, object], resp.json())
                    except ValueError:
                        return {"text": resp.text}
                return cast(dict[str, object], resp.json())
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
                body = getattr(e.response, "text", "") if e.response is not None else ""
                detail = f": {body}" if body else ""
                raise RuntimeError(f"HTTP error from {url}: {e}{detail}")
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"Timeout on {url}, retrying...")
                else:
                    raise RuntimeError(
                        f"Request to {url} timed out after {max_retries} attempts"
                    )
        raise RuntimeError(f"Request to {url} was not attempted")


def _float_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _response_choices(response: dict[str, object]) -> list[dict[str, object]]:
    raw_choices = response.get("choices", [])
    if not isinstance(raw_choices, list):
        raise RuntimeError("OpenAI-compatible response field 'choices' must be a list")
    choices: list[dict[str, object]] = []
    for index, raw_choice in enumerate(raw_choices):
        if not isinstance(raw_choice, dict):
            raise RuntimeError(
                "OpenAI-compatible response choices must be objects "
                f"(choice {index} was {type(raw_choice).__name__})"
            )
        choices.append(cast(dict[str, object], raw_choice))
    return choices
