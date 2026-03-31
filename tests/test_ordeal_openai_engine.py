"""Ordeal fault injection tests for retrain.inference_engine.openai_engine.

Tests the _post() retry logic and generate() response parsing under
network faults: connection errors, timeouts, HTTP errors, rate limiting.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from retrain.inference_engine.openai_engine import OpenAIEngine


@pytest.fixture
def engine():
    """Create an OpenAIEngine with a mocked tokenizer."""
    with patch(
        "retrain.inference_engine.openai_engine.AutoTokenizer"
    ) as mock_tok_cls:
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "test prompt"
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda tokens: list(
            range(len(tokens))
        )
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        eng = OpenAIEngine("http://fake:8000", "test-model", engine_type="vllm")
    return eng


# ═══════════════════════════════════════════
# _post() Retry Logic
# ═══════════════════════════════════════════


class TestPostRetryLogic:
    def test_connection_error_retries_then_fails(self, engine: OpenAIEngine) -> None:
        """ConnectionError retries max_retries times, then raises RuntimeError."""
        engine.session.post = MagicMock(
            side_effect=requests.exceptions.ConnectionError("refused")
        )
        with pytest.raises(RuntimeError, match="Failed to connect"):
            engine._post("/v1/completions", {}, max_retries=2)
        assert engine.session.post.call_count == 2

    def test_connection_error_recovers_on_retry(
        self, engine: OpenAIEngine
    ) -> None:
        """ConnectionError on first attempt, success on second."""
        ok_response = MagicMock()
        ok_response.json.return_value = {"choices": []}
        ok_response.raise_for_status.return_value = None

        engine.session.post = MagicMock(
            side_effect=[
                requests.exceptions.ConnectionError("refused"),
                ok_response,
            ]
        )
        result = engine._post("/v1/completions", {}, max_retries=3)
        assert result == {"choices": []}
        assert engine.session.post.call_count == 2

    def test_timeout_retries_then_fails(self, engine: OpenAIEngine) -> None:
        """Timeout retries max_retries times, then raises RuntimeError."""
        engine.session.post = MagicMock(
            side_effect=requests.exceptions.Timeout("120s")
        )
        with pytest.raises(RuntimeError, match="timed out"):
            engine._post("/v1/completions", {}, max_retries=3)
        assert engine.session.post.call_count == 3

    def test_timeout_recovers_on_retry(self, engine: OpenAIEngine) -> None:
        """Timeout on first attempt, success on second."""
        ok_response = MagicMock()
        ok_response.json.return_value = {"ok": True}
        ok_response.raise_for_status.return_value = None

        engine.session.post = MagicMock(
            side_effect=[requests.exceptions.Timeout("120s"), ok_response]
        )
        result = engine._post("/v1/completions", {}, max_retries=3)
        assert result == {"ok": True}

    def test_http_error_no_retry(self, engine: OpenAIEngine) -> None:
        """HTTPError (4xx/5xx) fails immediately, no retry."""
        bad_response = MagicMock()
        bad_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )
        engine.session.post = MagicMock(return_value=bad_response)

        with pytest.raises(RuntimeError, match="HTTP error"):
            engine._post("/v1/completions", {}, max_retries=3)
        # Should NOT retry on HTTP errors
        assert engine.session.post.call_count == 1

    def test_success_on_first_attempt(self, engine: OpenAIEngine) -> None:
        """Clean success returns parsed JSON."""
        ok_response = MagicMock()
        ok_response.json.return_value = {"choices": [{"text": "hello"}]}
        ok_response.raise_for_status.return_value = None
        engine.session.post = MagicMock(return_value=ok_response)

        result = engine._post("/v1/completions", {})
        assert result == {"choices": [{"text": "hello"}]}
        assert engine.session.post.call_count == 1


# ═══════════════════════════════════════════
# generate() Response Parsing
# ═══════════════════════════════════════════


class TestGenerateResponseParsing:
    def test_empty_choices(self, engine: OpenAIEngine) -> None:
        """Response with empty choices list returns empty results."""
        with patch.object(
            engine, "_post", return_value={"choices": []}
        ):
            results = engine.generate([[1, 2, 3]], num_samples=1, max_tokens=10,
                                      temperature=0.7, top_p=0.95)
        assert results == [[]]

    def test_missing_choices_key(self, engine: OpenAIEngine) -> None:
        """Response missing 'choices' returns empty results."""
        with patch.object(engine, "_post", return_value={"data": []}):
            results = engine.generate([[1, 2, 3]], num_samples=1, max_tokens=10,
                                      temperature=0.7, top_p=0.95)
        assert results == [[]]

    def test_valid_response_with_logprobs(self, engine: OpenAIEngine) -> None:
        """Valid response with logprobs returns SampleResults."""
        response = {
            "choices": [
                {
                    "text": "world",
                    "logprobs": {
                        "tokens": ["wor", "ld"],
                        "token_logprobs": [-0.5, -0.3],
                    },
                }
            ]
        }
        with patch.object(engine, "_post", return_value=response):
            results = engine.generate([[1, 2, 3]], num_samples=1, max_tokens=10,
                                      temperature=0.7, top_p=0.95)
        assert len(results) == 1
        assert len(results[0]) == 1
        assert len(results[0][0].logprobs) == 2

    def test_missing_logprobs_falls_back(self, engine: OpenAIEngine) -> None:
        """Response without logprobs falls back to re-encoding."""
        response = {"choices": [{"text": "hello world"}]}
        with patch.object(engine, "_post", return_value=response):
            results = engine.generate([[1, 2, 3]], num_samples=1, max_tokens=10,
                                      temperature=0.7, top_p=0.95)
        assert len(results) == 1
        assert len(results[0]) == 1
        # Fallback uses tokenizer.encode which returns [1, 2, 3]
        assert results[0][0].token_ids == [1, 2, 3]

    def test_none_logprob_values(self, engine: OpenAIEngine) -> None:
        """None logprob values (first token) become 0.0."""
        response = {
            "choices": [
                {
                    "text": "hello",
                    "logprobs": {
                        "tokens": ["hel", "lo"],
                        "token_logprobs": [None, -0.2],
                    },
                }
            ]
        }
        with patch.object(engine, "_post", return_value=response):
            results = engine.generate([[1]], num_samples=1, max_tokens=10,
                                      temperature=0.7, top_p=0.95)
        lps = results[0][0].logprobs
        assert lps[0] == 0.0
        assert lps[1] == -0.2


# ═══════════════════════════════════════════
# reload_weights() Fault Tolerance
# ═══════════════════════════════════════════


class TestReloadWeightsFaults:
    def test_adapter_reload_failure_is_warning(
        self, engine: OpenAIEngine
    ) -> None:
        """Failed adapter reload prints warning but doesn't crash."""
        with patch.object(
            engine, "_post", side_effect=RuntimeError("server down")
        ):
            # Should NOT raise
            engine.reload_weights("/path/to/adapter")
        # Adapter path should NOT be updated on failure
        assert engine._current_adapter_path is None

    def test_mlx_adapter_skips_http(self, engine: OpenAIEngine) -> None:
        """MLX engine stores adapter path locally, no HTTP call."""
        engine.engine_type = "mlx"
        engine.reload_weights("/path/to/adapter")
        assert engine._current_adapter_path == "/path/to/adapter"

    def test_same_adapter_skips_reload(self, engine: OpenAIEngine) -> None:
        """Reloading same adapter path is a no-op."""
        engine._current_adapter_path = "/existing"
        with patch.object(engine, "_post") as mock_post:
            engine.reload_weights("/existing")
        mock_post.assert_not_called()
