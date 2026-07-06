"""Tests for inference engine factory routing."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from retrain.inference_engine import create_engine


class TestCreateEngine:
    def test_pytorch_engine_receives_oscar_sampling_options(self):
        mock_cls = MagicMock(return_value=object())
        fake_mod = SimpleNamespace(PyTorchEngine=mock_cls)
        sample_oscar_options = object()

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "retrain.inference_engine.pytorch_engine", fake_mod)
            create_engine(
                engine_type="pytorch",
                model_name="Qwen/Qwen3-4B-Instruct-2507",
                device="cuda:0",
                peft_config=None,
                dtype=None,
                inference_url="",
                sample_kv_quantization="oscar",
                sample_oscar_options=sample_oscar_options,
            )

        mock_cls.assert_called_once()
        args, kwargs = mock_cls.call_args
        assert args == ("Qwen/Qwen3-4B-Instruct-2507", "cuda:0", None, None)
        assert kwargs["sample_kv_quantization"] == "oscar"
        assert kwargs["sample_oscar_options"] is sample_oscar_options

    def test_pytorch_engine_rejects_oscar_without_cache(self):
        from retrain.inference_engine.pytorch_engine import PyTorchEngine

        with pytest.raises(ValueError, match="requires sample_use_cache=true"):
            PyTorchEngine(
                "Qwen/Qwen3-4B-Instruct-2507",
                "cpu",
                None,
                None,
                existing_model=object(),
                sample_use_cache=False,
                sample_kv_quantization="oscar",
            )

    def test_pytorch_engine_rejects_oscar_existing_model(self):
        from retrain.inference_engine.pytorch_engine import PyTorchEngine

        with pytest.raises(ValueError, match="requires PyTorchEngine to load"):
            PyTorchEngine(
                "Qwen/Qwen3-4B-Instruct-2507",
                "cpu",
                None,
                None,
                existing_model=object(),
                sample_kv_quantization="oscar",
            )

    def test_mlx_engine_uses_default_url(self):
        mock_cls = MagicMock(return_value=object())
        fake_mod = SimpleNamespace(OpenAIEngine=mock_cls)

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "retrain.inference_engine.openai_engine", fake_mod)
            create_engine(
                engine_type="mlx",
                model_name="mlx-community/FakeModel",
                device="cpu",
                peft_config=None,
                dtype=None,
                inference_url="",
            )

        mock_cls.assert_called_once_with(
            base_url="http://localhost:8080",
            model_name="mlx-community/FakeModel",
            engine_type="mlx",
        )

    def test_mlx_engine_uses_explicit_url(self):
        mock_cls = MagicMock(return_value=object())
        fake_mod = SimpleNamespace(OpenAIEngine=mock_cls)

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "retrain.inference_engine.openai_engine", fake_mod)
            create_engine(
                engine_type="mlx",
                model_name="mlx-community/FakeModel",
                device="cpu",
                peft_config=None,
                dtype=None,
                inference_url="http://localhost:9999",
            )

        mock_cls.assert_called_once_with(
            base_url="http://localhost:9999",
            model_name="mlx-community/FakeModel",
            engine_type="mlx",
        )

    def test_trtllm_engine_uses_default_url(self):
        mock_cls = MagicMock(return_value=object())
        fake_mod = SimpleNamespace(OpenAIEngine=mock_cls)

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "retrain.inference_engine.openai_engine", fake_mod)
            create_engine(
                engine_type="trtllm",
                model_name="Qwen/Qwen3.5-2B",
                device="cuda:0",
                peft_config=None,
                dtype=None,
                inference_url="",
            )

        mock_cls.assert_called_once_with(
            base_url="http://localhost:31000",
            model_name="Qwen/Qwen3.5-2B",
            engine_type="trtllm",
        )

    def test_unknown_engine_error_lists_mlx(self):
        with pytest.raises(ValueError, match="Expected: pytorch, max, vllm, sglang, trtllm, mlx, openai"):
            create_engine(
                engine_type="unknown-engine",
                model_name="any",
                device="cpu",
                peft_config=None,
                dtype=None,
                inference_url="",
            )
