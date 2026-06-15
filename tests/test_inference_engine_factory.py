"""Tests for inference engine factory routing."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from retrain.inference_engine import create_engine


class TestCreateEngine:
    def test_pytorch_engine_receives_trust_remote_code(self):
        mock_cls = MagicMock(return_value=object())
        fake_mod = SimpleNamespace(PyTorchEngine=mock_cls)

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "retrain.inference_engine.pytorch_engine", fake_mod)
            create_engine(
                engine_type="pytorch",
                model_name="nvidia/FakeNemotron",
                device="cpu",
                peft_config=None,
                dtype=None,
                inference_url="",
                trust_remote_code=True,
            )

        mock_cls.assert_called_once_with(
            "nvidia/FakeNemotron",
            "cpu",
            None,
            None,
            trust_remote_code=True,
            use_cache=True,
        )

    def test_pytorch_engine_receives_use_cache(self):
        mock_cls = MagicMock(return_value=object())
        fake_mod = SimpleNamespace(PyTorchEngine=mock_cls)

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "retrain.inference_engine.pytorch_engine", fake_mod)
            create_engine(
                engine_type="pytorch",
                model_name="google/FakeGemma",
                device="cpu",
                peft_config=None,
                dtype=None,
                inference_url="",
                use_cache=False,
            )

        mock_cls.assert_called_once_with(
            "google/FakeGemma",
            "cpu",
            None,
            None,
            trust_remote_code=False,
            use_cache=False,
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

    def test_unknown_engine_error_lists_mlx(self):
        with pytest.raises(ValueError, match="Expected: pytorch, max, vllm, sglang, mlx, openai"):
            create_engine(
                engine_type="unknown-engine",
                model_name="any",
                device="cpu",
                peft_config=None,
                dtype=None,
                inference_url="",
            )
