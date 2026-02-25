"""Tests for retrain.registry — plug-and-play component registry."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from retrain.config import TrainConfig
from retrain.registry import (
    Registry,
    backend,
    backpressure,
    check_environment,
    data_source,
    get_registry,
    planning_detector,
    reward,
)


# ---------------------------------------------------------------------------
# Registry class basics
# ---------------------------------------------------------------------------

class TestRegistryCore:
    def test_register_and_create(self):
        reg = Registry("test")

        @reg.register("foo")
        def _make_foo(config):
            return {"type": "foo", "model": config.model}

        config = TrainConfig()
        result = reg.create("foo", config)
        assert result["type"] == "foo"

    def test_builtin_names(self):
        reg = Registry("test")

        @reg.register("b")
        def _b(config):
            return None

        @reg.register("a")
        def _a(config):
            return None

        assert reg.builtin_names == ["a", "b"]

    def test_unknown_name_raises_with_available(self):
        reg = Registry("widget")

        @reg.register("alpha")
        def _alpha(config):
            return None

        with pytest.raises(ValueError, match="Unknown widget 'nope'"):
            reg.create("nope", TrainConfig())

    def test_dotted_path_import(self):
        """A name with '.' should attempt dotted-path import."""
        config = TrainConfig()
        reg = Registry("test")

        mock_cls = MagicMock(return_value="plugin_instance")
        with patch("importlib.import_module") as mock_import:
            mock_mod = MagicMock()
            mock_mod.MyPlugin = mock_cls
            mock_import.return_value = mock_mod

            result = reg.create("mypkg.module.MyPlugin", config)

        mock_import.assert_called_once_with("mypkg.module")
        mock_cls.assert_called_once_with(config)
        assert result == "plugin_instance"

    def test_dotted_path_missing_module_raises(self):
        reg = Registry("test")
        with pytest.raises(ImportError, match="Could not import module"):
            reg.create("nonexistent.pkg.Cls", TrainConfig())

    def test_dotted_path_missing_attr_raises(self):
        reg = Registry("test")
        # importlib.import_module("retrain.config") works, but there's no "Nope"
        with pytest.raises(AttributeError, match="has no attribute 'Nope'"):
            reg.create("retrain.config.Nope", TrainConfig())


# ---------------------------------------------------------------------------
# get_registry
# ---------------------------------------------------------------------------

class TestGetRegistry:
    def test_known_registries(self):
        for name in ("backend", "reward", "planning_detector",
                     "data_source", "backpressure", "inference_engine"):
            reg = get_registry(name)
            assert reg.kind == name

    def test_unknown_registry_raises(self):
        with pytest.raises(KeyError, match="No registry 'bogus'"):
            get_registry("bogus")


# ---------------------------------------------------------------------------
# Built-in registrations — verify names are registered
# ---------------------------------------------------------------------------

class TestBuiltinNames:
    def test_backend_names(self):
        assert "local" in backend.builtin_names
        assert "tinker" in backend.builtin_names

    def test_reward_names(self):
        assert set(reward.builtin_names) >= {"match", "math", "judge", "custom"}

    def test_planning_detector_names(self):
        assert set(planning_detector.builtin_names) >= {"regex", "semantic"}

    def test_data_source_names(self):
        assert "math" in data_source.builtin_names

    def test_backpressure_names(self):
        assert set(backpressure.builtin_names) >= {"noop", "usl"}


# ---------------------------------------------------------------------------
# Built-in creation (mock heavy deps)
# ---------------------------------------------------------------------------

class TestBuiltinCreation:
    def test_reward_match(self):
        config = TrainConfig(reward_type="match")
        fn = reward.create("match", config)
        # BoxedMathReward has a .score method
        assert hasattr(fn, "score")

    def test_backpressure_noop(self):
        config = TrainConfig()
        bp = backpressure.create("noop", config)
        assert hasattr(bp, "observe")
        assert hasattr(bp, "recommend")

    def test_backpressure_usl(self):
        config = TrainConfig(bp_enabled=True)
        bp = backpressure.create("usl", config)
        assert hasattr(bp, "observe")

    def test_reward_custom_requires_module(self):
        config = TrainConfig(reward_type="custom", reward_custom_module="")
        with pytest.raises(ValueError, match="custom_module"):
            reward.create("custom", config)

    def test_tinker_uses_inference_url_before_base_url(self):
        mock_cls = MagicMock(return_value=object())
        fake_mod = SimpleNamespace(TinkerTrainHelper=mock_cls)
        config = TrainConfig(
            backend="tinker",
            inference_url="http://inference-url",
            base_url="http://model-base-url",
        )
        with patch.dict(sys.modules, {"retrain.tinker_backend": fake_mod}):
            backend.create("tinker", config)
        mock_cls.assert_called_once()
        assert mock_cls.call_args[0][1] == "http://inference-url"

    def test_tinker_falls_back_to_model_base_url(self):
        mock_cls = MagicMock(return_value=object())
        fake_mod = SimpleNamespace(TinkerTrainHelper=mock_cls)
        config = TrainConfig(
            backend="tinker",
            inference_url="",
            base_url="http://model-base-url",
        )
        with patch.dict(sys.modules, {"retrain.tinker_backend": fake_mod}):
            backend.create("tinker", config)
        mock_cls.assert_called_once()
        assert mock_cls.call_args[0][1] == "http://model-base-url"


# ---------------------------------------------------------------------------
# check_environment
# ---------------------------------------------------------------------------

class TestCheckEnvironment:
    def test_full_scan_returns_all_known(self):
        results = check_environment(config=None)
        names = {r[0] for r in results}
        assert names >= {"local", "tinker", "math", "judge", "semantic"}

    def test_config_scoped_check(self):
        config = TrainConfig(backend="local", reward_type="match",
                             planning_detector="regex")
        results = check_environment(config=config)
        names = {r[0] for r in results}
        # "match" and "regex" have no entry in _DEPENDENCY_MAP
        assert "local" in names
        # "match" is not in the dep map so shouldn't appear
        assert "match" not in names

    def test_result_tuple_shape(self):
        results = check_environment(config=None)
        for name, import_name, hint, available in results:
            assert isinstance(name, str)
            assert isinstance(import_name, str)
            assert isinstance(hint, str)
            assert isinstance(available, bool)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_existing_config_values_accepted(self):
        """All previously valid config combinations still work."""
        # These should not raise
        TrainConfig(backend="local")
        TrainConfig(backend="tinker")
        TrainConfig(reward_type="match")
        TrainConfig(reward_type="math")
        TrainConfig(planning_detector="regex")
        TrainConfig(planning_detector="semantic")

    def test_data_source_defaults_to_math(self):
        config = TrainConfig()
        assert config.data_source == "math"

    def test_dotted_planning_detector_accepted(self):
        """Dotted-path planning_detector names should not be rejected by config."""
        config = TrainConfig(planning_detector="mypackage.MyDetector")
        assert config.planning_detector == "mypackage.MyDetector"

    def test_create_reward_still_importable(self):
        """create_reward remains importable for external code."""
        from retrain.rewards import create_reward
        assert callable(create_reward)

    def test_create_planning_detector_still_importable(self):
        """create_planning_detector remains importable for external code."""
        from retrain.planning import create_planning_detector
        assert callable(create_planning_detector)
