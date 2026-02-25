"""Tests for TOML-selectable advantage plugins."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrain.advantages import (
    AdvantageSpec,
    compute_composable_advantages,
    get_advantage_spec,
    register_advantage_mode,
)
from retrain.config import TrainConfig


class TestAdvantageSpecs:
    def test_unknown_advantage_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown advantage_mode"):
            get_advantage_spec("not_a_real_mode")

    def test_custom_advantage_dotted_path(self, tmp_path, monkeypatch):
        module_name = "custom_advantage_plugin"
        plugin_file = Path(tmp_path) / f"{module_name}.py"
        plugin_file.write_text(
            "def hipa_like_advantages(rewards):\n"
            "    if not rewards:\n"
            "        return []\n"
            "    mean_r = sum(rewards) / len(rewards)\n"
            "    return [2.0 * (r - mean_r) for r in rewards]\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        mode = f"{module_name}.hipa_like_advantages"
        spec = get_advantage_spec(mode)
        assert spec.name == mode

        # Config accepts dotted advantage plugin from TOML.
        cfg = TrainConfig(advantage_mode=mode)
        assert cfg.advantage_mode == mode

        baseline = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[0, 0], [0, 0]],
            advantage_mode="grpo",
            transform_mode="none",
        )
        custom = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[0, 0], [0, 0]],
            advantage_mode=mode,
            transform_mode="none",
        )

        assert len(custom.token_advs) == len(baseline.token_advs)
        assert custom.token_advs != baseline.token_advs

    def test_custom_advantage_accepts_params(self, tmp_path, monkeypatch):
        module_name = "param_advantage_plugin"
        plugin_file = Path(tmp_path) / f"{module_name}.py"
        plugin_file.write_text(
            "def scaled_advantages(rewards, params):\n"
            "    if not rewards:\n"
            "        return []\n"
            "    scale = float(params.get('scale', 1.0))\n"
            "    mean_r = sum(rewards) / len(rewards)\n"
            "    return [scale * (r - mean_r) for r in rewards]\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        mode = f"{module_name}.scaled_advantages"
        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[0, 0], [0, 0]],
            advantage_mode=mode,
            transform_mode="none",
            advantage_params={"scale": 2.0},
        )

        assert result.token_advs[0] == pytest.approx([1.0, 1.0])
        assert result.token_advs[1] == pytest.approx([-1.0, -1.0])

    def test_custom_advantage_factory_returns_spec(self, tmp_path, monkeypatch):
        module_name = "factory_advantage_plugin"
        plugin_file = Path(tmp_path) / f"{module_name}.py"
        plugin_file.write_text(
            "from retrain.advantages import AdvantageSpec\n"
            "\n"
            "def _fixed_advantages(rewards):\n"
            "    return [0.25 for _ in rewards]\n"
            "\n"
            "def make_advantage_spec():\n"
            "    return AdvantageSpec(\n"
            "        name='factory_fixed',\n"
            "        compute=lambda rewards, params: _fixed_advantages(rewards),\n"
            "    )\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        mode = f"{module_name}.make_advantage_spec"
        spec = get_advantage_spec(mode)
        assert isinstance(spec, AdvantageSpec)
        assert spec.name == "factory_fixed"

        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[0, 0], [0, 0]],
            advantage_mode=mode,
            transform_mode="none",
        )
        assert result.token_advs == [[0.25, 0.25], [0.25, 0.25]]

    def test_runtime_short_name_registration(self):
        def _student_advantage(rewards):
            return [0.0 for _ in rewards]

        register_advantage_mode("student_zero", _student_advantage)
        try:
            cfg = TrainConfig(advantage_mode="student_zero")
            assert cfg.advantage_mode == "student_zero"

            result = compute_composable_advantages(
                rewards_G=[1.0, 0.0],
                logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
                planning_masks_G=[[0, 0], [0, 0]],
                advantage_mode="student_zero",
                transform_mode="none",
            )
            assert result.token_advs == [[0.0, 0.0], [0.0, 0.0]]
        finally:
            from retrain.advantages import (
                _ADVANTAGE_SPEC_CACHE,
                _BUILTIN_ADVANTAGE_SPECS,
            )

            _BUILTIN_ADVANTAGE_SPECS.pop("student_zero", None)
            _ADVANTAGE_SPEC_CACHE.pop("student_zero", None)
