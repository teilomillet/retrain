"""Tests for algorithm_mode plugins."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrain.advantages import (
    AlgorithmSpec,
    compute_algorithm_advantages,
    get_algorithm_spec,
)
from retrain.config import TrainConfig


class TestAlgorithmPlugins:
    def test_builtin_algorithm_mode_resolves(self):
        spec = get_algorithm_spec("maxrl_gtpo_sepa")
        assert isinstance(spec, AlgorithmSpec)
        assert spec.needs_planning is True
        assert spec.uses_sepa_controller is True

    def test_dotted_algorithm_callable(self, tmp_path, monkeypatch):
        module_name = "custom_algorithm_plugin"
        plugin_file = Path(tmp_path) / f"{module_name}.py"
        plugin_file.write_text(
            "from retrain import AlgorithmOutput\n\n"
            "def my_algorithm(ctx):\n"
            "    out = []\n"
            "    for i, logprobs in enumerate(ctx.logprobs_G):\n"
            "        out.append([ctx.rewards_G[i] for _ in logprobs])\n"
            "    return AlgorithmOutput(token_advs=out, has_stats=False)\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        mode = f"{module_name}.my_algorithm"

        cfg = TrainConfig(algorithm_mode=mode)
        assert cfg.algorithm_mode == mode

        result = compute_algorithm_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.2], [-0.7, -0.6]],
            planning_masks_G=[[0, 0], [0, 0]],
            algorithm_mode=mode,
            params={},
        )
        assert result.token_advs == [[1.0, 1.0], [0.0, 0.0]]

    def test_algorithm_factory_returning_spec(self, tmp_path, monkeypatch):
        module_name = "factory_algorithm_plugin"
        plugin_file = Path(tmp_path) / f"{module_name}.py"
        plugin_file.write_text(
            "from retrain import AlgorithmOutput, AlgorithmSpec\n\n"
            "def _impl(ctx):\n"
            "    return AlgorithmOutput(token_advs=[[0.25 for _ in s] for s in ctx.logprobs_G])\n"
            "\n"
            "def make_spec():\n"
            "    return AlgorithmSpec(name='factory_algo', compute=_impl, needs_planning=False, uses_sepa_controller=False)\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        mode = f"{module_name}.make_spec"

        spec = get_algorithm_spec(mode)
        assert spec.name == "factory_algo"

    def test_algorithm_output_validation(self, tmp_path, monkeypatch):
        module_name = "bad_algorithm_plugin"
        plugin_file = Path(tmp_path) / f"{module_name}.py"
        plugin_file.write_text(
            "def my_algorithm(ctx):\n"
            "    return [[float('inf') for _ in s] for s in ctx.logprobs_G]\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        mode = f"{module_name}.my_algorithm"

        with pytest.raises(ValueError, match="non-finite value"):
            compute_algorithm_advantages(
                rewards_G=[1.0, 0.0],
                logprobs_G=[[-0.5, -0.2], [-0.7, -0.6]],
                planning_masks_G=[[0, 0], [0, 0]],
                algorithm_mode=mode,
                params={},
            )
