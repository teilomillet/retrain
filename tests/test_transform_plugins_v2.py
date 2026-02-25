"""Tests for context-style transform plugins."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrain.advantages import compute_composable_advantages


class TestTransformPluginsV2:
    def test_context_style_transform_callable(self, tmp_path, monkeypatch):
        module_name = "ctx_transform_plugin"
        plugin_file = Path(tmp_path) / f"{module_name}.py"
        plugin_file.write_text(
            "from retrain import TransformOutput\n\n"
            "def my_transform(ctx):\n"
            "    scale = float(ctx.params.get('scale', 1.0))\n"
            "    out = []\n"
            "    for i, logprobs in enumerate(ctx.logprobs_G):\n"
            "        out.append([ctx.episode_advantages[i] * scale for _ in logprobs])\n"
            "    return TransformOutput(token_advs=out)\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        result = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.3, -0.1], [-0.8, -0.4]],
            planning_masks_G=[[0, 0], [0, 0]],
            advantage_mode="grpo",
            transform_mode=f"{module_name}.my_transform",
            transform_params={"scale": 2.0},
        )
        assert result.token_advs[0] == pytest.approx([1.0, 1.0])
        assert result.token_advs[1] == pytest.approx([-1.0, -1.0])

    def test_context_transform_shape_validation(self, tmp_path, monkeypatch):
        module_name = "bad_shape_transform_plugin"
        plugin_file = Path(tmp_path) / f"{module_name}.py"
        plugin_file.write_text(
            "def my_transform(ctx):\n"
            "    return [[0.0]]\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(ValueError, match="returned 1 sequences, expected 2"):
            compute_composable_advantages(
                rewards_G=[1.0, 0.0],
                logprobs_G=[[-0.3, -0.1], [-0.8, -0.4]],
                planning_masks_G=[[0, 0], [0, 0]],
                advantage_mode="grpo",
                transform_mode=f"{module_name}.my_transform",
            )

    def test_context_transform_non_finite_validation(self, tmp_path, monkeypatch):
        module_name = "nan_transform_plugin"
        plugin_file = Path(tmp_path) / f"{module_name}.py"
        plugin_file.write_text(
            "def my_transform(ctx):\n"
            "    return [[float('nan') for _ in s] for s in ctx.logprobs_G]\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(ValueError, match="non-finite value"):
            compute_composable_advantages(
                rewards_G=[1.0, 0.0],
                logprobs_G=[[-0.3, -0.1], [-0.8, -0.4]],
                planning_masks_G=[[0, 0], [0, 0]],
                advantage_mode="grpo",
                transform_mode=f"{module_name}.my_transform",
            )
