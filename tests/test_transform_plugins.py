"""Tests for TOML-selectable transform plugins."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrain.advantages import (
    TransformSpec,
    compute_composable_advantages,
    get_transform_spec,
)
from retrain.config import TrainConfig


class TestTransformSpecs:
    def test_builtin_spec_capabilities(self):
        sepa = get_transform_spec("gtpo_sepa")
        assert sepa.needs_planning is True
        assert sepa.uses_sepa_controller is True
        assert sepa.entropy_transform is not None

        hicra = get_transform_spec("gtpo_hicra")
        assert hicra.apply_hicra is True
        assert hicra.needs_planning is True
        assert hicra.uses_sepa_controller is False

    def test_unknown_transform_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown transform_mode"):
            get_transform_spec("not_a_real_mode")

    def test_custom_transform_dotted_path(self, tmp_path, monkeypatch):
        module_name = "custom_entropy_transform_plugin"
        plugin_file = Path(tmp_path) / f"{module_name}.py"
        plugin_file.write_text(
            "from retrain.advantages import TransformSpec\n"
            "\n"
            "def _entropy_shift(entropies, planning_mask, sepa_lambda):\n"
            "    return [e if m else e + sepa_lambda for e, m in zip(entropies, planning_mask)]\n"
            "\n"
            "def make_transform_spec():\n"
            "    return TransformSpec(\n"
            "        name='custom_entropy_shift',\n"
            "        use_gtpo=True,\n"
            "        needs_planning=True,\n"
            "        uses_sepa_controller=True,\n"
            "        entropy_transform=_entropy_shift,\n"
            "    )\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        mode = f"{module_name}.make_transform_spec"
        spec = get_transform_spec(mode)
        assert isinstance(spec, TransformSpec)
        assert spec.needs_planning is True
        assert spec.uses_sepa_controller is True
        assert spec.entropy_transform is not None

        # Config accepts dotted path directly from TOML.
        cfg = TrainConfig(transform_mode=mode)
        assert cfg.transform_mode == mode

        # Custom transform changes GTPO weighting while preserving shapes.
        baseline = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[1, 0], [0, 1]],
            advantage_mode="grpo",
            transform_mode="gtpo",
        )
        custom = compute_composable_advantages(
            rewards_G=[1.0, 0.0],
            logprobs_G=[[-0.5, -0.3], [-0.8, -0.6]],
            planning_masks_G=[[1, 0], [0, 1]],
            advantage_mode="grpo",
            transform_mode=mode,
            sepa_lambda=0.7,
        )

        assert len(custom.token_advs) == len(baseline.token_advs)
        assert custom.has_stats is True
        assert custom.token_advs != baseline.token_advs
