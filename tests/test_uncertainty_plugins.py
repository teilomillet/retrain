"""Tests for pluggable uncertainty kind registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrain.advantages import (
    UncertaintyContext,
    UncertaintySpec,
    get_uncertainty_spec,
    is_valid_uncertainty_kind_name,
)
from retrain.config import TrainConfig


class TestUncertaintySpecs:
    def test_builtin_spec_capabilities(self):
        surprisal = get_uncertainty_spec("surprisal")
        assert surprisal.needs_distributions is False

        shannon = get_uncertainty_spec("shannon_entropy")
        assert shannon.needs_distributions is True

        pred_var = get_uncertainty_spec("predictive_variance")
        assert pred_var.needs_distributions is False

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown uncertainty_kind"):
            get_uncertainty_spec("not_a_real_kind")

    def test_custom_dotted_path(self, tmp_path, monkeypatch):
        module_name = "custom_uncertainty_plugin"
        plugin_file = Path(tmp_path) / f"{module_name}.py"
        plugin_file.write_text(
            "from retrain.advantages import UncertaintySpec\n"
            "\n"
            "def _compute_constant(ctx):\n"
            "    return [1.0] * len(ctx.logprobs)\n"
            "\n"
            "def make_uncertainty_spec():\n"
            "    return UncertaintySpec(\n"
            "        name='custom_constant',\n"
            "        compute=_compute_constant,\n"
            "    )\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        kind = f"{module_name}.make_uncertainty_spec"
        spec = get_uncertainty_spec(kind)
        assert isinstance(spec, UncertaintySpec)
        assert spec.name == "custom_constant"

        ctx = UncertaintyContext(logprobs=[-0.5, -0.3, -0.1])
        assert spec.compute(ctx) == [1.0, 1.0, 1.0]

    def test_config_accepts_predictive_variance(self):
        cfg = TrainConfig(uncertainty_kind="predictive_variance")
        assert cfg.uncertainty_kind == "predictive_variance"

    def test_config_accepts_pred_var_alias(self):
        cfg = TrainConfig(uncertainty_kind="pred_var")
        assert cfg.uncertainty_kind == "predictive_variance"

    def test_config_accepts_bernoulli_variance_alias(self):
        cfg = TrainConfig(uncertainty_kind="bernoulli_variance")
        assert cfg.uncertainty_kind == "predictive_variance"

    def test_is_valid_covers_new_aliases(self):
        assert is_valid_uncertainty_kind_name("predictive_variance") is True
        assert is_valid_uncertainty_kind_name("pred_var") is True
        assert is_valid_uncertainty_kind_name("bernoulli_variance") is True
        assert is_valid_uncertainty_kind_name("my_pkg.my_unc") is True
        assert is_valid_uncertainty_kind_name("bogus") is False
