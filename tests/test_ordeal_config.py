"""Ordeal property tests for retrain.config.

Tests validation boundaries, default determinism, CLI override parsing,
and mode validation using boundary-biased property testing.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from ordeal.quickcheck import quickcheck

from retrain.config import TrainConfig, load_config, parse_cli_overrides


# ═══════════════════════════════════════════
# Default Initialization
# ═══════════════════════════════════════════


class TestDefaultInitialization:
    def test_deterministic(self) -> None:
        """Two default-constructed configs are identical."""
        a = TrainConfig()
        b = TrainConfig()
        for field in TrainConfig.__dataclass_fields__:
            assert getattr(a, field) == getattr(b, field), f"field {field} differs"

    def test_valid_defaults(self) -> None:
        """Default config passes all validation (no exception)."""
        TrainConfig()

    def test_default_advantage_mode(self) -> None:
        c = TrainConfig()
        assert c.advantage_mode == "maxrl"

    def test_default_transform_mode(self) -> None:
        c = TrainConfig()
        assert c.transform_mode == "gtpo_sepa"

    def test_default_backend(self) -> None:
        c = TrainConfig()
        assert c.backend == "local"


# ═══════════════════════════════════════════
# Validation: Positive Constraints
# ═══════════════════════════════════════════


class TestValidationPositive:
    """Fields that must be > 0."""

    @given(val=st.integers(min_value=-100, max_value=0))
    def test_batch_size_rejects_non_positive(self, val: int) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            TrainConfig(batch_size=val)

    @given(val=st.integers(min_value=-100, max_value=0))
    def test_group_size_rejects_non_positive(self, val: int) -> None:
        with pytest.raises(ValueError, match="group_size"):
            TrainConfig(group_size=val)

    @given(val=st.integers(min_value=-100, max_value=0))
    def test_max_steps_rejects_non_positive(self, val: int) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            TrainConfig(max_steps=val)

    @given(val=st.integers(min_value=-100, max_value=0))
    def test_max_tokens_rejects_non_positive(self, val: int) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            TrainConfig(max_tokens=val)

    @given(val=st.integers(min_value=-100, max_value=0))
    def test_lora_rank_rejects_non_positive(self, val: int) -> None:
        with pytest.raises(ValueError, match="lora_rank"):
            TrainConfig(lora_rank=val)

    @given(
        val=st.floats(
            min_value=-10.0,
            max_value=0.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    def test_lr_rejects_non_positive(self, val: float) -> None:
        with pytest.raises(ValueError, match="lr"):
            TrainConfig(lr=val)


class TestValidationAcceptsValid:
    """Valid values pass validation."""

    @given(val=st.integers(min_value=1, max_value=256))
    def test_batch_size_accepts_positive(self, val: int) -> None:
        c = TrainConfig(batch_size=val)
        assert c.batch_size == val

    @given(val=st.integers(min_value=1, max_value=256))
    def test_group_size_accepts_positive(self, val: int) -> None:
        c = TrainConfig(group_size=val)
        assert c.group_size == val

    @given(
        val=st.floats(
            min_value=1e-8,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    def test_lr_accepts_positive(self, val: float) -> None:
        c = TrainConfig(lr=val)
        assert c.lr == val


# ═══════════════════════════════════════════
# Validation: Range Constraints
# ═══════════════════════════════════════════


class TestValidationRanges:
    @given(
        val=st.floats(
            min_value=-10.0,
            max_value=-0.01,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    def test_temperature_rejects_negative(self, val: float) -> None:
        with pytest.raises(ValueError, match="temperature"):
            TrainConfig(temperature=val)

    @given(
        val=st.floats(
            min_value=0.0,
            max_value=2.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    def test_temperature_accepts_non_negative(self, val: float) -> None:
        c = TrainConfig(temperature=val)
        assert c.temperature == val

    def test_top_p_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="top_p"):
            TrainConfig(top_p=0.0)

    def test_top_p_rejects_above_one(self) -> None:
        with pytest.raises(ValueError, match="top_p"):
            TrainConfig(top_p=1.01)

    @given(
        val=st.floats(
            min_value=0.01,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    def test_top_p_accepts_valid(self, val: float) -> None:
        c = TrainConfig(top_p=val)
        assert c.top_p == val

    @given(
        val=st.floats(
            min_value=-1.0,
            max_value=-0.01,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    def test_clip_eps_rejects_negative(self, val: float) -> None:
        with pytest.raises(ValueError, match="clip_eps"):
            TrainConfig(clip_eps=val)

    def test_clip_eps_high_requires_clip_eps(self) -> None:
        """clip_eps_high > 0 requires clip_eps > 0."""
        with pytest.raises(ValueError, match="clip_eps_high"):
            TrainConfig(clip_eps=0.0, clip_eps_high=0.5)

    def test_clip_eps_high_with_clip_eps_valid(self) -> None:
        c = TrainConfig(clip_eps=0.2, clip_eps_high=0.5)
        assert c.clip_eps_high == 0.5

    @given(
        val=st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    def test_surprisal_mask_rho_valid(self, val: float) -> None:
        c = TrainConfig(surprisal_mask_rho=val)
        assert c.surprisal_mask_rho == val

    def test_surprisal_mask_rho_rejects_above_one(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            TrainConfig(surprisal_mask_rho=1.1)

    def test_surprisal_mask_rho_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            TrainConfig(surprisal_mask_rho=-0.1)


# ═══════════════════════════════════════════
# Mode Validation
# ═══════════════════════════════════════════


class TestModeValidation:
    @pytest.mark.parametrize("mode", ["grpo", "maxrl", "reinforce_pp"])
    def test_builtin_advantage_modes_accepted(self, mode: str) -> None:
        c = TrainConfig(advantage_mode=mode)
        assert c.advantage_mode == mode

    def test_dotted_advantage_mode_accepted(self) -> None:
        """Dotted plugin paths are accepted by validation."""
        c = TrainConfig(advantage_mode="some.module.function")
        assert c.advantage_mode == "some.module.function"

    def test_invalid_advantage_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="advantage_mode"):
            TrainConfig(advantage_mode="nonexistent_mode")

    @pytest.mark.parametrize(
        "mode", ["none", "gtpo", "gtpo_hicra", "gtpo_sepa", "delight"]
    )
    def test_builtin_transform_modes_accepted(self, mode: str) -> None:
        c = TrainConfig(transform_mode=mode)
        assert c.transform_mode == mode

    def test_invalid_transform_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="transform_mode"):
            TrainConfig(transform_mode="nonexistent_transform")


# ═══════════════════════════════════════════
# CLI Override Parsing
# ═══════════════════════════════════════════


class TestCLIOverrides:
    def test_batch_size_override_raw(self) -> None:
        """parse_cli_overrides stores regular flags as raw strings."""
        _, overrides = parse_cli_overrides(["--batch-size", "32"])
        assert overrides["batch_size"] == "32"

    def test_equals_syntax(self) -> None:
        _, overrides = parse_cli_overrides(["--batch-size=64"])
        assert overrides["batch_size"] == "64"

    def test_config_path_positional(self) -> None:
        path, overrides = parse_cli_overrides(["my_config.toml", "--lr", "1e-4"])
        assert path == "my_config.toml"
        assert overrides["lr"] == "1e-4"

    def test_unknown_flag_rejected(self) -> None:
        with pytest.raises(SystemExit):
            parse_cli_overrides(["--totally-made-up-flag", "42"])

    def test_backend_opt_stores_raw_strings(self) -> None:
        """--backend-opt stores raw strings; coercion happens in load_config."""
        _, overrides = parse_cli_overrides(
            ["--backend-opt", "key1=val1", "--backend-opt", "key2=42"]
        )
        assert overrides["backend_options"]["key1"] == "val1"
        assert overrides["backend_options"]["key2"] == "42"

    def test_algorithm_param_coerces_types(self) -> None:
        """--algorithm-param uses _parse_param_opt which auto-coerces."""
        _, overrides = parse_cli_overrides(
            ["--algorithm-param", "alpha=0.2", "--algorithm-param", "flag=true"]
        )
        assert overrides["algorithm_params"]["alpha"] == 0.2
        assert overrides["algorithm_params"]["flag"] is True

    def test_load_config_applies_cli_coercion(self) -> None:
        """load_config coerces raw CLI string overrides to correct types."""
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write("")
            f.flush()
            c = load_config(f.name, overrides={"batch_size": "32"})
        assert c.batch_size == 32


# ═══════════════════════════════════════════
# TOML Loading
# ═══════════════════════════════════════════


class TestTOMLLoading:
    def test_empty_toml_uses_defaults(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write("")
            f.flush()
            c = load_config(f.name)
        assert c.batch_size == TrainConfig().batch_size

    def test_minimal_toml(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write('[training]\nbatch_size = 32\n')
            f.flush()
            c = load_config(f.name)
        assert c.batch_size == 32

    def test_toml_override_by_cli(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write('[training]\nbatch_size = 32\n')
            f.flush()
            c = load_config(f.name, overrides={"batch_size": 64})
        assert c.batch_size == 64

    def test_algorithm_section(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write('[algorithm]\nadvantage_mode = "grpo"\ntransform_mode = "none"\n')
            f.flush()
            c = load_config(f.name)
        assert c.advantage_mode == "grpo"
        assert c.transform_mode == "none"


# ═══════════════════════════════════════════
# Environment Validation
# ═══════════════════════════════════════════


class TestEnvironmentValidation:
    def test_verifiers_requires_environment_id(self) -> None:
        with pytest.raises(ValueError, match="environment_id"):
            TrainConfig(environment_provider="verifiers")

    def test_verifiers_with_id_valid(self) -> None:
        c = TrainConfig(
            environment_provider="verifiers", environment_id="math_verify"
        )
        assert c.environment_provider == "verifiers"

    def test_invalid_environment_provider(self) -> None:
        with pytest.raises(ValueError, match="environment_provider"):
            TrainConfig(environment_provider="unknown_provider")

    def test_invalid_environment_args_json(self) -> None:
        with pytest.raises(ValueError, match="environment_args"):
            TrainConfig(
                environment_provider="verifiers",
                environment_id="test",
                environment_args="not valid json",
            )

    def test_valid_environment_args_json(self) -> None:
        c = TrainConfig(
            environment_provider="verifiers",
            environment_id="test",
            environment_args='{"key": "value"}',
        )
        assert c.environment_args == '{"key": "value"}'
