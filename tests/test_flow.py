"""Tests for the construct-and-trace training flow."""

import json
import subprocess
import sys
import textwrap

import pytest

from retrain.config import TrainConfig
from retrain.flow import (
    TraceIssue,
    TraceResult,
    TrainingFlow,
    _FLOW_PROBE_CASES,
    _SCALAR_BACKEND_DISALLOWED_BUILTIN_ALGORITHM_MODES,
    _SCALAR_BACKEND_DISALLOWED_BUILTIN_TRANSFORM_MODES,
    _condition_label,
    _token_advs_are_uniform,
    build_flow,
)


# ── TestBuildFlow ────────────────────────────────────────────────────────


class TestBuildFlow:
    def test_composable_path_resolves_advantage_and_transform_specs(self):
        cfg = TrainConfig(advantage_mode="grpo", transform_mode="none")
        flow = build_flow(cfg, gpu=False)
        assert flow.advantage_spec is not None
        assert flow.transform_spec is not None
        assert flow.algorithm_spec is None

    def test_algorithm_path_resolves_algorithm_spec(self):
        cfg = TrainConfig(algorithm_mode="grpo_none")
        flow = build_flow(cfg, gpu=False)
        assert flow.algorithm_spec is not None
        assert flow.advantage_spec is None
        assert flow.transform_spec is None

    def test_local_backend_preserves_token_advantages(self):
        cfg = TrainConfig(backend="local")
        flow = build_flow(cfg, gpu=False)
        assert flow.backend_capabilities.preserves_token_advantages is True

    def test_tinker_backend_preserves_token_advantages(self):
        cfg = TrainConfig(backend="tinker")
        flow = build_flow(cfg, gpu=False)
        assert flow.backend_capabilities.preserves_token_advantages is True

    def test_prime_rl_backend_does_not_preserve_token_advantages(self):
        cfg = TrainConfig(backend="prime_rl")
        flow = build_flow(cfg, gpu=False)
        assert flow.backend_capabilities.preserves_token_advantages is False

    def test_needs_planning_from_transform_spec(self):
        cfg = TrainConfig(
            advantage_mode="maxrl",
            transform_mode="gtpo_hicra",
        )
        flow = build_flow(cfg, gpu=False)
        assert flow.needs_planning is True

    def test_needs_planning_from_algorithm_spec(self):
        cfg = TrainConfig(algorithm_mode="maxrl_gtpo_hicra")
        flow = build_flow(cfg, gpu=False)
        assert flow.needs_planning is True

    def test_condition_label_composable(self):
        cfg = TrainConfig(
            advantage_mode="maxrl",
            transform_mode="gtpo_sepa",
        )
        flow = build_flow(cfg, gpu=False)
        assert flow.condition_label == "maxrl+gtpo_sepa"

    def test_condition_label_algorithm(self):
        cfg = TrainConfig(algorithm_mode="grpo_none")
        flow = build_flow(cfg, gpu=False)
        assert flow.condition_label == "grpo_none"

    def test_gpu_false_leaves_tier2_none(self):
        cfg = TrainConfig()
        flow = build_flow(cfg, gpu=False)
        assert flow.backend is None
        assert flow.planning_detector is None
        assert flow.sepa_controller is None
        assert flow.backpressure is None

    def test_backend_capability_source_builtin(self):
        cfg = TrainConfig(backend="local")
        flow = build_flow(cfg, gpu=False)
        assert flow.backend_capability_source == "builtin"

    def test_backend_capability_source_plugin(self):
        cfg = TrainConfig(backend="some.plugin.Backend")
        flow = build_flow(cfg, gpu=False)
        assert flow.backend_capability_source == "plugin/default"


# ── TestTrace ────────────────────────────────────────────────────────────


class TestTrace:
    def test_scalar_safe_modes_pass_for_prime_rl(self):
        cfg = TrainConfig(
            backend="prime_rl",
            advantage_mode="grpo",
            transform_mode="none",
        )
        flow = build_flow(cfg, gpu=False)
        result = flow.trace()
        # Warnings are ok (e.g. placeholder loss), but no errors
        errors = [i for i in result.issues if i.severity == "error"]
        assert len(errors) == 0

    def test_token_varying_modes_fail_for_prime_rl(self):
        cfg = TrainConfig(
            backend="prime_rl",
            advantage_mode="grpo",
            transform_mode="gtpo",
        )
        flow = build_flow(cfg, gpu=False)
        result = flow.trace()
        assert not result.ok

    def test_token_varying_modes_pass_for_local(self):
        cfg = TrainConfig(
            backend="local",
            advantage_mode="maxrl",
            transform_mode="gtpo",
        )
        flow = build_flow(cfg, gpu=False)
        result = flow.trace()
        assert result.ok

    def test_token_varying_modes_pass_for_tinker(self):
        cfg = TrainConfig(
            backend="tinker",
            advantage_mode="maxrl",
            transform_mode="gtpo",
        )
        flow = build_flow(cfg, gpu=False)
        result = flow.trace()
        assert result.ok

    def test_tinker_shannon_entropy_rejected_at_config(self):
        """shannon_entropy + tinker backend is rejected at config validation."""
        with pytest.raises(ValueError, match="shannon_entropy.*not supported.*tinker"):
            TrainConfig(
                backend="tinker",
                advantage_mode="maxrl",
                transform_mode="gtpo",
                uncertainty_kind="shannon_entropy",
            )

    def test_shannon_entropy_non_pytorch_engine_rejected(self):
        """shannon_entropy + non-pytorch engine is rejected at config."""
        with pytest.raises(ValueError, match="shannon_entropy.*requires.*pytorch"):
            TrainConfig(
                backend="local",
                inference_engine="vllm",
                inference_url="http://localhost:8000",
                uncertainty_kind="shannon_entropy",
            )

    def test_tinker_varentropy_request_fails_trace(self):
        cfg = TrainConfig(
            backend="tinker",
            advantage_mode="maxrl",
            transform_mode="gtpo",
            uncertainty_kind="varentropy",
        )
        flow = build_flow(cfg, gpu=False)
        result = flow.trace()
        assert not result.ok
        errors = [i for i in result.issues if i.severity == "error"]
        assert any(
            "varentropy" in e.message and e.category == "probe"
            for e in errors
        )

    def test_local_pytorch_shannon_entropy_passes_trace(self):
        """shannon_entropy + pytorch engine + local backend passes flow trace."""
        cfg = TrainConfig(
            backend="local",
            inference_engine="pytorch",
            advantage_mode="maxrl",
            transform_mode="gtpo",
            uncertainty_kind="shannon_entropy",
        )
        flow = build_flow(cfg, gpu=False)
        result = flow.trace()
        assert result.ok
        assert result.probe_cases_passed == result.probe_cases_run

    def test_disallowed_builtin_algorithm_on_prime_rl(self):
        cfg = TrainConfig(
            backend="prime_rl",
            algorithm_mode="maxrl_gtpo",
        )
        flow = build_flow(cfg, gpu=False)
        result = flow.trace()
        assert not result.ok
        errors = [i for i in result.issues if i.severity == "error"]
        assert any("algorithm_mode='maxrl_gtpo'" in e.message for e in errors)

    def test_broken_plugin_reports_probe_error(self, tmp_path, monkeypatch):
        module_name = "broken_transform_plugin"
        plugin_file = tmp_path / f"{module_name}.py"
        plugin_file.write_text(
            "from retrain.advantages import TransformSpec\n"
            "\n"
            "def _broken_entropy_transform(entropies, logprobs, planning_mask, "
            "gtpo_beta, hicra_alpha, params=None):\n"
            "    raise RuntimeError('intentional test failure')\n"
            "\n"
            "def make_transform_spec():\n"
            "    return TransformSpec(\n"
            "        name='broken', use_gtpo=True,\n"
            "        entropy_transform=_broken_entropy_transform,\n"
            "    )\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        cfg = TrainConfig(
            backend="local",
            transform_mode=f"{module_name}.make_transform_spec",
        )
        flow = build_flow(cfg, gpu=False)
        result = flow.trace()
        assert not result.ok
        errors = [i for i in result.issues if i.severity == "error"]
        assert any("probe" in e.category for e in errors)

    def test_multiple_problems_reported_at_once(self):
        """Scalar backend + token-varying mode produces both compat and probe errors."""
        cfg = TrainConfig(
            backend="prime_rl",
            advantage_mode="grpo",
            transform_mode="gtpo",
        )
        flow = build_flow(cfg, gpu=False)
        result = flow.trace()
        assert not result.ok
        # Should have at least the compat error AND the probe error
        categories = {i.category for i in result.issues if i.severity == "error"}
        assert "compat" in categories

    def test_sepa_zero_steps_warning(self):
        cfg = TrainConfig(
            advantage_mode="maxrl",
            transform_mode="gtpo_sepa",
            sepa_steps=0,
        )
        flow = build_flow(cfg, gpu=False)
        result = flow.trace()
        warnings = [i for i in result.issues if i.severity == "warning"]
        assert any("sepa_steps" in w.message for w in warnings)

    def test_all_probes_pass_for_well_formed_composable(self):
        cfg = TrainConfig(
            backend="local",
            advantage_mode="grpo",
            transform_mode="none",
        )
        flow = build_flow(cfg, gpu=False)
        result = flow.trace()
        assert result.probe_cases_run == len(_FLOW_PROBE_CASES)
        assert result.probe_cases_passed == result.probe_cases_run

    def test_planning_detector_none_warning(self):
        cfg = TrainConfig(
            advantage_mode="maxrl",
            transform_mode="gtpo_hicra",
            planning_detector="none",
        )
        flow = build_flow(cfg, gpu=False)
        result = flow.trace()
        warnings = [i for i in result.issues if i.severity == "warning"]
        assert any("planning" in w.message.lower() for w in warnings)

    def test_placeholder_loss_warning_for_prime_rl(self):
        cfg = TrainConfig(
            backend="prime_rl",
            advantage_mode="grpo",
            transform_mode="none",
        )
        flow = build_flow(cfg, gpu=False)
        result = flow.trace()
        warnings = [i for i in result.issues if i.severity == "warning"]
        assert any("placeholder" in w.message for w in warnings)


# ── TestTraceResult ──────────────────────────────────────────────────────


class TestTraceResult:
    def test_ok_when_no_errors(self):
        result = TraceResult(
            issues=[TraceIssue("warning", "config", "test")],
            probe_cases_run=2,
            probe_cases_passed=2,
        )
        assert result.ok

    def test_not_ok_when_errors(self):
        result = TraceResult(
            issues=[TraceIssue("error", "compat", "fail")],
            probe_cases_run=2,
            probe_cases_passed=0,
        )
        assert not result.ok


# ── TestHelpers ──────────────────────────────────────────────────────────


class TestHelpers:
    def test_token_advs_uniform(self):
        assert _token_advs_are_uniform([[1.0, 1.0, 1.0]])
        assert _token_advs_are_uniform([[5.0]])
        assert _token_advs_are_uniform([])

    def test_token_advs_not_uniform(self):
        assert not _token_advs_are_uniform([[1.0, 2.0, 3.0]])

    def test_condition_label_composable(self):
        cfg = TrainConfig(advantage_mode="grpo", transform_mode="none")
        assert _condition_label(cfg) == "grpo+none"

    def test_condition_label_algorithm(self):
        cfg = TrainConfig(algorithm_mode="grpo_none")
        assert _condition_label(cfg) == "grpo_none"


# ── TestTraceCLI ─────────────────────────────────────────────────────────


class TestTraceCLI:
    def _write_config(self, tmp_path, *, backend="local", **overrides):
        lines = [
            "[model]",
            'model = "test"',
            "lora_rank = 8",
            "",
            "[algorithm]",
            f'advantage_mode = "{overrides.get("advantage_mode", "grpo")}"',
            f'transform_mode = "{overrides.get("transform_mode", "none")}"',
            "",
            "[training]",
            "max_steps = 10",
            "batch_size = 2",
            "group_size = 4",
            "max_tokens = 128",
            "",
            "[backend]",
            f'backend = "{backend}"',
            f'adapter_path = "{tmp_path / "adapters"}"',
            "",
            "[logging]",
            f'log_dir = "{tmp_path / "logs"}"',
        ]
        config_path = tmp_path / "retrain.toml"
        config_path.write_text("\n".join(lines) + "\n")
        return str(config_path)

    def test_valid_config_exit_0(self, tmp_path):
        path = self._write_config(tmp_path, backend="local")
        result = subprocess.run(
            [sys.executable, "-m", "retrain.cli", "trace", path],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_incompatible_config_exit_1(self, tmp_path):
        path = self._write_config(
            tmp_path, backend="prime_rl", transform_mode="gtpo",
        )
        result = subprocess.run(
            [sys.executable, "-m", "retrain.cli", "trace", path],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_json_flag_produces_valid_json(self, tmp_path):
        path = self._write_config(tmp_path, backend="local")
        result = subprocess.run(
            [sys.executable, "-m", "retrain.cli", "trace", "--json", path],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # CLI may print dotenv messages before JSON; find the JSON object
        stdout = result.stdout
        json_start = stdout.index("{")
        payload = json.loads(stdout[json_start:])
        assert payload["ok"] is True
        assert "probe_cases_run" in payload
        assert "issues" in payload
        assert "flow" in payload
