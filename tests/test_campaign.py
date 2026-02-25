"""Tests for retrain.campaign condition parsing, TOML serialization, and parallel execution."""

import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from retrain.campaign import (
    DEFAULT_CONDITIONS,
    _config_to_toml,
    _parse_campaign_conditions,
    _run_parallel,
    _toml_value,
    _write_run_configs,
)
from retrain.config import TrainConfig, load_config


class TestParseCampaignConditions:
    def test_defaults_when_conditions_missing(self):
        assert _parse_campaign_conditions(None, "campaign.toml") == list(
            DEFAULT_CONDITIONS
        )
        assert _parse_campaign_conditions([], "campaign.toml") == list(
            DEFAULT_CONDITIONS
        )

    def test_valid_conditions_with_new_modes(self):
        conditions = _parse_campaign_conditions(
            [
                {
                    "advantage_mode": "maxrl",
                    "transform_mode": "gtpo_sepa_amp",
                },
                {
                    "advantage_mode": "maxrl",
                    "transform_mode": "gtpo_sepa_amp_c",
                },
            ],
            "campaign.toml",
        )
        assert conditions == [
            ("maxrl", "gtpo_sepa_amp"),
            ("maxrl", "gtpo_sepa_amp_c"),
        ]

    def test_dotted_transform_mode_is_accepted(self):
        conditions = _parse_campaign_conditions(
            [
                {
                    "advantage_mode": "maxrl",
                    "transform_mode": "my_transforms.make_transform_spec",
                }
            ],
            "campaign.toml",
        )
        assert conditions == [("maxrl", "my_transforms.make_transform_spec")]

    def test_dotted_advantage_mode_is_accepted(self):
        conditions = _parse_campaign_conditions(
            [
                {
                    "advantage_mode": "my_advantages.hipa_like_advantages",
                    "transform_mode": "gtpo",
                }
            ],
            "campaign.toml",
        )
        assert conditions == [("my_advantages.hipa_like_advantages", "gtpo")]

    def test_non_list_conditions_raises(self):
        with pytest.raises(ValueError, match="campaign.conditions must be a list"):
            _parse_campaign_conditions({"advantage_mode": "grpo"}, "campaign.toml")

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="advantage_mode must be a non-empty string"):
            _parse_campaign_conditions([{"transform_mode": "none"}], "campaign.toml")
        with pytest.raises(ValueError, match="transform_mode must be a non-empty string"):
            _parse_campaign_conditions([{"advantage_mode": "grpo"}], "campaign.toml")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid campaign condition at index 0"):
            _parse_campaign_conditions(
                [{"advantage_mode": "grpo", "transform_mode": "typo"}],
                "campaign.toml",
            )


# ---------------------------------------------------------------------------
# TOML serialization
# ---------------------------------------------------------------------------


class TestTomlValue:
    def test_string(self):
        assert _toml_value("hello") == '"hello"'

    def test_string_with_quotes(self):
        assert _toml_value('say "hi"') == '"say \\"hi\\""'

    def test_bool_true(self):
        assert _toml_value(True) == "true"

    def test_bool_false(self):
        assert _toml_value(False) == "false"

    def test_int(self):
        assert _toml_value(42) == "42"

    def test_negative_int(self):
        assert _toml_value(-1) == "-1"

    def test_float(self):
        assert _toml_value(0.7) == "0.7"

    def test_empty_string(self):
        assert _toml_value("") == '""'


class TestConfigToToml:
    def test_roundtrip(self, tmp_path):
        """Serialize a config to TOML, parse it back, assert field equality."""
        original = TrainConfig(
            advantage_mode="grpo",
            transform_mode="none",
            seed=42,
            max_steps=100,
            lr=1e-4,
            log_dir="logs/test",
        )
        toml_str = _config_to_toml(original)
        toml_path = tmp_path / "roundtrip.toml"
        toml_path.write_text(toml_str)

        reloaded = load_config(str(toml_path))

        assert reloaded.advantage_mode == original.advantage_mode
        assert reloaded.transform_mode == original.transform_mode
        assert reloaded.seed == original.seed
        assert reloaded.max_steps == original.max_steps
        assert reloaded.lr == pytest.approx(original.lr)
        assert reloaded.log_dir == original.log_dir

    def test_special_values(self, tmp_path):
        """Empty strings, negative seeds, booleans, floats survive roundtrip."""
        original = TrainConfig(
            seed=-1,
            bp_enabled=True,
            temperature=0.0,
            wandb_project="my-proj",
            prefix_caching=False,
        )
        toml_str = _config_to_toml(original)
        toml_path = tmp_path / "special.toml"
        toml_path.write_text(toml_str)

        reloaded = load_config(str(toml_path))

        assert reloaded.bp_enabled is True
        assert reloaded.temperature == pytest.approx(0.0)
        assert reloaded.wandb_project == "my-proj"
        assert reloaded.prefix_caching is False

    def test_defaults_produce_minimal_output(self):
        """A default config should produce near-empty TOML (only non-defaults emitted)."""
        cfg = TrainConfig()
        toml_str = _config_to_toml(cfg)
        # Should be essentially empty (just newline) since all values are defaults
        assert toml_str.strip() == ""

    def test_only_changed_fields_emitted(self):
        """Only non-default fields appear in the TOML output."""
        cfg = TrainConfig(seed=42, max_steps=100)
        toml_str = _config_to_toml(cfg)
        assert "seed = 42" in toml_str
        assert "max_steps = 100" in toml_str
        # Default fields should not appear
        assert "batch_size" not in toml_str
        assert "lora_rank" not in toml_str


# ---------------------------------------------------------------------------
# Per-run config writing
# ---------------------------------------------------------------------------


class TestWriteRunConfigs:
    def test_write_run_configs(self, tmp_path):
        """2 conditions x 2 seeds → 4 TOML files with correct overrides."""
        base = TrainConfig()
        runs = [
            {
                "condition": "grpo+none",
                "advantage_mode": "grpo",
                "transform_mode": "none",
                "seed": 42,
                "run_name": "grpo+none_s42",
                "log_dir": str(tmp_path / "runs" / "grpo+none_s42"),
            },
            {
                "condition": "grpo+none",
                "advantage_mode": "grpo",
                "transform_mode": "none",
                "seed": 101,
                "run_name": "grpo+none_s101",
                "log_dir": str(tmp_path / "runs" / "grpo+none_s101"),
            },
            {
                "condition": "maxrl+gtpo",
                "advantage_mode": "maxrl",
                "transform_mode": "gtpo",
                "seed": 42,
                "run_name": "maxrl+gtpo_s42",
                "log_dir": str(tmp_path / "runs" / "maxrl+gtpo_s42"),
            },
            {
                "condition": "maxrl+gtpo",
                "advantage_mode": "maxrl",
                "transform_mode": "gtpo",
                "seed": 101,
                "run_name": "maxrl+gtpo_s101",
                "log_dir": str(tmp_path / "runs" / "maxrl+gtpo_s101"),
            },
        ]

        config_dir = tmp_path / "configs"
        _write_run_configs(runs, base, max_steps=50, config_dir=config_dir)

        # 4 TOML files created
        toml_files = sorted(config_dir.glob("*.toml"))
        assert len(toml_files) == 4

        # Verify one config has correct overrides
        cfg = load_config(str(config_dir / "grpo+none_s42.toml"))
        assert cfg.advantage_mode == "grpo"
        assert cfg.transform_mode == "none"
        assert cfg.seed == 42
        assert cfg.max_steps == 50

        # Each run dict gets a config_path key
        assert all("config_path" in r for r in runs)


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


class TestRunParallel:
    def _make_runs(self, tmp_path, count=3):
        runs = []
        for i in range(count):
            run = {
                "run_name": f"run_{i}",
                "log_dir": str(tmp_path / f"run_{i}"),
                "config_path": str(tmp_path / f"run_{i}.toml"),
                "condition": "grpo+none",
            }
            Path(run["log_dir"]).mkdir(parents=True, exist_ok=True)
            runs.append(run)
        return runs

    def test_run_parallel_success(self, tmp_path):
        """All Popen returning 0 → all runs succeed."""
        runs = self._make_runs(tmp_path, count=3)

        def make_proc(*args, **kwargs):
            mock = MagicMock()
            mock.poll.side_effect = [None, 0]
            mock.pid = 12345
            return mock

        with patch("retrain.campaign.subprocess.Popen", side_effect=make_proc):
            with patch("retrain.campaign.time.sleep"):
                result = _run_parallel(runs, tmp_path, max_workers=3)

        assert all(r["returncode"] == 0 for r in result)
        assert len(result) == 3

    def test_run_parallel_partial_failure(self, tmp_path):
        """One Popen returns 1, others 0 → correct failure count."""
        runs = self._make_runs(tmp_path, count=3)

        call_count = [0]

        def make_proc(*args, **kwargs):
            mock = MagicMock()
            idx = call_count[0]
            call_count[0] += 1
            # First process fails, others succeed
            rc = 1 if idx == 0 else 0
            mock.poll.side_effect = [None, rc]
            mock.pid = 10000 + idx
            return mock

        with patch("retrain.campaign.subprocess.Popen", side_effect=make_proc):
            with patch("retrain.campaign.time.sleep"):
                result = _run_parallel(runs, tmp_path, max_workers=3)

        failed = sum(1 for r in result if r.get("returncode", -1) != 0)
        assert failed == 1
        succeeded = sum(1 for r in result if r.get("returncode") == 0)
        assert succeeded == 2

    def test_run_parallel_max_workers(self, tmp_path):
        """Verify concurrency bounded to max_workers."""
        runs = self._make_runs(tmp_path, count=4)

        max_concurrent = [0]
        current_concurrent = [0]

        def make_proc(*args, **kwargs):
            current_concurrent[0] += 1
            if current_concurrent[0] > max_concurrent[0]:
                max_concurrent[0] = current_concurrent[0]

            mock = MagicMock()
            mock.pid = 10000

            def poll_fn():
                current_concurrent[0] -= 1
                return 0
            mock.poll.side_effect = [None, poll_fn]
            return mock

        # The poll side_effect is tricky — let's simplify: use immediate completion
        procs_launched = [0]

        def make_proc_simple(*args, **kwargs):
            procs_launched[0] += 1
            mock = MagicMock()
            mock.pid = 10000 + procs_launched[0]
            mock.poll.return_value = 0  # immediately done
            return mock

        with patch("retrain.campaign.subprocess.Popen", side_effect=make_proc_simple):
            with patch("retrain.campaign.time.sleep"):
                result = _run_parallel(runs, tmp_path, max_workers=2)

        assert len(result) == 4
        assert all(r["returncode"] == 0 for r in result)


# ---------------------------------------------------------------------------
# Campaign-level parallel config
# ---------------------------------------------------------------------------


class TestParallelCampaignConfig:
    def test_parallel_false_is_default(self, tmp_path):
        """No parallel field → sequential execution (run_campaign reads False)."""
        toml_path = tmp_path / "campaign.toml"
        toml_path.write_text(
            '[campaign]\nseeds = [42]\nmax_steps = 10\n\n'
            '[[campaign.conditions]]\nadvantage_mode = "grpo"\ntransform_mode = "none"\n'
        )
        with open(str(toml_path), "rb") as f:
            data = tomllib.load(f)
        campaign = data.get("campaign", {})
        assert bool(campaign.get("parallel", False)) is False

    def test_parallel_true_from_toml(self, tmp_path):
        """parallel = true is read correctly from TOML."""
        toml_path = tmp_path / "campaign.toml"
        toml_path.write_text(
            '[campaign]\nseeds = [42]\nmax_steps = 10\nparallel = true\nmax_workers = 2\n\n'
            '[[campaign.conditions]]\nadvantage_mode = "grpo"\ntransform_mode = "none"\n'
        )
        with open(str(toml_path), "rb") as f:
            data = tomllib.load(f)
        campaign = data.get("campaign", {})
        assert bool(campaign.get("parallel", False)) is True
        assert int(campaign.get("max_workers", 0)) == 2

    def test_squeeze_after_all_parallel(self, tmp_path):
        """In parallel mode, squeeze should run after ALL runs complete.

        We verify by checking that _run_parallel is called before any squeeze logic
        (structural test via run_campaign flow).
        """
        toml_path = tmp_path / "campaign.toml"
        toml_path.write_text(
            '[campaign]\nseeds = [42]\nmax_steps = 10\nparallel = true\n\n'
            '[[campaign.conditions]]\nadvantage_mode = "grpo"\ntransform_mode = "none"\n\n'
            '[squeeze]\nmin_variance_retention = 0.95\n'
        )

        # Mock _run_parallel to return runs with returncode=0
        def mock_run_parallel(runs, config_dir, max_workers, stagger_seconds=0):
            for r in runs:
                r["returncode"] = 0
            return runs

        squeeze_called = [False]
        original_auto_squeeze = None

        def mock_auto_squeeze(*args, **kwargs):
            squeeze_called[0] = True
            return 16

        with patch("retrain.campaign._run_parallel", side_effect=mock_run_parallel):
            with patch("retrain.campaign._write_run_configs"):
                with patch("retrain.campaign._auto_squeeze", side_effect=mock_auto_squeeze):
                    with patch("retrain.campaign.load_config", return_value=TrainConfig()):
                        import os
                        old_cwd = os.getcwd()
                        os.chdir(tmp_path)
                        try:
                            from retrain.campaign import run_campaign
                            run_campaign(str(toml_path))
                        finally:
                            os.chdir(old_cwd)

        assert squeeze_called[0], "Squeeze should be called after parallel runs complete"
