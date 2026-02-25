"""Tests for retrain.config — TrainConfig and TOML loading."""

import tempfile
import warnings
from pathlib import Path

import pytest

from retrain.config import TrainConfig, load_config, parse_cli_overrides


class TestDefaults:
    def test_default_values(self):
        c = TrainConfig()
        assert c.advantage_mode == "maxrl"
        assert c.transform_mode == "gtpo_sepa"
        assert c.backend == "local"
        assert c.model == "Qwen/Qwen3-4B-Instruct-2507"
        assert c.lora_rank == 32
        assert c.max_steps == 500
        assert c.batch_size == 8
        assert c.group_size == 16
        assert c.max_tokens == 2048
        assert c.temperature == pytest.approx(0.7)
        assert c.lr == pytest.approx(4e-5)
        assert c.weight_decay == pytest.approx(0.0)
        assert c.sepa_steps == 500
        assert c.sepa_schedule == "linear"
        assert c.sepa_delay_steps == 50
        assert c.bp_enabled is False
        assert c.inference_engine == "pytorch"
        assert c.prefix_caching is True
        assert c.log_dir == "logs/train"


class TestLoadConfig:
    def test_load_minimal_toml(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text('[training]\nmax_steps = 100\n')
        c = load_config(str(toml))
        assert c.max_steps == 100
        # Other defaults preserved
        assert c.batch_size == 8

    def test_all_sections(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text("""\
[algorithm]
advantage_mode = "grpo"
transform_mode = "none"

[backend]
backend = "tinker"
devices = "gpu:0,gpu:1"

[model]
model = "meta-llama/Llama-3-8B"
lora_rank = 16

[training]
max_steps = 200
batch_size = 4
group_size = 8
max_tokens = 1024
temperature = 0.9
lr = 1e-4
weight_decay = 0.01
save_every = 50

[gtpo]
beta = 0.2

[hicra]
alpha = 0.3

[sepa]
steps = 300
schedule = "auto"
delay_steps = 100
correct_rate_gate = 0.2

[inference]
engine = "vllm"
url = "http://localhost:8000"
attention_kernel = "flash"
dtype = "bf16"
kv_cache_dtype = "fp8"
prefix_caching = false

[backpressure]
enabled = true
warmup_steps = 20
ema_decay = 0.95

[logging]
log_dir = "logs/custom"
wandb_project = "my-project"
wandb_run_name = "run-1"
""")
        c = load_config(str(toml))
        assert c.advantage_mode == "grpo"
        assert c.transform_mode == "none"
        assert c.backend == "tinker"
        assert c.devices == "gpu:0,gpu:1"
        assert c.model == "meta-llama/Llama-3-8B"
        assert c.lora_rank == 16
        assert c.max_steps == 200
        assert c.batch_size == 4
        assert c.temperature == pytest.approx(0.9)
        assert c.lr == pytest.approx(1e-4)
        assert c.weight_decay == pytest.approx(0.01)
        assert c.gtpo_beta == pytest.approx(0.2)
        assert c.hicra_alpha == pytest.approx(0.3)
        assert c.sepa_steps == 300
        assert c.sepa_schedule == "auto"
        assert c.sepa_delay_steps == 100
        assert c.sepa_correct_rate_gate == pytest.approx(0.2)
        assert c.inference_engine == "vllm"
        assert c.inference_url == "http://localhost:8000"
        assert c.attention_kernel == "flash"
        assert c.inference_dtype == "bf16"
        assert c.kv_cache_dtype == "fp8"
        assert c.prefix_caching is False
        assert c.bp_enabled is True
        assert c.bp_warmup_steps == 20
        assert c.bp_ema_decay == pytest.approx(0.95)
        assert c.log_dir == "logs/custom"
        assert c.wandb_project == "my-project"
        assert c.wandb_run_name == "run-1"

    def test_empty_string_ignored(self, tmp_path):
        """Empty-string TOML values should keep the default (match Mojo behavior)."""
        toml = tmp_path / "config.toml"
        toml.write_text('[model]\nmodel = ""\n')
        c = load_config(str(toml))
        assert c.model == "Qwen/Qwen3-4B-Instruct-2507"

    def test_type_coercion_float_from_int(self, tmp_path):
        """TOML integer should be coerced to float for float fields."""
        toml = tmp_path / "config.toml"
        toml.write_text('[training]\ntemperature = 1\n')
        c = load_config(str(toml))
        assert c.temperature == pytest.approx(1.0)
        assert isinstance(c.temperature, float)

    def test_no_file_returns_defaults(self):
        """When no path given and no retrain.toml in cwd, returns defaults."""
        c = load_config(None)
        assert c.max_steps == 500

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.toml")

    def test_unknown_sections_ignored(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text('[unknown_section]\nfoo = "bar"\n\n[training]\nmax_steps = 42\n')
        c = load_config(str(toml))
        assert c.max_steps == 42

    def test_unknown_keys_ignored(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text('[training]\nmax_steps = 42\nunknown_key = "value"\n')
        c = load_config(str(toml))
        assert c.max_steps == 42

    def test_resume_from_default(self):
        c = TrainConfig()
        assert c.resume_from == ""

    def test_resume_from_toml(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text('[resume]\nfrom = "logs/my_run"\n')
        c = load_config(str(toml))
        assert c.resume_from == "logs/my_run"

    def test_seed_default(self):
        c = TrainConfig()
        assert c.seed == -1

    def test_seed_from_toml(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text("[training]\nseed = 42\n")
        c = load_config(str(toml))
        assert c.seed == 42

    def test_wandb_entity_tags_group_defaults(self):
        c = TrainConfig()
        assert c.wandb_entity == ""
        assert c.wandb_group == ""
        assert c.wandb_tags == ""

    def test_wandb_extended_from_toml(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text(
            '[logging]\n'
            'wandb_entity = "my-team"\n'
            'wandb_group = "sweep-1"\n'
            'wandb_tags = "baseline,seed42"\n'
        )
        c = load_config(str(toml))
        assert c.wandb_entity == "my-team"
        assert c.wandb_group == "sweep-1"
        assert c.wandb_tags == "baseline,seed42"

    def test_new_fields_defaults(self):
        c = TrainConfig()
        assert c.top_p == pytest.approx(0.95)
        assert c.optim_beta1 == pytest.approx(0.9)
        assert c.optim_beta2 == pytest.approx(0.95)
        assert c.optim_eps == pytest.approx(1e-8)
        assert c.lora_alpha == 0
        assert c.lora_dropout == pytest.approx(0.0)
        assert c.prime_rl_transport == "filesystem"
        assert c.prime_rl_zmq_host == "localhost"
        assert c.prime_rl_zmq_port == 5555
        assert c.prime_rl_zmq_hwm == 10
        assert c.prime_rl_strict_advantages is True
        assert c.prime_rl_sync_wait_s == 30
        assert c.prime_rl_sync_poll_s == pytest.approx(0.2)
        assert c.environment_provider == ""
        assert c.environment_id == ""
        assert c.environment_args == ""
        assert c.environment_max_turns == -1
        assert c.environment_auto_install is False

    def test_new_fields_from_toml(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text(
            '[training]\ntop_p = 0.9\n\n'
            '[optimizer]\nbeta1 = 0.85\nbeta2 = 0.99\neps = 1e-6\n\n'
            '[lora]\nalpha = 64\ndropout = 0.05\n\n'
            '[backend]\n'
            'prime_rl_transport = "zmq"\n'
            'prime_rl_zmq_host = "127.0.0.1"\n'
            'prime_rl_zmq_port = 7777\n'
            'prime_rl_zmq_hwm = 32\n'
            'prime_rl_strict_advantages = false\n'
            'prime_rl_sync_wait_s = 5\n'
            'prime_rl_sync_poll_s = 0.5\n'
        )
        c = load_config(str(toml))
        assert c.top_p == pytest.approx(0.9)
        assert c.optim_beta1 == pytest.approx(0.85)
        assert c.optim_beta2 == pytest.approx(0.99)
        assert c.optim_eps == pytest.approx(1e-6)
        assert c.lora_alpha == 64
        assert c.lora_dropout == pytest.approx(0.05)
        assert c.prime_rl_transport == "zmq"
        assert c.prime_rl_zmq_host == "127.0.0.1"
        assert c.prime_rl_zmq_port == 7777
        assert c.prime_rl_zmq_hwm == 32
        assert c.prime_rl_strict_advantages is False
        assert c.prime_rl_sync_wait_s == 5
        assert c.prime_rl_sync_poll_s == pytest.approx(0.5)

    def test_environment_fields_from_toml(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text(
            '[environment]\n'
            'provider = "verifiers"\n'
            'id = "primeintellect/aime"\n'
            'args = "{\\"split\\": \\"train\\"}"\n'
            'max_turns = 8\n'
            'auto_install = true\n'
        )
        c = load_config(str(toml))
        assert c.environment_provider == "verifiers"
        assert c.environment_id == "primeintellect/aime"
        assert c.environment_args == '{"split": "train"}'
        assert c.environment_max_turns == 8
        assert c.environment_auto_install is True

    def test_environment_args_table_from_toml(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text(
            '[environment]\n'
            'provider = "verifiers"\n'
            'id = "primeintellect/aime"\n'
            'args = { split = "train", seed = 7 }\n'
        )
        c = load_config(str(toml))
        assert c.environment_provider == "verifiers"
        assert c.environment_id == "primeintellect/aime"
        assert c.environment_args == '{"split": "train", "seed": 7}'


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_advantage_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid advantage_mode"):
            TrainConfig(advantage_mode="invalid")

    def test_invalid_transform_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid transform_mode"):
            TrainConfig(transform_mode="typo")

    def test_valid_modes_accepted(self):
        for am in ("grpo", "maxrl"):
            for tm in (
                "none",
                "gtpo",
                "entropy_mask",
                "gtpo_hicra",
                "gtpo_sepa",
                "gtpo_sepa_amp",
                "gtpo_sepa_amp_c",
            ):
                c = TrainConfig(advantage_mode=am, transform_mode=tm)
                assert c.advantage_mode == am
                assert c.transform_mode == tm

    def test_invalid_mode_from_toml_raises(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text('[algorithm]\nadvantage_mode = "wrong"\n')
        with pytest.raises(ValueError, match="Invalid advantage_mode"):
            load_config(str(toml))

    def test_dotted_transform_mode_accepted(self):
        c = TrainConfig(transform_mode="custom_transforms.make_transform_spec")
        assert c.transform_mode == "custom_transforms.make_transform_spec"

    def test_malformed_dotted_transform_mode_rejected(self):
        with pytest.raises(ValueError, match="Invalid transform_mode"):
            TrainConfig(transform_mode="custom_transforms.")

    def test_invalid_environment_provider_raises(self):
        with pytest.raises(ValueError, match="Invalid environment_provider"):
            TrainConfig(environment_provider="unknown")

    def test_environment_provider_requires_id(self):
        with pytest.raises(ValueError, match="environment_id is required"):
            TrainConfig(environment_provider="verifiers", environment_id="")

    def test_environment_args_must_be_object_when_provider_set(self):
        with pytest.raises(ValueError, match="must decode to a JSON object"):
            TrainConfig(
                environment_provider="verifiers",
                environment_id="primeintellect/aime",
                environment_args='["not","an","object"]',
            )


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    def test_positional_and_flags(self):
        path, overrides = parse_cli_overrides(["config.toml", "--seed", "42"])
        assert path == "config.toml"
        assert overrides == {"seed": "42"}

    def test_flags_only(self):
        path, overrides = parse_cli_overrides(["--seed", "42", "--lr", "1e-4"])
        assert path is None
        assert overrides == {"seed": "42", "lr": "1e-4"}

    def test_flag_equals_syntax(self):
        path, overrides = parse_cli_overrides(["--seed=42"])
        assert path is None
        assert overrides == {"seed": "42"}

    def test_unknown_flag_exits(self):
        with pytest.raises(SystemExit):
            parse_cli_overrides(["--sead", "42"])

    def test_resume_alias(self):
        _, overrides = parse_cli_overrides(["--resume", "/path/to/logs"])
        assert overrides == {"resume_from": "/path/to/logs"}

    def test_kebab_to_snake(self):
        _, overrides = parse_cli_overrides(["--batch-size", "4"])
        assert overrides == {"batch_size": "4"}

    def test_overrides_applied_to_config(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text("[training]\nmax_steps = 200\n")
        c = load_config(str(toml), overrides={"seed": "42"})
        assert c.seed == 42
        assert c.max_steps == 200

    def test_cli_beats_toml(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text("[training]\nmax_steps = 200\n")
        c = load_config(str(toml), overrides={"max_steps": "50"})
        assert c.max_steps == 50

    def test_bool_override(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text("")
        c = load_config(str(toml), overrides={"bp_enabled": "true"})
        assert c.bp_enabled is True

    def test_missing_value_exits(self):
        with pytest.raises(SystemExit):
            parse_cli_overrides(["--seed"])


# ---------------------------------------------------------------------------
# Numeric validation
# ---------------------------------------------------------------------------


class TestNumericValidation:
    def test_batch_size_zero(self):
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            TrainConfig(batch_size=0)

    def test_group_size_zero(self):
        with pytest.raises(ValueError, match="group_size must be > 0"):
            TrainConfig(group_size=0)

    def test_lr_zero(self):
        with pytest.raises(ValueError, match="lr must be > 0"):
            TrainConfig(lr=0)

    def test_lr_negative(self):
        with pytest.raises(ValueError, match="lr must be > 0"):
            TrainConfig(lr=-1e-5)

    def test_max_steps_zero(self):
        with pytest.raises(ValueError, match="max_steps must be > 0"):
            TrainConfig(max_steps=0)

    def test_max_tokens_zero(self):
        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            TrainConfig(max_tokens=0)

    def test_lora_rank_zero(self):
        with pytest.raises(ValueError, match="lora_rank must be > 0"):
            TrainConfig(lora_rank=0)

    def test_temperature_negative(self):
        with pytest.raises(ValueError, match="temperature must be >= 0"):
            TrainConfig(temperature=-0.1)

    def test_temperature_zero_ok(self):
        c = TrainConfig(temperature=0.0)
        assert c.temperature == 0.0

    def test_top_p_zero(self):
        with pytest.raises(ValueError, match="top_p must be in"):
            TrainConfig(top_p=0.0)

    def test_top_p_above_one(self):
        with pytest.raises(ValueError, match="top_p must be in"):
            TrainConfig(top_p=1.1)

    def test_multiple_errors_batched(self):
        with pytest.raises(ValueError) as exc_info:
            TrainConfig(batch_size=0, lr=-1)
        msg = str(exc_info.value)
        assert "batch_size" in msg
        assert "lr" in msg


# ---------------------------------------------------------------------------
# Validation warnings
# ---------------------------------------------------------------------------


class TestValidationWarnings:
    def test_tmp_adapter_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TrainConfig(adapter_path="/tmp/foo")
        msgs = [str(x.message) for x in w]
        assert any("/tmp" in m for m in msgs)

    def test_high_temperature_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TrainConfig(temperature=3.0)
        msgs = [str(x.message) for x in w]
        assert any("unusually high" in m for m in msgs)

    def test_save_every_zero_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TrainConfig(save_every=0)
        msgs = [str(x.message) for x in w]
        assert any("disables periodic" in m for m in msgs)


# ---------------------------------------------------------------------------
# Entropy mask rho
# ---------------------------------------------------------------------------


class TestEntropyMaskRho:
    def test_default_value(self):
        c = TrainConfig()
        assert c.entropy_mask_rho == pytest.approx(0.0)

    def test_toml_loading(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text('[algorithm]\nentropy_mask_rho = 0.2\n')
        c = load_config(str(toml))
        assert c.entropy_mask_rho == pytest.approx(0.2)

    def test_out_of_range_negative(self):
        with pytest.raises(ValueError, match="entropy_mask_rho"):
            TrainConfig(entropy_mask_rho=-0.1)

    def test_out_of_range_above_one(self):
        with pytest.raises(ValueError, match="entropy_mask_rho"):
            TrainConfig(entropy_mask_rho=1.1)

    def test_boundary_zero(self):
        c = TrainConfig(entropy_mask_rho=0.0)
        assert c.entropy_mask_rho == 0.0

    def test_boundary_one(self):
        c = TrainConfig(entropy_mask_rho=1.0)
        assert c.entropy_mask_rho == 1.0

    def test_post_process_params_property(self):
        c = TrainConfig(entropy_mask_rho=0.3)
        params = c.post_process_params
        assert params == {"entropy_mask_rho": 0.3}

    def test_post_process_params_default(self):
        c = TrainConfig()
        params = c.post_process_params
        assert params == {"entropy_mask_rho": 0.0}
