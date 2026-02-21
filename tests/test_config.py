"""Tests for retrain.config — TrainConfig and TOML loading."""

import tempfile
from pathlib import Path

import pytest

from retrain.config import TrainConfig, load_config


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
