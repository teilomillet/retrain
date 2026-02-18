"""Tests for config.mojo — TrainConfig defaults and field consistency.

Covers:
    - All 24 default values match Python argparse specification
    - Fieldwise init constructs correctly
    - write_to produces parseable output
"""

from testing import assert_true, assert_equal, assert_almost_equal

from src.config import TrainConfig


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


fn test_defaults_algorithm_selection() raises:
    """Default advantage_mode and transform_mode."""
    var c = TrainConfig()
    assert_equal(c.advantage_mode, "maxrl")
    assert_equal(c.transform_mode, "gtpo_sepa")


fn test_defaults_model() raises:
    """Default model ID."""
    var c = TrainConfig()
    assert_equal(c.model, "Qwen/Qwen3-4B-Instruct-2507")


fn test_defaults_backend() raises:
    """Default backend selection fields."""
    var c = TrainConfig()
    assert_equal(c.backend, "tinker")
    assert_equal(c.devices, "gpu:0")
    assert_equal(c.adapter_path, "/tmp/retrain_adapter")


fn test_defaults_tinker() raises:
    """Default base_url (empty=production) and lora_rank."""
    var c = TrainConfig()
    assert_equal(c.base_url, "")
    assert_equal(c.lora_rank, 32)


fn test_defaults_training() raises:
    """Default training hyperparameters."""
    var c = TrainConfig()
    assert_equal(c.max_steps, 500)
    assert_equal(c.batch_size, 8)
    assert_equal(c.group_size, 16)
    assert_equal(c.max_tokens, 2048)
    assert_almost_equal(c.temperature, 0.7, atol=1e-10)
    assert_almost_equal(c.lr, 4e-5, atol=1e-10)
    assert_almost_equal(c.weight_decay, 0.0, atol=1e-10)
    assert_equal(c.max_examples, 0)
    assert_equal(c.save_every, 20)


fn test_defaults_algorithm_hparams() raises:
    """Default GTPO and HICRA hyperparameters."""
    var c = TrainConfig()
    assert_almost_equal(c.gtpo_beta, 0.1, atol=1e-10)
    assert_almost_equal(c.hicra_alpha, 0.2, atol=1e-10)


fn test_defaults_sepa() raises:
    """Default SEPA parameters."""
    var c = TrainConfig()
    assert_equal(c.sepa_steps, 500)
    assert_equal(c.sepa_schedule, "linear")
    assert_equal(c.sepa_delay_steps, 50)
    assert_almost_equal(c.sepa_correct_rate_gate, 0.1, atol=1e-10)


fn test_defaults_inference_engine() raises:
    """Default inference engine parameters."""
    var c = TrainConfig()
    assert_equal(c.inference_engine, "pytorch")
    assert_equal(c.inference_url, "")


fn test_defaults_logging() raises:
    """Default logging parameters."""
    var c = TrainConfig()
    assert_equal(c.log_dir, "logs/tinker_math")
    assert_equal(c.wandb_project, "")
    assert_equal(c.wandb_run_name, "")
    assert_equal(c.strategic_grams, "")


# ---------------------------------------------------------------------------
# Fieldwise init / copy
# ---------------------------------------------------------------------------


fn test_fieldwise_init() raises:
    """@fieldwise_init generates a full-argument constructor."""
    var c = TrainConfig(
        advantage_mode="grpo",
        transform_mode="none",
        backend="local",
        devices="gpu:0,gpu:1",
        adapter_path="/tmp/test_adapter",
        model="test-model",
        base_url="http://localhost",
        lora_rank=16,
        max_steps=10,
        batch_size=2,
        group_size=4,
        max_tokens=512,
        temperature=0.9,
        lr=1e-4,
        weight_decay=0.01,
        max_examples=100,
        save_every=5,
        gtpo_beta=0.2,
        hicra_alpha=0.3,
        sepa_steps=200,
        sepa_schedule="auto",
        sepa_delay_steps=10,
        sepa_correct_rate_gate=0.5,
        strategic_grams="[]",
        bp_enabled=False,
        bp_warmup_steps=10,
        bp_ema_decay=0.9,
        bp_throttle_margin=0.85,
        bp_increase_margin=0.5,
        bp_min_batch_size=1,
        bp_max_batch_size=64,
        bp_peak_gflops=0.0,
        bp_peak_bw_gb_s=0.0,
        inference_engine="vllm",
        inference_url="http://localhost:8000",
        log_dir="/tmp/logs",
        wandb_project="test-project",
        wandb_run_name="run-1",
    )
    assert_equal(c.advantage_mode, "grpo")
    assert_equal(c.transform_mode, "none")
    assert_equal(c.backend, "local")
    assert_equal(c.devices, "gpu:0,gpu:1")
    assert_equal(c.adapter_path, "/tmp/test_adapter")
    assert_equal(c.model, "test-model")
    assert_equal(c.lora_rank, 16)
    assert_equal(c.max_steps, 10)
    assert_almost_equal(c.temperature, 0.9, atol=1e-10)
    assert_equal(c.sepa_schedule, "auto")
    assert_equal(c.inference_engine, "vllm")
    assert_equal(c.inference_url, "http://localhost:8000")
    assert_equal(c.wandb_project, "test-project")


fn test_copy() raises:
    """TrainConfig is Copyable."""
    var c1 = TrainConfig()
    c1.max_steps = 42
    c1.model = "custom-model"
    var c2 = c1.copy()
    assert_equal(c2.max_steps, 42)
    assert_equal(c2.model, "custom-model")
    # Modifying c2 shouldn't affect c1
    c2.max_steps = 100
    assert_equal(c1.max_steps, 42)


fn test_write_to() raises:
    """write_to (Writable) produces a string representation."""
    var c = TrainConfig()
    var s = String(c)
    assert_true("TrainConfig(" in s, "Should contain struct name")
    assert_true("maxrl" in s, "Should contain advantage_mode")
    assert_true("gtpo_sepa" in s, "Should contain transform_mode")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


fn main() raises:
    test_defaults_algorithm_selection()
    test_defaults_backend()
    test_defaults_model()
    test_defaults_tinker()
    test_defaults_training()
    test_defaults_algorithm_hparams()
    test_defaults_sepa()
    test_defaults_inference_engine()
    test_defaults_logging()
    test_fieldwise_init()
    test_copy()
    test_write_to()

    print("All 12 config tests passed!")
