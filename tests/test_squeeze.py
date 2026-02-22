"""Tests for retrain.squeeze — LoRA-Squeeze rank analysis and compression."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from retrain.config import SqueezeConfig, load_squeeze_config
from retrain.squeeze import (
    SqueezeAnalysis,
    analyze_adapter,
    compress_adapter,
    compress_layer,
    load_adapter_matrices,
    squeeze_layer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lora_pair(m: int, n: int, r: int, seed: int = 0):
    """Create random A (m, r) and B (r, n) matrices."""
    gen = torch.Generator().manual_seed(seed)
    A = torch.randn(m, r, generator=gen)
    B = torch.randn(r, n, generator=gen)
    return A, B


def _make_peft_adapter(tmp_path: Path, m: int, n: int, r: int, modules: list[str] | None = None):
    """Create a mock PEFT adapter directory with safetensors + config."""
    if modules is None:
        modules = ["model.layers.0.self_attn.q_proj"]

    state_dict = {}
    for mod in modules:
        gen = torch.Generator().manual_seed(42)
        # PEFT convention: lora_A is (r, m), lora_B is (n, r)
        a = torch.randn(r, m, generator=gen)
        b = torch.randn(n, r, generator=gen)
        state_dict[f"{mod}.lora_A.weight"] = a
        state_dict[f"{mod}.lora_B.weight"] = b

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    save_file(state_dict, str(adapter_dir / "adapter_model.safetensors"))

    config = {"r": r, "lora_alpha": r * 2}
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

    return adapter_dir


# ---------------------------------------------------------------------------
# SVD correctness
# ---------------------------------------------------------------------------

class TestSVDCorrectness:
    def test_singular_values_match_direct_svd(self):
        """Singular values from memory-efficient path match torch.linalg.svd(A @ B)."""
        A, B = _make_lora_pair(64, 64, 16, seed=7)

        # Direct SVD of the full product
        full = A @ B
        _, S_direct, _ = torch.linalg.svd(full)

        # Memory-efficient path
        result = squeeze_layer(A, B, list(range(1, 17)))
        S_efficient = result.singular_values

        # They should match (both are sorted descending)
        assert S_efficient.shape == (16,)
        torch.testing.assert_close(S_efficient, S_direct[:16], atol=1e-4, rtol=1e-4)

    def test_singular_values_non_negative(self):
        A, B = _make_lora_pair(32, 32, 8)
        result = squeeze_layer(A, B, [1, 4, 8])
        assert (result.singular_values >= 0).all()

    def test_singular_values_sorted_descending(self):
        A, B = _make_lora_pair(64, 48, 16)
        result = squeeze_layer(A, B, [1, 8, 16])
        svs = result.singular_values
        for i in range(len(svs) - 1):
            assert svs[i] >= svs[i + 1] - 1e-6


# ---------------------------------------------------------------------------
# Variance
# ---------------------------------------------------------------------------

class TestVariance:
    def test_full_rank_variance_is_one(self):
        A, B = _make_lora_pair(64, 64, 16)
        result = squeeze_layer(A, B, [16])
        assert result.variance_at_rank[16] == pytest.approx(1.0)

    def test_monotonically_non_decreasing(self):
        A, B = _make_lora_pair(64, 64, 16)
        ranks = list(range(1, 17))
        result = squeeze_layer(A, B, ranks)
        for i in range(len(ranks) - 1):
            assert result.variance_at_rank[ranks[i]] <= result.variance_at_rank[ranks[i + 1]] + 1e-9

    def test_single_rank_positive(self):
        A, B = _make_lora_pair(32, 32, 8)
        result = squeeze_layer(A, B, [1])
        assert result.variance_at_rank[1] > 0.0

    def test_zero_matrix_variance(self):
        """Zero matrices should give variance 1.0 (degenerate case)."""
        A = torch.zeros(32, 8)
        B = torch.zeros(8, 32)
        result = squeeze_layer(A, B, [1, 4, 8])
        for k in [1, 4, 8]:
            assert result.variance_at_rank[k] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Compression fidelity
# ---------------------------------------------------------------------------

class TestCompressionFidelity:
    def test_full_rank_exact(self):
        """Compression at full rank should reproduce A @ B exactly."""
        A, B = _make_lora_pair(64, 48, 16, seed=3)
        original = A @ B

        A_tgt, B_tgt = compress_layer(A, B, 16)
        reconstructed = A_tgt @ B_tgt

        torch.testing.assert_close(reconstructed, original, atol=1e-4, rtol=1e-4)

    def test_compressed_shapes(self):
        A, B = _make_lora_pair(64, 48, 16)
        A_tgt, B_tgt = compress_layer(A, B, 4)
        assert A_tgt.shape == (64, 4)
        assert B_tgt.shape == (4, 48)

    def test_compression_reduces_error_monotonically(self):
        """Higher target rank -> lower reconstruction error."""
        A, B = _make_lora_pair(64, 64, 16, seed=5)
        original = A @ B

        errors = []
        for k in [2, 4, 8, 16]:
            A_tgt, B_tgt = compress_layer(A, B, k)
            err = torch.norm(original - A_tgt @ B_tgt).item()
            errors.append(err)

        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-6

    def test_target_rank_larger_than_source(self):
        """Target rank > source rank should clamp to source rank."""
        A, B = _make_lora_pair(32, 32, 8)
        A_tgt, B_tgt = compress_layer(A, B, 32)
        assert A_tgt.shape == (32, 8)
        assert B_tgt.shape == (8, 32)


# ---------------------------------------------------------------------------
# PEFT format round-trip
# ---------------------------------------------------------------------------

class TestPEFTFormat:
    def test_load_adapter_matrices(self, tmp_path):
        """Load mock PEFT safetensors with correct pairing/transposition."""
        r, m, n = 8, 64, 64
        adapter_dir = _make_peft_adapter(tmp_path, m, n, r)
        pairs = load_adapter_matrices(str(adapter_dir))

        assert len(pairs) == 1
        name, A, B = pairs[0]
        assert "q_proj" in name
        assert A.shape == (m, r)  # transposed from PEFT's (r, m)
        assert B.shape == (r, n)  # transposed from PEFT's (n, r)

    def test_load_multiple_modules(self, tmp_path):
        modules = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.mlp.gate_proj",
        ]
        adapter_dir = _make_peft_adapter(tmp_path, 64, 64, 8, modules)
        pairs = load_adapter_matrices(str(adapter_dir))
        assert len(pairs) == 3
        names = [p[0] for p in pairs]
        for mod in modules:
            assert mod in names

    def test_missing_safetensors_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="safetensors"):
            load_adapter_matrices(str(tmp_path))

    def test_compress_adapter_round_trip(self, tmp_path):
        """Compress adapter and verify output is valid PEFT format."""
        r, m, n = 16, 64, 64
        adapter_dir = _make_peft_adapter(tmp_path, m, n, r)
        output_dir = tmp_path / "compressed"

        compress_adapter(str(adapter_dir), str(output_dir), target_rank=4)

        # Verify output files exist
        assert (output_dir / "adapter_model.safetensors").is_file()
        assert (output_dir / "adapter_config.json").is_file()

        # Verify config
        with open(output_dir / "adapter_config.json") as f:
            config = json.load(f)
        assert config["r"] == 4
        # alpha scaled: 32 * 4 / 16 = 8
        assert config["lora_alpha"] == 8

        # Verify we can reload the compressed adapter
        pairs = load_adapter_matrices(str(output_dir))
        assert len(pairs) == 1
        _, A, B = pairs[0]
        assert A.shape == (m, 4)
        assert B.shape == (4, n)


# ---------------------------------------------------------------------------
# Adapter-level analysis
# ---------------------------------------------------------------------------

class TestAnalyzeAdapter:
    def test_analyze_basic(self, tmp_path):
        modules = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
        ]
        adapter_dir = _make_peft_adapter(tmp_path, 64, 64, 16, modules)

        analysis = analyze_adapter(str(adapter_dir), min_variance_retention=0.95)

        assert len(analysis.layers) == 2
        assert analysis.recommended_rank <= 16
        assert analysis.min_variance_retention == 0.95
        # Auto target ranks should be powers of 2
        assert 1 in analysis.target_ranks
        assert 16 in analysis.target_ranks

    def test_recommended_rank_respects_threshold(self, tmp_path):
        adapter_dir = _make_peft_adapter(tmp_path, 64, 64, 16)

        # With threshold 0.0, should recommend rank 1
        analysis = analyze_adapter(str(adapter_dir), min_variance_retention=0.0)
        assert analysis.recommended_rank == 1

        # With threshold 1.0, should recommend full rank
        analysis = analyze_adapter(str(adapter_dir), min_variance_retention=1.0)
        assert analysis.recommended_rank == 16


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestSqueezeConfig:
    def test_defaults(self):
        cfg = SqueezeConfig()
        assert cfg.adapter_path == ""
        assert cfg.source_rank == 0
        assert cfg.target_ranks == []
        assert cfg.min_variance_retention == 0.95
        assert cfg.output_path == ""
        assert cfg.compress_to == 0
        assert cfg.device == "cpu"

    def test_load_squeeze_config(self, tmp_path):
        toml = tmp_path / "squeeze.toml"
        toml.write_text("""\
[squeeze]
adapter_path = "/tmp/my_adapter"
min_variance_retention = 0.90
output_path = "/tmp/compressed"
compress_to = 8

[model]
lora_rank = 128
""")
        cfg = load_squeeze_config(str(toml))
        assert cfg.adapter_path == "/tmp/my_adapter"
        assert cfg.min_variance_retention == pytest.approx(0.90)
        assert cfg.output_path == "/tmp/compressed"
        assert cfg.compress_to == 8
        assert cfg.source_rank == 128  # fallback from [model]

    def test_missing_squeeze_section_raises(self, tmp_path):
        toml = tmp_path / "no_squeeze.toml"
        toml.write_text('[model]\nlora_rank = 32\n')
        with pytest.raises(ValueError, match="No \\[squeeze\\] section"):
            load_squeeze_config(str(toml))

    def test_missing_adapter_path_raises(self, tmp_path):
        toml = tmp_path / "no_path.toml"
        toml.write_text('[squeeze]\nmin_variance_retention = 0.9\n')
        with pytest.raises(ValueError, match="adapter_path is required"):
            load_squeeze_config(str(toml))

    def test_source_rank_explicit(self, tmp_path):
        toml = tmp_path / "explicit.toml"
        toml.write_text("""\
[squeeze]
adapter_path = "/tmp/adapter"
source_rank = 64
""")
        cfg = load_squeeze_config(str(toml))
        assert cfg.source_rank == 64

    def test_source_rank_fallback_to_model(self, tmp_path):
        toml = tmp_path / "fallback.toml"
        toml.write_text("""\
[squeeze]
adapter_path = "/tmp/adapter"

[model]
lora_rank = 128
""")
        cfg = load_squeeze_config(str(toml))
        assert cfg.source_rank == 128


# ---------------------------------------------------------------------------
# Campaign auto-squeeze integration
# ---------------------------------------------------------------------------

class TestCampaignAutoSqueeze:
    def test_auto_squeeze_with_local_adapter(self, tmp_path):
        """_auto_squeeze analyzes a local adapter and returns recommended rank."""
        from retrain.campaign import _auto_squeeze

        adapter_dir = _make_peft_adapter(tmp_path, 64, 64, 16)
        squeeze_cfg = {"min_variance_retention": 0.95}

        rank = _auto_squeeze(str(adapter_dir), squeeze_cfg, lora_rank=16)
        assert isinstance(rank, int)
        assert 1 <= rank <= 16

    def test_auto_squeeze_respects_threshold(self, tmp_path):
        """Lower threshold -> lower or equal recommended rank."""
        from retrain.campaign import _auto_squeeze

        adapter_dir = _make_peft_adapter(tmp_path, 64, 64, 16)

        rank_strict = _auto_squeeze(
            str(adapter_dir), {"min_variance_retention": 0.99}, lora_rank=16
        )
        rank_loose = _auto_squeeze(
            str(adapter_dir), {"min_variance_retention": 0.5}, lora_rank=16
        )
        assert rank_loose <= rank_strict

    def test_campaign_toml_with_squeeze_parsed(self, tmp_path):
        """Campaign TOML with [squeeze] section is parsed correctly."""
        import tomllib

        toml_path = tmp_path / "campaign.toml"
        toml_path.write_text("""\
[campaign]
seeds = [42]
max_steps = 5

[[campaign.conditions]]
advantage_mode = "grpo"
transform_mode = "none"

[squeeze]
min_variance_retention = 0.90

[model]
lora_rank = 128
""")
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        assert "squeeze" in data
        assert "campaign" in data
        assert data["squeeze"]["min_variance_retention"] == pytest.approx(0.90)
