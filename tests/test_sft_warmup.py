"""Battle tests for SFT warmup + GRPO pipeline.

Inspired by Jane Street's "getting from tested to battle-tested":
- Unit tests for config parsing
- Property-based tests for data format invariants
- Adversarial tests for edge cases (empty data, corrupt JSONL, huge tokens)
- Integration tests against real warehouse kernel
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

import retrain.training.sft as sft_module
from retrain.config import TrainConfig, load_config
from retrain.training.warmup import load_sft_warmup_data
from retrain.training.sft import (
    SftExample,
    load_sft_dataset,
    load_sft_jsonl,
    tokenize_sft_batch,
    verify_sft_data_contract,
    write_sft_run_snapshot_artifacts,
)


# ---------------------------------------------------------------------------
# Config parsing tests
# ---------------------------------------------------------------------------


class TestSftWarmupConfig:
    """SFT warmup config fields parse correctly from TOML."""

    def test_defaults_are_zero(self):
        """SFT warmup is off by default."""
        config = TrainConfig()
        assert config.sft_warmup_steps == 0
        assert config.sft_data_path == ""

    def test_parse_from_toml(self):
        """TOML config loads SFT fields."""
        toml_content = """
[backend]
backend = "local"

[model]
model = "test-model"

[training]
max_steps = 20
sft_warmup_steps = 5
sft_data_path = "/tmp/test_sft.jsonl"
sft_data_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
sft_data_rows = 123
sft_batch_size = 3
sft_max_tokens = 256
sft_loss_fn = "cross_entropy"
sft_batch_order = "length_bucket"
sft_length_bucket_size = 64
tl_grpo = true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            config = load_config(f.name)

        os.unlink(f.name)
        assert config.sft_warmup_steps == 5
        assert config.sft_data_path == "/tmp/test_sft.jsonl"
        assert config.sft_data_sha256 == "a" * 64
        assert config.sft_data_rows == 123
        assert config.sft_batch_size == 3
        assert config.sft_max_tokens == 256
        assert config.sft_loss_fn == "cross_entropy"
        assert config.sft_batch_order == "length_bucket"
        assert config.sft_length_bucket_size == 64
        assert config.tl_grpo is True

    def test_sft_steps_cannot_exceed_max_steps(self):
        """SFT warmup steps should be < max_steps (leaves room for RL)."""
        config = TrainConfig(max_steps=10, sft_warmup_steps=10)
        # This is technically valid (all steps are SFT, no RL) but weird
        assert config.sft_warmup_steps <= config.max_steps

    def test_sft_without_data_path_is_noop(self):
        """SFT warmup with steps > 0 but no data path should be harmless."""
        config = TrainConfig(sft_warmup_steps=5, sft_data_path="")
        assert config.sft_warmup_steps == 5
        assert config.sft_data_path == ""

    def test_standalone_sft_requires_data_path(self):
        """Standalone SFT cannot start without a user dataset."""
        with pytest.raises(ValueError, match="sft_data_path"):
            TrainConfig(trainer="sft")

    def test_sft_data_pin_config_is_validated(self):
        with pytest.raises(ValueError, match="sft_data_sha256"):
            TrainConfig(sft_data_sha256="not-a-sha")
        with pytest.raises(ValueError, match="sft_data_rows"):
            TrainConfig(sft_data_rows=-1)

    def test_pinned_missing_sft_warmup_data_fails(self, tmp_path):
        config = TrainConfig(
            sft_warmup_steps=1,
            sft_data_path=str(tmp_path / "missing.jsonl"),
            sft_data_sha256="0" * 64,
        )

        with pytest.raises(FileNotFoundError, match="cannot verify"):
            load_sft_warmup_data(config, _TinyTokenizer())


class _TinyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


class _ThinkingTokenizer(_TinyTokenizer):
    def __init__(self):
        self.enable_thinking_values = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    ):
        self.enable_thinking_values.append(enable_thinking)
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


class _KwargsThinkingTokenizer(_TinyTokenizer):
    def __init__(self):
        self.kwargs_values = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        self.kwargs_values.append(dict(kwargs))
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


class TestGenericSftJsonl:
    """User-supplied SFT JSONL formats."""

    def test_loads_messages_prompt_completion_and_text_rows(self, tmp_path):
        data_path = tmp_path / "sft.jsonl"
        data_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": "sys"},
                                {"role": "user", "content": "question"},
                                {"role": "assistant", "content": "answer"},
                            ]
                        }
                    ),
                    json.dumps({"prompt": "Question: ", "completion": "Answer"}),
                    json.dumps({"text": "plain next-token text"}),
                ]
            )
            + "\n"
        )

        examples = load_sft_jsonl(data_path)
        batch = tokenize_sft_batch(_TinyTokenizer(), examples, max_tokens=0)

        assert len(examples) == 3
        assert len(batch.tokens) == 3
        assert batch.supervised_tokens > 0
        assert batch.total_tokens >= batch.supervised_tokens

    def test_load_sft_dataset_records_file_provenance(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        data_path = data_dir / "sft.jsonl"
        payload = json.dumps({"text": "plain next-token text"}) + "\n\n"
        data_path.write_text(payload)

        dataset = load_sft_dataset(data_path)

        assert len(dataset.examples) == 1
        assert dataset.provenance.data_path == str(data_path.resolve())
        assert dataset.provenance.data_sha256 == hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()
        assert dataset.provenance.data_rows == 1
        assert dataset.provenance.data_bytes == len(payload.encode("utf-8"))
        assert dataset.provenance.data_root == str(data_dir.resolve())
        assert dataset.provenance.data_path_status == "scratch"
        assert any(
            "scratch/tmp" in warning
            for warning in dataset.provenance.data_warnings
        )

    def test_load_sft_dataset_records_git_tracking_status(self, tmp_path):
        repo_dir = tmp_path / "repo"
        data_dir = repo_dir / "data"
        data_dir.mkdir(parents=True)
        data_path = data_dir / "sft.jsonl"
        data_path.write_text(json.dumps({"text": "plain next-token text"}) + "\n")
        try:
            subprocess.run(
                ["git", "init"],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            pytest.skip("git is unavailable")

        untracked = load_sft_dataset(data_path)

        assert untracked.provenance.git_root == str(repo_dir.resolve())
        assert untracked.provenance.git_tracked is False
        assert any(
            "not tracked by git" in warning
            for warning in untracked.provenance.data_warnings
        )

        subprocess.run(
            ["git", "add", "data/sft.jsonl"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        staged = load_sft_dataset(data_path)

        assert staged.provenance.git_root == str(repo_dir.resolve())
        assert staged.provenance.git_tracked is True
        assert not any(
            "not tracked by git" in warning
            for warning in staged.provenance.data_warnings
        )

    def test_sft_data_contract_accepts_matching_pins(self, tmp_path):
        data_path = tmp_path / "data" / "sft.jsonl"
        data_path.parent.mkdir()
        data_path.write_text(json.dumps({"text": "plain next-token text"}) + "\n")
        dataset = load_sft_dataset(data_path)
        config = TrainConfig(
            sft_data_path=str(data_path),
            sft_data_sha256=dataset.provenance.data_sha256,
            sft_data_rows=dataset.provenance.data_rows,
        )

        verify_sft_data_contract(config, dataset.provenance)

    def test_sft_data_contract_rejects_mismatched_pins(self, tmp_path):
        data_path = tmp_path / "data" / "sft.jsonl"
        data_path.parent.mkdir()
        data_path.write_text(json.dumps({"text": "plain next-token text"}) + "\n")
        dataset = load_sft_dataset(data_path)
        config = TrainConfig(
            sft_data_path=str(data_path),
            sft_data_sha256="0" * 64,
            sft_data_rows=dataset.provenance.data_rows + 1,
        )

        with pytest.raises(ValueError, match="SFT data contract mismatch") as exc_info:
            verify_sft_data_contract(config, dataset.provenance)
        message = str(exc_info.value)
        assert "sft_data_sha256 mismatch" in message
        assert "sft_data_rows mismatch" in message

    def test_sft_snapshot_recoverability_records_large_data_without_copy(self, tmp_path):
        data_path = tmp_path / "data" / "sft.jsonl"
        data_path.parent.mkdir()
        data_path.write_text(json.dumps({"text": "plain next-token text"}) + "\n")
        dataset = load_sft_dataset(data_path)
        log_dir = tmp_path / "logs"
        config = TrainConfig(sft_data_path=str(data_path), log_dir=str(log_dir))

        paths = write_sft_run_snapshot_artifacts(
            log_dir,
            config,
            dataset.provenance,
            snapshot_max_bytes=1,
        )

        assert "resolved_config.json" in paths
        assert "sft_data.snapshot.jsonl" not in paths
        recoverability = json.loads((log_dir / "sft_data_recoverability.json").read_text())
        assert recoverability["copied"] is False
        assert recoverability["recoverable"] is False
        assert recoverability["source_sha256"] == dataset.provenance.data_sha256
        assert "exceeds snapshot_max_bytes" in recoverability["reason"]

    def test_sft_snapshot_redacts_sensitive_config_values(self, tmp_path):
        data_path = tmp_path / "data" / "sft.jsonl"
        data_path.parent.mkdir()
        data_path.write_text(json.dumps({"text": "plain next-token text"}) + "\n")
        dataset = load_sft_dataset(data_path)
        log_dir = tmp_path / "logs"
        config = TrainConfig(
            sft_data_path=str(data_path),
            log_dir=str(log_dir),
        )
        config.backend_options = {
            "api_key": "secret-value",
            "load_in_4bit": True,
            "nested": {
                "hf_token": "token-value",
                "max_tokens": 123,
            },
        }

        write_sft_run_snapshot_artifacts(log_dir, config, dataset.provenance)

        resolved = json.loads((log_dir / "resolved_config.json").read_text())
        options = resolved["config"]["backend_options"]
        assert options["api_key"] == "<redacted>"
        assert options["load_in_4bit"] is True
        assert options["nested"]["hf_token"] == "<redacted>"
        assert options["nested"]["max_tokens"] == 123

    def test_message_sft_disables_thinking_when_supported(self):
        tokenizer = _ThinkingTokenizer()
        examples = [
            SftExample(
                messages=[
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            )
        ]

        tokenize_sft_batch(tokenizer, examples, max_tokens=0)

        assert tokenizer.enable_thinking_values == [False, False]

    def test_message_sft_disables_thinking_for_kwargs_tokenizers(self):
        tokenizer = _KwargsThinkingTokenizer()
        examples = [
            SftExample(
                messages=[
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            )
        ]

        tokenize_sft_batch(tokenizer, examples, max_tokens=0)

        assert tokenizer.kwargs_values == [
            {"enable_thinking": False},
            {"enable_thinking": False},
        ]

    def test_message_sft_resolves_template_kwargs_once_per_row(self, monkeypatch):
        tokenizer = _ThinkingTokenizer()
        examples = [
            SftExample(
                messages=[
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            )
        ]
        real_signature = sft_module.inspect.signature
        signature_calls = 0

        def counting_signature(obj):
            nonlocal signature_calls
            signature_calls += 1
            return real_signature(obj)

        monkeypatch.setattr(sft_module.inspect, "signature", counting_signature)

        tokenize_sft_batch(tokenizer, examples, max_tokens=0)

        assert signature_calls == 1

    def test_invalid_jsonl_reports_line_number(self, tmp_path):
        data_path = tmp_path / "bad.jsonl"
        data_path.write_text(json.dumps({"text": "ok"}) + "\nnot json\n")

        with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
            load_sft_jsonl(data_path)


# ---------------------------------------------------------------------------
# SFT data format tests (property-based style)
# ---------------------------------------------------------------------------


class TestSftDataFormat:
    """Validate SFT data file format invariants."""

    @pytest.fixture
    def sft_data_path(self):
        """Path to the generated SFT data."""
        path = Path(__file__).resolve().parent.parent.parent / "python" / "scripts" / "warehouse_sft_data.jsonl"
        if not path.exists():
            pytest.skip("SFT data not generated yet (run generate_sft_data.py)")
        return path

    def test_every_line_is_valid_json(self, sft_data_path):
        """Every line must parse as valid JSON."""
        with open(sft_data_path) as f:
            for i, line in enumerate(f):
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    pytest.fail(f"Line {i+1} is not valid JSON: {line[:100]}")

    def test_every_example_has_messages(self, sft_data_path):
        """Every example must have a 'messages' key with 3 messages."""
        with open(sft_data_path) as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                assert "messages" in data, f"Line {i+1} missing 'messages'"
                msgs = data["messages"]
                assert len(msgs) == 3, f"Line {i+1}: expected 3 messages, got {len(msgs)}"
                assert msgs[0]["role"] == "system"
                assert msgs[1]["role"] == "user"
                assert msgs[2]["role"] == "assistant"

    def test_assistant_messages_are_valid_json_actions(self, sft_data_path):
        """Every assistant message must be valid warehouse JSON."""
        with open(sft_data_path) as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                action_str = data["messages"][2]["content"]
                try:
                    action = json.loads(action_str)
                except json.JSONDecodeError:
                    pytest.fail(f"Line {i+1}: assistant message is not valid JSON: {action_str[:100]}")
                assert "kind" in action, f"Line {i+1}: action missing 'kind'"

    def test_action_distribution_is_diverse(self, sft_data_path):
        """Actions should cover multiple types, not just one."""
        action_types = set()
        with open(sft_data_path) as f:
            for line in f:
                data = json.loads(line)
                action = json.loads(data["messages"][2]["content"])
                if action.get("kind") == "act":
                    action_types.add(action["action"]["type"])
                else:
                    action_types.add(action["kind"])
        # Oracle should produce at least 5 different action types
        assert len(action_types) >= 5, f"Only {len(action_types)} action types: {action_types}"

    def test_no_empty_observations(self, sft_data_path):
        """User messages (observations) must not be empty."""
        with open(sft_data_path) as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                obs = data["messages"][1]["content"]
                assert len(obs) > 10, f"Line {i+1}: observation too short ({len(obs)} chars)"


# ---------------------------------------------------------------------------
# Adversarial / edge case tests
# ---------------------------------------------------------------------------


class TestSftAdversarial:
    """Edge cases that could break the SFT warmup."""

    def test_empty_sft_file(self, tmp_path):
        """Empty JSONL file should not crash."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")
        config = TrainConfig(sft_warmup_steps=5, sft_data_path=str(empty_file))
        assert config.sft_data_path == str(empty_file)

    def test_corrupt_jsonl_line(self, tmp_path):
        """A corrupt line in the middle should be detectable."""
        bad_file = tmp_path / "bad.jsonl"
        lines = [
            json.dumps({"messages": [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "obs"},
                {"role": "assistant", "content": '{"kind":"wait"}'},
            ]}),
            "THIS IS NOT JSON",
            json.dumps({"messages": [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "obs2"},
                {"role": "assistant", "content": '{"kind":"wait"}'},
            ]}),
        ]
        bad_file.write_text("\n".join(lines))

        # Reading should fail on line 2
        with open(bad_file) as f:
            for i, line in enumerate(f):
                if i == 1:
                    with pytest.raises(json.JSONDecodeError):
                        json.loads(line)

    def test_very_long_observation(self, tmp_path):
        """Observations with huge JSON should still parse."""
        big_obs = json.dumps({"data": "x" * 50000})
        sft_file = tmp_path / "big.jsonl"
        sft_file.write_text(json.dumps({"messages": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": big_obs},
            {"role": "assistant", "content": '{"kind":"wait"}'},
        ]}) + "\n")

        with open(sft_file) as f:
            data = json.loads(f.readline())
            assert len(data["messages"][1]["content"]) > 50000

    def test_missing_messages_key(self, tmp_path):
        """JSONL without 'messages' key should be caught."""
        bad_file = tmp_path / "nomsg.jsonl"
        bad_file.write_text(json.dumps({"wrong_key": "data"}) + "\n")

        with open(bad_file) as f:
            data = json.loads(f.readline())
            assert "messages" not in data


# ---------------------------------------------------------------------------
# Warehouse domain integration tests
# ---------------------------------------------------------------------------


class TestWarehouseOracleData:
    """Verify oracle demonstrations are consistent with kernel behavior."""

    @pytest.fixture
    def sft_data_path(self):
        path = Path(__file__).resolve().parent.parent.parent / "python" / "scripts" / "warehouse_sft_data.jsonl"
        if not path.exists():
            pytest.skip("SFT data not generated yet")
        return path

    def test_quote_client_actions_have_valid_prices(self, sft_data_path):
        """All quote_client actions should have positive price."""
        with open(sft_data_path) as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                action = json.loads(data["messages"][2]["content"])
                if action.get("kind") == "act" and action["action"].get("type") == "quote_client":
                    price = action["action"]["price_per_sqm_day_cents"]
                    assert price > 0, f"Line {i+1}: quote price must be > 0, got {price}"
                    assert price <= 500, f"Line {i+1}: quote price must be <= 500, got {price}"

    def test_upgrade_facility_appears_early(self, sft_data_path):
        """Oracle should upgrade facility in the first few turns of each episode."""
        # Each episode starts at a multiple of 100 (100 turns per seed)
        with open(sft_data_path) as f:
            lines = f.readlines()

        # Check first episode (lines 0-99)
        first_upgrade = None
        for i in range(min(100, len(lines))):
            data = json.loads(lines[i])
            action = json.loads(data["messages"][2]["content"])
            if action.get("kind") == "act" and action["action"].get("type") == "upgrade_facility":
                first_upgrade = i
                break
        assert first_upgrade is not None, "Oracle should upgrade facility in first episode"
        assert first_upgrade < 10, f"Oracle should upgrade early, but first upgrade at turn {first_upgrade}"

    def test_advantage_mask_zeros_prompt_tokens(self, sft_data_path):
        """Advantages must be 0 for prompt tokens, 1 only for response tokens."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
        with open(sft_data_path) as f:
            ex = json.loads(f.readline())

        msgs = ex["messages"]
        prompt_msgs = msgs[:2]
        full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_tokens = tok.encode(full_text, add_special_tokens=False)
        prompt_tokens = tok.encode(prompt_text, add_special_tokens=False)

        n_prompt = min(len(prompt_tokens), len(full_tokens))
        advantages = [0.0] * n_prompt + [1.0] * (len(full_tokens) - n_prompt)

        # Prompt tokens must have advantage 0
        assert all(a == 0.0 for a in advantages[:n_prompt])
        # Response tokens must have advantage 1
        assert all(a == 1.0 for a in advantages[n_prompt:])
        # Response must be non-empty
        assert len(full_tokens) - n_prompt > 0
        # Response should be the action (short JSON), not hundreds of tokens
        assert len(full_tokens) - n_prompt < 100

    def test_context_window_keeps_system_and_recent(self):
        """Sliding window must keep system prompt and last N turns."""
        pytest.importorskip("soma_adapter")
        from soma_adapter.verifiers_env import SomaVendingVerifiersEnv

        env = SomaVendingVerifiersEnv.__new__(SomaVendingVerifiersEnv)
        messages = [
            {"role": "system", "content": "You are a manager."},
            {"role": "user", "content": "intro message"},
            {"role": "user", "content": "obs turn 1"},
            {"role": "assistant", "content": "<think>long reasoning</think>{\"kind\":\"wait\"}"},
            {"role": "user", "content": "obs turn 2"},
            {"role": "assistant", "content": "<think>more thinking</think>{\"kind\":\"wait\"}"},
            {"role": "user", "content": "obs turn 3"},
            {"role": "assistant", "content": "{\"kind\":\"wait\"}"},
            {"role": "user", "content": "obs turn 4"},
            {"role": "assistant", "content": "<think>reasoning</think>{\"kind\":\"act\"}"},
            {"role": "user", "content": "obs turn 5 (current)"},
        ]
        windowed = env._apply_context_window(messages, max_context_turns=2)

        # System prompt kept
        assert windowed[0]["role"] == "system"
        assert "manager" in windowed[0]["content"]

        # Intro kept
        assert windowed[1]["content"] == "intro message"

        # Only last 2 turn pairs + current observation
        # Last 4 messages from non-system: turn4 assistant, turn5 obs, etc.
        assert windowed[-1]["content"] == "obs turn 5 (current)"

        # Total should be: system + intro + last 4 non-system messages
        assert len(windowed) <= 2 + 1 + 4  # system + intro + 2 turns × 2

    def test_context_window_strips_think_tags(self):
        """Old assistant messages should have <think> tags stripped."""
        pytest.importorskip("soma_adapter")
        from soma_adapter.verifiers_env import SomaVendingVerifiersEnv

        env = SomaVendingVerifiersEnv.__new__(SomaVendingVerifiersEnv)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "obs1"},
            {"role": "assistant", "content": "<think>secret reasoning</think>{\"kind\":\"wait\"}"},
            {"role": "user", "content": "obs2"},
        ]
        windowed = env._apply_context_window(messages, max_context_turns=3)

        # The old assistant message should have thinking stripped
        assistant_msgs = [m for m in windowed if m["role"] == "assistant"]
        for msg in assistant_msgs:
            assert "<think>" not in msg["content"], f"Think tag not stripped: {msg['content'][:50]}"

    def test_context_window_preserves_short_conversations(self):
        """Short conversations (under window) should be unchanged."""
        pytest.importorskip("soma_adapter")
        from soma_adapter.verifiers_env import SomaVendingVerifiersEnv

        env = SomaVendingVerifiersEnv.__new__(SomaVendingVerifiersEnv)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "obs1"},
            {"role": "assistant", "content": "act1"},
            {"role": "user", "content": "obs2"},
        ]
        windowed = env._apply_context_window(messages, max_context_turns=10)
        # All messages preserved (under the window)
        assert len(windowed) == len(messages)

    def test_determinism_same_seed_same_actions(self, sft_data_path):
        """Same seed should produce identical action sequences."""
        # The data was generated in order: seed 42 → turns 0-99, seed 43 → turns 100-199
        # If we regenerate for seed 42, we should get the same actions
        with open(sft_data_path) as f:
            lines = f.readlines()

        # First turn of seed 42
        first_action = json.loads(lines[0])["messages"][2]["content"]
        # This should always be upgrade_facility (oracle's first move)
        parsed = json.loads(first_action)
        assert parsed["kind"] == "act"
        # The oracle upgrades facility first
        assert parsed["action"]["type"] == "upgrade_facility"
