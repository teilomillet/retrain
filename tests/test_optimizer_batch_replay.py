"""Deterministic optimizer-batch capture/replay tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import orjson
import pytest
from safetensors.numpy import load_file, save_file

from retrain.config import TrainConfig, load_config
from retrain.registry.builtin import get_registry
from retrain.training.batch_digest import logical_optimizer_batch_sha256
from retrain.training.optimizer_batch import (
    AdapterProvenance,
    OptimizerBatch,
    TorchRngState,
    load_optimizer_batch_capture,
    resolve_initial_adapter,
    save_optimizer_batch_capture,
    validate_capture_resume_step,
    validate_replay_contract,
)
from retrain.training.optimizer_batch.codec import (
    batch_from_tensors,
    validate_batch,
)
from retrain.training.runner.optimizer_replay import OptimizerReplayRunner


def _adapter_provenance(sha: str = "a" * 64) -> AdapterProvenance:
    return AdapterProvenance(
        requested_ref="/adapter",
        adapter_dir="/adapter",
        weight_file="adapter_model.safetensors",
        weight_bytes=3,
        weight_sha256=sha,
    )


def _source_config(tmp_path: Path, *, checkpointing: bool = True) -> TrainConfig:
    return TrainConfig(
        max_steps=1,
        save_every=0,
        resume_from=str(tmp_path / "resume"),
        log_dir=str(tmp_path / "capture"),
        adapter_path=str(tmp_path / "capture-adapter"),
        optimizer_batch_capture=True,
        backend_options={"gradient_checkpointing": checkpointing},
    )


def _batch(*, echo: bool = False) -> OptimizerBatch:
    kwargs = {}
    if echo:
        kwargs = {
            "echo_advantages": [[0.0, 0.25, 0.0], [0.0, 0.0]],
            "echo_full_observation_counts": [1, 1],
            "echo_rollout_denominator": 2,
        }
    return OptimizerBatch(
        tokens=[[10, 11, 12], [20, 21]],
        old_logprobs=[[0.0, -0.125, -1.75], [0.0, -0.5]],
        advantages=[[0.0, 1.25, -0.75], [0.0, 0.5]],
        torch_rng=TorchRngState(cpu=b"ignored-by-capture"),
        **kwargs,
    )


@pytest.mark.parametrize("echo", [False, True])
def test_safetensors_round_trip_preserves_logical_batch_exactly(
    tmp_path: Path,
    echo: bool,
) -> None:
    source = _source_config(tmp_path)
    captured = save_optimizer_batch_capture(
        source.log_dir,
        step=0,
        batch=_batch(echo=echo),
        config=source,
        initial_adapter=_adapter_provenance(),
    )

    loaded = load_optimizer_batch_capture(captured.manifest_path)

    assert loaded.batch.tokens == _batch(echo=echo).tokens
    assert loaded.batch.old_logprobs == _batch(echo=echo).old_logprobs
    assert loaded.batch.advantages == _batch(echo=echo).advantages
    assert loaded.batch.echo_advantages == _batch(echo=echo).echo_advantages
    assert loaded.batch.echo_full_observation_counts == (
        _batch(echo=echo).echo_full_observation_counts
    )
    assert loaded.batch.echo_rollout_denominator == (
        _batch(echo=echo).echo_rollout_denominator
    )
    assert loaded.batch.torch_rng.cpu
    assert loaded.logical_batch_sha256 == logical_optimizer_batch_sha256(
        loaded.batch.tokens,
        loaded.batch.old_logprobs,
        loaded.batch.advantages,
        echo_observation_masks=loaded.batch.echo_advantages,
        echo_full_observation_counts=loaded.batch.echo_full_observation_counts,
        echo_rollout_denominator=loaded.batch.echo_rollout_denominator,
    )


def test_capture_refuses_overwrite_and_writes_manifest_after_payload(
    tmp_path: Path,
) -> None:
    source = _source_config(tmp_path)
    captured = save_optimizer_batch_capture(
        source.log_dir,
        step=0,
        batch=_batch(),
        config=source,
        initial_adapter=_adapter_provenance(),
    )
    manifest = orjson.loads(captured.manifest_path.read_bytes())
    payload = captured.payload_path.read_bytes()

    assert manifest["payload"]["sha256"] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_optimizer_batch_capture(
            source.log_dir,
            step=0,
            batch=_batch(),
            config=source,
            initial_adapter=_adapter_provenance(),
        )


def test_capture_manifest_redacts_nested_and_opaque_credentials(
    tmp_path: Path,
) -> None:
    source = _source_config(tmp_path)
    source.backend_options = {
        "gradient_checkpointing": True,
        "api_key": "backend-secret",
        "nested": {"hf_token": "nested-secret"},
        "token": 17,
        "endpoint": (
            "https://alice:url-secret@example.com/v1?token=query-secret#fragment-secret"
        ),
    }
    source.environment_args = json.dumps(
        {
            "task": "safe",
            "credentials": {"api_key": "environment-secret"},
            "token": "environment-token-secret",
            "auth": "environment-auth-secret",
            "sig": "environment-signature-secret",
        }
    )
    source.trainer_command = "trainer --token command-secret"
    source.base_url = (
        "https://bob:model-secret@example.com/v1?api_key=model-query-secret"
    )

    captured = save_optimizer_batch_capture(
        source.log_dir,
        step=0,
        batch=_batch(),
        config=source,
        initial_adapter=_adapter_provenance(),
    )

    manifest_text = captured.manifest_path.read_text()
    for secret in (
        "backend-secret",
        "nested-secret",
        "url-secret",
        "query-secret",
        "fragment-secret",
        "environment-secret",
        "environment-token-secret",
        "environment-auth-secret",
        "environment-signature-secret",
        "command-secret",
        "model-secret",
        "model-query-secret",
    ):
        assert secret not in manifest_text
    manifest = json.loads(manifest_text)
    snapshot = manifest["config"]["snapshot"]
    assert snapshot["backend_options"]["api_key"] == "<redacted>"
    assert snapshot["backend_options"]["nested"]["hf_token"] == "<redacted>"
    assert snapshot["trainer_command"] == "<redacted>"
    assert "environment-secret" not in snapshot["environment_args"]
    assert (
        manifest["config"]["optimizer_contract"]["backend"]["options"]["api_key"]
        == "<redacted>"
    )
    assert manifest["config"]["snapshot"]["backend_options"]["token"] == ("<redacted>")
    assert manifest["config"]["optimizer_contract"]["backend"]["options"]["token"] == 17


def test_payload_tamper_fails_before_deserialization(tmp_path: Path) -> None:
    source = _source_config(tmp_path)
    captured = save_optimizer_batch_capture(
        source.log_dir,
        step=0,
        batch=_batch(),
        config=source,
        initial_adapter=_adapter_provenance(),
    )
    payload = bytearray(captured.payload_path.read_bytes())
    payload[-1] ^= 1
    captured.payload_path.write_bytes(payload)

    with pytest.raises(ValueError, match="payload SHA256 mismatch"):
        load_optimizer_batch_capture(captured.manifest_path)


def test_external_manifest_pin_covers_rng_payload_state(tmp_path: Path) -> None:
    source = _source_config(tmp_path)
    captured = save_optimizer_batch_capture(
        source.log_dir,
        step=0,
        batch=_batch(),
        config=source,
        initial_adapter=_adapter_provenance(),
    )
    tensors = load_file(str(captured.payload_path))
    tensors["torch_cpu_rng"][0] ^= 1
    save_file(
        tensors,
        str(captured.payload_path),
        metadata={"format": "retrain.optimizer-batch.v1"},
    )
    manifest = orjson.loads(captured.manifest_path.read_bytes())
    payload = captured.payload_path.read_bytes()
    manifest["payload"]["bytes"] = len(payload)
    manifest["payload"]["sha256"] = hashlib.sha256(payload).hexdigest()
    captured.manifest_path.write_bytes(orjson.dumps(manifest) + b"\n")

    # The logical rows are unchanged, and the coordinated internal hashes are
    # self-consistent. The external exact-manifest pin must still reject the
    # changed RNG stream before parsing the manifest.
    assert manifest["logical_batch_sha256"] == captured.logical_batch_sha256
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        load_optimizer_batch_capture(
            captured.manifest_path,
            expected_manifest_sha256=captured.manifest_sha256,
        )


def test_config_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    source = _source_config(tmp_path)
    captured = save_optimizer_batch_capture(
        source.log_dir,
        step=0,
        batch=_batch(),
        config=source,
        initial_adapter=_adapter_provenance(),
    )
    manifest = orjson.loads(captured.manifest_path.read_bytes())
    manifest["config"]["snapshot"]["lr"] = 9.0
    captured.manifest_path.write_bytes(orjson.dumps(manifest))

    with pytest.raises(ValueError, match="config hash mismatch"):
        load_optimizer_batch_capture(captured.manifest_path)


def test_logical_digest_and_ragged_offset_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    source = _source_config(tmp_path)
    captured = save_optimizer_batch_capture(
        source.log_dir,
        step=0,
        batch=_batch(),
        config=source,
        initial_adapter=_adapter_provenance(),
    )
    manifest = orjson.loads(captured.manifest_path.read_bytes())
    manifest["logical_batch_sha256"] = "0" * 64
    captured.manifest_path.write_bytes(orjson.dumps(manifest))
    with pytest.raises(ValueError, match="logical SHA256 mismatch"):
        load_optimizer_batch_capture(captured.manifest_path)

    tensors = load_file(str(captured.payload_path))
    tensors["row_offsets"][-1] -= 1
    with pytest.raises(ValueError, match="offsets do not span"):
        batch_from_tensors(tensors)


def test_nonfinite_optimizer_rows_are_rejected() -> None:
    invalid = replace(
        _batch(),
        advantages=[[0.0, float("nan"), 0.0], [0.0, 1.0]],
    )

    with pytest.raises(ValueError, match="non-finite"):
        validate_batch(invalid)


def test_replay_contract_allows_only_declared_checkpointing_difference(
    tmp_path: Path,
) -> None:
    source = _source_config(tmp_path, checkpointing=True)
    captured = save_optimizer_batch_capture(
        source.log_dir,
        step=0,
        batch=_batch(),
        config=source,
        initial_adapter=_adapter_provenance(),
    )
    replay = TrainConfig(
        trainer="optimizer_replay",
        max_steps=1,
        save_every=0,
        resume_from=str(tmp_path / "other-path"),
        optimizer_batch_replay_path=str(captured.manifest_path),
        optimizer_batch_expected_logical_sha256=captured.logical_batch_sha256,
        optimizer_batch_expected_manifest_sha256=captured.manifest_sha256,
        optimizer_batch_allow_config_differences=[
            "backend.options.gradient_checkpointing"
        ],
        backend_options={"gradient_checkpointing": False},
    )
    contract = validate_replay_contract(
        captured.manifest,
        replay,
        _adapter_provenance(),
    )

    assert contract.observed_differences == ("backend.options.gradient_checkpointing",)
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_replay_contract(
            captured.manifest,
            replace(replay, lr=1e-4),
            _adapter_provenance(),
        )


def test_replay_contract_rejects_unused_allowance_and_adapter_mismatch(
    tmp_path: Path,
) -> None:
    source = _source_config(tmp_path)
    captured = save_optimizer_batch_capture(
        source.log_dir,
        step=0,
        batch=_batch(),
        config=source,
        initial_adapter=_adapter_provenance(),
    )
    same = TrainConfig(
        trainer="optimizer_replay",
        max_steps=1,
        save_every=0,
        resume_from="same",
        optimizer_batch_replay_path=str(captured.manifest_path),
        optimizer_batch_expected_logical_sha256=captured.logical_batch_sha256,
        optimizer_batch_expected_manifest_sha256=captured.manifest_sha256,
        optimizer_batch_allow_config_differences=[
            "backend.options.gradient_checkpointing"
        ],
        backend_options={"gradient_checkpointing": True},
    )

    with pytest.raises(ValueError, match="unused"):
        validate_replay_contract(
            captured.manifest,
            same,
            _adapter_provenance(),
        )
    with pytest.raises(ValueError, match="initial adapter hash mismatch"):
        validate_replay_contract(
            captured.manifest,
            replace(same, optimizer_batch_allow_config_differences=[]),
            _adapter_provenance("b" * 64),
        )


class _ReplayHelper:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.loaded: list[str] = []
        self.train_calls: list[dict[str, object]] = []
        self.sample_calls = 0
        self.closed = False

    def load_state(self, ref: str) -> None:
        self.loaded.append(ref)

    def train_step(
        self,
        tokens,
        logprobs,
        advantages,
        lr,
        weight_decay,
    ) -> float:
        self.train_calls.append(
            {
                "tokens": tokens,
                "logprobs": logprobs,
                "advantages": advantages,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
        return 0.25

    def sample(self, *args, **kwargs):
        self.sample_calls += 1
        raise AssertionError("optimizer replay must not sample")

    def save_adapter(self, path: str, name: str) -> str:
        output = Path(path) / name
        output.mkdir(parents=True)
        (output / "adapter_model.safetensors").write_bytes(b"final")
        return str(output)

    def runtime_metrics(self) -> dict[str, object]:
        return {
            "optimizer/local_effective_rows_sha256": "e" * 64,
            "local_gradient_checkpointing_enabled": 0,
            "local_train_wall_s": 0.2,
        }

    def shutdown(self) -> None:
        self.closed = True


def _write_resume(tmp_path: Path) -> Path:
    adapter = tmp_path / "initial-adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"initial")
    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / "trainer_state.json").write_text(
        json.dumps(
            {
                "step": -1,
                "example_idx": 0,
                "total_correct": 0,
                "total_completions": 0,
                "current_batch_size": 1,
                "current_group_size": 2,
                "checkpoint_name": "init",
                "checkpoint_path": str(adapter),
                "sepa": {},
            }
        )
    )
    return resume


def test_replay_runner_skips_rollout_and_submits_exact_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume = _write_resume(tmp_path)
    source = _source_config(tmp_path, checkpointing=True)
    source.resume_from = str(resume)
    initial_sha = hashlib.sha256(b"initial").hexdigest()
    captured = save_optimizer_batch_capture(
        source.log_dir,
        step=0,
        batch=_batch(),
        config=source,
        initial_adapter=AdapterProvenance(
            requested_ref="init",
            adapter_dir=str(tmp_path / "initial-adapter"),
            weight_file="adapter_model.safetensors",
            weight_bytes=7,
            weight_sha256=initial_sha,
        ),
    )
    replay = TrainConfig(
        trainer="optimizer_replay",
        max_steps=1,
        save_every=0,
        resume_from=str(resume),
        log_dir=str(tmp_path / "replay"),
        adapter_path=str(tmp_path / "replay-adapter"),
        optimizer_batch_replay_path=str(captured.manifest_path),
        optimizer_batch_expected_logical_sha256=captured.logical_batch_sha256,
        optimizer_batch_expected_manifest_sha256=captured.manifest_sha256,
        optimizer_batch_allow_config_differences=[
            "backend.options.gradient_checkpointing"
        ],
        backend_options={"gradient_checkpointing": False},
    )
    replay.environment_args = json.dumps(
        {
            "credentials": {"hf_token": "replay-environment-secret"},
            "token": "replay-token-secret",
        }
    )
    replay.trainer_command = "trainer --api-key replay-command-secret"
    helper = _ReplayHelper(tmp_path)
    registry = get_registry("backend")
    monkeypatch.setattr(registry, "create", lambda name, config: helper)

    result = OptimizerReplayRunner().run(replay)

    assert result.ok
    assert helper.sample_calls == 0
    assert helper.closed is True
    assert helper.loaded == [str((tmp_path / "initial-adapter").resolve())]
    assert len(helper.train_calls) == 1
    assert helper.train_calls[0]["tokens"] == _batch().tokens
    metrics = json.loads((Path(replay.log_dir) / "metrics.jsonl").read_text())
    assert metrics["optimizer/logical_batch_sha256"] == captured.logical_batch_sha256
    assert metrics["optimizer/local_effective_rows_sha256"] == "e" * 64
    assert metrics["optimizer_batch/environment_skipped"] == 1
    assert metrics["optimizer_batch/sampling_skipped"] == 1
    assert metrics["sample_time_s"] == 0.0
    assert metrics["optimizer_batch/observed_config_differences"] == (
        '["backend.options.gradient_checkpointing"]'
    )
    provenance_text = (
        Path(replay.log_dir) / "optimizer_batch_replay_manifest.json"
    ).read_text()
    assert "replay-environment-secret" not in provenance_text
    assert "replay-command-secret" not in provenance_text
    assert "replay-token-secret" not in provenance_text
    provenance = json.loads(provenance_text)
    assert provenance["source"]["payload_sha256"] == captured.payload_sha256
    assert provenance["replay"]["config"]["trainer_command"] == "<redacted>"

    repeated = OptimizerReplayRunner().run(replay)
    assert not repeated.ok
    assert "requires fresh output paths" in repeated.error_message
    assert len(helper.train_calls) == 1


def test_optimizer_batch_toml_and_v1_guards(tmp_path: Path) -> None:
    path = tmp_path / "replay.toml"
    path.write_text(
        """
[training]
trainer = "optimizer_replay"
max_steps = 1
save_every = 0

[resume]
from = "resume"

[optimizer_batch]
replay_path = "batch.manifest.json"
expected_logical_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
expected_manifest_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
allow_config_differences = ["backend.options.gradient_checkpointing"]
"""
    )

    config = load_config(str(path))

    assert config.optimizer_batch_replay_path == "batch.manifest.json"
    assert config.optimizer_batch_expected_manifest_sha256 == "b" * 64
    assert config.optimizer_batch_allow_config_differences == [
        "backend.options.gradient_checkpointing"
    ]
    with pytest.raises(ValueError, match="exactly one local device"):
        replace(config, devices="gpu:0,gpu:1")
    with pytest.raises(ValueError, match="inference_engine='pytorch'"):
        replace(config, inference_engine="vllm")
    with pytest.raises(ValueError, match="only permits"):
        TrainConfig(
            trainer="optimizer_replay",
            max_steps=1,
            save_every=0,
            resume_from="resume",
            optimizer_batch_replay_path="batch.manifest.json",
            optimizer_batch_expected_logical_sha256="a" * 64,
            optimizer_batch_expected_manifest_sha256="b" * 64,
            optimizer_batch_allow_config_differences=["optimizer.lr"],
        )


def test_optimizer_batch_cli_list_override(tmp_path: Path) -> None:
    path = tmp_path / "base.toml"
    path.write_text("[training]\nmax_steps = 1\n")

    config = load_config(
        str(path),
        overrides={
            "optimizer_batch_allow_config_differences": (
                '["backend.options.gradient_checkpointing"]'
            )
        },
    )

    assert config.optimizer_batch_allow_config_differences == [
        "backend.options.gradient_checkpointing"
    ]


@pytest.mark.parametrize("saved_step", [0, 4, -2])
def test_capture_resume_step_must_produce_exactly_one_iteration(
    saved_step: int,
) -> None:
    with pytest.raises(ValueError, match="trainer_state.step = -1"):
        validate_capture_resume_step(saved_step=saved_step, max_steps=1)

    validate_capture_resume_step(saved_step=-1, max_steps=1)


def test_initial_adapter_resolution_matches_local_backend_precedence(
    tmp_path: Path,
) -> None:
    resume = tmp_path / "resume"
    resume.mkdir()
    resume_named = resume / "init"
    resume_named.mkdir()
    (resume_named / "adapter_model.safetensors").write_bytes(b"wrong")
    output = tmp_path / "adapters"
    output_named = output / "init"
    output_named.mkdir(parents=True)
    (output_named / "adapter_model.safetensors").write_bytes(b"loaded")
    (resume / "trainer_state.json").write_text(
        json.dumps(
            {
                "step": -1,
                "example_idx": 0,
                "total_correct": 0,
                "total_completions": 0,
                "current_batch_size": 1,
                "current_group_size": 1,
                "checkpoint_name": "init",
                "sepa": {},
            }
        )
    )
    config = _source_config(tmp_path)
    config.resume_from = str(resume)
    config.adapter_path = str(output)

    provenance = resolve_initial_adapter(config)

    assert provenance.adapter_dir == str(output_named.resolve())
    assert provenance.weight_sha256 == hashlib.sha256(b"loaded").hexdigest()
