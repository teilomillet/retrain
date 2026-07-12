"""Strict deterministic local-backend setup and claim-boundary tests."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from retrain.backends import determinism as determinism_module
from retrain.backends.determinism import (
    add_model_attention_proof,
    establish_strict_determinism,
    prepare_strict_determinism_environment,
    seed_strict_determinism,
)
from retrain.backends.local import metrics as local_metrics
from retrain.config import TrainConfig
from retrain.training.runner import SftRunner
from retrain.training.sft import build_sft_artifact_manifest


@pytest.fixture(autouse=True)
def _reset_determinism_setup_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        determinism_module,
        "_STRICT_SETUP_ESTABLISHED_BEFORE_CUDA",
        False,
    )


class _FakeCuda:
    def __init__(self, *, initialized: bool = False) -> None:
        self.initialized = initialized

    def is_initialized(self) -> bool:
        return self.initialized

    def is_available(self) -> bool:
        return False

    def manual_seed_all(self, seed: int) -> None:
        _ = seed


class _FakeCudnn:
    deterministic = False
    benchmark = True


class _FakeBackends:
    def __init__(self) -> None:
        self.cudnn = _FakeCudnn()


class _FakeTorch:
    __version__ = "test-torch"

    def __init__(self, *, cuda_initialized: bool = False) -> None:
        self.cuda = _FakeCuda(initialized=cuda_initialized)
        self.backends = _FakeBackends()
        self._deterministic = False
        self._warn_only = False

    def use_deterministic_algorithms(
        self,
        mode: bool,
        *,
        warn_only: bool = False,
    ) -> None:
        self._deterministic = mode
        self._warn_only = warn_only

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self._deterministic

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self._warn_only

    def manual_seed(self, seed: int) -> object:
        return seed


class _IgnoringFakeTorch(_FakeTorch):
    def use_deterministic_algorithms(
        self,
        mode: bool,
        *,
        warn_only: bool = False,
    ) -> None:
        _ = mode, warn_only


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, fake: _FakeTorch) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_strict_determinism_establishes_and_records_runtime_proof(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake = _FakeTorch()
    _install_fake_torch(monkeypatch, fake)

    proof = establish_strict_determinism(enabled=True)

    assert proof["local_strict_deterministic_established"] == 1
    assert proof["local_torch_deterministic_algorithms_enabled"] == 1
    assert proof["local_torch_deterministic_warn_only"] == 0
    assert proof["local_cudnn_deterministic"] == 1
    assert proof["local_cudnn_benchmark"] == 0
    assert proof["local_strict_deterministic_setup_before_cuda"] == 1
    assert proof["local_cublas_workspace_config"] == ":4096:8"
    assert proof["local_cublas_workspace_config_valid"] == 1
    assert proof["local_third_party_kernel_determinism_guaranteed"] == 0
    assert proof["local_two_run_adapter_hash_canary_required"] == 1


def test_default_mode_records_without_mutating_torch_or_environment(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake = _FakeTorch()
    _install_fake_torch(monkeypatch, fake)

    proof = establish_strict_determinism(enabled=False)

    assert proof["local_strict_deterministic_requested"] == 0
    assert proof["local_strict_deterministic_established"] == 0
    assert fake.backends.cudnn.benchmark is True
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_default_mode_fails_after_strict_mode_in_same_process(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake = _FakeTorch()
    _install_fake_torch(monkeypatch, fake)

    establish_strict_determinism(enabled=True)

    with pytest.raises(RuntimeError, match="cannot follow.*same process"):
        establish_strict_determinism(enabled=False)


def test_strict_seed_is_explicit_and_invalid_seeds_fail_closed(monkeypatch) -> None:
    fake = _FakeTorch()
    _install_fake_torch(monkeypatch, fake)

    proof = seed_strict_determinism(123)

    assert proof["local_strict_deterministic_seed"] == 123
    assert proof["local_strict_deterministic_torch_seeded"] == 1
    assert (
        seed_strict_determinism((1 << 32) - 1)["local_strict_deterministic_seed"]
        == (1 << 32) - 1
    )
    for invalid_seed in (-1, 1 << 32, 1.5, True):
        with pytest.raises(RuntimeError, match="between 0 and 4294967295"):
            seed_strict_determinism(invalid_seed)  # type: ignore[arg-type]


def test_strict_mode_rejects_conflicting_cublas_environment(monkeypatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "bad")

    with pytest.raises(RuntimeError, match="requires CUBLAS_WORKSPACE_CONFIG"):
        prepare_strict_determinism_environment(enabled=True)


def test_strict_mode_fails_if_cuda_was_initialized_without_controls(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake = _FakeTorch(cuda_initialized=True)
    _install_fake_torch(monkeypatch, fake)

    with pytest.raises(RuntimeError, match="before CUDA is initialized"):
        establish_strict_determinism(enabled=True)


def test_strict_mode_fails_if_torch_controls_do_not_stick(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    _install_fake_torch(monkeypatch, _IgnoringFakeTorch())

    with pytest.raises(RuntimeError, match="did not remain enabled"):
        establish_strict_determinism(enabled=True)


def test_strict_mode_is_idempotent_after_its_own_cuda_setup(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake = _FakeTorch()
    _install_fake_torch(monkeypatch, fake)

    establish_strict_determinism(enabled=True)
    fake.cuda.initialized = True
    proof = establish_strict_determinism(enabled=True)

    assert proof["local_strict_deterministic_established"] == 1
    assert proof["local_cublas_workspace_config_preconfigured"] == 1


def test_attention_proof_binds_resolved_sdpa_without_claiming_triton() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            _attn_implementation="sdpa",
            text_config=None,
        )
    )
    proof = add_model_attention_proof(
        {
            "local_strict_deterministic_requested": 1,
            "local_torch_deterministic_algorithms_enabled": 1,
            "local_torch_deterministic_warn_only": 0,
            "local_third_party_kernel_determinism_guaranteed": 0,
        },
        model=model,
        requested_attention_kernel="default",
    )

    assert proof["local_attention_kernel_requested"] == "default"
    assert proof["local_attention_implementation_resolved"] == "sdpa"
    assert proof["local_sdpa_strict_torch_guard_enabled"] == 1
    assert proof["local_third_party_kernel_determinism_guaranteed"] == 0


def test_strict_attention_proof_fails_closed_when_model_path_is_unknown() -> None:
    with pytest.raises(RuntimeError, match="could not resolve"):
        add_model_attention_proof(
            {"local_strict_deterministic_requested": 1},
            model=SimpleNamespace(config=SimpleNamespace()),
            requested_attention_kernel="default",
        )


def test_local_runtime_metrics_exposes_determinism_proof() -> None:
    owner = SimpleNamespace(
        _determinism_metrics={
            "local_strict_deterministic_requested": 1,
            "local_strict_deterministic_established": 1,
            "local_cublas_workspace_config": ":4096:8",
            "local_attention_implementation_resolved": "sdpa",
            "local_two_run_adapter_hash_canary_required": 1,
        }
    )

    metrics = local_metrics.runtime_metrics(owner)

    assert metrics["local_strict_deterministic_established"] == 1
    assert metrics["local_cublas_workspace_config"] == ":4096:8"
    assert metrics["local_attention_implementation_resolved"] == "sdpa"
    assert metrics["local_two_run_adapter_hash_canary_required"] == 1


def test_sft_manifest_preserves_runtime_determinism_proof(tmp_path) -> None:
    config = TrainConfig(
        trainer="sft",
        backend="local",
        backend_options={"strict_deterministic": True},
        seed=123,
        sft_data_path="future-sft.jsonl",
    )
    runtime = {
        "backend/local_strict_deterministic_established": 1,
        "backend/local_cublas_workspace_config": ":4096:8",
        "backend/local_attention_implementation_resolved": "sdpa",
        "backend/local_two_run_adapter_hash_canary_required": 1,
    }

    manifest = build_sft_artifact_manifest(
        config,
        policy_ref=str(tmp_path / "adapter"),
        examples_count=1,
        batch_size=1,
        max_tokens=32,
        loss_fn="cross_entropy",
        latest_metrics=runtime,
    )

    assert manifest["resource_metrics"] == runtime


def test_sft_establishes_strict_mode_before_any_dataset_or_cuda_work(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        "retrain.backends.determinism.establish_strict_determinism",
        lambda *, enabled: calls.append(enabled) or {},
    )
    config = TrainConfig(
        trainer="sft",
        backend="local",
        backend_options={"strict_deterministic": True},
        seed=123,
        sft_data_path="missing-determinism-test.jsonl",
    )

    result = SftRunner().run(config)

    assert calls == [True]
    assert result.failure_status == "exception:FileNotFoundError"
    assert "missing-determinism-test.jsonl" in result.error_message
