"""Typed values shared by optimizer-batch capture and replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TorchRngState:
    """Torch RNG bytes at the optimizer boundary.

    Capturing these bytes keeps replay valid for models that use dropout or
    another stochastic training-time operation. Model construction happens
    before restore, so initialization cannot advance the replay RNG stream.
    """

    cpu: bytes
    cuda: tuple[bytes, ...] = ()


@dataclass(frozen=True)
class OptimizerBatch:
    """Exact trainer-logical rows passed to one RL optimizer update."""

    tokens: list[list[int]]
    old_logprobs: list[list[float]]
    advantages: list[list[float]]
    echo_advantages: list[list[float]] | None = None
    echo_full_observation_counts: list[int] | None = None
    echo_rollout_denominator: int | None = None
    torch_rng: TorchRngState = field(
        default_factory=lambda: TorchRngState(cpu=b"")
    )


@dataclass(frozen=True)
class AdapterProvenance:
    """Content identity of the local adapter weights actually loaded."""

    requested_ref: str
    adapter_dir: str
    weight_file: str
    weight_bytes: int
    weight_sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "requested_ref": self.requested_ref,
            "adapter_dir": self.adapter_dir,
            "weight_file": self.weight_file,
            "weight_bytes": self.weight_bytes,
            "weight_sha256": self.weight_sha256,
        }


@dataclass(frozen=True)
class CapturedOptimizerBatch:
    """Paths and verified hashes returned after an atomic capture."""

    manifest_path: Path
    payload_path: Path
    manifest: dict[str, object]
    manifest_sha256: str
    payload_sha256: str
    logical_batch_sha256: str
    config_sha256: str
    optimizer_contract_sha256: str
    initial_adapter_sha256: str


@dataclass(frozen=True)
class LoadedOptimizerBatch:
    """A verified capture ready for one optimizer replay."""

    batch: OptimizerBatch
    manifest_path: Path
    payload_path: Path
    manifest: dict[str, object]
    manifest_sha256: str
    payload_sha256: str
    logical_batch_sha256: str


@dataclass(frozen=True)
class ReplayContract:
    """Validated config and adapter contract for one replay."""

    current_config_sha256: str
    current_optimizer_contract_sha256: str
    allowed_differences: tuple[str, ...]
    observed_differences: tuple[str, ...]
    initial_adapter: AdapterProvenance
