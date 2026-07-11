"""High-level optimizer-batch artifact capture and verified loading."""

from __future__ import annotations

import hashlib
import platform
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import orjson
from safetensors.numpy import load_file

from retrain.config import TrainConfig
from retrain.training.optimizer_batch.codec import (
    batch_from_tensors,
    batch_summary,
    batch_tensors,
    logical_sha,
    validate_batch,
)
from retrain.training.optimizer_batch.contract import (
    canonical_json_bytes,
    config_snapshot,
    optimizer_contract,
    sha256_json,
)
from retrain.training.optimizer_batch.rng import capture_torch_rng_state
from retrain.training.optimizer_batch.storage import (
    sha256_file,
    write_bytes_atomic,
    write_safetensors_atomic,
)
from retrain.training.optimizer_batch.types import (
    AdapterProvenance,
    CapturedOptimizerBatch,
    LoadedOptimizerBatch,
    OptimizerBatch,
)


_FORMAT = "retrain.optimizer-batch.v1"
_KIND = "rl"


def save_optimizer_batch_capture(
    log_dir: str | Path,
    *,
    step: int,
    batch: OptimizerBatch,
    config: TrainConfig,
    initial_adapter: AdapterProvenance,
) -> CapturedOptimizerBatch:
    """Write payload first and manifest last using atomic sibling renames."""

    batch = replace(batch, torch_rng=capture_torch_rng_state())
    validate_batch(batch)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    stem = f"optimizer_batch_step_{step:06d}"
    payload_path = log_path / f"{stem}.safetensors"
    manifest_path = log_path / f"{stem}.manifest.json"
    if payload_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Refusing to overwrite optimizer-batch capture artifact at "
            f"{manifest_path}."
        )

    logical_batch_sha = logical_sha(batch)
    write_safetensors_atomic(
        payload_path,
        batch_tensors(batch),
        format_name=_FORMAT,
    )
    payload_sha = sha256_file(payload_path)
    manifest = _build_manifest(
        log_path=log_path,
        payload_path=payload_path,
        payload_sha=payload_sha,
        logical_batch_sha=logical_batch_sha,
        step=step,
        batch=batch,
        config=config,
        initial_adapter=initial_adapter,
    )
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    # The manifest is the commit marker: an interrupted payload write never
    # leaves a loadable capture contract.
    write_bytes_atomic(manifest_path, manifest_bytes)
    config_map = cast(dict[str, object], manifest["config"])
    return CapturedOptimizerBatch(
        manifest_path=manifest_path,
        payload_path=payload_path,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        payload_sha256=payload_sha,
        logical_batch_sha256=logical_batch_sha,
        config_sha256=cast(str, config_map["sha256"]),
        optimizer_contract_sha256=cast(
            str,
            config_map["optimizer_contract_sha256"],
        ),
        initial_adapter_sha256=initial_adapter.weight_sha256,
    )


def load_optimizer_batch_capture(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str = "",
) -> LoadedOptimizerBatch:
    """Load a manifest-pinned safetensors payload and verify every digest.

    Replay callers provide an external manifest digest. It is checked before
    parsing so every manifest field, including the RNG-bearing payload hash,
    is transitively pinned.
    """

    path = Path(manifest_path).expanduser().resolve()
    manifest_bytes = path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    expected_manifest_sha = expected_manifest_sha256.strip().lower()
    if expected_manifest_sha and manifest_sha != expected_manifest_sha:
        raise ValueError(
            "optimizer-batch manifest SHA256 mismatch: expected "
            f"{expected_manifest_sha}, got {manifest_sha}."
        )
    raw = orjson.loads(manifest_bytes)
    if not isinstance(raw, dict):
        raise ValueError("optimizer-batch manifest must contain a JSON object.")
    manifest = cast(dict[str, object], raw)
    _validate_format(manifest)

    payload_info = _require_object(manifest, "payload")
    payload_name = _require_string(payload_info, "file")
    if Path(payload_name).name != payload_name:
        raise ValueError("optimizer-batch payload path must be a sibling filename.")
    payload_path = path.parent / payload_name
    expected_bytes = _require_int(payload_info, "bytes")
    expected_payload_sha = _require_string(payload_info, "sha256")
    if payload_path.stat().st_size != expected_bytes:
        raise ValueError("optimizer-batch payload byte-size mismatch.")
    payload_sha = sha256_file(payload_path)
    if payload_sha != expected_payload_sha:
        raise ValueError(
            "optimizer-batch payload SHA256 mismatch: expected "
            f"{expected_payload_sha}, got {payload_sha}."
        )

    batch = batch_from_tensors(load_file(str(payload_path)))
    validate_batch(batch)
    logical_batch_sha = logical_sha(batch)
    expected_logical_sha = _require_string(manifest, "logical_batch_sha256")
    if logical_batch_sha != expected_logical_sha:
        raise ValueError(
            "optimizer-batch logical SHA256 mismatch: expected "
            f"{expected_logical_sha}, got {logical_batch_sha}."
        )
    _validate_manifest_provenance(manifest, batch)
    return LoadedOptimizerBatch(
        batch=batch,
        manifest_path=path,
        payload_path=payload_path,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        payload_sha256=payload_sha,
        logical_batch_sha256=logical_batch_sha,
    )


def _build_manifest(
    *,
    log_path: Path,
    payload_path: Path,
    payload_sha: str,
    logical_batch_sha: str,
    step: int,
    batch: OptimizerBatch,
    config: TrainConfig,
    initial_adapter: AdapterProvenance,
) -> dict[str, object]:
    snapshot = config_snapshot(config)
    contract = optimizer_contract(config)
    return {
        "format": _FORMAT,
        "kind": _KIND,
        "payload": {
            "file": payload_path.name,
            "bytes": payload_path.stat().st_size,
            "sha256": payload_sha,
        },
        "logical_batch_sha256": logical_batch_sha,
        "batch": batch_summary(batch),
        "source": {
            "step": step,
            "seed": config.seed,
            "log_dir": str(log_path.resolve()),
        },
        "config": {
            "snapshot": snapshot,
            "sha256": sha256_json(snapshot),
            "optimizer_contract": contract,
            "optimizer_contract_sha256": sha256_json(contract),
        },
        "initial_adapter": initial_adapter.to_dict(),
        "runtime": _runtime_provenance(),
    }


def _validate_format(manifest: dict[str, object]) -> None:
    if manifest.get("format") != _FORMAT or manifest.get("kind") != _KIND:
        raise ValueError(
            "Unsupported optimizer-batch manifest format/kind: "
            f"{manifest.get('format')!r}/{manifest.get('kind')!r}."
        )


def _validate_manifest_provenance(
    manifest: dict[str, object],
    batch: OptimizerBatch,
) -> None:
    config = _require_object(manifest, "config")
    snapshot = _require_object(config, "snapshot")
    if sha256_json(snapshot) != _require_string(config, "sha256"):
        raise ValueError("optimizer-batch manifest config hash mismatch.")
    contract = _require_object(config, "optimizer_contract")
    if sha256_json(contract) != _require_string(
        config,
        "optimizer_contract_sha256",
    ):
        raise ValueError("optimizer-batch manifest optimizer contract hash mismatch.")
    if _require_object(manifest, "batch") != batch_summary(batch):
        raise ValueError("optimizer-batch manifest batch summary mismatch.")
    adapter = _require_object(manifest, "initial_adapter")
    sha = _require_string(adapter, "weight_sha256")
    if len(sha) != 64:
        raise ValueError("optimizer-batch initial adapter SHA256 is invalid.")


def _runtime_provenance() -> dict[str, object]:
    try:
        retrain_version = version("retrain")
    except PackageNotFoundError:
        retrain_version = "uninstalled"
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda or ""
        cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        torch_version = "unavailable"
        cuda_version = ""
        cuda_devices = 0
    return {
        "retrain_version": retrain_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "cuda_devices": cuda_devices,
    }


def _require_object(parent: dict[str, object], key: str) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"optimizer-batch manifest field {key!r} must be an object.")
    return cast(dict[str, object], value)


def _require_string(parent: dict[str, object], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"optimizer-batch manifest field {key!r} must be a non-empty string."
        )
    return value


def _require_int(parent: dict[str, object], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"optimizer-batch manifest field {key!r} must be a non-negative integer."
        )
    return value
