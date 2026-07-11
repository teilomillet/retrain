"""Atomic filesystem primitives for optimizer-batch artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def write_safetensors_atomic(
    path: Path,
    tensors: dict[str, np.ndarray],
    *,
    format_name: str,
) -> None:
    """Commit a safetensors payload through an atomic sibling rename."""

    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    save_file(tensors, str(tmp), metadata={"format": format_name})
    os.replace(tmp, path)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Fsync and atomically commit a small manifest-like payload."""

    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
