"""Atomic filesystem primitives for optimizer-batch artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from retrain.io.digest import sha256_file as sha256_file

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
