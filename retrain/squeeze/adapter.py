"""Locate and load LoRA adapter matrices (local or Tinker-hosted)."""

from __future__ import annotations

import json
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file

from retrain.backends.tinker.runtime import load_tinker


def _resolve_tinker_path(tinker_path: str, local_dir: str | None = None) -> str:
    """Download a tinker:// checkpoint to a local directory.

    Uses the Tinker SDK to get a signed archive URL, downloads and extracts
    the PEFT adapter files (adapter_model.safetensors + adapter_config.json).

    Args:
        tinker_path: tinker:// URI (e.g. tinker://run-id/weights/checkpoint-name)
        local_dir: where to extract (default: tempdir under /tmp/retrain_squeeze/)

    Returns:
        Local directory path containing the extracted adapter files.
    """
    try:
        tinker = load_tinker()
    except ImportError as exc:
        raise RuntimeError(
            "Tinker SDK required for tinker:// paths. Install with: uv add tinker"
        ) from exc

    if local_dir is None:
        local_dir = tempfile.mkdtemp(prefix="retrain_squeeze_")

    out = Path(local_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (out / "adapter_model.safetensors").is_file():
        print(f"Using cached adapter at {out}")
        return str(out)

    print(f"Downloading Tinker checkpoint: {tinker_path}")
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()

    url_response = rest_client.get_checkpoint_archive_url_from_tinker_path(
        tinker_path
    ).result()

    # Download archive
    archive_path = out / "checkpoint.tar"
    with urllib.request.urlopen(url_response.url, timeout=300) as response:
        with open(archive_path, "wb") as f:
            f.write(response.read())

    # Extract
    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(path=out, filter="data")

    archive_path.unlink()

    # Verify expected files
    if not (out / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(
            f"Downloaded checkpoint missing adapter_model.safetensors: {out}"
        )

    print(f"Checkpoint extracted to {out}")
    return str(out)


def _resolve_adapter_path(adapter_path: str) -> str:
    """Resolve adapter_path: download if tinker://, otherwise return as-is."""
    if adapter_path.startswith("tinker://"):
        return _resolve_tinker_path(adapter_path)
    return adapter_path


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------

def load_adapter_matrices(
    adapter_path: str, device: str = "cpu"
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Load PEFT safetensors, pair lora_A/lora_B by module name.

    PEFT stores:
      lora_A.weight: (r, in_features)   — we transpose to (in_features, r)
      lora_B.weight: (out_features, r)   — kept as-is

    Returns list of (module_name, A, B) where:
      A: (m, r) = in_features × rank
      B: (r, n) = rank × out_features  (transposed from PEFT convention)

    So the effective weight delta is A @ B = (m, n).
    """
    safetensors_path = Path(adapter_path) / "adapter_model.safetensors"
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_path}")

    state_dict = load_file(str(safetensors_path), device=device)

    # Group by module name
    a_matrices: dict[str, torch.Tensor] = {}
    b_matrices: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if "lora_A" in key:
            # Extract module name: everything before .lora_A
            module_name = key.split(".lora_A")[0]
            # PEFT stores lora_A as (r, in_features), transpose to (in_features, r)
            a_matrices[module_name] = tensor.float().t()
        elif "lora_B" in key:
            module_name = key.split(".lora_B")[0]
            # PEFT stores lora_B as (out_features, r), transpose to (r, out_features)
            b_matrices[module_name] = tensor.float().t()

    # Pair them up
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for name in sorted(a_matrices.keys()):
        if name not in b_matrices:
            continue
        pairs.append((name, a_matrices[name], b_matrices[name]))

    if not pairs:
        raise ValueError(f"No lora_A/lora_B pairs found in {adapter_path}")

    return pairs


# ---------------------------------------------------------------------------
# Core algorithm (Algorithm 2 from paper)
# ---------------------------------------------------------------------------
