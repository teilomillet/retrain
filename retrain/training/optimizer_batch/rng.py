"""Torch RNG capture and restore at the optimizer boundary."""

from __future__ import annotations

from retrain.training.optimizer_batch.types import TorchRngState


def capture_torch_rng_state() -> TorchRngState:
    """Capture CPU and visible-CUDA RNG bytes without consuming randomness."""

    import torch

    cpu = bytes(torch.get_rng_state().cpu().tolist())
    cuda = (
        tuple(bytes(state.cpu().tolist()) for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    return TorchRngState(cpu=cpu, cuda=cuda)


def restore_torch_rng_state(state: TorchRngState) -> None:
    """Restore optimizer-boundary RNG after model construction and loading."""

    import torch

    if not state.cpu:
        raise ValueError("optimizer-batch artifact has no Torch CPU RNG state.")
    torch.set_rng_state(torch.tensor(list(state.cpu), dtype=torch.uint8))
    if not state.cuda:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "optimizer-batch artifact contains CUDA RNG state but CUDA is unavailable."
        )
    current_devices = torch.cuda.device_count()
    if len(state.cuda) != current_devices:
        raise RuntimeError(
            "optimizer-batch CUDA RNG device-count mismatch: expected "
            f"{len(state.cuda)}, got {current_devices}."
        )
    torch.cuda.set_rng_state_all(
        [torch.tensor(list(item), dtype=torch.uint8) for item in state.cuda]
    )
