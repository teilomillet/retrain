"""Local adapter content provenance for deterministic batch replay."""

from __future__ import annotations

import re
from pathlib import Path

from retrain.config import TrainConfig
from retrain.training.optimizer_batch.types import AdapterProvenance
from retrain.training.optimizer_batch.storage import sha256_file
from retrain.training.state import load_trainer_state


_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def adapter_provenance(
    adapter_dir: str | Path,
    *,
    requested_ref: str | None = None,
) -> AdapterProvenance:
    """Hash the exact local weight file preferred by the local backend."""

    directory = Path(adapter_dir).expanduser().resolve()
    for filename in _WEIGHT_FILES:
        weight_path = directory / filename
        if not weight_path.is_file():
            continue
        return AdapterProvenance(
            requested_ref=requested_ref or str(adapter_dir),
            adapter_dir=str(directory),
            weight_file=filename,
            weight_bytes=weight_path.stat().st_size,
            weight_sha256=sha256_file(weight_path),
        )
    raise FileNotFoundError(
        f"No local adapter weights in {directory}. Expected one of "
        f"{list(_WEIGHT_FILES)}."
    )


def resolve_initial_adapter(config: TrainConfig) -> AdapterProvenance:
    """Resolve and hash the same resume adapter the trainer will load."""

    if not config.resume_from:
        raise ValueError("optimizer-batch replay requires resume.from.")
    state = load_trainer_state(config.resume_from)
    requested_ref = state.get("checkpoint_path", "") or state.get(
        "checkpoint_name",
        "",
    )
    if not requested_ref:
        raise ValueError(
            "resume trainer state does not identify an initial adapter checkpoint."
        )
    if _URI_RE.match(requested_ref):
        raise ValueError(
            "optimizer-batch replay v1 requires a local resume adapter, got "
            f"{requested_ref!r}."
        )

    direct = Path(requested_ref).expanduser()
    if _contains_weights(direct):
        return adapter_provenance(direct, requested_ref=requested_ref)

    # Mirror local.state.resolve_adapter_dir exactly: a direct adapter wins;
    # otherwise a relative checkpoint name resolves under adapter_path. The
    # resume directory is not searched here because the backend does not
    # search it either. load_trainer_state has already made a saved relative
    # checkpoint_path artifact-local when appropriate.
    relative_to_output = Path(config.adapter_path).expanduser() / direct
    if _contains_weights(relative_to_output):
        return adapter_provenance(
            relative_to_output,
            requested_ref=requested_ref,
        )
    raise FileNotFoundError(
        "Could not resolve local initial adapter for optimizer-batch replay: "
        f"{requested_ref!r}."
    )


def validate_capture_resume_step(*, saved_step: int, max_steps: int) -> None:
    """Fail unless v1 capture will execute exactly logical step zero once."""

    start_step = saved_step + 1
    if start_step != 0 or max_steps != 1:
        raise ValueError(
            "optimizer-batch capture v1 requires resume trainer_state.step = -1 "
            "and max_steps = 1 so exactly one optimizer-boundary iteration runs; "
            f"got saved step {saved_step} (start step {start_step}) and "
            f"max_steps {max_steps}."
        )


def preflight_optimizer_batch_capture(config: TrainConfig) -> AdapterProvenance:
    """Validate one-step capture state and hash its adapter before GPU setup."""

    if not config.resume_from:
        raise ValueError("optimizer-batch capture requires resume.from.")
    state = load_trainer_state(config.resume_from)
    validate_capture_resume_step(
        saved_step=state["step"],
        max_steps=config.max_steps,
    )
    return resolve_initial_adapter(config)


def _contains_weights(path: Path) -> bool:
    return path.is_dir() and any((path / name).is_file() for name in _WEIGHT_FILES)
