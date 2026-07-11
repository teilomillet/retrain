"""Checkpoint state serialization for trainer-style runners."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from retrain.training.sepa import SEPAStateDict


TRAINER_STATE_FILE = "trainer_state.json"
_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


class TrainerState(TypedDict):
    """Serialized trainer state stored in checkpoint directories."""

    step: int
    example_idx: int
    total_correct: int
    total_completions: int
    current_batch_size: int
    current_group_size: int
    checkpoint_name: str
    checkpoint_path: NotRequired[str]
    resume_mode: NotRequired[str]
    resume_warning: NotRequired[str]
    sepa: SEPAStateDict
    tl_grpo_ema: NotRequired[float]
    delight_eta_ema: NotRequired[float]
    sft_schedule: NotRequired[dict[str, object]]


def save_trainer_state(
    path: Path,
    *,
    step: int,
    example_idx: int,
    total_correct: int,
    total_completions: int,
    current_batch_size: int,
    current_group_size: int,
    checkpoint_name: str,
    checkpoint_path: str | None = None,
    resume_mode: str = "",
    resume_warning: str = "",
    sepa_state: SEPAStateDict,
    tl_grpo_ema: float | None = None,
    delight_eta_ema: float | None = None,
    sft_schedule: Mapping[str, object] | None = None,
) -> None:
    """Write trainer-side state to JSON for checkpoint resume."""
    path.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {
        "step": step,
        "example_idx": example_idx,
        "total_correct": total_correct,
        "total_completions": total_completions,
        "current_batch_size": current_batch_size,
        "current_group_size": current_group_size,
        "checkpoint_name": checkpoint_name,
        "sepa": sepa_state,
    }
    if checkpoint_path:
        state["checkpoint_path"] = checkpoint_path
    if resume_mode:
        state["resume_mode"] = resume_mode
    if resume_warning:
        state["resume_warning"] = resume_warning
    if tl_grpo_ema is not None:
        state["tl_grpo_ema"] = tl_grpo_ema
    if delight_eta_ema is not None:
        state["delight_eta_ema"] = delight_eta_ema
    if sft_schedule is not None:
        state["sft_schedule"] = dict(sft_schedule)
    tmp = path / f"{TRAINER_STATE_FILE}.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path / TRAINER_STATE_FILE)
    if checkpoint_path:
        latest_tmp = path / "latest_sampler_path.txt.tmp"
        latest_tmp.write_text(f"{checkpoint_path}\n")
        latest_tmp.replace(path / "latest_sampler_path.txt")


def load_trainer_state(resume_dir: str) -> TrainerState:
    """Load trainer state from a checkpoint directory."""
    path = Path(resume_dir)
    state_file = path / TRAINER_STATE_FILE
    if not state_file.is_file():
        raise FileNotFoundError(
            f"No {TRAINER_STATE_FILE} found in {resume_dir}. "
            f"Cannot resume without trainer state."
        )
    payload = json.loads(state_file.read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid trainer state file {state_file}: expected JSON object."
        )
    payload_map = cast(Mapping[str, object], payload)
    state: TrainerState = {
        "step": _require_int_field(payload_map, "step"),
        "example_idx": _require_int_field(payload_map, "example_idx"),
        "total_correct": _require_int_field(payload_map, "total_correct"),
        "total_completions": _require_int_field(payload_map, "total_completions"),
        "current_batch_size": _require_int_field(payload_map, "current_batch_size"),
        "current_group_size": _require_int_field(payload_map, "current_group_size"),
        "checkpoint_name": _optional_str_field(payload_map, "checkpoint_name"),
        "sepa": _optional_sepa_state(payload_map),
    }
    checkpoint_path = _optional_str_field(payload_map, "checkpoint_path")
    if checkpoint_path:
        state["checkpoint_path"] = _resolve_checkpoint_path(path, checkpoint_path)
    else:
        latest_sampler_path = path / "latest_sampler_path.txt"
        if latest_sampler_path.is_file():
            fallback_path = latest_sampler_path.read_text().strip()
            if fallback_path:
                state["checkpoint_path"] = _resolve_checkpoint_path(path, fallback_path)
    resume_mode = _optional_str_field(payload_map, "resume_mode")
    if resume_mode:
        state["resume_mode"] = resume_mode
    resume_warning = _optional_str_field(payload_map, "resume_warning")
    if resume_warning:
        state["resume_warning"] = resume_warning
    tl_grpo_ema = _optional_float_field(payload_map, "tl_grpo_ema")
    if tl_grpo_ema is not None:
        state["tl_grpo_ema"] = tl_grpo_ema
    delight_eta_ema = _optional_float_field(payload_map, "delight_eta_ema")
    if delight_eta_ema is not None:
        state["delight_eta_ema"] = delight_eta_ema
    sft_schedule = _optional_object_field(payload_map, "sft_schedule")
    if sft_schedule is not None:
        state["sft_schedule"] = sft_schedule
    return state


def _require_int_field(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Trainer state field '{key}' must be an integer.")
    return value


def _optional_str_field(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"Trainer state field '{key}' must be a string.")
    return value


def _optional_sepa_state(payload: Mapping[str, object]) -> SEPAStateDict:
    value = payload.get("sepa", {})
    if not isinstance(value, dict):
        raise ValueError("Trainer state field 'sepa' must be an object.")
    return cast(SEPAStateDict, value)


def _optional_float_field(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Trainer state field '{key}' must be a number.")
    return float(value)


def _optional_object_field(
    payload: Mapping[str, object],
    key: str,
) -> dict[str, object] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Trainer state field '{key}' must be an object.")
    return dict(cast(Mapping[str, object], value))


def _resolve_checkpoint_path(resume_dir: Path, checkpoint_path: str) -> str:
    """Prefer artifact-local adapter payloads when original paths disappeared."""
    if _URI_RE.match(checkpoint_path):
        return checkpoint_path
    original_path = Path(checkpoint_path).expanduser()
    if original_path.exists():
        return checkpoint_path
    if not original_path.is_absolute():
        relative_to_resume = resume_dir / original_path
        if relative_to_resume.exists():
            return str(relative_to_resume)
    artifact_adapter = resume_dir / "adapter"
    if artifact_adapter.exists():
        return str(artifact_adapter)
    return checkpoint_path
