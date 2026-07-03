"""Checkpoint state serialization for trainer-style runners."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from retrain.training.sepa import SEPAStateDict


TRAINER_STATE_FILE = "trainer_state.json"


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
    sepa: SEPAStateDict
    tl_grpo_ema: NotRequired[float]
    delight_eta_ema: NotRequired[float]


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
    sepa_state: SEPAStateDict,
    tl_grpo_ema: float | None = None,
    delight_eta_ema: float | None = None,
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
    if tl_grpo_ema is not None:
        state["tl_grpo_ema"] = tl_grpo_ema
    if delight_eta_ema is not None:
        state["delight_eta_ema"] = delight_eta_ema
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
        state["checkpoint_path"] = checkpoint_path
    else:
        latest_sampler_path = path / "latest_sampler_path.txt"
        if latest_sampler_path.is_file():
            fallback_path = latest_sampler_path.read_text().strip()
            if fallback_path:
                state["checkpoint_path"] = fallback_path
    tl_grpo_ema = _optional_float_field(payload_map, "tl_grpo_ema")
    if tl_grpo_ema is not None:
        state["tl_grpo_ema"] = tl_grpo_ema
    delight_eta_ema = _optional_float_field(payload_map, "delight_eta_ema")
    if delight_eta_ema is not None:
        state["delight_eta_ema"] = delight_eta_ema
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
