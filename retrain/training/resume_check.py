"""Preflight checks for checkpoint resume directories."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from retrain.backends.local.state import ADAPTER_WEIGHT_FILES
from retrain.config import TrainConfig
from retrain.training.state import TRAINER_STATE_FILE, load_trainer_state

IssueSeverity = Literal["error", "warning", "info"]

_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_PRIME_STATE_FILE = "prime_rl_backend_state.json"


@dataclass(frozen=True)
class ResumeCheckIssue:
    severity: IssueSeverity
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }


@dataclass
class ResumeCheckResult:
    path: str
    ok: bool = False
    trainer_state_present: bool = False
    trainer_state_valid: bool = False
    checkpoint_name: str = ""
    checkpoint_path: str = ""
    resolved_checkpoint_path: str = ""
    checkpoint_payload: str = "none"
    checkpoint_exists: bool | None = None
    adapter_payload_ok: bool | None = None
    adapter_payload_files: list[str] = field(default_factory=list)
    step: int | None = None
    next_step: int | None = None
    max_steps: int | None = None
    max_steps_ok: bool | None = None
    config_path: str = ""
    config_source: str = ""
    trainer: str = ""
    resume_mode: str = ""
    resume_warning: str = ""
    sft_data: dict[str, object] = field(default_factory=dict)
    recoverability_files: dict[str, bool] = field(default_factory=dict)
    issues: list[ResumeCheckIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "ok": self.ok,
            "trainer_state_present": self.trainer_state_present,
            "trainer_state_valid": self.trainer_state_valid,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_path": self.checkpoint_path,
            "resolved_checkpoint_path": self.resolved_checkpoint_path,
            "checkpoint_payload": self.checkpoint_payload,
            "checkpoint_exists": self.checkpoint_exists,
            "adapter_payload_ok": self.adapter_payload_ok,
            "adapter_payload_files": list(self.adapter_payload_files),
            "step": self.step,
            "next_step": self.next_step,
            "max_steps": self.max_steps,
            "max_steps_ok": self.max_steps_ok,
            "config_path": self.config_path,
            "config_source": self.config_source,
            "trainer": self.trainer,
            "resume_mode": self.resume_mode,
            "resume_warning": self.resume_warning,
            "sft_data": dict(self.sft_data),
            "recoverability_files": dict(self.recoverability_files),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def check_resume_dir(
    resume_dir: str | Path,
    *,
    config: TrainConfig | None = None,
    config_path: str = "",
) -> ResumeCheckResult:
    """Check whether a log/artifact directory has enough state to resume."""
    path = Path(resume_dir).expanduser()
    result = ResumeCheckResult(path=str(path))
    if config is not None:
        result.config_path = config_path
        result.config_source = "config"
        result.trainer = config.trainer
        result.max_steps = int(config.max_steps)
    else:
        _read_resolved_config(path, result)

    if not path.exists():
        _add_issue(result, "error", "resume_dir_missing", f"path does not exist: {path}")
        _finalize(result)
        return result
    if not path.is_dir():
        _add_issue(
            result,
            "error",
            "resume_dir_not_directory",
            f"path is not a directory: {path}",
        )
        _finalize(result)
        return result

    _check_recoverability_files(path, result)
    _check_trainer_state(path, result)
    _check_checkpoint_payload(result)
    _check_max_steps(result)
    _check_sft_data(path, result, config)
    _finalize(result)
    return result


def _check_trainer_state(path: Path, result: ResumeCheckResult) -> None:
    state_path = path / TRAINER_STATE_FILE
    result.trainer_state_present = state_path.is_file()
    if not result.trainer_state_present:
        _add_issue(
            result,
            "error",
            "trainer_state_missing",
            f"missing {TRAINER_STATE_FILE}",
        )
        return

    try:
        state = load_trainer_state(str(path))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _add_issue(
            result,
            "error",
            "trainer_state_invalid",
            f"{TRAINER_STATE_FILE} is invalid: {exc}",
        )
        return

    result.trainer_state_valid = True
    result.step = state["step"]
    result.next_step = state["step"] + 1
    result.checkpoint_name = state.get("checkpoint_name", "")
    result.resolved_checkpoint_path = state.get("checkpoint_path", "")
    result.resume_mode = state.get("resume_mode", "")
    result.resume_warning = state.get("resume_warning", "")

    raw_payload = _read_json_object(state_path)
    if raw_payload is not None:
        result.checkpoint_path = _optional_str(raw_payload, "checkpoint_path")

    if not result.checkpoint_path:
        latest_path = path / "latest_sampler_path.txt"
        if latest_path.is_file():
            try:
                result.checkpoint_path = latest_path.read_text().strip()
            except OSError:
                result.checkpoint_path = ""
    if not result.checkpoint_path:
        result.checkpoint_path = result.checkpoint_name

    if not result.resolved_checkpoint_path and result.checkpoint_name:
        result.resolved_checkpoint_path = result.checkpoint_name

    if not result.checkpoint_path and not result.checkpoint_name:
        _add_issue(
            result,
            "error",
            "checkpoint_ref_missing",
            "trainer state does not name a checkpoint path or checkpoint name",
        )


def _check_checkpoint_payload(result: ResumeCheckResult) -> None:
    checkpoint_ref = result.resolved_checkpoint_path or result.checkpoint_path
    if not checkpoint_ref:
        result.checkpoint_payload = "none"
        result.checkpoint_exists = False
        result.adapter_payload_ok = False
        return

    if _URI_RE.match(checkpoint_ref):
        result.checkpoint_payload = "remote_ref"
        result.checkpoint_exists = None
        result.adapter_payload_ok = None
        _add_issue(
            result,
            "warning",
            "remote_checkpoint_unverified",
            f"remote checkpoint ref is not locally verified: {checkpoint_ref}",
        )
        return

    path = Path(checkpoint_ref).expanduser()
    result.checkpoint_exists = path.exists()
    if not path.exists():
        result.checkpoint_payload = "missing_local_path"
        result.adapter_payload_ok = False
        _add_issue(
            result,
            "error",
            "checkpoint_missing",
            f"checkpoint path does not exist: {path}",
        )
        return

    if path.is_file():
        result.checkpoint_payload = "local_file"
        result.adapter_payload_files = [path.name]
        result.adapter_payload_ok = path.name in {*ADAPTER_WEIGHT_FILES, _PRIME_STATE_FILE}
        if not result.adapter_payload_ok:
            _add_issue(
                result,
                "warning",
                "checkpoint_file_backend_specific",
                f"checkpoint is a local file, not a standard adapter directory: {path}",
            )
        return

    result.checkpoint_payload = "local_dir"
    files = [name for name in (*ADAPTER_WEIGHT_FILES, _PRIME_STATE_FILE) if (path / name).is_file()]
    result.adapter_payload_files = files
    result.adapter_payload_ok = bool(files)
    if not files:
        _add_issue(
            result,
            "error",
            "adapter_payload_missing",
            f"checkpoint directory has no adapter payload files: {path}",
        )


def _check_max_steps(result: ResumeCheckResult) -> None:
    if result.next_step is None or result.max_steps is None:
        return
    result.max_steps_ok = result.next_step <= result.max_steps
    if not result.max_steps_ok:
        _add_issue(
            result,
            "error",
            "resume_beyond_max_steps",
            f"checkpoint resumes at step {result.next_step}, beyond max_steps={result.max_steps}",
        )


def _check_sft_data(
    path: Path,
    result: ResumeCheckResult,
    config: TrainConfig | None,
) -> None:
    existing_sft_data = dict(result.sft_data)
    recoverability_path = path / "sft_data_recoverability.json"
    recoverability = _read_json_object(recoverability_path)
    local_snapshot = path / "sft_data.snapshot.jsonl"
    source_path = _path_from_mapping(recoverability, "source_path")
    snapshot_path = _path_from_mapping(recoverability, "snapshot_path")
    if not snapshot_path and local_snapshot.is_file():
        snapshot_path = local_snapshot
    resolved_config_data_path = _path_from_value(
        existing_sft_data.get("resolved_config_data_path")
    )

    source_exists = source_path.exists() if source_path else False
    snapshot_exists = snapshot_path.exists() if snapshot_path else False
    resolved_config_data_exists = (
        resolved_config_data_path.is_file() if resolved_config_data_path else False
    )
    recoverable = source_exists or snapshot_exists or resolved_config_data_exists
    sft_relevant = result.trainer == "sft" or recoverability is not None
    if recoverability is not None or sft_relevant:
        existing_sft_data.update(
            {
                "recoverability_file": str(recoverability_path),
                "recoverability_present": recoverability is not None,
                "source_path": str(source_path) if source_path else "",
                "source_exists": source_exists,
                "snapshot_path": str(snapshot_path) if snapshot_path else "",
                "snapshot_exists": snapshot_exists,
                "resolved_config_data_exists": resolved_config_data_exists,
                "recoverable": recoverable,
            }
        )
        result.sft_data = existing_sft_data

    if recoverability_path.is_file() and recoverability is None:
        _add_issue(
            result,
            "warning",
            "sft_recoverability_invalid",
            "sft_data_recoverability.json is not a valid JSON object",
        )

    config_can_supply_sft_data = (
        config is not None
        and result.trainer == "sft"
        and bool(config.sft_data_path)
    )
    if sft_relevant and not recoverable and not config_can_supply_sft_data:
        severity: IssueSeverity = "error" if result.trainer == "sft" else "warning"
        _add_issue(
            result,
            severity,
            "sft_data_not_recoverable",
            "SFT data is neither present at source_path nor snapshotted in the run directory",
        )

    if config is None or result.trainer != "sft":
        return

    if not config.sft_data_path:
        _add_issue(
            result,
            "error",
            "sft_config_missing_data_path",
            "trainer='sft' config does not set sft_data_path",
        )
        return

    config_data_path = Path(config.sft_data_path).expanduser()
    if not config_data_path.is_file():
        _add_issue(
            result,
            "error",
            "sft_config_data_missing",
            f"configured SFT data does not exist: {config_data_path}",
        )
        return

    _verify_sft_config_data(result, config)


def _verify_sft_config_data(result: ResumeCheckResult, config: TrainConfig) -> None:
    try:
        from retrain.training.sft import load_sft_dataset, verify_sft_data_contract

        dataset = load_sft_dataset(config.sft_data_path)
        verify_sft_data_contract(config, dataset.provenance)
        sft_data = dict(result.sft_data)
        sft_data.update(
            {
                "config_data_path": dataset.provenance.data_path,
                "config_data_sha256": dataset.provenance.data_sha256,
                "config_data_rows": dataset.provenance.data_rows,
                "config_data_contract_ok": True,
                "recoverable": True,
            }
        )
        result.sft_data = sft_data
    except Exception as exc:
        _add_issue(
            result,
            "error",
            "sft_config_data_invalid",
            f"configured SFT data contract failed: {exc}",
        )


def _check_recoverability_files(path: Path, result: ResumeCheckResult) -> None:
    names = (
        TRAINER_STATE_FILE,
        "latest_sampler_path.txt",
        "adapter",
        "resolved_config.json",
        "sft_data.snapshot.jsonl",
        "sft_data_recoverability.json",
        "sft_manifest.json",
    )
    result.recoverability_files = {name: (path / name).exists() for name in names}


def _read_resolved_config(path: Path, result: ResumeCheckResult) -> None:
    payload = _read_json_object(path / "resolved_config.json")
    if payload is None:
        return
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        return
    config_map = cast(Mapping[str, object], config_payload)
    result.config_source = "resolved_config.json"
    result.max_steps = _optional_int(config_map, "max_steps")
    result.trainer = _optional_str(config_map, "trainer")
    sft_data_path = _optional_str(config_map, "sft_data_path")
    if sft_data_path:
        result.sft_data = {
            "resolved_config_data_path": sft_data_path,
            "resolved_config_data_sha256": _optional_str(config_map, "sft_data_sha256"),
            "resolved_config_data_rows": _optional_int(config_map, "sft_data_rows"),
        }


def _read_json_object(path: Path) -> Mapping[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return cast(Mapping[str, object], payload) if isinstance(payload, dict) else None


def _path_from_mapping(payload: Mapping[str, object] | None, key: str) -> Path | None:
    if payload is None:
        return None
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser()


def _path_from_value(value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser()


def _optional_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _add_issue(
    result: ResumeCheckResult,
    severity: IssueSeverity,
    code: str,
    message: str,
) -> None:
    result.issues.append(ResumeCheckIssue(severity=severity, code=code, message=message))


def _finalize(result: ResumeCheckResult) -> None:
    result.ok = not any(issue.severity == "error" for issue in result.issues)
