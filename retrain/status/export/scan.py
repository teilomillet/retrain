"""Filesystem scanning for live run snapshots."""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import cast

from retrain.io.json import JSONDecodeError, loads
from retrain.metrics.scan import JsonObject, float_or_none, int_or_none
from retrain.status.export.types import RunSnapshot
from retrain.training.state import TRAINER_STATE_FILE


_RECENT_DIAG_LIMIT = 256
_LATEST_JSONL_ENTRY_SEARCH_LIMIT = 16
_ACTIVE_AGE_SECONDS = 10 * 60


def _tail_jsonl(path: Path, limit: int) -> list[JsonObject]:
    if not path.is_file():
        return []
    if limit <= 0:
        return []

    lines: list[bytes]
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            chunks: list[bytes] = []
            newline_count = 0
            while position > 0 and newline_count <= limit:
                read_size = min(8192, position)
                position -= read_size
                handle.seek(position)
                chunk = handle.read(read_size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
    except OSError:
        return []

    if not chunks:
        return []

    lines = b"".join(reversed(chunks)).splitlines()[-limit:]
    rows: deque[JsonObject] = deque(maxlen=limit)
    for raw_line in lines:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            payload = loads(raw_line)
        except JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(cast(JsonObject, payload))
    return list(rows)


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int) and not isinstance(item, bool)]


def _infer_phase(run_name: str, metrics_entry: JsonObject | None) -> str:
    if metrics_entry and isinstance(metrics_entry.get("phase"), str):
        phase = str(metrics_entry["phase"]).strip()
        if phase:
            return phase
    lowered = run_name.lower()
    if "sft" in lowered:
        return "sft"
    if "rl" in lowered:
        return "rl"
    return "train"


def _completed_from_state(path: Path) -> bool:
    state_path = path / TRAINER_STATE_FILE
    if not state_path.is_file():
        return False
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return payload.get("checkpoint_name") == "final"


def _trainer_name(path: Path) -> str:
    meta_path = path / "run_meta.json"
    if not meta_path.is_file():
        return ""
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    trainer = payload.get("trainer")
    return str(trainer) if isinstance(trainer, str) else ""


def _scan_run(path: Path, now: float) -> RunSnapshot | None:
    metrics_path = path / "metrics.jsonl"
    diag_path = path / "tinker_sample_diagnostics.jsonl"
    run_meta_path = path / "run_meta.json"
    if not (metrics_path.is_file() or diag_path.is_file() or run_meta_path.is_file()):
        return None

    latest_metrics_rows = _tail_jsonl(metrics_path, _LATEST_JSONL_ENTRY_SEARCH_LIMIT)
    latest_metrics = latest_metrics_rows[-1] if latest_metrics_rows else None
    diag_rows = _tail_jsonl(diag_path, _RECENT_DIAG_LIMIT)
    result_rows = [row for row in diag_rows if row.get("event") == "result"]
    dispatch_rows = [row for row in diag_rows if row.get("event") == "dispatch"]
    latest_diag_row = diag_rows[-1] if diag_rows else None
    latest_result_row = result_rows[-1] if result_rows else None

    metrics_age_seconds: float | None = None
    if metrics_path.is_file():
        try:
            metrics_age_seconds = max(0.0, now - metrics_path.stat().st_mtime)
        except OSError:
            metrics_age_seconds = None

    def _event_age(row: JsonObject | None) -> float | None:
        if row is None:
            return None
        ts = float_or_none(row.get("ts"))
        if ts is None:
            return None
        return max(0.0, now - ts)

    sample_event_age_seconds = _event_age(latest_diag_row)
    sample_result_age_seconds = _event_age(latest_result_row)
    completed = _completed_from_state(path)
    active = (
        not completed
        and (
            (metrics_age_seconds is not None and metrics_age_seconds <= _ACTIVE_AGE_SECONDS)
            or (
                sample_event_age_seconds is not None
                and sample_event_age_seconds <= _ACTIVE_AGE_SECONDS
            )
        )
    )

    prompt_tokens = [
        token_count
        for token_count in (
            int_or_none(row.get("prompt_tokens")) for row in dispatch_rows + result_rows
        )
        if token_count is not None
    ]
    completion_tokens = [
        max(seq_tokens)
        for seq_tokens in (
            _int_list(row.get("completion_tokens")) for row in result_rows
        )
        if seq_tokens
    ]
    result_latencies = [
        latency
        for latency in (float_or_none(row.get("result_latency_s")) for row in result_rows)
        if latency is not None
    ]
    error_counter = Counter(
        str(row.get("error_type") or "")
        for row in result_rows
        if str(row.get("status")) == "error"
    )

    return RunSnapshot(
        run=path.name,
        path=str(path),
        trainer=_trainer_name(path),
        phase=_infer_phase(path.name, latest_metrics),
        metrics_present=latest_metrics is not None,
        completed=completed,
        active=active,
        latest_step=int_or_none(latest_metrics.get("step")) if latest_metrics else None,
        latest_mean_reward=float_or_none(latest_metrics.get("mean_reward")) if latest_metrics else None,
        latest_loss=float_or_none(latest_metrics.get("loss")) if latest_metrics else None,
        latest_max_token_hit_rate=float_or_none(latest_metrics.get("max_token_hit_rate")) if latest_metrics else None,
        latest_invalid_action_rate=float_or_none(latest_metrics.get("behavior/invalid_action_rate")) if latest_metrics else None,
        latest_avg_response_chars=float_or_none(latest_metrics.get("behavior/avg_response_chars")) if latest_metrics else None,
        latest_step_time_s=float_or_none(latest_metrics.get("step_time_s")) if latest_metrics else None,
        latest_tokens_per_second=float_or_none(latest_metrics.get("tokens_per_second")) if latest_metrics else None,
        latest_sample_share=float_or_none(latest_metrics.get("sample_share")) if latest_metrics else None,
        latest_train_share=float_or_none(latest_metrics.get("train_share")) if latest_metrics else None,
        latest_process_max_rss_mb=float_or_none(latest_metrics.get("process_max_rss_mb")) if latest_metrics else None,
        metrics_age_seconds=metrics_age_seconds,
        sample_event_age_seconds=sample_event_age_seconds,
        sample_result_age_seconds=sample_result_age_seconds,
        recent_dispatch_count=len(dispatch_rows),
        recent_result_count=len(result_rows),
        recent_error_count=sum(error_counter.values()),
        recent_timeout_count=error_counter.get("TimeoutError", 0),
        recent_prompt_tokens_last=prompt_tokens[-1] if prompt_tokens else None,
        recent_prompt_tokens_max=max(prompt_tokens) if prompt_tokens else None,
        recent_completion_tokens_last=completion_tokens[-1] if completion_tokens else None,
        recent_completion_tokens_max=max(completion_tokens) if completion_tokens else None,
        recent_result_latency_last_s=result_latencies[-1] if result_latencies else None,
        recent_result_latency_max_s=max(result_latencies) if result_latencies else None,
    )


def collect_run_snapshots(root: Path) -> list[RunSnapshot]:
    now = time.time()
    snapshots: list[RunSnapshot] = []
    if not root.is_dir():
        return snapshots
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        snapshot = _scan_run(child, now)
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots
