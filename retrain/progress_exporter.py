"""Prometheus/JSON exporter for live retrain runs.

Scans retrain log directories, surfaces the latest step metrics, and exposes
high-frequency sampling diagnostics from ``tinker_sample_diagnostics.jsonl`` so
Grafana/Prometheus can follow long-running jobs before ``metrics.jsonl`` lands.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from retrain.json_utils import JSONDecodeError, loads


_RECENT_DIAG_LIMIT = 256
_ACTIVE_AGE_SECONDS = 10 * 60
JsonObject = dict[str, object]


@dataclass
class RunSnapshot:
    run: str
    path: str
    trainer: str
    phase: str
    metrics_present: bool
    completed: bool
    active: bool
    latest_step: int | None
    latest_mean_reward: float | None
    latest_loss: float | None
    latest_max_token_hit_rate: float | None
    latest_invalid_action_rate: float | None
    latest_avg_response_chars: float | None
    latest_step_time_s: float | None
    latest_tokens_per_second: float | None
    latest_sample_share: float | None
    latest_train_share: float | None
    latest_process_max_rss_mb: float | None
    metrics_age_seconds: float | None
    sample_event_age_seconds: float | None
    sample_result_age_seconds: float | None
    recent_dispatch_count: int
    recent_result_count: int
    recent_error_count: int
    recent_timeout_count: int
    recent_prompt_tokens_last: int | None
    recent_prompt_tokens_max: int | None
    recent_completion_tokens_last: int | None
    recent_completion_tokens_max: int | None
    recent_result_latency_last_s: float | None
    recent_result_latency_max_s: float | None


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
            rows.append(payload)
    return list(rows)


def _latest_jsonl_entry(path: Path) -> JsonObject | None:
    rows = _tail_jsonl(path, 1)
    return rows[-1] if rows else None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


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
    state_path = path / "trainer_state.json"
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

    latest_metrics = _latest_jsonl_entry(metrics_path)
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
        ts = _float_or_none(row.get("ts"))
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
            _int_or_none(row.get("prompt_tokens")) for row in dispatch_rows + result_rows
        )
        if token_count is not None
    ]
    completion_tokens = [
        max(seq_tokens)
        for seq_tokens in (
            [
                token
                for token in row.get("completion_tokens", [])
                if isinstance(token, int) and not isinstance(token, bool)
            ]
            for row in result_rows
        )
        if seq_tokens
    ]
    result_latencies = [
        latency
        for latency in (_float_or_none(row.get("result_latency_s")) for row in result_rows)
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
        latest_step=_int_or_none(latest_metrics.get("step")) if latest_metrics else None,
        latest_mean_reward=_float_or_none(latest_metrics.get("mean_reward")) if latest_metrics else None,
        latest_loss=_float_or_none(latest_metrics.get("loss")) if latest_metrics else None,
        latest_max_token_hit_rate=_float_or_none(latest_metrics.get("max_token_hit_rate")) if latest_metrics else None,
        latest_invalid_action_rate=_float_or_none(latest_metrics.get("behavior/invalid_action_rate")) if latest_metrics else None,
        latest_avg_response_chars=_float_or_none(latest_metrics.get("behavior/avg_response_chars")) if latest_metrics else None,
        latest_step_time_s=_float_or_none(latest_metrics.get("step_time_s")) if latest_metrics else None,
        latest_tokens_per_second=_float_or_none(latest_metrics.get("tokens_per_second")) if latest_metrics else None,
        latest_sample_share=_float_or_none(latest_metrics.get("sample_share")) if latest_metrics else None,
        latest_train_share=_float_or_none(latest_metrics.get("train_share")) if latest_metrics else None,
        latest_process_max_rss_mb=_float_or_none(latest_metrics.get("process_max_rss_mb")) if latest_metrics else None,
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


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _metric_line(name: str, value: float | int, labels: dict[str, str]) -> str:
    rendered_labels = ",".join(
        f'{key}="{_escape_label(label_value)}"' for key, label_value in labels.items()
    )
    return f"{name}{{{rendered_labels}}} {value}"


def render_prometheus_text(snapshots: list[RunSnapshot]) -> str:
    lines = [
        "# HELP soma_retrain_run_info Static info for a retrain run.",
        "# TYPE soma_retrain_run_info gauge",
    ]
    for snapshot in snapshots:
        labels = {
            "run": snapshot.run,
            "phase": snapshot.phase,
            "trainer": snapshot.trainer or "unknown",
            "path": snapshot.path,
        }
        lines.append(_metric_line("soma_retrain_run_info", 1, labels))

    metric_map: list[tuple[str, str, str, callable[[RunSnapshot], float | int | None]]] = [
        ("soma_retrain_run_active", "Whether the run appears active.", "gauge", lambda s: 1 if s.active else 0),
        ("soma_retrain_run_completed", "Whether the run finished with a final checkpoint.", "gauge", lambda s: 1 if s.completed else 0),
        ("soma_retrain_metrics_present", "Whether metrics.jsonl exists for the run.", "gauge", lambda s: 1 if s.metrics_present else 0),
        ("soma_retrain_latest_step", "Latest completed training step.", "gauge", lambda s: s.latest_step),
        ("soma_retrain_latest_mean_reward", "Latest mean reward from metrics.jsonl.", "gauge", lambda s: s.latest_mean_reward),
        ("soma_retrain_latest_loss", "Latest loss from metrics.jsonl.", "gauge", lambda s: s.latest_loss),
        ("soma_retrain_latest_max_token_hit_rate", "Latest max-token hit rate from metrics.jsonl.", "gauge", lambda s: s.latest_max_token_hit_rate),
        ("soma_retrain_latest_invalid_action_rate", "Latest invalid action rate from metrics.jsonl.", "gauge", lambda s: s.latest_invalid_action_rate),
        ("soma_retrain_latest_avg_response_chars", "Latest average response length from metrics.jsonl.", "gauge", lambda s: s.latest_avg_response_chars),
        ("soma_retrain_latest_step_time_seconds", "Latest full training-step duration.", "gauge", lambda s: s.latest_step_time_s),
        ("soma_retrain_latest_tokens_per_second", "Latest tokens-per-second throughput from metrics.jsonl.", "gauge", lambda s: s.latest_tokens_per_second),
        ("soma_retrain_latest_sample_share", "Latest sampling share of step time.", "gauge", lambda s: s.latest_sample_share),
        ("soma_retrain_latest_train_share", "Latest training share of step time.", "gauge", lambda s: s.latest_train_share),
        ("soma_retrain_latest_process_max_rss_megabytes", "Latest reported process peak RSS.", "gauge", lambda s: s.latest_process_max_rss_mb),
        ("soma_retrain_metrics_age_seconds", "Seconds since metrics.jsonl was updated.", "gauge", lambda s: s.metrics_age_seconds),
        ("soma_retrain_sample_event_age_seconds", "Seconds since any sample diagnostic event was recorded.", "gauge", lambda s: s.sample_event_age_seconds),
        ("soma_retrain_sample_result_age_seconds", "Seconds since the latest sample result was recorded.", "gauge", lambda s: s.sample_result_age_seconds),
        ("soma_retrain_recent_dispatch_count", "Recent dispatch events seen in diagnostics.", "gauge", lambda s: s.recent_dispatch_count),
        ("soma_retrain_recent_result_count", "Recent result events seen in diagnostics.", "gauge", lambda s: s.recent_result_count),
        ("soma_retrain_recent_error_count", "Recent sample errors seen in diagnostics.", "gauge", lambda s: s.recent_error_count),
        ("soma_retrain_recent_timeout_count", "Recent TimeoutError count seen in diagnostics.", "gauge", lambda s: s.recent_timeout_count),
        ("soma_retrain_recent_prompt_tokens_last", "Latest prompt token count seen in diagnostics.", "gauge", lambda s: s.recent_prompt_tokens_last),
        ("soma_retrain_recent_prompt_tokens_max", "Max recent prompt token count seen in diagnostics.", "gauge", lambda s: s.recent_prompt_tokens_max),
        ("soma_retrain_recent_completion_tokens_last", "Latest completion token count seen in diagnostics.", "gauge", lambda s: s.recent_completion_tokens_last),
        ("soma_retrain_recent_completion_tokens_max", "Max recent completion token count seen in diagnostics.", "gauge", lambda s: s.recent_completion_tokens_max),
        ("soma_retrain_recent_result_latency_seconds_last", "Latest sample result latency from diagnostics.", "gauge", lambda s: s.recent_result_latency_last_s),
        ("soma_retrain_recent_result_latency_seconds_max", "Max sample result latency in recent diagnostics.", "gauge", lambda s: s.recent_result_latency_max_s),
    ]
    for name, help_text, metric_type, getter in metric_map:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {metric_type}")
        for snapshot in snapshots:
            value = getter(snapshot)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            lines.append(
                _metric_line(
                    name,
                    value,
                    {"run": snapshot.run, "phase": snapshot.phase},
                )
            )
    return "\n".join(lines) + "\n"


class _ExporterHandler(BaseHTTPRequestHandler):
    root: Path = Path(".")

    def do_GET(self) -> None:  # noqa: N802
        snapshots = collect_run_snapshots(self.root)
        if self.path in ("/metrics", "/metrics/"):
            payload = render_prometheus_text(snapshots).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path in ("/v1/runs", "/v1/runs/"):
            payload = json.dumps(
                {
                    "generated_at": time.time(),
                    "root": str(self.root),
                    "runs": [asdict(snapshot) for snapshot in snapshots],
                },
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path in ("/healthz", "/healthz/"):
            payload = json.dumps(
                {
                    "status": "ok",
                    "root": str(self.root),
                    "run_count": len(snapshots),
                }
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expose live retrain run progress.")
    parser.add_argument("--root", default="logs", help="Root directory containing retrain run dirs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9108)
    args = parser.parse_args(argv)

    handler_cls = type(
        "RetrainProgressHandler",
        (_ExporterHandler,),
        {"root": Path(args.root).resolve()},
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(
        f"retrain progress exporter listening on http://{args.host}:{args.port} "
        f"(root={handler_cls.root})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
