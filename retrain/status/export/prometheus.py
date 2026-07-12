"""Prometheus text rendering for exported run snapshots."""

from __future__ import annotations

import math
from collections.abc import Callable

from retrain.status.export.types import RunSnapshot


MetricGetter = Callable[[RunSnapshot], float | int | None]


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
            "train_time_semantics": (
                snapshot.latest_train_time_semantics or "unspecified"
            ),
            "path": snapshot.path,
        }
        lines.append(_metric_line("soma_retrain_run_info", 1, labels))

    metric_map: list[tuple[str, str, str, MetricGetter]] = [
        (
            "soma_retrain_run_active",
            "Whether the run appears active.",
            "gauge",
            lambda s: 1 if s.active else 0,
        ),
        (
            "soma_retrain_run_completed",
            "Whether the run finished with a final checkpoint.",
            "gauge",
            lambda s: 1 if s.completed else 0,
        ),
        (
            "soma_retrain_metrics_present",
            "Whether metrics.jsonl exists for the run.",
            "gauge",
            lambda s: 1 if s.metrics_present else 0,
        ),
        (
            "soma_retrain_latest_step",
            "Latest completed training step.",
            "gauge",
            lambda s: s.latest_step,
        ),
        (
            "soma_retrain_latest_mean_reward",
            "Latest mean reward from metrics.jsonl.",
            "gauge",
            lambda s: s.latest_mean_reward,
        ),
        (
            "soma_retrain_latest_loss",
            "Latest loss from metrics.jsonl.",
            "gauge",
            lambda s: s.latest_loss,
        ),
        (
            "soma_retrain_latest_max_token_hit_rate",
            "Latest max-token hit rate from metrics.jsonl.",
            "gauge",
            lambda s: s.latest_max_token_hit_rate,
        ),
        (
            "soma_retrain_latest_invalid_action_rate",
            "Latest invalid action rate from metrics.jsonl.",
            "gauge",
            lambda s: s.latest_invalid_action_rate,
        ),
        (
            "soma_retrain_latest_avg_response_chars",
            "Latest average response length from metrics.jsonl.",
            "gauge",
            lambda s: s.latest_avg_response_chars,
        ),
        (
            "soma_retrain_latest_step_time_seconds",
            "Latest full training-step duration.",
            "gauge",
            lambda s: s.latest_step_time_s,
        ),
        (
            "soma_retrain_latest_tokens_per_second",
            "Latest tokens-per-second throughput from metrics.jsonl.",
            "gauge",
            lambda s: s.latest_tokens_per_second,
        ),
        (
            "soma_retrain_latest_sample_share",
            "Latest sampling share of step time.",
            "gauge",
            lambda s: s.latest_sample_share,
        ),
        (
            "soma_retrain_latest_train_share",
            "Latest synchronous training share of step time.",
            "gauge",
            lambda s: s.latest_train_share,
        ),
        (
            "soma_retrain_latest_train_submit_enqueue_time_seconds",
            "Latest PRIME-RL submit/enqueue latency.",
            "gauge",
            lambda s: s.latest_train_submit_enqueue_time_s,
        ),
        (
            "soma_retrain_latest_train_submit_enqueue_share",
            "Latest PRIME-RL submit/enqueue share of step time.",
            "gauge",
            lambda s: s.latest_train_submit_enqueue_share,
        ),
        (
            "soma_retrain_latest_process_max_rss_megabytes",
            "Latest reported process peak RSS.",
            "gauge",
            lambda s: s.latest_process_max_rss_mb,
        ),
        (
            "soma_retrain_metrics_age_seconds",
            "Seconds since metrics.jsonl was updated.",
            "gauge",
            lambda s: s.metrics_age_seconds,
        ),
        (
            "soma_retrain_sample_event_age_seconds",
            "Seconds since any sample diagnostic event was recorded.",
            "gauge",
            lambda s: s.sample_event_age_seconds,
        ),
        (
            "soma_retrain_sample_result_age_seconds",
            "Seconds since the latest sample result was recorded.",
            "gauge",
            lambda s: s.sample_result_age_seconds,
        ),
        (
            "soma_retrain_recent_dispatch_count",
            "Recent dispatch events seen in diagnostics.",
            "gauge",
            lambda s: s.recent_dispatch_count,
        ),
        (
            "soma_retrain_recent_result_count",
            "Recent result events seen in diagnostics.",
            "gauge",
            lambda s: s.recent_result_count,
        ),
        (
            "soma_retrain_recent_error_count",
            "Recent sample errors seen in diagnostics.",
            "gauge",
            lambda s: s.recent_error_count,
        ),
        (
            "soma_retrain_recent_timeout_count",
            "Recent TimeoutError count seen in diagnostics.",
            "gauge",
            lambda s: s.recent_timeout_count,
        ),
        (
            "soma_retrain_recent_prompt_tokens_last",
            "Latest prompt token count seen in diagnostics.",
            "gauge",
            lambda s: s.recent_prompt_tokens_last,
        ),
        (
            "soma_retrain_recent_prompt_tokens_max",
            "Max recent prompt token count seen in diagnostics.",
            "gauge",
            lambda s: s.recent_prompt_tokens_max,
        ),
        (
            "soma_retrain_recent_completion_tokens_last",
            "Latest completion token count seen in diagnostics.",
            "gauge",
            lambda s: s.recent_completion_tokens_last,
        ),
        (
            "soma_retrain_recent_completion_tokens_max",
            "Max recent completion token count seen in diagnostics.",
            "gauge",
            lambda s: s.recent_completion_tokens_max,
        ),
        (
            "soma_retrain_recent_result_latency_seconds_last",
            "Latest sample result latency from diagnostics.",
            "gauge",
            lambda s: s.recent_result_latency_last_s,
        ),
        (
            "soma_retrain_recent_result_latency_seconds_max",
            "Max sample result latency in recent diagnostics.",
            "gauge",
            lambda s: s.recent_result_latency_max_s,
        ),
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
