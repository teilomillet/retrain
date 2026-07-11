"""Snapshot types for live run exports."""

from __future__ import annotations

from dataclasses import dataclass, fields

from retrain.metrics.scan import JsonObject


@dataclass
class RunSnapshot:
    run: str
    path: str
    trainer: str
    phase: str
    metrics_present: bool
    completed: bool
    active: bool
    resume_mode: str
    resume_warning: str
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
    latest_train_time_semantics: str
    latest_train_submit_enqueue_time_s: float | None
    latest_train_submit_enqueue_share: float | None
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

    def to_dict(self) -> JsonObject:
        """Serialize the flat snapshot without ``asdict`` deep-copy overhead."""
        return {name: getattr(self, name) for name in _RUN_SNAPSHOT_FIELD_NAMES}


_RUN_SNAPSHOT_FIELD_NAMES = tuple(field.name for field in fields(RunSnapshot))
