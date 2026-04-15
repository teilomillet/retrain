"""Shared helpers for scanning retrain metrics JSONL files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


JsonObject = dict[str, object]


def float_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def int_or_none(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def iter_jsonl_objects(path: Path) -> Iterator[JsonObject]:
    """Yield JSON-object rows from a JSONL file, skipping blank/bad lines."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


@dataclass
class MetricsScanResult:
    """Single-pass summary of a metrics.jsonl file."""

    rows: int = 0
    last: JsonObject | None = None
    wall_time_s: float = 0.0
    step_time_count: int = 0
    step_times: list[float] = field(default_factory=list)
    correct_rates: list[float] = field(default_factory=list)
    mean_sums: dict[str, float] = field(default_factory=dict)
    mean_counts: dict[str, int] = field(default_factory=dict)
    maxima: dict[str, float] = field(default_factory=dict)

    def mean(self, key: str) -> float | None:
        count = self.mean_counts.get(key, 0)
        if count <= 0:
            return None
        return self.mean_sums[key] / count

    def maximum(self, key: str) -> float | None:
        return self.maxima.get(key)


def scan_metrics_file(
    path: Path,
    *,
    mean_fields: tuple[str, ...] = (),
    max_fields: tuple[str, ...] = (),
    collect_step_times: bool = False,
    collect_correct_rates: bool = False,
) -> MetricsScanResult:
    """Scan a metrics JSONL file once and collect selected aggregates."""
    result = MetricsScanResult()

    for row in iter_jsonl_objects(path):
        result.rows += 1
        result.last = row

        step_time = float_or_none(row.get("step_time_s"))
        if step_time is None:
            step_time = float_or_none(row.get("time_s"))
        if step_time is not None:
            result.wall_time_s += step_time
            result.step_time_count += 1
            if collect_step_times:
                result.step_times.append(step_time)

        if collect_correct_rates:
            correct_rate = float_or_none(row.get("correct_rate"))
            if correct_rate is not None:
                result.correct_rates.append(correct_rate)

        for key in mean_fields:
            value = float_or_none(row.get(key))
            if value is None:
                continue
            result.mean_sums[key] = result.mean_sums.get(key, 0.0) + value
            result.mean_counts[key] = result.mean_counts.get(key, 0) + 1

        for key in max_fields:
            value = float_or_none(row.get(key))
            if value is None:
                continue
            current = result.maxima.get(key)
            if current is None or value > current:
                result.maxima[key] = value

    return result
