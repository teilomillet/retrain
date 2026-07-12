"""Timing extraction for verifiers environment rollouts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

from retrain.environments.coerce import field as _object_field


def collect_observation_timing(
    state: dict[str, object],
    totals: dict[str, float] | None,
) -> None:
    if totals is None:
        return
    trajectory = state.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        return
    step = trajectory[-1]
    extras = _object_field(step, "extras")
    candidates: list[tuple[object, bool]] = []
    if isinstance(extras, Mapping):
        extras_map = cast(Mapping[str, object], extras)
        candidates.extend(
            [
                (extras_map.get("openenv_info"), False),
                (extras_map.get("info"), False),
                (extras_map, False),
            ]
        )
    candidates.append((_object_field(step, "timing"), True))

    for candidate, direct_timing in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_map = cast(Mapping[str, object], candidate)
        timing = candidate_map.get("timing")
        if isinstance(timing, Mapping):
            _accumulate_numeric_timing(cast(Mapping[object, object], timing), totals)
        elif direct_timing:
            _accumulate_numeric_timing(
                cast(Mapping[object, object], candidate_map),
                totals,
            )


def _accumulate_numeric_timing(
    timing: Mapping[object, object],
    totals: dict[str, float],
) -> None:
    for raw_key, raw_value in timing.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            continue
        if not math.isfinite(raw_value):
            continue
        key = str(raw_key)
        totals[key] = totals.get(key, 0.0) + float(raw_value)
