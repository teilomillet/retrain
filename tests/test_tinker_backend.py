"""Tests for Tinker backend support that does not need the optional SDK."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from retrain.backends.tinker import _mean_loss_from_result


@pytest.mark.parametrize(
    ("metrics", "datum_count", "expected"),
    [
        ({"loss:sum": 6.0}, 3, 2.0),
        ({"loss:sum": 6.0}, 0, 6.0),
        ({"other": 6.0}, 3, 0.0),
        ({}, 3, 0.0),
        (None, 3, 0.0),
    ],
)
def test_mean_loss_from_tinker_result(
    metrics: dict[str, float] | None,
    datum_count: int,
    expected: float,
) -> None:
    assert _mean_loss_from_result(
        SimpleNamespace(metrics=metrics),
        datum_count,
    ) == pytest.approx(expected)


def test_mean_loss_from_tinker_result_tolerates_missing_metrics() -> None:
    assert _mean_loss_from_result(SimpleNamespace(), datum_count=3) == 0.0
