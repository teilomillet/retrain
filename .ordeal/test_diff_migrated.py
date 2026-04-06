"""Auto-generated ordeal test for retrain.diff.

fuzz() tests crash safety only — it does NOT verify correctness.
Property tests assert mined invariants (confirmed by sampling).
"""

from ordeal.auto import fuzz
import retrain.diff
from ordeal.quickcheck import quickcheck
import math


def test_diff_conditions_no_crash():
    """Crash safety: retrain.diff.diff_conditions does not raise."""
    result = fuzz(retrain.diff.diff_conditions, max_examples=20)
    assert result.passed, result.summary()

def test_diff_runs_no_crash():
    """Crash safety: retrain.diff.diff_runs does not raise."""
    result = fuzz(retrain.diff.diff_runs, max_examples=20)
    assert result.passed, result.summary()

def test_format_diff_no_crash():
    """Crash safety: retrain.diff.format_diff does not raise."""
    result = fuzz(retrain.diff.format_diff, max_examples=20)
    assert result.passed, result.summary()

def test_format_time_no_crash():
    """Crash safety: retrain.diff.format_time does not raise."""
    result = fuzz(retrain.diff.format_time, max_examples=20)
    assert result.passed, result.summary()

def test_load_metrics_no_crash():
    """Crash safety: retrain.diff.load_metrics does not raise."""
    result = fuzz(retrain.diff.load_metrics, max_examples=20)
    assert result.passed, result.summary()

@quickcheck
def test_format_diff_properties(result: DiffResult):
    """Mined properties for retrain.diff.format_diff."""
    result = retrain.diff.format_diff(result)
    assert type(result).__name__ == 'str'  # >=83.9% CI
    assert result is not None  # >=83.9% CI
    assert not (isinstance(result, float) and math.isnan(result))  # >=83.9% CI
    assert len(result) > 0  # >=83.9% CI
    assert retrain.diff.format_diff(result) == result  # >=83.9% CI

@quickcheck
def test_format_time_properties(seconds: float):
    """Mined properties for retrain.diff.format_time."""
    result = retrain.diff.format_time(seconds)
    assert type(result).__name__ == 'str'  # >=83.9% CI
    assert result is not None  # >=83.9% CI
    assert not (isinstance(result, float) and math.isnan(result))  # >=83.9% CI
    assert len(result) > 0  # >=83.9% CI
    assert retrain.diff.format_time(seconds) == result  # >=83.9% CI
