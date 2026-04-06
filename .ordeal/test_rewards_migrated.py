"""Auto-generated ordeal test for retrain.rewards.

fuzz() tests crash safety only — it does NOT verify correctness.
Property tests assert mined invariants (confirmed by sampling).
"""

from ordeal.auto import fuzz
import retrain.rewards
from ordeal.quickcheck import quickcheck
import math


def test_extract_boxed_no_crash():
    """Crash safety: retrain.rewards.extract_boxed does not raise."""
    result = fuzz(retrain.rewards.extract_boxed, max_examples=20)
    assert result.passed, result.summary()

@quickcheck
def test_extract_boxed_properties(text: str):
    """Mined properties for retrain.rewards.extract_boxed."""
    result = retrain.rewards.extract_boxed(text)
    assert type(result).__name__ == 'str'  # >=83.9% CI
    assert result is not None  # >=83.9% CI
    assert not (isinstance(result, float) and math.isnan(result))  # >=83.9% CI
    assert retrain.rewards.extract_boxed(text) == result  # >=83.9% CI
    assert retrain.rewards.extract_boxed(result) == result  # >=83.9% CI
    assert len(result) <= len(text)  # >=83.9% CI
