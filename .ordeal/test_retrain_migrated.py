"""Auto-generated ordeal test for retrain.

fuzz() tests crash safety only — it does NOT verify correctness.
Property tests assert mined invariants (confirmed by sampling).
"""

from ordeal.auto import fuzz
import retrain
from ordeal.quickcheck import quickcheck
import math


def test_build_flow_no_crash():
    """Crash safety: retrain.build_flow does not raise."""
    result = fuzz(retrain.build_flow, max_examples=20)
    assert result.passed, result.summary()

def test_entropy_mask_post_process_no_crash():
    """Crash safety: retrain.entropy_mask_post_process does not raise."""
    result = fuzz(retrain.entropy_mask_post_process, max_examples=20)
    assert result.passed, result.summary()

def test_register_advantage_mode_no_crash():
    """Crash safety: retrain.register_advantage_mode does not raise."""
    result = fuzz(retrain.register_advantage_mode, max_examples=20)
    assert result.passed, result.summary()

def test_register_algorithm_mode_no_crash():
    """Crash safety: retrain.register_algorithm_mode does not raise."""
    result = fuzz(retrain.register_algorithm_mode, max_examples=20)
    assert result.passed, result.summary()

def test_register_transform_mode_no_crash():
    """Crash safety: retrain.register_transform_mode does not raise."""
    result = fuzz(retrain.register_transform_mode, max_examples=20)
    assert result.passed, result.summary()

def test_register_uncertainty_kind_no_crash():
    """Crash safety: retrain.register_uncertainty_kind does not raise."""
    result = fuzz(retrain.register_uncertainty_kind, max_examples=20)
    assert result.passed, result.summary()

def test_surprisal_mask_post_process_no_crash():
    """Crash safety: retrain.surprisal_mask_post_process does not raise."""
    result = fuzz(retrain.surprisal_mask_post_process, max_examples=20)
    assert result.passed, result.summary()

@quickcheck
def test_entropy_mask_post_process_properties(all_token_advs: list[list[float]], all_raw_surprisals: list[list[float]], params: Mapping[str, object]):
    """Mined properties for retrain.entropy_mask_post_process."""
    result = retrain.entropy_mask_post_process(all_token_advs, all_raw_surprisals, params)
    assert type(result).__name__ == 'tuple'  # >=83.9% CI
    assert result is not None  # >=83.9% CI
    assert not (isinstance(result, float) and math.isnan(result))  # >=83.9% CI
    assert len(result) > 0  # >=83.9% CI
    assert retrain.entropy_mask_post_process(all_token_advs, all_raw_surprisals, params) == result  # >=83.9% CI

@quickcheck
def test_register_advantage_mode_properties(name: str, compute: Union[Callable[[list[float]], list[float]], Callable[[list[float], Mapping[str, object]], list[float]]]):
    """Mined properties for retrain.register_advantage_mode."""
    result = retrain.register_advantage_mode(name, compute)
    assert type(result).__name__ == 'NoneType'  # >=83.9% CI
    assert not (isinstance(result, float) and math.isnan(result))  # >=83.9% CI
    assert retrain.register_advantage_mode(name, compute) == result  # >=83.9% CI

def test_register_algorithm_mode_properties():
    """Mined properties for retrain.register_algorithm_mode."""
    # output type is NoneType: 20/20 (>=83.9% at 95% CI)
    # no NaN: 20/20 (>=83.9% at 95% CI)
    # deterministic: 20/20 (>=83.9% at 95% CI)
    result = fuzz(retrain.register_algorithm_mode, max_examples=20)
    assert result.passed

def test_register_transform_mode_properties():
    """Mined properties for retrain.register_transform_mode."""
    # output type is NoneType: 20/20 (>=83.9% at 95% CI)
    # no NaN: 20/20 (>=83.9% at 95% CI)
    # deterministic: 20/20 (>=83.9% at 95% CI)
    result = fuzz(retrain.register_transform_mode, max_examples=20)
    assert result.passed

def test_register_uncertainty_kind_properties():
    """Mined properties for retrain.register_uncertainty_kind."""
    # output type is NoneType: 20/20 (>=83.9% at 95% CI)
    # no NaN: 20/20 (>=83.9% at 95% CI)
    # deterministic: 20/20 (>=83.9% at 95% CI)
    result = fuzz(retrain.register_uncertainty_kind, max_examples=20)
    assert result.passed

@quickcheck
def test_surprisal_mask_post_process_properties(all_token_advs: list[list[float]], all_raw_surprisals: list[list[float]], params: Mapping[str, object]):
    """Mined properties for retrain.surprisal_mask_post_process."""
    result = retrain.surprisal_mask_post_process(all_token_advs, all_raw_surprisals, params)
    assert type(result).__name__ == 'tuple'  # >=83.9% CI
    assert result is not None  # >=83.9% CI
    assert not (isinstance(result, float) and math.isnan(result))  # >=83.9% CI
    assert len(result) > 0  # >=83.9% CI
    assert retrain.surprisal_mask_post_process(all_token_advs, all_raw_surprisals, params) == result  # >=83.9% CI
