import pytest
import torch

from retrain.backends.local import loss as local_loss


def test_normalize_compile_mode_aliases() -> None:
    assert local_loss.normalize_compile_mode(True, option_name="x") == "auto"
    assert local_loss.normalize_compile_mode(False, option_name="x") == "off"
    assert local_loss.normalize_compile_mode("required", option_name="x") == "require"
    assert local_loss.normalize_compile_mode("disabled", option_name="x") == "off"
    with pytest.raises(ValueError, match="x"):
        local_loss.normalize_compile_mode("bad", option_name="x")


def test_normalize_unsloth_fused_ce_mode_aliases() -> None:
    assert local_loss.normalize_unsloth_fused_ce_mode(True) == "require"
    assert local_loss.normalize_unsloth_fused_ce_mode(False) == "off"
    assert local_loss.normalize_unsloth_fused_ce_mode("auto") == "auto"
    assert local_loss.normalize_unsloth_fused_ce_mode("on") == "require"
    with pytest.raises(ValueError, match="train_unsloth_fused_ce"):
        local_loss.normalize_unsloth_fused_ce_mode("bad")


def test_constant_positive_weight_only_accepts_constant_selected_values() -> None:
    weights = torch.tensor([[0.0, 0.5, 0.5, 0.0]])
    target_mask = torch.tensor([[False, True, True, False]])

    assert local_loss.constant_positive_weight(weights, target_mask).item() == 0.5

    weights[:, 2] = 1.0
    assert local_loss.constant_positive_weight(weights, target_mask) is None
