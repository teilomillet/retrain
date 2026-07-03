"""Tests for shared token id parsing."""

from __future__ import annotations

from types import SimpleNamespace

from retrain.models.gemma4 import eos_token_ids
from retrain.tokens.ids import coerce_token_id_set, model_eos_token_ids


def test_coerce_token_id_set_handles_none_scalar_and_iterables() -> None:
    assert coerce_token_id_set(None) == set()
    assert coerce_token_id_set(3) == {3}
    assert coerce_token_id_set([1, None, "2"]) == {1, 2}
    assert coerce_token_id_set(("4", 5)) == {4, 5}


def test_model_eos_token_ids_prefers_generation_config() -> None:
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[7, 8]),
        config=SimpleNamespace(eos_token_id=9),
    )

    assert model_eos_token_ids(model) == {7, 8}


def test_model_eos_token_ids_falls_back_when_generation_ids_are_empty() -> None:
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[]),
        config=SimpleNamespace(eos_token_id=9),
    )

    assert model_eos_token_ids(model) == {9}


def test_model_eos_token_ids_uses_unwrapped_config_for_peft_models() -> None:
    base = SimpleNamespace(config=SimpleNamespace(eos_token_id=11))
    wrapped = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=None),
        config=SimpleNamespace(eos_token_id=None),
        base_model=SimpleNamespace(model=base),
    )

    assert eos_token_ids(wrapped) == {11}
