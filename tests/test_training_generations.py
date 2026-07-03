"""Tests for generation logging helpers."""

from __future__ import annotations

from retrain.training.generations import generation_log_indices, top_surprisal_payload
from retrain.training.rollouts import TokenTextLookup


class _Tokenizer:
    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"tok{token_id}" for token_id in ids]


def test_generation_log_indices_prefers_high_reward_deterministically() -> None:
    assert generation_log_indices(
        4,
        samples_per_prompt=2,
        rewards=[0.1, 0.9, 0.9, 0.2],
    ) == [1, 2]


def test_generation_log_indices_falls_back_to_prefix_without_rewards() -> None:
    assert generation_log_indices(4, samples_per_prompt=2) == [0, 1]
    assert generation_log_indices(3, samples_per_prompt=0) == [0, 1, 2]
    assert generation_log_indices(0, samples_per_prompt=2) == []


def test_top_surprisal_payload_is_optional_and_uses_lookup() -> None:
    lookup = TokenTextLookup(_Tokenizer())

    assert top_surprisal_payload([-0.2], [10], lookup, limit=0) == []
    assert top_surprisal_payload([], [10], lookup, limit=2) == []
    assert top_surprisal_payload([-0.2, -1.4, -0.7], [10, 11, 12], lookup, limit=2) == [
        {"pos": 1, "surprisal": 1.4, "token": "tok11"},
        {"pos": 2, "surprisal": 0.7, "token": "tok12"},
    ]
