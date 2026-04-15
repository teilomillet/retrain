"""Tests for retrain.runtime_support helper abstractions."""

from __future__ import annotations

from retrain.runtime_support import (
    DecodedSequence,
    ExamplePromptCache,
    RuntimeCounters,
    TokenTextLookup,
    decode_sequence_groups,
    top_surprisal_entries,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.convert_calls = 0

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        self.convert_calls += 1
        return [f"tok-{token_id}" for token_id in ids]

    def batch_decode(
        self,
        token_ids: list[list[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        _ = skip_special_tokens
        return [" ".join(str(token_id) for token_id in seq) for seq in token_ids]


class _FakeDetector:
    def detect(self, token_strs: list[str]) -> list[int]:
        return [1 if token.endswith("7") else 0 for token in token_strs]


def test_token_text_lookup_batches_cache_misses() -> None:
    tokenizer = _FakeTokenizer()
    counters = RuntimeCounters()
    lookup = TokenTextLookup(tokenizer, counters=counters)

    first = lookup.get_many([1, 2, 1])
    second = lookup.get_many([2, 3])

    assert first == ["tok-1", "tok-2", "tok-1"]
    assert second == ["tok-2", "tok-3"]
    assert tokenizer.convert_calls == 2
    assert counters.token_lookup_requests == 5
    assert counters.token_lookup_convert_calls == 2
    assert counters.token_lookup_cache_misses == 3


def test_example_prompt_cache_uses_supplied_callables() -> None:
    calls: list[tuple[object, object]] = []
    previews: list[object] = []
    counters = RuntimeCounters()
    cache = ExamplePromptCache(
        tokenizer=object(),
        prompts=["alpha"],
        encoder=lambda tokenizer, prompt: calls.append((tokenizer, prompt)) or [1, 2, 3],
        preview_renderer=lambda prompt: previews.append(prompt) or "preview",
        counters=counters,
    )

    assert cache.prompt_ids(0) == [1, 2, 3]
    assert cache.prompt_ids(0) == [1, 2, 3]
    assert cache.preview(0) == "preview"
    assert cache.preview(0) == "preview"
    assert len(calls) == 1
    assert len(previews) == 1
    assert counters.prompt_encode_calls == 1
    assert counters.prompt_preview_calls == 1


def test_decode_sequence_groups_and_top_surprisal_entries() -> None:
    tokenizer = _FakeTokenizer()
    detector = _FakeDetector()
    counters = RuntimeCounters()
    lookup = TokenTextLookup(tokenizer, counters=counters)
    groups = decode_sequence_groups(
        tokenizer,
        [[([7, 8], [-0.1, -0.9]), ([9], [-0.2])]],
        needs_planning=True,
        token_lookup=lookup,
        detector=detector,
        counters=counters,
    )

    assert groups == [
        [
            DecodedSequence(
                token_ids=[7, 8],
                logprobs=[-0.1, -0.9],
                text="7 8",
                planning_mask=[1, 0],
            ),
            DecodedSequence(
                token_ids=[9],
                logprobs=[-0.2],
                text="9",
                planning_mask=[0],
            ),
        ]
    ]

    top = top_surprisal_entries([-0.1, -1.2, -0.4], [10, 11, 12], lookup, limit=2)
    assert top == [
        {"pos": 1, "surprisal": 1.2, "token": "tok-11"},
        {"pos": 2, "surprisal": 0.4, "token": "tok-12"},
    ]
    assert counters.batch_decode_calls == 1
    assert counters.batch_decoded_sequences == 2
