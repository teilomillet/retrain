"""Tests for training prompt batch selection."""

from __future__ import annotations

from retrain.data.source import Example
from retrain.training.prompts import PromptBatch, select_prompt_batch
from retrain.training.rollouts import ExamplePromptCache, RuntimeCounters


def test_select_prompt_batch_wraps_round_robin_and_uses_cache() -> None:
    examples = [
        Example(prompt="alpha", reference="A", task="t0", info={"i": 0}),
        Example(prompt="beta", reference="B", task="t1", info={"i": 1}),
        Example(prompt="gamma", reference="C", task="t2", info={"i": 2}),
    ]
    counters = RuntimeCounters()
    cache = ExamplePromptCache(
        tokenizer=object(),
        prompts=[ex.prompt for ex in examples],
        encoder=lambda _tokenizer, prompt: [ord(str(prompt)[0])],
        preview_renderer=lambda prompt: f"preview:{prompt}",
        counters=counters,
    )

    batch, cursor = select_prompt_batch(
        examples,
        cache,
        start_index=2,
        batch_size=4,
    )

    assert isinstance(batch, PromptBatch)
    assert cursor == 6
    assert batch.objs == ["gamma", "alpha", "beta", "gamma"]
    assert batch.previews == [
        "preview:gamma",
        "preview:alpha",
        "preview:beta",
        "preview:gamma",
    ]
    assert batch.ids == [[ord("g")], [ord("a")], [ord("b")], [ord("g")]]
    assert batch.answers == ["C", "A", "B", "C"]
    assert batch.tasks == ["t2", "t0", "t1", "t2"]
    assert batch.infos == [{"i": 2}, {"i": 0}, {"i": 1}, {"i": 2}]
    assert counters.prompt_encode_calls == 3
    assert counters.prompt_preview_calls == 3
