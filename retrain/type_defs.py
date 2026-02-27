"""Shared type aliases used across retrain modules."""

from __future__ import annotations

from typing import TypeAlias

# JSON-like values used for config/state/payload annotations.
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]

# Prompt/input payloads used by dataset + verifiers bridge.
PromptMessage: TypeAlias = dict[str, object]
PromptLike: TypeAlias = str | list[PromptMessage]
ExampleInfoLike: TypeAlias = dict[str, JSONValue] | str | None

# Sampling output shared by helper protocols and trainer loops.
SampleGroup: TypeAlias = list[tuple[list[int], list[float]]]
SampleBatch: TypeAlias = list[SampleGroup]

# Enriched sampling output with optional per-token entropy.
# Each tuple: (token_ids, logprobs, token_entropies | None)
EnrichedSampleGroup: TypeAlias = list[tuple[list[int], list[float], list[float] | None]]
EnrichedSampleBatch: TypeAlias = list[EnrichedSampleGroup]
