"""Runtime helpers for prompt/token caching and decoded rollout batches."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from retrain.type_defs import PromptLike
from retrain.verifiers_bridge import encode_prompt_for_sampling, prompt_preview


class _TokenizerWithIds(Protocol):
    def convert_ids_to_tokens(self, ids: list[int]) -> list[object]: ...

    def batch_decode(
        self,
        token_ids: Sequence[Sequence[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]: ...


class _PlanningDetector(Protocol):
    def detect(self, token_strs: list[str]) -> list[int]: ...


class TokenTextLookup:
    """Lazy token-id to string lookup with incremental caching."""

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer
        self._cache: dict[int, str] = {}

    def get_many(self, token_ids: Sequence[int]) -> list[str]:
        """Resolve token ids to strings, batching only the cache misses."""
        missing: list[int] = []
        seen_missing: set[int] = set()
        for token_id in token_ids:
            if token_id < 0 or token_id in self._cache or token_id in seen_missing:
                continue
            missing.append(token_id)
            seen_missing.add(token_id)

        if missing:
            tokenizer = self._tokenizer
            if not hasattr(tokenizer, "convert_ids_to_tokens"):
                raise TypeError(
                    "Tokenizer must expose convert_ids_to_tokens() for planning detection."
                )
            tokenizer_typed = tokenizer
            raw_tokens = tokenizer_typed.convert_ids_to_tokens(missing)  # type: ignore[unresolved-attribute]
            for token_id, token in zip(missing, raw_tokens):
                self._cache[token_id] = str(token) if token is not None else ""

        return [self._cache.get(token_id, "") if token_id >= 0 else "" for token_id in token_ids]

    def get(self, token_id: int) -> str:
        """Resolve a single token id to text."""
        values = self.get_many([token_id])
        return values[0] if values else ""


class ExamplePromptCache:
    """Lazy prompt encoding/preview cache for stable training datasets."""

    def __init__(
        self,
        tokenizer: object,
        prompts: Sequence[PromptLike],
        *,
        encoder: Callable[[object, PromptLike], list[int]] = encode_prompt_for_sampling,
        preview_renderer: Callable[[PromptLike], str] = prompt_preview,
    ) -> None:
        self._tokenizer = tokenizer
        self._prompts = list(prompts)
        self._encoder = encoder
        self._preview_renderer = preview_renderer
        self._prompt_ids: dict[int, list[int]] = {}
        self._previews: dict[int, str] = {}

    def prompt_ids(self, index: int) -> list[int]:
        """Return cached prompt ids for ``index``, encoding on first use."""
        if index not in self._prompt_ids:
            self._prompt_ids[index] = self._encoder(
                self._tokenizer,
                self._prompts[index],
            )
        return self._prompt_ids[index]

    def preview(self, index: int) -> str:
        """Return cached prompt preview text for ``index``."""
        if index not in self._previews:
            self._previews[index] = self._preview_renderer(self._prompts[index])
        return self._previews[index]


@dataclass
class DecodedSequence:
    """Decoded sampling result with local metadata attached once."""

    token_ids: list[int]
    logprobs: list[float]
    text: str
    planning_mask: list[int]


def decode_sequence_groups(
    tokenizer: object,
    all_group_sequences: Sequence[Sequence[tuple[list[int], list[float]]]],
    *,
    needs_planning: bool,
    token_lookup: TokenTextLookup | None = None,
    detector: object | None = None,
) -> list[list[DecodedSequence]]:
    """Decode grouped token sequences once and attach planning masks."""
    flat_token_ids: list[list[int]] = []
    group_offsets: list[int] = []

    for group in all_group_sequences:
        group_offsets.append(len(flat_token_ids))
        for token_ids, _logprobs in group:
            flat_token_ids.append(list(token_ids))

    decoded_texts: list[str]
    if flat_token_ids:
        if not hasattr(tokenizer, "batch_decode"):
            raise TypeError("Tokenizer must expose batch_decode().")
        tokenizer_typed = tokenizer
        decoded_texts = tokenizer_typed.batch_decode(  # type: ignore[unresolved-attribute]
            flat_token_ids,
            skip_special_tokens=True,
        )
    else:
        decoded_texts = []

    decoded_groups: list[list[DecodedSequence]] = []
    for group_idx, group in enumerate(all_group_sequences):
        flat_offset = group_offsets[group_idx]
        decoded_group: list[DecodedSequence] = []
        for seq_idx, (token_ids, seq_logprobs) in enumerate(group):
            token_ids_list = list(token_ids)
            logprobs = list(seq_logprobs)
            if needs_planning:
                if token_lookup is None or detector is None:
                    raise ValueError(
                        "Planning decode requires both token_lookup and detector."
                    )
                detector_typed = detector
                planning_mask = detector_typed.detect(  # type: ignore[unresolved-attribute]
                    token_lookup.get_many(token_ids_list)
                )
            else:
                planning_mask = [0] * len(logprobs)
            decoded_group.append(
                DecodedSequence(
                    token_ids=token_ids_list,
                    logprobs=logprobs,
                    text=decoded_texts[flat_offset + seq_idx],
                    planning_mask=planning_mask,
                )
            )
        decoded_groups.append(decoded_group)

    return decoded_groups


def top_surprisal_entries(
    logprobs: Sequence[float],
    token_ids: Sequence[int],
    token_lookup: TokenTextLookup,
    *,
    limit: int = 10,
) -> list[dict[str, int | float | str]]:
    """Return the top surprisal tokens without sorting the full sequence."""
    n_tokens = min(len(logprobs), len(token_ids))
    top_k = min(limit, n_tokens)
    if top_k <= 0:
        return []

    top_indices = heapq.nlargest(top_k, range(n_tokens), key=lambda idx: -logprobs[idx])
    top_token_ids = [token_ids[idx] for idx in top_indices]
    top_token_texts = token_lookup.get_many(top_token_ids)

    entries: list[dict[str, int | float | str]] = []
    for idx, token_id, token_text in zip(top_indices, top_token_ids, top_token_texts):
        entries.append(
            {
                "pos": idx,
                "surprisal": round(-logprobs[idx], 4),
                "token": token_text or f"<{token_id}>",
            }
        )
    return entries
