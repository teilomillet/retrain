"""Prompt batch selection for training steps."""

from __future__ import annotations

from dataclasses import dataclass, field

from retrain.data.source import Example
from retrain.training.rollouts import ExamplePromptCache
from retrain.types import ExampleInfoLike, PromptLike


@dataclass
class PromptBatch:
    """Parallel per-prompt arrays for one training batch."""

    objs: list[PromptLike] = field(default_factory=list)
    previews: list[str] = field(default_factory=list)
    ids: list[list[int]] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    infos: list[ExampleInfoLike] = field(default_factory=list)


def select_prompt_batch(
    examples: list[Example],
    prompt_cache: ExamplePromptCache,
    start_index: int,
    batch_size: int,
) -> tuple[PromptBatch, int]:
    """Take ``batch_size`` examples round-robin and return the new cursor."""
    prompts = PromptBatch()
    example_idx = start_index
    for _ in range(batch_size):
        ex_idx = example_idx % len(examples)
        example_idx += 1
        ex = examples[ex_idx]
        prompts.objs.append(ex.prompt)
        prompts.previews.append(prompt_cache.preview(ex_idx))
        prompts.ids.append(list(prompt_cache.prompt_ids(ex_idx)))
        prompts.answers.append(ex.reference)
        prompts.tasks.append(ex.task)
        prompts.infos.append(ex.info)
    return prompts, example_idx
