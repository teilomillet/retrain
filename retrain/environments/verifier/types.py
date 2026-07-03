"""Structural types for environment rollouts."""

from __future__ import annotations

from typing import Protocol

from retrain.types import PromptLike


StateDict = dict[str, object]


class Rubric(Protocol):
    async def score_group(self, states: list[StateDict]) -> object: ...


class SingleTurnEnvironment(Protocol):
    message_type: str
    rubric: Rubric


class MultiTurnEnvironment(Protocol):
    message_type: str
    rubric: Rubric

    async def init_state(
        self,
        *,
        input: dict[str, object],
        client: object,
        model: str,
        sampling_args: object,
    ) -> StateDict: ...

    async def setup_state(self, state: StateDict) -> StateDict: ...
    async def is_completed(self, state: StateDict) -> bool: ...
    async def get_prompt_messages(self, state: StateDict) -> PromptLike: ...
    async def add_trajectory_step(self, state: StateDict, step: object) -> object: ...
    async def render_completion(self, state: StateDict) -> object: ...
    async def cleanup(self, state: StateDict) -> object: ...


class Tokenizer(Protocol):
    def encode(self, text: str) -> object: ...
    def batch_decode(
        self, token_ids: list[list[int]], *, skip_special_tokens: bool = True
    ) -> list[str]: ...
