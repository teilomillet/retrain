"""Training example shape and data-source protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from retrain.type_defs import ExampleInfoLike, PromptLike


@dataclass
class Example:
    """A single training example: prompt + reference answer."""
    prompt: PromptLike
    reference: str
    task: str = "default"
    info: ExampleInfoLike = None
    example_id: int | str = -1


@runtime_checkable
class DataSource(Protocol):
    """Any object that can load training examples."""

    def load(self) -> list[Example]: ...
