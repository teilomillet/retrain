"""Typed boundary for the optional Unsloth runtime."""

from __future__ import annotations

import importlib
from typing import Protocol, cast


class FastLanguageModel(Protocol):
    def from_pretrained(self, **kwargs: object) -> tuple[object, object]: ...
    def get_peft_model(self, model: object, **kwargs: object) -> object: ...
    def for_inference(self, model: object) -> None: ...
    def for_training(self, model: object) -> None: ...


def load_fast_language_model() -> FastLanguageModel:
    unsloth = importlib.import_module("unsloth")
    try:
        fast_language_model = getattr(unsloth, "FastLanguageModel")
    except AttributeError as exc:
        raise ImportError("Unsloth module does not expose FastLanguageModel.") from exc
    return cast(FastLanguageModel, fast_language_model)
