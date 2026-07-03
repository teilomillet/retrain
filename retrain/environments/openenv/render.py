"""Render OpenEnv observations into chat messages.

A renderer is any callable ``(observation, **context) -> messages``. The
context kwargs match verifiers' OpenEnv integration (``context``,
``action_schema``, ``contract``, ``seed``) and are filtered against the
callable's signature, so renderers written for that integration — e.g.
quaero's ``render_prompt`` — work here unchanged.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from typing import cast

from retrain.plugins.resolve import resolve_dotted_attribute
from retrain.types import JSONObject, PromptMessage

Renderer = Callable[..., object]


def default_renderer(
    observation: object,
    *,
    context: str = "step",
    action_schema: Mapping[str, object] | None = None,
    **_: object,
) -> list[PromptMessage]:
    """JSON-dump the observation; on reset, lead with the action contract.

    Adequate for smoke tests and simple text envs. Real training runs should
    pass a task-specific renderer via ``[environment] args.renderer`` — prompt
    shape is part of the harness contract the model is trained into.
    """
    parts: list[str] = []
    if context == "reset" and action_schema:
        parts.append(
            "Respond with exactly one JSON object matching this action "
            "schema:\n" + json.dumps(action_schema, ensure_ascii=True)
        )
    if isinstance(observation, str):
        parts.append(observation)
    else:
        parts.append(json.dumps(observation, ensure_ascii=True, default=str))
    return [{"role": "user", "content": "\n\n".join(parts)}]


def resolve_renderer(target: str) -> Renderer:
    """Load a renderer callable from a dotted path."""
    resolution = resolve_dotted_attribute(
        target,
        selector="openenv renderer",
        expected="callable(observation, **context) -> chat messages",
    )
    renderer = resolution.obj
    if not callable(renderer):
        raise TypeError(
            f"OpenEnv renderer '{target}' resolved to a non-callable "
            f"{type(renderer).__name__}."
        )
    return cast(Renderer, renderer)


def render_messages(
    renderer: Renderer,
    observation: object,
    *,
    context: str,
    action_schema: Mapping[str, object] | None,
    seed: int | None,
) -> list[PromptMessage]:
    """Invoke *renderer* with only the context kwargs it accepts."""
    observation = _normalize_observation(observation)
    kwargs: dict[str, object] = {
        "context": context,
        "action_schema": action_schema,
        "contract": "gym",
        "seed": seed,
    }
    try:
        signature = inspect.signature(renderer)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if not accepts_var_kwargs:
            kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
    rendered = renderer(observation, **kwargs)
    return _coerce_messages(rendered)


def _normalize_observation(observation: object) -> object:
    """Unwrap typed (pydantic) observations to plain data."""
    model_dump = getattr(observation, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:
            return observation
    return observation


def _coerce_messages(rendered: object) -> list[PromptMessage]:
    if isinstance(rendered, str):
        return [{"role": "user", "content": rendered}]
    if not isinstance(rendered, list) or not rendered:
        raise ValueError(
            "OpenEnv renderer must return a non-empty chat messages list "
            f"or string, got {type(rendered).__name__}."
        )
    messages: list[PromptMessage] = []
    for message in cast(list[object], rendered):
        if not isinstance(message, dict):
            raise ValueError(
                "OpenEnv renderer messages must be dicts with role/content, "
                f"got {type(message).__name__}."
            )
        message_obj = cast(dict[str, object], message)
        content = message_obj.get("content")
        if content is None:
            raise ValueError("OpenEnv renderer message content cannot be null.")
        messages.append(
            {
                "role": str(message_obj.get("role", "user")),
                "content": str(content),
            }
        )
    return messages
