"""Single-turn scoring against verifiers rubrics."""

from __future__ import annotations

import asyncio
import types
from collections.abc import Callable, Mapping
from typing import cast

from retrain.environments import load as env_load
from retrain.environments.verifier import coerce
from retrain.environments.verifier.types import SingleTurnEnvironment, StateDict
from retrain.types import ExampleInfoLike, PromptLike


def completion_messages(
    env: SingleTurnEnvironment, completion_text: str
) -> list[dict[str, str]] | str:
    if getattr(env, "message_type", "chat") == "chat":
        return [{"role": "assistant", "content": completion_text}]
    return completion_text


def messages_to_text(messages: object) -> str:
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return str(messages)
    chunks: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            chunks.append(str(msg))
            continue
        msg_data = cast(Mapping[str, object], msg)
        content = msg_data.get("content")
        if content:
            chunks.append(str(content))
    return "\n".join(chunks)


def score_singleturn_group(
    env: object,
    *,
    prompt: PromptLike,
    answer: str,
    task: str,
    info: ExampleInfoLike,
    completion_texts: list[str],
    require_fn: Callable[[], types.ModuleType] = env_load.require_verifiers,
) -> list[float]:
    """Score a group of single-turn completions with the environment rubric."""
    vf = require_fn()
    env_typed = cast(SingleTurnEnvironment, env)

    states: list[StateDict] = []
    for i, text in enumerate(completion_texts):
        input_payload: dict[str, object] = {
            "prompt": prompt,
            "answer": answer,
            "task": task,
            "example_id": i,
        }
        if info is not None:
            input_payload["info"] = info

        state = cast(StateDict, vf.State(input=input_payload))
        state["completion"] = completion_messages(env_typed, text)
        state["trajectory"] = []
        state["reward"] = None
        state["advantage"] = None
        state["metrics"] = {}
        state["error"] = None
        state["is_completed"] = True
        state["is_truncated"] = False
        state["timing"] = {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}
        states.append(state)

    asyncio.run(env_typed.rubric.score_group(states))
    return [coerce.reward(s.get("reward")) for s in states]
