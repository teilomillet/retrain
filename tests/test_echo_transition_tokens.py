from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from retrain.environments.echo_tokens import (
    EchoTokenBridgeError,
    bridge_observation_tokens,
)
from retrain.environments.prompt import encode_for_sampling


class _BodyRenderer:
    def bridge_to_next_turn(
        self,
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        *,
        tools=None,
    ):
        del tools
        body = str(new_messages[0]["content"])
        return SimpleNamespace(
            token_ids=(
                list(previous_prompt_ids)
                + list(previous_completion_ids)
                + [900, 901]
                + [ord(char) for char in body]
                + [902, 903]
            )
        )


def test_counterfactual_bridge_marks_only_new_observation_body() -> None:
    result = bridge_observation_tokens(
        _BodyRenderer(),
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        observation_messages=[{"role": "user", "content": "ok"}],
    )

    assert result.token_ids == [1, 2, 3, 4, 900, 901, ord("o"), ord("k")]
    assert result.observation_mask == [0, 0, 0, 0, 0, 0, 1, 1]


def test_bridge_rejects_a_renderer_that_changes_sampled_action_tokens() -> None:
    class _BadRenderer(_BodyRenderer):
        def bridge_to_next_turn(self, *args, **kwargs):
            rendered = super().bridge_to_next_turn(*args, **kwargs)
            rendered.token_ids[0] = 999
            return rendered

    with pytest.raises(EchoTokenBridgeError, match="changed or truncated"):
        bridge_observation_tokens(
            _BadRenderer(),
            prompt_ids=[1, 2],
            completion_ids=[3],
            observation_messages=[{"role": "user", "content": "result"}],
        )


def test_bridge_wraps_renderer_failures_for_action_only_fallback() -> None:
    class _FailingRenderer(_BodyRenderer):
        def bridge_to_next_turn(self, *args, **kwargs):
            del args, kwargs
            raise ValueError("renderer could not bridge")

    with pytest.raises(EchoTokenBridgeError, match="failed while bridging") as error:
        bridge_observation_tokens(
            _FailingRenderer(),
            prompt_ids=[1, 2],
            completion_ids=[3],
            observation_messages=[{"role": "user", "content": "result"}],
        )

    assert isinstance(error.value.__cause__, ValueError)


def _cached_qwen35_snapshot() -> Path | None:
    root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen3.5-4B"
        / "snapshots"
    )
    if not root.is_dir():
        return None
    for candidate in sorted(root.iterdir()):
        if (candidate / "tokenizer.json").exists():
            return candidate
    return None


@pytest.mark.skipif(
    _cached_qwen35_snapshot() is None,
    reason="Qwen3.5 tokenizer is not cached locally",
)
def test_real_qwen35_bridge_recovers_observation_after_nonprefix_rerender() -> None:
    from renderers import Qwen35Renderer
    from transformers import AutoTokenizer

    snapshot = _cached_qwen35_snapshot()
    assert snapshot is not None
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    renderer = Qwen35Renderer(tokenizer, enable_thinking=False)
    initial_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    action_text = '{"tool":"query","sql":"select 1"}'
    observation_text = "query ok\nanswer = 1"
    prompt_ids = encode_for_sampling(tokenizer, initial_messages)
    completion_ids = tokenizer.encode(action_text, add_special_tokens=False)
    next_prompt = encode_for_sampling(
        tokenizer,
        [
            *initial_messages,
            {"role": "assistant", "content": action_text},
            {"role": "user", "content": observation_text},
        ],
    )

    sampled_prefix = prompt_ids + completion_ids
    assert next_prompt[: len(sampled_prefix)] != sampled_prefix

    result = bridge_observation_tokens(
        renderer,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        observation_messages=[{"role": "user", "content": observation_text}],
    )

    assert result.token_ids[: len(sampled_prefix)] == sampled_prefix
    selected_ids = [
        token
        for token, include in zip(result.token_ids, result.observation_mask)
        if include
    ]
    assert tokenizer.decode(selected_ids) == observation_text
    assert not any(result.observation_mask[: len(sampled_prefix)])
