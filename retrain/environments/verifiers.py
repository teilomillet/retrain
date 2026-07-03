"""Public verifiers environment bridge.

This module keeps the historical import path stable while the implementation
lives in focused modules under :mod:`retrain.environments.verifier`.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, cast

from retrain.environments import load as env_load
from retrain.environments import prompt as prompt_utils
from retrain.environments.rollout import VerifiersRolloutTiming, VerifiersTurnSample
from retrain.environments.verifier import client as verifier_client
from retrain.environments.verifier import coerce
from retrain.environments.verifier import multi as verifier_multi
from retrain.environments.verifier import score as verifier_score
from retrain.environments.verifier.score import (
    completion_messages as _completion_messages_for_env,
)
from retrain.environments.verifier.score import messages_to_text as _messages_to_text
from retrain.environments.verifier.types import (
    MultiTurnEnvironment as _MultiTurnEnvironment,
)
from retrain.environments.verifier.types import Rubric as _Rubric
from retrain.environments.verifier.types import (
    SingleTurnEnvironment as _SingleTurnEnvironment,
)
from retrain.environments.verifier.types import StateDict
from retrain.environments.verifier.types import Tokenizer as _Tokenizer
from retrain.types import ExampleInfoLike, JSONObject, PromptLike

if TYPE_CHECKING:
    from retrain.backends import TrainHelper
    from retrain.config import TrainConfig
    from retrain.data.source import Example


_FALLBACK_TRAINING_ENVS = env_load.FALLBACK_TRAINING_ENVS


def _require_verifiers() -> types.ModuleType:
    return env_load.require_verifiers()


def _optional_verifiers() -> types.ModuleType | None:
    return verifier_client.optional(_require_verifiers)


def _make_env_client() -> object | None:
    return verifier_client.make()


def _coerce_int(raw: object) -> int:
    try:
        return int(cast(int | str | float, raw))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected int-like value, got {raw!r}.") from exc


def _coerce_float_list(raw: object) -> list[float]:
    return coerce.float_list(raw)


def _coerce_reward(raw: object) -> float:
    return coerce.reward(raw)


def parse_environment_args(raw: str | JSONObject | None) -> JSONObject:
    return env_load.parse_args(raw)


def _hub_env_suggestions(env_id: str, limit: int = 5) -> list[str]:
    return env_load.hub_suggestions(env_id, limit=limit)


def _format_hub_suggestions(env_id: str) -> str:
    return env_load.format_hub_suggestions(env_id)


def load_verifiers_environment(config: "TrainConfig") -> object:
    return env_load.load_environment(config, require_fn=_require_verifiers)


def load_examples_from_environment(env: object, config: "TrainConfig") -> list[Example]:
    return env_load.examples_from_environment(env, config)


def prompt_preview(prompt: PromptLike, max_chars: int = 200) -> str:
    return prompt_utils.preview(prompt, max_chars=max_chars)


def encode_prompt_for_sampling(tokenizer: object, prompt: PromptLike) -> list[int]:
    return prompt_utils.encode_for_sampling(tokenizer, prompt)


def observation_mask_for_prompt(
    tokenizer: object,
    prompt: PromptLike,
    prompt_ids: list[int],
) -> list[int] | None:
    return prompt_utils.observation_mask(tokenizer, prompt, prompt_ids)


def score_singleturn_group(
    env: object,
    *,
    prompt: PromptLike,
    answer: str,
    task: str,
    info: ExampleInfoLike,
    completion_texts: list[str],
) -> list[float]:
    return verifier_score.score_singleturn_group(
        env,
        prompt=prompt,
        answer=answer,
        task=task,
        info=info,
        completion_texts=completion_texts,
        require_fn=_require_verifiers,
    )


def is_multiturn_environment(env: object) -> bool:
    return verifier_multi.is_multiturn_environment(env, require_fn=_require_verifiers)


def run_multiturn_group(
    env: object,
    *,
    helper: "TrainHelper",
    tokenizer: object,
    model_name: str,
    prompt: PromptLike,
    answer: str,
    task: str,
    info: ExampleInfoLike,
    num_rollouts: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_turns_override: int = -1,
    tl_grpo: bool = False,
    tl_grpo_branch_mode: str = "action_space",
    tl_grpo_branch_size: int = 4,
    tl_grpo_lookahead_steps: int = 0,
    tl_grpo_outcome_baseline: float | None = None,
    temperature_spread: float = 0.0,
    rollout_env_workers: int = 1,
    rollout_buffer_size: int = 0,
) -> tuple[
    list[float],
    list[list[VerifiersTurnSample]],
    list[str],
    list[list[float]],
    list[list[float]],
    list[list[dict[str, object]]],
    list[list[list[float]]],
    VerifiersRolloutTiming,
]:
    return verifier_multi.run_multiturn_group(
        env,
        helper=helper,
        tokenizer=tokenizer,
        model_name=model_name,
        prompt=prompt,
        answer=answer,
        task=task,
        info=info,
        num_rollouts=num_rollouts,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        max_turns_override=max_turns_override,
        tl_grpo=tl_grpo,
        tl_grpo_branch_mode=tl_grpo_branch_mode,
        tl_grpo_branch_size=tl_grpo_branch_size,
        tl_grpo_lookahead_steps=tl_grpo_lookahead_steps,
        tl_grpo_outcome_baseline=tl_grpo_outcome_baseline,
        temperature_spread=temperature_spread,
        rollout_env_workers=rollout_env_workers,
        rollout_buffer_size=rollout_buffer_size,
        verifiers_loader=_optional_verifiers,
        env_client_factory=_make_env_client,
    )
