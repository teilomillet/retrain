"""Bridge utilities to run verifiers environments from retrain.

This keeps the user-facing workflow TOML-first:
- select env in [environment]
- keep training through retrain backends (local/tinker)

Supports:
- dataset loading from verifiers environments
- rubric scoring for single-turn and multi-turn rollouts
- multi-turn rollouts driven by retrain sampling backends
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import types
from typing import TYPE_CHECKING, Protocol, cast

from retrain.data import Example
from retrain.type_defs import ExampleInfoLike, JSONObject, PromptLike

if TYPE_CHECKING:
    from retrain.backends import TrainHelper
    from retrain.config import TrainConfig


_FALLBACK_TRAINING_ENVS = (
    "primeintellect/gsm8k",
    "primeintellect/wordle",
    "primeintellect/hendrycks-math",
)


StateDict = dict[str, object]


class _Rubric(Protocol):
    async def score_group(self, states: list[StateDict]) -> object: ...


class _DatasetEnvironment(Protocol):
    env_id: str

    def get_dataset(self, *, n: int, seed: int | None) -> Iterable[Mapping[str, object]]: ...


class _SingleTurnEnvironment(Protocol):
    message_type: str
    rubric: _Rubric


class _MultiTurnEnvironment(Protocol):
    message_type: str
    rubric: _Rubric

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


class _Tokenizer(Protocol):
    def encode(self, text: str) -> object: ...
    def batch_decode(
        self, token_ids: list[list[int]], *, skip_special_tokens: bool = True
    ) -> list[str]: ...


def _require_verifiers() -> types.ModuleType:
    try:
        import verifiers as vf
    except ModuleNotFoundError:
        raise ImportError(
            "Verifiers environment bridge requires the verifiers package.\n"
            "Install it with: pip install 'retrain[verifiers]'"
        ) from None
    return vf


def _coerce_prompt(raw: object) -> PromptLike:
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return str(raw)
    messages: list[dict[str, object]] = []
    for msg in raw:
        if isinstance(msg, Mapping):
            messages.append(dict(cast(Mapping[str, object], msg)))
        else:
            messages.append({"role": "", "content": str(msg)})
    return messages


def _coerce_example_info(raw: object) -> ExampleInfoLike:
    if raw is None or isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        return cast(ExampleInfoLike, dict(cast(Mapping[str, object], raw)))
    return str(raw)


def _coerce_int_ids(raw: object) -> list[int]:
    if not isinstance(raw, list):
        raise TypeError(f"Expected list of token ids, got {type(raw).__name__}")
    return [_coerce_int(tok) for tok in raw]


def _coerce_int(raw: object) -> int:
    try:
        return int(cast(int | str | float, raw))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected int-like value, got {raw!r}.") from exc


def _coerce_reward(raw: object) -> float:
    if raw is None:
        return 0.0
    try:
        return float(cast(int | float | str, raw))
    except (TypeError, ValueError):
        return 0.0


def _coerce_example_id(raw: object) -> int | str:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int | str):
        return raw
    return str(raw)


def parse_environment_args(raw: str | JSONObject | None) -> JSONObject:
    """Parse [environment].args from TOML/CLI into a dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return cast(JSONObject, raw)
    if not isinstance(raw, str):
        raise ValueError(
            f"[environment].args must be a JSON string/object, got {type(raw).__name__}"
        )
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"[environment].args must be valid JSON, got: {raw}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("[environment].args must decode to a JSON object.")
    return cast(JSONObject, parsed)


def _hub_env_suggestions(env_id: str, limit: int = 5) -> list[str]:
    """Best-effort suggestions for similar Hub environment IDs."""
    if "/" not in env_id:
        return []

    query = env_id.rsplit("/", 1)[-1].split("@", 1)[0].strip()
    if not query:
        return []

    try:
        import requests
        from verifiers.utils.install_utils import ENVIRONMENTS_HUB_URL
    except Exception:
        return []

    try:
        response = requests.get(
            ENVIRONMENTS_HUB_URL,
            params={"search": query, "limit": max(1, limit)},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    suggestions: list[str] = []
    for row in payload.get("data", []):
        owner = (row.get("owner") or {}).get("name")
        name = row.get("name")
        if owner and name:
            env_key = f"{owner}/{name}"
            if env_key not in suggestions:
                suggestions.append(env_key)
        if len(suggestions) >= limit:
            break
    return suggestions


def _format_hub_suggestions(env_id: str) -> str:
    suggestions = _hub_env_suggestions(env_id)
    if not suggestions:
        return ""
    return " Similar IDs: " + ", ".join(suggestions) + "."


def load_verifiers_environment(config: "TrainConfig") -> object:
    """Load a verifiers environment from local install or Prime Hub package."""
    vf = _require_verifiers()
    env_id = config.environment_id
    env_args = parse_environment_args(config.environment_args)

    if config.environment_auto_install:
        try:
            from verifiers.utils.install_utils import (
                check_hub_env_installed,
                install_from_hub,
                is_hub_env,
            )
        except Exception:
            # Keep loading path robust even if helper APIs change in verifiers.
            check_hub_env_installed = None  # type: ignore[assignment]
            install_from_hub = None  # type: ignore[assignment]
            is_hub_env = None  # type: ignore[assignment]

        if (
            check_hub_env_installed is not None
            and install_from_hub is not None
            and is_hub_env is not None
            and is_hub_env(env_id)
            and not check_hub_env_installed(env_id)
        ):
            ok = install_from_hub(env_id)
            if not ok:
                suggestion_hint = _format_hub_suggestions(env_id)
                raise RuntimeError(
                    f"Failed to auto-install verifiers environment '{env_id}'. "
                    "The environment ID may be invalid, private, or inaccessible."
                    f"{suggestion_hint} "
                    "Try manually: uv run python -m verifiers.cli.commands.install "
                    f"{env_id}"
                )

    try:
        return vf.load_environment(env_id, **env_args)
    except Exception as exc:
        suggestion_hint = _format_hub_suggestions(env_id)
        raise RuntimeError(
            f"Failed to load verifiers environment '{env_id}': {exc}."
            f"{suggestion_hint}"
        ) from exc


def load_examples_from_environment(env: object, config: "TrainConfig") -> list[Example]:
    """Convert verifiers dataset rows into retrain Example objects."""
    n = config.max_examples if config.max_examples > 0 else -1
    seed = config.seed if config.seed >= 0 else None
    env_id = str(getattr(env, "env_id", config.environment_id or "unknown"))
    dataset_env = cast(_DatasetEnvironment, env)
    try:
        dataset = dataset_env.get_dataset(n=n, seed=seed)
    except Exception as exc:
        msg = str(exc).lower()
        if "dataset is not set" in msg:
            fallback = ", ".join(_FALLBACK_TRAINING_ENVS)
            raise RuntimeError(
                f"Environment '{env_id}' does not expose a training dataset "
                "(likely eval-only). Use a trainable environment such as "
                f"{fallback}. If you intended evaluation, use verifiers eval flow."
            ) from None
        raise RuntimeError(
            f"Failed to load dataset from verifiers environment '{env_id}': {exc}"
        ) from exc

    examples: list[Example] = []
    for row in dataset:
        row_data = row
        prompt = row_data.get("prompt")
        if prompt is None:
            # Fallback for raw pre-format datasets.
            question = row_data.get("question", "")
            prompt = str(question)
        answer = row_data.get("answer", "")
        task = row_data.get("task", getattr(env, "env_id", "") or "default")
        info = row_data.get("info", None)
        example_id = _coerce_example_id(row_data.get("example_id", -1))
        examples.append(
            Example(
                prompt=_coerce_prompt(prompt),
                reference=str(answer),
                task=str(task),
                info=_coerce_example_info(info),
                example_id=example_id,
            )
        )
    return examples


def prompt_preview(prompt: PromptLike, max_chars: int = 200) -> str:
    """Render a compact text preview for logs."""
    if isinstance(prompt, str):
        text = prompt
    else:
        parts: list[str] = []
        for msg in prompt:
            role = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            if content:
                parts.append(f"{role}: {content}")
        text = "\n".join(parts)
    return text[:max_chars]


def encode_prompt_for_sampling(tokenizer: object, prompt: PromptLike) -> list[int]:
    """Encode a prompt object (string or chat messages) for model sampling."""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    encode = getattr(tokenizer, "encode", None)
    if isinstance(prompt, str):
        if callable(apply_chat_template):
            messages = [{"role": "user", "content": prompt}]
            ids = apply_chat_template(messages, add_generation_prompt=True)
        else:
            if not callable(encode):
                raise TypeError("Tokenizer must expose encode() for string prompts.")
            ids = encode(prompt)
    else:
        if callable(apply_chat_template):
            ids = apply_chat_template(prompt, add_generation_prompt=True)
        else:
            text = "\n".join(str(msg.get("content", "")) for msg in prompt)
            if not callable(encode):
                raise TypeError("Tokenizer must expose encode() for chat prompts.")
            ids = encode(text)

    if isinstance(ids, Mapping):
        input_ids = ids.get("input_ids")
        if input_ids is not None:
            return _coerce_int_ids(input_ids)
    if hasattr(ids, "input_ids"):
        return _coerce_int_ids(getattr(ids, "input_ids"))
    return _coerce_int_ids(ids)


def _completion_messages_for_env(
    env: _SingleTurnEnvironment, completion_text: str
) -> list[dict[str, str]] | str:
    if getattr(env, "message_type", "chat") == "chat":
        return [{"role": "assistant", "content": completion_text}]
    return completion_text


def _messages_to_text(messages: object) -> str:
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
) -> list[float]:
    """Score a group of single-turn completions with env rubric."""
    vf = _require_verifiers()
    env_typed = cast(_SingleTurnEnvironment, env)

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
        state["completion"] = _completion_messages_for_env(env_typed, text)
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
    return [_coerce_reward(s.get("reward")) for s in states]


@dataclass
class VerifiersTurnSample:
    """One assistant turn sampled by retrain for a rollout."""

    prompt_ids: list[int]
    completion_ids: list[int]
    completion_logprobs: list[float]
    completion_text: str


def is_multiturn_environment(env: object) -> bool:
    """Whether env is a verifiers MultiTurnEnv."""
    vf = _require_verifiers()
    return isinstance(env, vf.MultiTurnEnv)


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
) -> tuple[list[float], list[list[VerifiersTurnSample]], list[str]]:
    """Run group rollouts for verifiers MultiTurnEnv using retrain sampling."""

    async def _run() -> tuple[list[float], list[list[VerifiersTurnSample]], list[str]]:
        vf = _require_verifiers()
        env_typed = cast(_MultiTurnEnvironment, env)
        tokenizer_typed = cast(_Tokenizer, tokenizer)
        states: list[StateDict] = []
        per_rollout_turns: list[list[VerifiersTurnSample]] = [[] for _ in range(num_rollouts)]

        for i in range(num_rollouts):
            input_payload: dict[str, object] = {
                "prompt": prompt,
                "answer": answer,
                "task": task,
                "example_id": i,
            }
            if info is not None:
                input_payload["info"] = info
            state = await env_typed.init_state(
                input=input_payload,
                client=None,  # retrain handles model calls through helper.sample()
                model=model_name,
                sampling_args=None,
            )
            state = await env_typed.setup_state(state)
            states.append(state)

        turn_count = 0
        while True:
            active: list[tuple[int, PromptLike, list[int]]] = []
            for idx, state in enumerate(states):
                if await env_typed.is_completed(state):
                    continue
                prompt_messages = await env_typed.get_prompt_messages(state)
                if state.get("final_env_response") is not None:
                    continue
                prompt_ids = encode_prompt_for_sampling(tokenizer, prompt_messages)
                active.append((idx, prompt_messages, prompt_ids))

            if not active:
                break

            if max_turns_override > 0 and turn_count >= max_turns_override:
                for idx, _messages, _prompt_ids in active:
                    states[idx]["is_completed"] = True
                    states[idx]["is_truncated"] = True
                    states[idx]["stop_condition"] = "retrain_max_turns"
                break

            prompt_ids_batch = [item[2] for item in active]
            sampled_groups = helper.sample(
                prompt_ids_batch,
                1,
                max_tokens,
                temperature,
                top_p,
            )
            completion_ids_batch = [list(group[0][0]) for group in sampled_groups]
            completion_logprobs_batch = [
                [float(lp) for lp in group[0][1]] for group in sampled_groups
            ]
            completion_texts = tokenizer_typed.batch_decode(
                completion_ids_batch, skip_special_tokens=True
            )

            for pos, (idx, prompt_messages, prompt_ids) in enumerate(active):
                completion_ids = completion_ids_batch[pos]
                completion_logprobs = completion_logprobs_batch[pos]
                completion_text = completion_texts[pos]
                completion_messages = _completion_messages_for_env(env_typed, completion_text)

                tokens_payload = {
                    "prompt_ids": list(prompt_ids),
                    "prompt_mask": [0] * len(prompt_ids),
                    "completion_ids": list(completion_ids),
                    "completion_mask": [1] * len(completion_ids),
                    "completion_logprobs": list(completion_logprobs),
                    "overlong_prompt": False,
                    "is_truncated": False,
                }
                trajectory_step = vf.TrajectoryStep(
                    prompt=prompt_messages,
                    completion=completion_messages,
                    response=None,
                    tokens=tokens_payload,
                    reward=None,
                    advantage=None,
                    is_truncated=False,
                    trajectory_id=states[idx]["trajectory_id"],
                    extras={},
                )
                await env_typed.add_trajectory_step(states[idx], trajectory_step)
                per_rollout_turns[idx].append(
                    VerifiersTurnSample(
                        prompt_ids=list(prompt_ids),
                        completion_ids=list(completion_ids),
                        completion_logprobs=list(completion_logprobs),
                        completion_text=completion_text,
                    )
                )

            turn_count += 1

        for state in states:
            await env_typed.render_completion(state)
        await env_typed.rubric.score_group(states)

        rewards = [_coerce_reward(s.get("reward")) for s in states]
        completions_text = [_messages_to_text(s.get("completion")) for s in states]
        return rewards, per_rollout_turns, completions_text

    return asyncio.run(_run())
