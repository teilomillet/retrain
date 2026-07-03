"""Verifiers environment loading and dataset conversion."""

from __future__ import annotations

import importlib
import json
import types
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, cast

from retrain.data.source import Example
from retrain.types import ExampleInfoLike, JSONObject, PromptLike


FALLBACK_TRAINING_ENVS = (
    "primeintellect/gsm8k",
    "primeintellect/wordle",
    "primeintellect/hendrycks-math",
)


class DatasetEnvironment(Protocol):
    env_id: str

    def get_dataset(
        self,
        *,
        n: int,
        seed: int | None,
    ) -> Iterable[Mapping[str, object]]: ...


def require_verifiers() -> types.ModuleType:
    try:
        return importlib.import_module("verifiers")
    except ModuleNotFoundError:
        raise ImportError(
            "Verifiers environment bridge requires the verifiers package.\n"
            "Install it with: pip install 'retrain[verifiers]'"
        ) from None


def parse_args(raw: str | JSONObject | None) -> JSONObject:
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


def hub_suggestions(env_id: str, limit: int = 5) -> list[str]:
    """Best-effort suggestions for similar Hub environment IDs."""
    if "/" not in env_id:
        return []

    query = env_id.rsplit("/", 1)[-1].split("@", 1)[0].strip()
    if not query:
        return []

    try:
        import requests

        install_utils = importlib.import_module("verifiers.utils.install_utils")
        environments_hub_url = str(getattr(install_utils, "ENVIRONMENTS_HUB_URL"))
    except Exception:
        return []

    try:
        response = requests.get(
            environments_hub_url,
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


def format_hub_suggestions(env_id: str) -> str:
    suggestions = hub_suggestions(env_id)
    if not suggestions:
        return ""
    return " Similar IDs: " + ", ".join(suggestions) + "."


def load_environment(
    config: object,
    *,
    require_fn: Callable[[], types.ModuleType] = require_verifiers,
) -> object:
    """Load a verifiers environment from local install or Prime Hub package."""
    vf = require_fn()
    env_id = str(getattr(config, "environment_id"))
    env_args = parse_args(getattr(config, "environment_args"))

    if bool(getattr(config, "environment_auto_install")):
        check_hub_env_installed_fn: Callable[[str], bool] | None
        install_from_hub_fn: Callable[[str], bool] | None
        is_hub_env_fn: Callable[[str], bool] | None
        try:
            install_utils = importlib.import_module("verifiers.utils.install_utils")
            check_hub_env_installed_obj = getattr(
                install_utils,
                "check_hub_env_installed",
                None,
            )
            install_from_hub_obj = getattr(install_utils, "install_from_hub", None)
            is_hub_env_obj = getattr(install_utils, "is_hub_env", None)
            check_hub_env_installed_fn = (
                cast(Callable[[str], bool], check_hub_env_installed_obj)
                if callable(check_hub_env_installed_obj)
                else None
            )
            install_from_hub_fn = (
                cast(Callable[[str], bool], install_from_hub_obj)
                if callable(install_from_hub_obj)
                else None
            )
            is_hub_env_fn = (
                cast(Callable[[str], bool], is_hub_env_obj)
                if callable(is_hub_env_obj)
                else None
            )
        except Exception:
            # Auto-install is best-effort; normal environment loading still works.
            check_hub_env_installed_fn = None
            install_from_hub_fn = None
            is_hub_env_fn = None

        if (
            check_hub_env_installed_fn is not None
            and install_from_hub_fn is not None
            and is_hub_env_fn is not None
            and is_hub_env_fn(env_id)
            and not check_hub_env_installed_fn(env_id)
        ):
            ok = install_from_hub_fn(env_id)
            if not ok:
                suggestion_hint = format_hub_suggestions(env_id)
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
        suggestion_hint = format_hub_suggestions(env_id)
        raise RuntimeError(
            f"Failed to load verifiers environment '{env_id}': {exc}."
            f"{suggestion_hint}"
        ) from exc


def examples_from_environment(env: object, config: object) -> list[Example]:
    """Convert verifiers dataset rows into retrain Example objects."""
    max_examples = int(getattr(config, "max_examples"))
    n = max_examples if max_examples > 0 else -1
    seed_value = int(getattr(config, "seed"))
    seed = seed_value if seed_value >= 0 else None
    env_id = str(getattr(env, "env_id", getattr(config, "environment_id") or "unknown"))
    dataset_env = cast(DatasetEnvironment, env)
    try:
        dataset = dataset_env.get_dataset(n=n, seed=seed)
    except Exception as exc:
        msg = str(exc).lower()
        if "dataset is not set" in msg:
            fallback = ", ".join(FALLBACK_TRAINING_ENVS)
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


def _coerce_example_id(raw: object) -> int | str:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int | str):
        return raw
    return str(raw)
