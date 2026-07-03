"""Reward functions for RLVR training.

Provides pluggable reward functions dispatched via ``create_reward(config)``.

Supported reward types (set via ``[reward] type`` in TOML):
  match  — string-match on \\boxed{} (default, no extra deps)
  math   — symbolic math equivalence via ``math_verify``
  judge  — LLM-based evaluation via ``verifiers.JudgeRubric``
  custom — user-provided ``module:function``
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Protocol, SupportsFloat, SupportsIndex, cast, runtime_checkable

if TYPE_CHECKING:
    from retrain.config import TrainConfig


class _MathParse(Protocol):
    def __call__(
        self,
        text: str,
        *,
        extraction_config: list[object],
        parsing_timeout: int,
    ) -> list[object]: ...


class _MathVerify(Protocol):
    def __call__(self, reference: list[object], response: list[object]) -> bool: ...


class _JudgeRubric(Protocol):
    def judge(
        self,
        *,
        prompt: list[dict[str, str]],
        completion: list[dict[str, str]],
        answer: str,
    ) -> Coroutine[object, object, str]: ...


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class RewardFunction(Protocol):
    """Any object that can score a completion against a reference answer."""

    def score(self, response: str, reference: str) -> float: ...


# ---------------------------------------------------------------------------
# Boxed string-match (original, zero deps)
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str:
    """Extract \\boxed{...} answer from MATH solution text."""
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return ""

    start = idx + len(marker)
    if start >= len(text):
        return ""

    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        pos += 1

    end = pos - 1 if depth == 0 else pos
    if end <= start:
        return ""
    return text[start:end].strip()


class BoxedMathReward:
    """Binary correctness reward: extract \\boxed{...} and string-match."""

    def score(self, response: str, reference: str) -> float:
        given = extract_boxed(response)
        if given.strip() == reference.strip():
            return 1.0
        return 0.0


# ---------------------------------------------------------------------------
# Verifiers adapters (lazy-imported so ``verifiers`` is optional)
# ---------------------------------------------------------------------------

class VerifiersMathReward:
    """Symbolic math equivalence via ``math_verify.parse`` + ``math_verify.verify``.

    Calls the sync ``parse()``/``verify()`` API directly, avoiding the
    MathRubric async wrapper and its per-call ``asyncio.run()`` overhead.

    Extracts the last ``\\boxed{}`` answer from the response (matching our
    convention), then uses ``math_verify`` for symbolic comparison only.
    Reference parses are cached since the same problem is scored G times.
    """

    def __init__(self) -> None:
        math_verify = importlib.import_module("math_verify")
        extraction_config = cast(
            Callable[[], object],
            getattr(math_verify, "LatexExtractionConfig"),
        )
        self._parse = cast(_MathParse, getattr(math_verify, "parse"))
        self._verify = cast(_MathVerify, getattr(math_verify, "verify"))
        self._extraction_config = [extraction_config()]
        self._ref_cache: dict[str, list[object]] = {}

    def _parse_expr(self, text: str) -> list[object]:
        return self._parse(
            f"\\boxed{{{text}}}",
            extraction_config=self._extraction_config,
            parsing_timeout=5,
        )

    def _parse_ref(self, reference: str) -> list[object]:
        cached = self._ref_cache.get(reference)
        if cached is not None:
            return cached
        parsed = self._parse_expr(reference)
        self._ref_cache[reference] = parsed
        return parsed

    def score(self, response: str, reference: str) -> float:
        try:
            # Extract last \boxed{} (same convention as BoxedMathReward)
            given = extract_boxed(response)
            if not given:
                return 0.0
            parsed_response = self._parse_expr(given)
            if not parsed_response:
                return 0.0
            parsed_ref = self._parse_ref(reference)
            if not parsed_ref:
                return 0.0
            return 1.0 if self._verify(parsed_ref, parsed_response) else 0.0
        except Exception:
            return 0.0


class VerifiersJudgeReward:
    """LLM-based evaluation via ``verifiers.JudgeRubric``.

    Calls ``rubric.judge(prompt, completion, answer)`` which sends the
    completion to an LLM judge, then checks for "yes" in the response.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        judge_module = importlib.import_module("verifiers.rubrics.judge_rubric")
        judge_cls = cast(
            Callable[..., _JudgeRubric],
            getattr(judge_module, "JudgeRubric"),
        )
        self._rubric = judge_cls(judge_model=model)

    def score(self, response: str, reference: str) -> float:
        # judge() accesses prompt[-1] so we must pass a non-empty list.
        # The original user prompt isn't available at this level, so we
        # send an empty-content message — the judge still sees the ground
        # truth answer and the model response.
        prompt = [{"role": "user", "content": ""}]
        completion = [{"role": "assistant", "content": response}]
        try:
            judge_response = asyncio.run(
                self._rubric.judge(
                    prompt=prompt,
                    completion=completion,
                    answer=reference,
                )
            )
            return 1.0 if "yes" in judge_response.lower() else 0.0
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Custom user-provided reward
# ---------------------------------------------------------------------------

class CustomReward:
    """Loads a user-provided ``module:function`` as the reward function."""

    def __init__(self, module_path: str, function_name: str) -> None:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, function_name, None)
        if fn is None:
            raise AttributeError(
                f"Module '{module_path}' has no attribute '{function_name}'"
            )
        self._fn = fn

    def score(self, response: str, reference: str) -> float:
        result = self._fn(response, reference)
        if asyncio.iscoroutine(result):
            result = asyncio.run(cast(Coroutine[object, object, object], result))
        return _float_score(result)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_VERIFIERS_TYPES = {"math", "judge"}


def create_reward(config: TrainConfig) -> RewardFunction:
    """Create the reward function specified by *config.reward_type*."""
    rtype = config.reward_type

    if rtype == "match":
        return BoxedMathReward()

    if rtype in _VERIFIERS_TYPES:
        try:
            importlib.import_module("verifiers")
        except ModuleNotFoundError:
            raise ImportError(
                f"Reward type '{rtype}' requires the verifiers library. "
                "Install it with:  pip install verifiers"
            ) from None

        if rtype == "math":
            return VerifiersMathReward()
        if rtype == "judge":
            model = config.reward_judge_model or "gpt-4o-mini"
            return VerifiersJudgeReward(model=model)

    if rtype == "custom":
        if not config.reward_custom_module:
            raise ValueError(
                "Reward type 'custom' requires [reward] custom_module to be set."
            )
        return CustomReward(
            config.reward_custom_module,
            config.reward_custom_function or "score",
        )

    raise ValueError(
        f"Unknown reward type '{rtype}'. "
        "Choose from: match, math, judge, custom"
    )


def _float_score(value: object) -> float:
    if isinstance(value, str):
        return float(value)
    return float(cast(SupportsFloat | SupportsIndex, value))
