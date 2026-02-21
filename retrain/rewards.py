"""Reward functions for RLVR training.

Provides pluggable reward functions dispatched via ``create_reward(config)``.

Supported reward types (set via ``[reward] type`` in TOML):
  match  — string-match on \\boxed{} (default, no extra deps)
  math   — symbolic math equivalence via ``verifiers.MathRubric``
  judge  — LLM-based evaluation via ``verifiers.JudgeRubric``
  custom — user-provided ``module:function``
"""

from __future__ import annotations

import asyncio
import importlib
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from retrain.config import TrainConfig


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
    """Symbolic math equivalence via ``verifiers.MathRubric``.

    ``MathRubric.correct_answer`` expects ``(parser, completion, answer)``.
    We pass the rubric's own parser (``MaybeThinkParser`` with
    ``extract_boxed_answer``) so the framework can extract the answer
    from the completion messages.
    """

    def __init__(self) -> None:
        from verifiers.rubrics.math_rubric import MathRubric
        self._rubric = MathRubric()

    def score(self, response: str, reference: str) -> float:
        completion = [{"role": "assistant", "content": response}]
        try:
            return asyncio.run(
                self._rubric.correct_answer(
                    parser=self._rubric.parser,
                    completion=completion,
                    answer=reference,
                )
            )
        except Exception:
            return 0.0


class VerifiersJudgeReward:
    """LLM-based evaluation via ``verifiers.JudgeRubric``.

    Calls ``rubric.judge(prompt, completion, answer)`` which sends the
    completion to an LLM judge, then checks for "yes" in the response.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from verifiers.rubrics.judge_rubric import JudgeRubric
        self._rubric = JudgeRubric(judge_model=model)

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
            return float(asyncio.run(result))
        return float(result)


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
            import verifiers as _verifiers  # noqa: F841
        except ModuleNotFoundError:
            raise ImportError(
                f"Reward type '{rtype}' requires the verifiers library. "
                "Install it with:  pip install retrain[verifiers]"
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
