"""Verifiers-backed rewards: symbolic math equivalence and LLM judge.

Lazy-imported so the ``verifiers`` and ``math_verify`` deps stay optional.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable, Coroutine
from typing import Protocol, cast

from retrain.rewards.boxed import extract_boxed


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
