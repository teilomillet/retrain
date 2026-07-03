"""Built-in MATH dataset source."""

from __future__ import annotations

from retrain.data.source import Example
from retrain.rewards.boxed import extract_boxed

_DEFAULT_SUBJECTS = [
    "intermediate_algebra",
    "precalculus",
    "number_theory",
    "counting_and_probability",
    "geometry",
]


class MathDataSource:
    """Loads hendrycks/MATH via EleutherAI mirror."""

    def __init__(self, max_examples: int = 0) -> None:
        self.max_examples = max_examples

    def load(self) -> list[Example]:
        from datasets import load_dataset

        examples: list[Example] = []
        for subject in _DEFAULT_SUBJECTS:
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
            for item in ds:
                solution = item["solution"]
                answer = extract_boxed(solution)
                problem = item["problem"]
                examples.append(Example(prompt=problem, reference=answer))

                if 0 < self.max_examples <= len(examples):
                    break
            if 0 < self.max_examples <= len(examples):
                break

        return examples
