"""SROIE receipt extraction dataset for RLVR training.

Loads the SROIE dataset (ICDAR 2019) from HuggingFace and formats each receipt
as a structured extraction task. The model must return a JSON object with four
fields: company, date, address, total.

Register with retrain via plugin:
    [data]
    source = "campaigns.sroie_dataset:SROIEDataSource"
"""

from __future__ import annotations

import json

from retrain.data import Example

_SYSTEM_PROMPT = (
    "You are a document extraction assistant. "
    "Extract structured fields from receipt text and return only valid JSON."
)

_USER_TEMPLATE = """\
Extract the following fields from this receipt and return them as a JSON object:
- company: the name of the store or company
- date: the date of the transaction (as written on the receipt)
- address: the full address of the store
- total: the total amount paid (as written, e.g. "12.50")

Receipt:
{text}

Return only a JSON object with keys: company, date, address, total."""


def _build_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(text=text)},
    ]


class SROIEDataSource:
    """Loads SROIE v2 from HuggingFace for structured receipt extraction."""

    def __init__(self, config: object = None, max_examples: int = 0) -> None:
        self.max_examples = max_examples

    def load(self) -> list[Example]:
        from datasets import load_dataset

        ds = load_dataset("jsdnrs/ICDAR2019-SROIE", split="train", columns=["words", "entities"])

        examples: list[Example] = []
        for i, item in enumerate(ds):
            words = item.get("words") or []
            text = " ".join(words).strip()
            if not text:
                continue

            entities = item.get("entities") or {}
            reference = json.dumps({
                "company": entities.get("company", ""),
                "date": entities.get("date", ""),
                "address": entities.get("address", ""),
                "total": entities.get("total", ""),
            }, ensure_ascii=False)

            examples.append(Example(
                prompt=_build_prompt(text),
                reference=reference,
                task="sroie",
                example_id=i,
            ))

            if 0 < self.max_examples <= len(examples):
                break

        return examples


def score(response: str, reference: str) -> float:
    """Field-level F1 reward for retrain custom reward.

    Each of the 4 fields contributes 0.25. Matching is case-insensitive and
    strips leading/trailing whitespace. Partial credit: if the model extracts
    3/4 fields correctly it gets 0.75.
    """
    try:
        # Extract JSON from response (model may wrap it in markdown)
        text = response.strip()
        if "```" in text:
            lines = text.split("\n")
            text = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            ).strip()

        pred = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return 0.0

    try:
        ref = json.loads(reference)
    except (json.JSONDecodeError, ValueError):
        return 0.0

    fields = ["company", "date", "address", "total"]
    correct = sum(
        1 for f in fields
        if pred.get(f, "").strip().lower() == ref.get(f, "").strip().lower()
    )
    return correct / len(fields)
