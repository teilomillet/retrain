"""Prompt rendering for test-time discovery."""

from __future__ import annotations

from retrain.training.discovery.archive import DiscoverEntry
from retrain.types import PromptLike


def truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def build_discovery_prompt(
    base_prompt: PromptLike,
    *,
    start_text: str,
    context_entries: list[DiscoverEntry],
    candidate_char_limit: int = 1200,
    context_char_limit: int = 600,
) -> PromptLike:
    """Append reusable discovery memory to a prompt."""

    start_text = truncate_text(start_text.strip(), candidate_char_limit)
    context_entries = [entry for entry in context_entries if entry.text.strip()]
    if not start_text and not context_entries:
        return base_prompt

    blocks: list[str] = []
    if start_text:
        blocks.append(f"Current candidate to improve:\n{start_text}")
    if context_entries:
        rendered = []
        for idx, entry in enumerate(context_entries, start=1):
            rendered.append(
                f"{idx}. reward={entry.reward:.4f}\n"
                f"{truncate_text(entry.text.strip(), context_char_limit)}"
            )
        blocks.append("Other promising attempts:\n" + "\n\n".join(rendered))
    blocks.append(
        "Improve on the current candidate if possible. Return only the full improved solution."
    )
    memory = "Discovery memory:\n\n" + "\n\n".join(blocks)

    if isinstance(base_prompt, str):
        return base_prompt.rstrip() + "\n\n" + memory

    messages = [dict(msg) for msg in base_prompt]
    messages.append({"role": "user", "content": memory})
    return messages
