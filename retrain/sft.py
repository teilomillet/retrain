"""Supervised fine-tuning dataset and tokenization helpers."""

from __future__ import annotations

import inspect
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, cast

if TYPE_CHECKING:
    from retrain.config import TrainConfig


@dataclass(frozen=True)
class SftExample:
    """One supervised fine-tuning datum.

    Supported JSONL row shapes:
    - {"messages": [{"role": ..., "content": ...}, ...]}
    - {"prompt": "...", "completion": "..."}
    - {"text": "..."}
    """

    messages: list[dict[str, str]] | None = None
    prompt: str = ""
    completion: str = ""
    text: str = ""


@dataclass(frozen=True)
class SftTokenizedBatch:
    """Tokenized SFT batch plus useful accounting."""

    tokens: list[list[int]]
    advantages: list[list[float]]
    total_tokens: int
    supervised_tokens: int


@dataclass(frozen=True)
class SftTokenizedExample:
    """One already-tokenized SFT datum."""

    tokens: list[int]
    advantages: list[float]

    @property
    def total_tokens(self) -> int:
        return len(self.tokens)

    @property
    def supervised_tokens(self) -> int:
        return sum(1 for value in self.advantages if value > 0.0)


def _row_error(path: Path, line_no: int, message: str) -> ValueError:
    return ValueError(f"Invalid SFT JSONL row in {path}:{line_no}: {message}")


def _coerce_messages(raw: object, *, path: Path, line_no: int) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise _row_error(path, line_no, "'messages' must be a non-empty list.")
    messages: list[dict[str, str]] = []
    for idx, msg in enumerate(raw):
        if not isinstance(msg, Mapping):
            raise _row_error(
                path,
                line_no,
                f"messages[{idx}] must be an object with role/content strings.",
            )
        msg_map = cast(Mapping[str, object], msg)
        role = msg_map.get("role")
        content = msg_map.get("content")
        if not isinstance(role, str) or not role.strip():
            raise _row_error(path, line_no, f"messages[{idx}].role must be a string.")
        if not isinstance(content, str):
            raise _row_error(path, line_no, f"messages[{idx}].content must be a string.")
        messages.append({"role": role, "content": content})
    if not any(msg["role"] == "assistant" for msg in messages):
        raise _row_error(path, line_no, "'messages' must include an assistant target.")
    return messages


def load_sft_jsonl(path: str | Path) -> list[SftExample]:
    """Load supported SFT JSONL rows with line-numbered validation errors."""
    sft_path = Path(path)
    examples: list[SftExample] = []
    with open(sft_path) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _row_error(sft_path, line_no, f"invalid JSON: {exc.msg}") from exc
            if not isinstance(row, Mapping):
                raise _row_error(sft_path, line_no, "row must be a JSON object.")
            row_map = cast(Mapping[str, object], row)

            if "messages" in row_map:
                examples.append(
                    SftExample(
                        messages=_coerce_messages(
                            row_map["messages"],
                            path=sft_path,
                            line_no=line_no,
                        )
                    )
                )
                continue

            if "prompt" in row_map or "completion" in row_map:
                prompt = row_map.get("prompt")
                completion = row_map.get("completion")
                if not isinstance(prompt, str):
                    raise _row_error(sft_path, line_no, "'prompt' must be a string.")
                if not isinstance(completion, str) or not completion:
                    raise _row_error(
                        sft_path,
                        line_no,
                        "'completion' must be a non-empty string.",
                    )
                examples.append(SftExample(prompt=prompt, completion=completion))
                continue

            if "text" in row_map:
                text = row_map.get("text")
                if not isinstance(text, str) or not text:
                    raise _row_error(sft_path, line_no, "'text' must be a non-empty string.")
                examples.append(SftExample(text=text))
                continue

            raise _row_error(
                sft_path,
                line_no,
                "expected one of 'messages', 'prompt'+'completion', or 'text'.",
            )
    return examples


def build_sft_example_order(
    example_count: int,
    seed: int,
    *,
    lengths: list[int] | None = None,
    batch_order: str = "shuffle",
    length_bucket_size: int = 0,
) -> list[int]:
    """Return a deterministic SFT traversal.

    ``shuffle`` preserves the historical behavior. Length-aware modes operate
    after tokenization, matching the cost signal that controls padding and VRAM.
    """
    if example_count <= 0:
        return []
    order = list(range(example_count))
    rng = random.Random(seed)
    rng.shuffle(order)
    if batch_order == "shuffle":
        return order
    if lengths is None or len(lengths) != example_count:
        raise ValueError("length-aware SFT ordering requires one token length per example.")
    if batch_order in ("length", "length_asc"):
        return sorted(order, key=lambda idx: (lengths[idx], idx))
    if batch_order == "length_desc":
        return sorted(order, key=lambda idx: (-lengths[idx], idx))
    if batch_order == "length_bucket":
        bucket_size = int(length_bucket_size or example_count)
        bucket_size = max(1, bucket_size)
        bucketed: list[int] = []
        for start in range(0, example_count, bucket_size):
            bucket = order[start : start + bucket_size]
            bucketed.extend(sorted(bucket, key=lambda idx: (lengths[idx], idx)))
        return bucketed
    raise ValueError(
        "batch_order must be 'shuffle', 'length', 'length_asc', "
        "'length_desc', or 'length_bucket'."
    )


def select_sft_batch_indices(
    example_order: list[int],
    *,
    batch_size: int,
    step: int,
) -> list[int]:
    """Select a deterministic SFT batch from the shuffled traversal."""
    if batch_size <= 0 or not example_order:
        return []
    start = step * batch_size
    size = len(example_order)
    return [example_order[(start + offset) % size] for offset in range(batch_size)]


def _last_assistant_index(messages: list[dict[str, str]]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx]["role"] == "assistant":
            return idx
    raise ValueError("SFT messages must include an assistant target.")


type _ChatTemplate = Callable[..., object]


def _chat_template(tokenizer: object) -> _ChatTemplate | None:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return None
    return apply_chat_template


def _chat_template_kwargs(apply_chat_template: _ChatTemplate) -> dict[str, object]:
    try:
        sig = inspect.signature(apply_chat_template)
    except (TypeError, ValueError):
        return {}
    if "enable_thinking" in sig.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    ):
        return {"enable_thinking": False}
    return {}


def _render_sft_text(tokenizer: object, example: SftExample) -> tuple[str, str]:
    """Return (full_text, prompt_text) for one SFT example."""
    if example.messages is not None:
        apply_chat_template = _chat_template(tokenizer)
        if apply_chat_template is None:
            raise RuntimeError(
                "SFT rows with 'messages' require a tokenizer with apply_chat_template()."
            )
        chat_template_kwargs = _chat_template_kwargs(apply_chat_template)
        target_idx = _last_assistant_index(example.messages)
        full_messages = example.messages[: target_idx + 1]
        prompt_messages = example.messages[:target_idx]
        full_text = apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            **chat_template_kwargs,
        )
        prompt_text = apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        return str(full_text), str(prompt_text)

    if example.prompt or example.completion:
        return f"{example.prompt}{example.completion}", example.prompt

    return example.text, ""


def tokenize_sft_example(
    tokenizer: object,
    example: SftExample,
    *,
    max_tokens: int,
) -> tuple[list[int], list[float]]:
    """Tokenize one SFT datum and build the supervised-token mask."""
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("SFT tokenization requires a tokenizer with encode().")

    full_text, prompt_text = _render_sft_text(tokenizer, example)
    full_tokens = list(encode(full_text, add_special_tokens=False))
    prompt_tokens = list(encode(prompt_text, add_special_tokens=False))

    if max_tokens > 0 and len(full_tokens) > max_tokens:
        full_tokens = full_tokens[:max_tokens]
    n_prompt = min(len(prompt_tokens), len(full_tokens))
    advantages = [0.0] * n_prompt + [1.0] * (len(full_tokens) - n_prompt)
    if not full_tokens:
        raise ValueError("SFT example tokenized to zero tokens.")
    if not any(value > 0.0 for value in advantages):
        raise ValueError("SFT example has no supervised target tokens after truncation.")
    return full_tokens, advantages


def tokenize_sft_batch(
    tokenizer: object,
    examples: list[SftExample],
    *,
    max_tokens: int,
) -> SftTokenizedBatch:
    """Tokenize a batch of SFT examples."""
    tokens: list[list[int]] = []
    advantages: list[list[float]] = []
    supervised_tokens = 0
    total_tokens = 0
    for example in examples:
        row_tokens, row_advantages = tokenize_sft_example(
            tokenizer,
            example,
            max_tokens=max_tokens,
        )
        tokens.append(row_tokens)
        advantages.append(row_advantages)
        total_tokens += len(row_tokens)
        supervised_tokens += sum(1 for value in row_advantages if value > 0.0)
    return SftTokenizedBatch(
        tokens=tokens,
        advantages=advantages,
        total_tokens=total_tokens,
        supervised_tokens=supervised_tokens,
    )


def tokenize_sft_dataset(
    tokenizer: object,
    examples: list[SftExample],
    *,
    max_tokens: int,
) -> list[SftTokenizedExample]:
    """Tokenize an SFT dataset once so batching can use true lengths."""
    tokenized: list[SftTokenizedExample] = []
    for example in examples:
        row_tokens, row_advantages = tokenize_sft_example(
            tokenizer,
            example,
            max_tokens=max_tokens,
        )
        tokenized.append(
            SftTokenizedExample(tokens=row_tokens, advantages=row_advantages)
        )
    return tokenized


def build_sft_tokenized_batch(
    examples: list[SftTokenizedExample],
) -> SftTokenizedBatch:
    """Build a batch view from pre-tokenized examples."""
    tokens = [example.tokens for example in examples]
    advantages = [example.advantages for example in examples]
    return SftTokenizedBatch(
        tokens=tokens,
        advantages=advantages,
        total_tokens=sum(example.total_tokens for example in examples),
        supervised_tokens=sum(example.supervised_tokens for example in examples),
    )


def build_sft_artifact_manifest(
    config: "TrainConfig",
    *,
    policy_ref: str,
    examples_count: int,
    batch_size: int,
    max_tokens: int,
    loss_fn: str,
    latest_metrics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a self-describing manifest for an SFT LoRA adapter."""
    adapter_dir = Path(policy_ref)
    adapter_files: list[str] = []
    if adapter_dir.is_dir():
        adapter_files = sorted(path.name for path in adapter_dir.iterdir() if path.is_file())

    return {
        "schema_version": 1,
        "kind": "retrain_sft_adapter",
        "trainer": "sft",
        "backend": config.backend,
        "base_model": config.model,
        "adapter_path": str(adapter_dir),
        "adapter_root": config.adapter_path,
        "log_dir": config.log_dir,
        "resume": {
            "from": config.log_dir,
            "checkpoint_path": str(adapter_dir),
        },
        "sft": {
            "data_path": config.sft_data_path,
            "examples": int(examples_count),
            "batch_size": int(batch_size),
            "max_tokens": int(max_tokens),
            "max_steps": int(config.max_steps),
            "lr": float(config.sft_lr if config.sft_lr > 0 else config.lr),
            "loss_fn": loss_fn,
            "batch_order": config.sft_batch_order,
            "length_bucket_size": int(config.sft_length_bucket_size),
        },
        "lora": {
            "rank": int(config.lora_rank),
            "alpha": int(config.lora_alpha if config.lora_alpha else config.lora_rank * 2),
            "dropout": float(config.lora_dropout),
        },
        "huggingface": {
            "format": "peft_lora_adapter",
            "adapter_files": adapter_files,
            "load_snippet": (
                "from transformers import AutoModelForCausalLM\n"
                "from peft import PeftModel\n"
                f"base = AutoModelForCausalLM.from_pretrained({config.model!r})\n"
                f"model = PeftModel.from_pretrained(base, {str(adapter_dir)!r})"
            ),
            "publish_hint": (
                "Use `huggingface-cli upload <repo-id> "
                f"{str(adapter_dir)} .` or load the directory with PEFT."
            ),
        },
        "resource_metrics": dict(latest_metrics or {}),
        "ergonomics": {
            "no_rl_rollouts": True,
            "no_reward_model": True,
            "no_environment_bridge": True,
            "resume_into_retrain": True,
        },
    }


def write_sft_artifact_manifest(
    log_dir: str | Path,
    policy_ref: str,
    manifest: Mapping[str, object],
) -> dict[str, str]:
    """Write the SFT manifest beside logs and inside the adapter directory."""
    payload = json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n"
    log_path = Path(log_dir) / "sft_manifest.json"
    log_path.write_text(payload)

    paths = {"log_manifest": str(log_path)}
    adapter_dir = Path(policy_ref)
    if adapter_dir.is_dir():
        adapter_path = adapter_dir / "retrain_sft_manifest.json"
        adapter_path.write_text(payload)
        paths["adapter_manifest"] = str(adapter_path)
    return paths
