"""Supervised fine-tuning dataset and tokenization helpers."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, cast

if TYPE_CHECKING:
    from retrain.config import TrainConfig

from retrain.config.snapshot import config_snapshot
from retrain.training.sft_audit import SFT_AUDIT_SCHEMA

SFT_DATA_SNAPSHOT_MAX_BYTES = 16 * 1024 * 1024
SFT_RESUME_SCHEDULE_CONTRACT_VERSION = 2
SFT_RESUME_SCHEDULE_ALGORITHM = "absolute_sample_seed_plus_epoch_v1"


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
class SftDataProvenance:
    """Provenance for the exact SFT JSONL file used by a run."""

    data_path: str
    data_sha256: str
    data_rows: int
    data_bytes: int
    data_path_status: str
    data_root: str = ""
    git_root: str = ""
    git_tracked: bool | None = None
    data_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SftDataset:
    """Loaded SFT examples plus file-level provenance."""

    examples: list[SftExample]
    provenance: SftDataProvenance


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
        return _loss_bearing_supervised_tokens(self.advantages)


def _loss_bearing_supervised_tokens(advantages: list[float]) -> int:
    """Count supervised targets that survive the causal one-token shift."""

    return sum(1 for value in advantages[1:] if value > 0.0)


def build_sft_batch_metrics(
    batch: SftTokenizedBatch,
) -> dict[str, int | float]:
    """Describe logical-batch sequence shape without changing SFT tensors.

    These metrics are backend-independent. Local-backend telemetry separately
    reports the padding actually materialized after microbatching.
    """

    lengths = [len(row) for row in batch.tokens]
    if not lengths:
        return {
            "sft_sequence_length_min": 0,
            "sft_sequence_length_mean": 0.0,
            "sft_sequence_length_p50": 0,
            "sft_sequence_length_p95": 0,
            "sft_sequence_length_max": 0,
            "sft_logical_padded_tokens": 0,
            "sft_logical_padding_tokens": 0,
            "sft_logical_padding_fraction": 0.0,
            "sft_supervised_token_fraction": 0.0,
        }

    sorted_lengths = sorted(lengths)
    count = len(sorted_lengths)

    def nearest_rank(percent: int) -> int:
        index = max(0, (percent * count + 99) // 100 - 1)
        return sorted_lengths[index]

    padded_tokens = count * sorted_lengths[-1]
    padding_tokens = max(0, padded_tokens - batch.total_tokens)
    return {
        "sft_sequence_length_min": sorted_lengths[0],
        "sft_sequence_length_mean": batch.total_tokens / count,
        "sft_sequence_length_p50": nearest_rank(50),
        "sft_sequence_length_p95": nearest_rank(95),
        "sft_sequence_length_max": sorted_lengths[-1],
        "sft_logical_padded_tokens": padded_tokens,
        "sft_logical_padding_tokens": padding_tokens,
        "sft_logical_padding_fraction": (
            padding_tokens / padded_tokens if padded_tokens else 0.0
        ),
        "sft_supervised_token_fraction": (
            batch.supervised_tokens / batch.total_tokens
            if batch.total_tokens
            else 0.0
        ),
    }


def effective_sft_loss_fn(config: "TrainConfig") -> str:
    """Resolve SFT loss without changing legacy warmup defaults."""

    if config.sft_loss_fn != "auto":
        return config.sft_loss_fn
    if config.trainer == "sft":
        return "cross_entropy"
    return "importance_sampling"


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


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _named_data_root(path: Path) -> Path | None:
    for parent in (path.parent, *path.parents):
        if parent.name in {"data", "datasets"}:
            return parent
    return None


def _is_scratch_path(path: Path) -> bool:
    scratch_roots = {
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp").resolve(),
        Path("/private/tmp").resolve(),
    }
    return any(_is_relative_to(path, root) for root in scratch_roots)


def _git_tracking(path: Path) -> tuple[str, bool | None]:
    try:
        root_proc = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "", None
    if root_proc.returncode != 0:
        return "", None

    git_root = Path(root_proc.stdout.strip()).resolve()
    try:
        relative_path = path.relative_to(git_root)
    except ValueError:
        return str(git_root), None

    tracked_proc = subprocess.run(
        ["git", "-C", str(git_root), "ls-files", "--error-unmatch", str(relative_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return str(git_root), tracked_proc.returncode == 0


def _build_sft_data_provenance(
    path: Path,
    *,
    raw_bytes: bytes,
    data_rows: int,
) -> SftDataProvenance:
    data_root = _named_data_root(path)
    git_root, git_tracked = _git_tracking(path)
    scratch = _is_scratch_path(path)
    if scratch:
        status = "scratch"
    elif data_root is None:
        status = "outside_data_root"
    else:
        status = "data_root"

    warnings: list[str] = []
    if scratch:
        warnings.append(
            f"SFT data resolves under scratch/tmp ({path}); move durable training "
            "data under a tracked data/ or datasets/ root."
        )
    if data_root is None:
        warnings.append(
            f"SFT data resolves outside a data/ or datasets/ root ({path})."
        )
    if git_tracked is None:
        warnings.append(
            f"SFT data git tracking could not be verified for {path}; "
            "keep durable training data in a git checkout when possible."
        )
    if git_tracked is False:
        warnings.append(
            f"SFT data is not tracked by git in {git_root}: {path}."
        )

    return SftDataProvenance(
        data_path=str(path),
        data_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        data_rows=int(data_rows),
        data_bytes=len(raw_bytes),
        data_path_status=status,
        data_root=str(data_root) if data_root else "",
        git_root=git_root,
        git_tracked=git_tracked,
        data_warnings=tuple(warnings),
    )


def _load_sft_jsonl_rows(path: Path, text: str) -> list[SftExample]:
    examples: list[SftExample] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _row_error(path, line_no, f"invalid JSON: {exc.msg}") from exc
        if not isinstance(row, Mapping):
            raise _row_error(path, line_no, "row must be a JSON object.")
        row_map = cast(Mapping[str, object], row)

        if "messages" in row_map:
            examples.append(
                SftExample(
                    messages=_coerce_messages(
                        row_map["messages"],
                        path=path,
                        line_no=line_no,
                    )
                )
            )
            continue

        if "prompt" in row_map or "completion" in row_map:
            prompt = row_map.get("prompt")
            completion = row_map.get("completion")
            if not isinstance(prompt, str):
                raise _row_error(path, line_no, "'prompt' must be a string.")
            if not isinstance(completion, str) or not completion:
                raise _row_error(
                    path,
                    line_no,
                    "'completion' must be a non-empty string.",
                )
            examples.append(SftExample(prompt=prompt, completion=completion))
            continue

        if "text" in row_map:
            text_value = row_map.get("text")
            if not isinstance(text_value, str) or not text_value:
                raise _row_error(path, line_no, "'text' must be a non-empty string.")
            examples.append(SftExample(text=text_value))
            continue

        raise _row_error(
            path,
            line_no,
            "expected one of 'messages', 'prompt'+'completion', or 'text'.",
        )
    return examples


def load_sft_dataset(path: str | Path) -> SftDataset:
    """Load SFT JSONL rows and provenance from the exact parsed bytes."""

    sft_path = Path(path).expanduser().resolve(strict=True)
    raw_bytes = sft_path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid SFT JSONL file {sft_path}: expected UTF-8 text.") from exc
    examples = _load_sft_jsonl_rows(sft_path, text)
    return SftDataset(
        examples=examples,
        provenance=_build_sft_data_provenance(
            sft_path,
            raw_bytes=raw_bytes,
            data_rows=len(examples),
        ),
    )


def verify_sft_data_contract(
    config: "TrainConfig",
    provenance: SftDataProvenance,
) -> dict[str, object] | None:
    """Fail fast when configured SFT data pins do not match loaded data."""

    errors: list[str] = []
    expected_sha = str(getattr(config, "sft_data_sha256", "")).strip().lower()
    if expected_sha and expected_sha != provenance.data_sha256:
        errors.append(
            "sft_data_sha256 mismatch: "
            f"expected {expected_sha}, got {provenance.data_sha256} "
            f"for {provenance.data_path}"
        )

    expected_rows = int(getattr(config, "sft_data_rows", 0) or 0)
    if expected_rows > 0 and expected_rows != provenance.data_rows:
        errors.append(
            "sft_data_rows mismatch: "
            f"expected {expected_rows}, got {provenance.data_rows} "
            f"for {provenance.data_path}"
        )

    if errors:
        raise ValueError("SFT data contract mismatch:\n- " + "\n- ".join(errors))

    from retrain.training.sft_audit import verify_sft_audit_contract

    return verify_sft_audit_contract(config, provenance)


def build_sft_resume_schedule_contract(
    config: "TrainConfig",
    provenance: SftDataProvenance,
    *,
    batch_size: int,
    max_tokens: int,
    example_order: list[int],
) -> dict[str, object]:
    """Bind every input that can change standalone-SFT example traversal.

    The epoch-zero order fingerprint also binds tokenizer-derived lengths for
    length-aware policies. Dataset paths are intentionally excluded so an
    identical, hash-pinned dataset remains relocatable.
    """

    configured_bucket_size = int(config.sft_length_bucket_size)
    effective_bucket_size = max(
        1,
        configured_bucket_size or len(example_order),
    )
    return {
        "version": SFT_RESUME_SCHEDULE_CONTRACT_VERSION,
        "algorithm": SFT_RESUME_SCHEDULE_ALGORITHM,
        "seed": int(config.seed),
        "batch_size": int(batch_size),
        "batch_order": str(config.sft_batch_order),
        "length_bucket_size": configured_bucket_size,
        "effective_length_bucket_size": effective_bucket_size,
        "reshuffle_each_epoch": bool(config.sft_reshuffle_each_epoch),
        "data_sha256": provenance.data_sha256,
        "data_rows": int(provenance.data_rows),
        "audit_sha256": config.sft_audit_sha256.strip().lower(),
        "audit_schema": (
            SFT_AUDIT_SCHEMA if config.sft_audit_sha256 else ""
        ),
        "sft_warmup_steps": int(config.sft_warmup_steps),
        "example_count": len(example_order),
        "model": str(config.model),
        "max_tokens": int(max_tokens),
        "epoch_zero_order_sha256": sft_indices_sha256(example_order),
    }


def verify_sft_resume_schedule_contract(
    saved: Mapping[str, object] | None,
    current: Mapping[str, object],
) -> None:
    """Fail closed when checkpoint continuation would change SFT traversal."""

    if saved is None:
        raise ValueError(
            "SFT resume schedule contract is missing from trainer_state.json; "
            "refusing checkpoint continuation because the original traversal "
            "cannot be verified. Restart SFT, or use the checkpoint adapter "
            "directly as step-0 initialization instead."
        )

    errors: list[str] = []
    for key, expected in current.items():
        if key not in saved:
            errors.append(f"{key}: missing (expected {expected!r})")
            continue
        actual = saved[key]
        if type(actual) is not type(expected) or actual != expected:
            errors.append(f"{key}: saved {actual!r}, current {expected!r}")
    if errors:
        raise ValueError(
            "SFT resume schedule contract mismatch; refusing to change example "
            "traversal:\n- "
            + "\n- ".join(errors)
        )


def load_sft_jsonl(path: str | Path) -> list[SftExample]:
    """Load supported SFT JSONL rows with line-numbered validation errors."""

    return load_sft_dataset(path).examples


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


def build_sft_epoch_order(
    example_order: list[int],
    *,
    epoch: int,
    seed: int,
    lengths: list[int] | None = None,
    batch_order: str = "shuffle",
    length_bucket_size: int = 0,
    reshuffle_each_epoch: bool = False,
    epoch_order_cache: dict[int, list[int]] | None = None,
) -> list[int]:
    """Resolve one deterministic epoch order, optionally through a cache."""
    if epoch < 0:
        raise ValueError("SFT epoch must be >= 0.")
    if epoch == 0 or not reshuffle_each_epoch:
        return example_order
    if epoch_order_cache is not None and epoch in epoch_order_cache:
        return epoch_order_cache[epoch]
    order = build_sft_example_order(
        len(example_order),
        seed + epoch,
        lengths=lengths,
        batch_order=batch_order,
        length_bucket_size=length_bucket_size,
    )
    if epoch_order_cache is not None:
        epoch_order_cache[epoch] = order
    return order


def select_sft_batch_indices(
    example_order: list[int],
    *,
    batch_size: int,
    step: int,
    seed: int = 0,
    lengths: list[int] | None = None,
    batch_order: str = "shuffle",
    length_bucket_size: int = 0,
    reshuffle_each_epoch: bool = False,
    epoch_order_cache: dict[int, list[int]] | None = None,
) -> list[int]:
    """Select a deterministic SFT batch by absolute sample position.

    With ``reshuffle_each_epoch=False`` this exactly preserves the historical
    fixed-permutation cycle. When enabled, epoch zero uses ``example_order``
    and each later epoch rebuilds the same ordering policy with ``seed +
    epoch``. Deriving the epoch and in-epoch offset from ``step * batch_size``
    makes a resumed step select the same examples without serialized RNG
    state, including batches that cross a non-divisible epoch boundary.
    """
    if batch_size <= 0 or not example_order:
        return []
    if step < 0:
        raise ValueError("SFT batch step must be >= 0.")
    start = step * batch_size
    size = len(example_order)
    epoch_orders = epoch_order_cache if epoch_order_cache is not None else {}
    epoch_orders.setdefault(0, example_order)
    indices: list[int] = []
    for absolute_position in range(start, start + batch_size):
        epoch, epoch_offset = divmod(absolute_position, size)
        order = build_sft_epoch_order(
            example_order,
            epoch=epoch,
            seed=seed,
            lengths=lengths,
            batch_order=batch_order,
            length_bucket_size=length_bucket_size,
            reshuffle_each_epoch=reshuffle_each_epoch,
            epoch_order_cache=epoch_orders,
        )
        indices.append(order[epoch_offset])
    if epoch_order_cache is not None:
        start_epoch = start // size
        end_epoch = (start + batch_size - 1) // size
        keep_epochs = {0, start_epoch, end_epoch}
        for cached_epoch in tuple(epoch_orders):
            if cached_epoch not in keep_epochs:
                del epoch_orders[cached_epoch]
    return indices


def describe_sft_batch_position(
    example_count: int,
    *,
    batch_size: int,
    step: int,
) -> dict[str, int]:
    """Describe the absolute epoch span of one deterministic SFT batch."""
    if example_count <= 0 or batch_size <= 0:
        return {
            "sft_epoch": 0,
            "sft_epoch_end": 0,
            "sft_epoch_sample_offset": 0,
            "sft_absolute_sample": 0,
        }
    if step < 0:
        raise ValueError("SFT batch step must be >= 0.")
    start = step * batch_size
    end = start + batch_size - 1
    epoch, epoch_offset = divmod(start, example_count)
    return {
        "sft_epoch": epoch,
        "sft_epoch_end": end // example_count,
        "sft_epoch_sample_offset": epoch_offset,
        "sft_absolute_sample": start,
    }


def sft_indices_sha256(indices: list[int]) -> str:
    """Hash an exact index sequence as concatenated unsigned 64-bit big endian."""
    digest = hashlib.sha256()
    for index in indices:
        if index < 0 or index >= 1 << 64:
            raise ValueError("SFT schedule indices must fit unsigned 64-bit encoding.")
        digest.update(index.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def build_sft_schedule_metrics(
    example_order: list[int],
    selected_indices: list[int],
    *,
    batch_size: int,
    step: int,
    seed: int,
    lengths: list[int] | None = None,
    batch_order: str = "shuffle",
    length_bucket_size: int = 0,
    reshuffle_each_epoch: bool = False,
    epoch_order_cache: dict[int, list[int]] | None = None,
    epoch_order_sha256_cache: dict[int, str] | None = None,
) -> dict[str, int | str]:
    """Return reconstructable position and SHA256 evidence for one SFT batch."""
    position = describe_sft_batch_position(
        len(example_order),
        batch_size=batch_size,
        step=step,
    )
    start_epoch = position["sft_epoch"]
    end_epoch = position["sft_epoch_end"]
    start_order = build_sft_epoch_order(
        example_order,
        epoch=start_epoch,
        seed=seed,
        lengths=lengths,
        batch_order=batch_order,
        length_bucket_size=length_bucket_size,
        reshuffle_each_epoch=reshuffle_each_epoch,
        epoch_order_cache=epoch_order_cache,
    )
    start_order_sha256 = None
    if epoch_order_sha256_cache is not None:
        start_order_sha256 = epoch_order_sha256_cache.get(start_epoch)
    if start_order_sha256 is None:
        start_order_sha256 = sft_indices_sha256(start_order)
        if epoch_order_sha256_cache is not None:
            epoch_order_sha256_cache[start_epoch] = start_order_sha256
    metrics: dict[str, int | str] = {
        **position,
        "sft_epoch_seed": seed + start_epoch if reshuffle_each_epoch else seed,
        "sft_epoch_end_seed": seed + end_epoch if reshuffle_each_epoch else seed,
        "sft_batch_indices_sha256": sft_indices_sha256(selected_indices),
        "sft_epoch_start_order_sha256": start_order_sha256,
    }
    if end_epoch != start_epoch:
        end_order = build_sft_epoch_order(
            example_order,
            epoch=end_epoch,
            seed=seed,
            lengths=lengths,
            batch_order=batch_order,
            length_bucket_size=length_bucket_size,
            reshuffle_each_epoch=reshuffle_each_epoch,
            epoch_order_cache=epoch_order_cache,
        )
        end_order_sha256 = None
        if epoch_order_sha256_cache is not None:
            end_order_sha256 = epoch_order_sha256_cache.get(end_epoch)
        if end_order_sha256 is None:
            end_order_sha256 = sft_indices_sha256(end_order)
            if epoch_order_sha256_cache is not None:
                epoch_order_sha256_cache[end_epoch] = end_order_sha256
        metrics["sft_epoch_end_order_sha256"] = end_order_sha256
    if epoch_order_sha256_cache is not None:
        keep_epochs = {0, start_epoch, end_epoch}
        for cached_epoch in tuple(epoch_order_sha256_cache):
            if cached_epoch not in keep_epochs:
                del epoch_order_sha256_cache[cached_epoch]
    return metrics


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

    if example.messages is None and (example.prompt or example.completion):
        # Tokenize the two fields independently. BPE tokenization is not
        # prefix-stable: encoding ``prompt + completion`` can merge across the
        # boundary and make a completion token look like prompt context.
        prompt_tokens = list(encode(example.prompt, add_special_tokens=False))
        completion_tokens = list(
            encode(example.completion, add_special_tokens=False)
        )
        full_tokens = prompt_tokens + completion_tokens
        n_prompt = len(prompt_tokens)
    else:
        full_text, prompt_text = _render_sft_text(tokenizer, example)
        full_tokens = list(encode(full_text, add_special_tokens=False))
        prompt_tokens = list(encode(prompt_text, add_special_tokens=False))
        n_prompt = min(len(prompt_tokens), len(full_tokens))
    if max_tokens > 0 and len(full_tokens) > max_tokens:
        crop_start = len(full_tokens) - max_tokens
        full_tokens = full_tokens[crop_start:]
        n_prompt = max(0, n_prompt - crop_start)
    advantages = [0.0] * n_prompt + [1.0] * (len(full_tokens) - n_prompt)
    if not full_tokens:
        raise ValueError("SFT example tokenized to zero tokens.")
    # Causal LM loss predicts token i from token i - 1, so position zero can
    # never contribute loss. A suffix crop that lands inside the target must
    # reserve its first retained token as causal context instead of reporting
    # it as supervised work that the backend silently shifts away.
    advantages[0] = 0.0
    if _loss_bearing_supervised_tokens(advantages) == 0:
        raise ValueError(
            "SFT example has no loss-bearing supervised target tokens after "
            "causal shifting/truncation; retain at least two tokens (for "
            "example, set sft_max_tokens >= 2)."
        )
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
        supervised_tokens += _loss_bearing_supervised_tokens(row_advantages)
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
    zero_loss_rows = [
        index for index, example in enumerate(examples)
        if example.supervised_tokens == 0
    ]
    if zero_loss_rows:
        raise ValueError(
            "SFT tokenized batch contains rows with no loss-bearing supervised "
            f"tokens after causal shifting: {zero_loss_rows}."
        )
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
    data_provenance: SftDataProvenance | None = None,
    snapshot_artifacts: Mapping[str, str] | None = None,
    latest_metrics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a self-describing manifest for an SFT LoRA adapter."""
    adapter_dir = Path(policy_ref)
    adapter_files: list[str] = []
    if adapter_dir.is_dir():
        adapter_files = sorted(path.name for path in adapter_dir.iterdir() if path.is_file())

    sft_payload: dict[str, object] = {
        "data_path": config.sft_data_path,
        "examples": int(examples_count),
        "batch_size": int(batch_size),
        "max_tokens": int(max_tokens),
        "max_steps": int(config.max_steps),
        "lr": float(config.sft_lr if config.sft_lr > 0 else config.lr),
        "loss_fn": loss_fn,
        "batch_order": config.sft_batch_order,
        "length_bucket_size": int(config.sft_length_bucket_size),
        "reshuffle_each_epoch": bool(config.sft_reshuffle_each_epoch),
        "seed": int(config.seed),
        "epoch_seed_rule": (
            "seed_plus_epoch" if config.sft_reshuffle_each_epoch else "fixed_seed"
        ),
        "schedule_hash_algorithm": "sha256",
        "schedule_hash_encoding": "uint64_be_concatenation",
    }
    if data_provenance is not None:
        sft_payload.update(
            {
                "configured_data_path": config.sft_data_path,
                "data_path": data_provenance.data_path,
                "data_sha256": data_provenance.data_sha256,
                "data_rows": data_provenance.data_rows,
                "data_bytes": data_provenance.data_bytes,
                "data_path_status": data_provenance.data_path_status,
                "data_root": data_provenance.data_root,
                "git_root": data_provenance.git_root,
                "git_tracked": data_provenance.git_tracked,
                "data_warnings": list(data_provenance.data_warnings),
            }
        )
    if config.sft_audit_path:
        sft_payload.update(
            {
                "audit_path": config.sft_audit_path,
                "audit_sha256": config.sft_audit_sha256.strip().lower(),
                "audit_schema": SFT_AUDIT_SCHEMA,
            }
        )

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
        "sft": sft_payload,
        "reproducibility": {
            "artifacts": dict(snapshot_artifacts or {}),
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


def write_sft_run_snapshot_artifacts(
    log_dir: str | Path,
    config: "TrainConfig",
    provenance: SftDataProvenance,
    *,
    snapshot_max_bytes: int = SFT_DATA_SNAPSHOT_MAX_BYTES,
) -> dict[str, str]:
    """Write reproducibility artifacts beside an SFT run's logs."""

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    config_path = log_path / "resolved_config.json"
    config_payload = {
        "schema_version": 1,
        "kind": "retrain_resolved_config",
        "config": config_snapshot(config),
    }
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True, default=str) + "\n")
    paths["resolved_config.json"] = str(config_path)

    source_path = Path(provenance.data_path)
    snapshot_path = log_path / "sft_data.snapshot.jsonl"
    copied = False
    reason = ""
    if provenance.data_bytes <= snapshot_max_bytes:
        if source_path.resolve() != snapshot_path.resolve():
            shutil.copyfile(source_path, snapshot_path)
        copied = True
        copied_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        if copied_sha256 != provenance.data_sha256:
            raise RuntimeError(
                "SFT data snapshot hash mismatch: "
                f"expected {provenance.data_sha256}, got {copied_sha256}"
            )
        paths["sft_data.snapshot.jsonl"] = str(snapshot_path)
    else:
        reason = (
            f"data_bytes {provenance.data_bytes} exceeds snapshot_max_bytes "
            f"{snapshot_max_bytes}"
        )

    recoverability_path = log_path / "sft_data_recoverability.json"
    recoverability = {
        "schema_version": 1,
        "kind": "retrain_sft_data_recoverability",
        "source_path": provenance.data_path,
        "source_sha256": provenance.data_sha256,
        "source_rows": provenance.data_rows,
        "source_bytes": provenance.data_bytes,
        "source_path_status": provenance.data_path_status,
        "source_git_root": provenance.git_root,
        "source_git_tracked": provenance.git_tracked,
        "snapshot_max_bytes": int(snapshot_max_bytes),
        "copied": copied,
        "snapshot_path": str(snapshot_path) if copied else "",
        "recoverable": copied,
        "reason": reason,
    }
    recoverability_path.write_text(json.dumps(recoverability, indent=2, sort_keys=True) + "\n")
    paths["sft_data_recoverability.json"] = str(recoverability_path)
    return paths
