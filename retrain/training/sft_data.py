"""SFT dataset loading, provenance, and verified tokenizer configuration."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

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
            raise _row_error(
                path, line_no, f"messages[{idx}].content must be a string."
            )
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
        warnings.append(f"SFT data is not tracked by git in {git_root}: {path}.")

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
        raise ValueError(
            f"Invalid SFT JSONL file {sft_path}: expected UTF-8 text."
        ) from exc
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
    from retrain.training.sft_token_audit import (
        verify_sft_token_audit_contract,
    )

    verify_sft_token_audit_contract(config, provenance)
    return verify_sft_audit_contract(config, provenance)


def sft_tokenizer_load_kwargs(
    config: "TrainConfig",
    provenance: SftDataProvenance | None = None,
) -> dict[str, object]:
    """Return tokenizer kwargs bound to the verified SFT token audit."""

    kwargs: dict[str, object] = {"trust_remote_code": True}
    configured_revision = str(config.model_revision).strip()
    if configured_revision:
        kwargs["revision"] = configured_revision
    if config.model_local_files_only:
        kwargs["local_files_only"] = True
    if not config.sft_token_audit_path:
        return kwargs
    if provenance is None:
        if not config.sft_data_path:
            raise ValueError(
                "sft_token_audit_path requires sft_data_path before tokenizer load"
            )
        provenance = load_sft_dataset(config.sft_data_path).provenance

    from retrain.training.sft_token_audit import (
        verify_sft_token_audit_contract,
    )

    verified = verify_sft_token_audit_contract(config, provenance)
    if verified is None:
        raise RuntimeError("configured SFT token audit was not verified")
    revision = verified.get("tokenizer_revision")
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("verified SFT token audit has no tokenizer revision")
    if configured_revision != revision:
        raise RuntimeError(
            "verified SFT token audit revision does not match [model] revision"
        )
    if not config.model_local_files_only:
        raise RuntimeError(
            "verified SFT token audit requires [model] local_files_only=true"
        )
    return kwargs


def load_sft_jsonl(path: str | Path) -> list[SftExample]:
    """Load supported SFT JSONL rows with line-numbered validation errors."""

    return load_sft_dataset(path).examples
