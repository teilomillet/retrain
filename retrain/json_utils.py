"""JSON helpers with optional orjson acceleration for hot paths."""

from __future__ import annotations

import json as _json

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover - exercised only without orjson installed
    _orjson = None


JSONDecodeError = (
    _orjson.JSONDecodeError if _orjson is not None else _json.JSONDecodeError
)


def loads(data: str | bytes | bytearray | memoryview) -> object:
    """Parse JSON from text or bytes, preferring orjson when available."""
    if _orjson is not None:
        return _orjson.loads(data)
    if isinstance(data, memoryview):
        data = data.tobytes()
    return _json.loads(data)


def dumps_jsonl(entry: object) -> str:
    """Encode one compact JSONL row, preferring orjson when available."""
    if _orjson is not None:
        return _orjson.dumps(
            entry,
            option=_orjson.OPT_APPEND_NEWLINE,
        ).decode("utf-8")
    return _json.dumps(
        entry,
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"
