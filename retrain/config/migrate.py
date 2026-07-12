"""Legacy backend config migration."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass

from retrain.config.table import as_object_table


_LEGACY_PRIME_RL_KEYS: dict[str, str] = {
    "prime_rl_transport": "transport",
    "prime_rl_zmq_host": "zmq_host",
    "prime_rl_zmq_port": "zmq_port",
    "prime_rl_zmq_hwm": "zmq_hwm",
    "prime_rl_strict_advantages": "strict_advantages",
    "prime_rl_sync_wait_s": "sync_wait_s",
    "prime_rl_sync_poll_s": "sync_poll_s",
}

_TOML_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$")
_TOML_KEY_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=")


@dataclass(frozen=True)
class BackendConfigMigrationResult:
    """Result payload for legacy [backend] PRIME-RL key migration."""

    changed: bool
    legacy_keys: tuple[str, ...]
    merged_backend_options: dict[str, object]
    output_text: str


def detect_legacy_prime_rl_backend_keys(
    backend_sec: object,
) -> dict[str, object]:
    """Detect legacy PRIME-RL keys still present in [backend]."""
    backend = as_object_table(backend_sec)
    if backend is None:
        return {}
    return {key: backend[key] for key in _LEGACY_PRIME_RL_KEYS if key in backend}


def _toml_literal(value: object) -> str:
    """Render a python value as a TOML literal for migration hints."""
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


def _migration_error_for_legacy_prime_rl_keys(
    backend_sec: dict[str, object],
) -> ValueError:
    """Build a concrete rewrite message for legacy PRIME-RL backend keys."""
    lines = [
        "Legacy PRIME-RL keys are no longer supported in [backend].",
        "Move them under [backend.options].",
        "",
        "Rewrite this section as:",
        "[backend]",
        f"backend = {_toml_literal(backend_sec.get('backend', 'prime_rl'))}",
    ]
    if "devices" in backend_sec:
        lines.append(f"devices = {_toml_literal(backend_sec['devices'])}")
    if "adapter_path" in backend_sec:
        lines.append(f"adapter_path = {_toml_literal(backend_sec['adapter_path'])}")
    lines.append("")
    lines.append("[backend.options]")
    for old_key, new_key in _LEGACY_PRIME_RL_KEYS.items():
        if old_key in backend_sec:
            lines.append(f"{new_key} = {_toml_literal(backend_sec[old_key])}")
    return ValueError("\n".join(lines))


def _section_bounds(lines: list[str], section: str) -> tuple[int | None, int]:
    """Return (start,end) bounds for one top-level TOML section."""
    start: int | None = None
    end = len(lines)
    for idx, line in enumerate(lines):
        m = _TOML_SECTION_RE.match(line)
        if not m:
            continue
        if m.group(1).strip() == section:
            start = idx
            break
    if start is None:
        return None, end
    for idx in range(start + 1, len(lines)):
        if _TOML_SECTION_RE.match(lines[idx]):
            end = idx
            break
    return start, end


def _line_key(line: str) -> str | None:
    m = _TOML_KEY_RE.match(line)
    if not m:
        return None
    return m.group(1)


def _ordered_backend_option_keys(options: dict[str, object]) -> list[str]:
    """Stable key order for synthesized [backend.options] blocks."""
    ordered: list[str] = []
    seen: set[str] = set()
    for new_key in _LEGACY_PRIME_RL_KEYS.values():
        if new_key in options:
            ordered.append(new_key)
            seen.add(new_key)
    for key in sorted(options):
        if key not in seen:
            ordered.append(key)
    return ordered


def migrate_legacy_backend_keys_toml_text(
    toml_text: str,
) -> BackendConfigMigrationResult:
    """Migrate legacy PRIME-RL [backend] keys to [backend.options] textually.

    Existing keys under [backend.options] win over migrated legacy values.
    """
    data = tomllib.loads(toml_text)
    backend_sec = data.get("backend")
    legacy_values = detect_legacy_prime_rl_backend_keys(backend_sec)
    legacy_keys = tuple(sorted(legacy_values))
    if not legacy_values:
        return BackendConfigMigrationResult(
            changed=False,
            legacy_keys=legacy_keys,
            merged_backend_options={},
            output_text=toml_text,
        )

    backend = as_object_table(backend_sec)
    if backend is None:
        return BackendConfigMigrationResult(
            changed=False,
            legacy_keys=legacy_keys,
            merged_backend_options={},
            output_text=toml_text,
        )

    existing_opts_raw = backend.get("options", {})
    if existing_opts_raw is None:
        existing_options: dict[str, object] = {}
    else:
        existing_options = as_object_table(existing_opts_raw)
        if existing_options is None:
            raise ValueError(
                "Invalid [backend].options value. "
                'Use a TOML table, e.g. [backend.options] transport = "filesystem".'
            )
        existing_options = dict(existing_options)

    migrated_options: dict[str, object] = {}
    for old_key, new_key in _LEGACY_PRIME_RL_KEYS.items():
        if old_key in legacy_values:
            migrated_options[new_key] = legacy_values[old_key]

    merged_options = dict(migrated_options)
    merged_options.update(existing_options)

    had_trailing_newline = toml_text.endswith("\n")
    lines = toml_text.splitlines()

    backend_start, backend_end = _section_bounds(lines, "backend")
    if backend_start is None:
        return BackendConfigMigrationResult(
            changed=False,
            legacy_keys=legacy_keys,
            merged_backend_options=merged_options,
            output_text=toml_text,
        )

    options_start, _ = _section_bounds(lines, "backend.options")
    creating_options_section = options_start is None

    stripped_lines: list[str] = []
    for idx, line in enumerate(lines):
        if backend_start < idx < backend_end:
            key = _line_key(line)
            if key in legacy_values:
                continue
            # Avoid duplicate definition conflict when we create [backend.options].
            if creating_options_section and key == "options":
                continue
        stripped_lines.append(line)

    lines = stripped_lines
    backend_start, backend_end = _section_bounds(lines, "backend")
    options_start, options_end = _section_bounds(lines, "backend.options")

    if options_start is not None:
        existing_option_keys = set(existing_options)
        to_add = [
            k
            for k in _ordered_backend_option_keys(migrated_options)
            if k not in existing_option_keys
        ]
        if to_add:
            insertion = options_end
            for key in to_add:
                lines.insert(insertion, f"{key} = {_toml_literal(merged_options[key])}")
                insertion += 1
    elif backend_start is not None and merged_options:
        insertion = backend_end
        block: list[str] = [
            "",
            "[backend.options]",
        ]
        for key in _ordered_backend_option_keys(merged_options):
            block.append(f"{key} = {_toml_literal(merged_options[key])}")
        lines[insertion:insertion] = block

    migrated_text = "\n".join(lines)
    if had_trailing_newline:
        migrated_text += "\n"

    changed = migrated_text != toml_text
    return BackendConfigMigrationResult(
        changed=changed,
        legacy_keys=legacy_keys,
        merged_backend_options=merged_options,
        output_text=migrated_text,
    )


def _extract_backend_options(backend_sec: object) -> dict[str, object] | None:
    """Extract [backend.options] table when present."""
    if backend_sec is None:
        return None
    backend = as_object_table(backend_sec)
    if backend is None:
        return None

    legacy_hits = sorted(detect_legacy_prime_rl_backend_keys(backend))
    if legacy_hits:
        raise _migration_error_for_legacy_prime_rl_keys(backend)

    raw_options = backend.get("options")
    if raw_options is None:
        return None
    options = as_object_table(raw_options)
    if options is None:
        raise ValueError(
            "Invalid [backend].options value. "
            'Use a TOML table, e.g. [backend.options] transport = "filesystem".'
        )
    return dict(options)
