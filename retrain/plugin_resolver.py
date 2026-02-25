"""Shared plugin resolution utilities.

This module centralizes dotted-path plugin loading for all pluggable
components (algorithm/advantage/transform + registries).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PluginRuntimeConfig:
    """Runtime plugin loading settings."""

    search_paths: tuple[str, ...] = ("plugins",)
    strict: bool = True


@dataclass(frozen=True)
class PluginResolution:
    """Resolved plugin payload with source metadata."""

    target: str
    source: str
    resolved_module: str
    attr_name: str
    obj: object


_runtime_config = PluginRuntimeConfig()
_resolution_cache: dict[tuple[PluginRuntimeConfig, str], PluginResolution] = {}


def set_plugin_runtime(search_paths: list[str], strict: bool) -> None:
    """Update global plugin loading behavior and clear stale cache."""
    normalized: list[str] = []
    for path in search_paths:
        p = path.strip()
        if not p:
            continue
        normalized.append(p.strip("."))
    if not normalized:
        normalized = ["plugins"]

    global _runtime_config
    _runtime_config = PluginRuntimeConfig(
        search_paths=tuple(dict.fromkeys(normalized)),
        strict=bool(strict),
    )
    _resolution_cache.clear()


def get_plugin_runtime() -> PluginRuntimeConfig:
    """Return active plugin loading settings."""
    return _runtime_config


def _parse_dotted_target(target: str) -> tuple[str, str]:
    module_path, sep, attr_name = target.rpartition(".")
    if not sep or not module_path or not attr_name:
        raise ValueError(
            f"Invalid plugin target '{target}'. Expected dotted path "
            "(example: 'my_plugins.my_transform')."
        )
    return module_path, attr_name


def _candidate_modules(module_path: str) -> list[str]:
    """Return module import candidates, preferring configured search paths."""
    cfg = get_plugin_runtime()
    candidates: list[str] = []
    for prefix in cfg.search_paths:
        if prefix:
            candidates.append(f"{prefix}.{module_path}")
    candidates.append(module_path)
    # stable unique
    return list(dict.fromkeys(candidates))


def resolve_dotted_attribute(
    target: str,
    *,
    selector: str,
    expected: str,
) -> PluginResolution:
    """Resolve a dotted target to an attribute with standardized errors."""
    cfg = get_plugin_runtime()
    cache_key = (cfg, target)
    cached = _resolution_cache.get(cache_key)
    if cached is not None:
        return cached

    module_path, attr_name = _parse_dotted_target(target)
    candidates = _candidate_modules(module_path)

    module = None
    imported_name = ""
    import_errors: list[str] = []
    for candidate in candidates:
        try:
            module = importlib.import_module(candidate)
            imported_name = candidate
            break
        except ModuleNotFoundError as exc:
            import_errors.append(f"{candidate}: {exc}")

    if module is None:
        raise ImportError(
            f"Failed to resolve {selector} plugin '{target}'. "
            f"Could not import module candidates: {candidates}. "
            f"Expected {expected}. "
            "Fix: place your file under one configured plugin search path "
            "(default: ./plugins) or use a fully importable module path."
        )

    obj = getattr(module, attr_name, None)
    if obj is None:
        raise AttributeError(
            f"Failed to resolve {selector} plugin '{target}'. "
            f"Module '{imported_name}' has no attribute '{attr_name}'. "
            f"Expected {expected}. "
            f"Fix: export '{attr_name}' from {imported_name}."
        )

    source = "plugin/dotted"
    if imported_name != module_path:
        source = f"plugin/path:{imported_name}"

    resolved = PluginResolution(
        target=target,
        source=source,
        resolved_module=imported_name,
        attr_name=attr_name,
        obj=obj,
    )
    _resolution_cache[cache_key] = resolved
    return resolved


def discover_plugin_modules(search_paths: list[str]) -> list[str]:
    """Best-effort discovery of plugin modules in configured search paths."""
    modules: list[str] = []
    for raw_path in search_paths:
        path = raw_path.strip().strip("/")
        if not path:
            continue
        try:
            import pathlib

            root = pathlib.Path(path)
            if not root.is_dir():
                continue
            for py_file in sorted(root.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue
                modules.append(f"{path.replace('/', '.')}.{py_file.stem}")
        except OSError:
            continue
    return sorted(dict.fromkeys(modules))
