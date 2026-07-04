"""`retrain backends` command."""

from __future__ import annotations

import json
import sys
from typing import cast


def _object_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Invalid {name}: expected object mapping")
    return cast(dict[str, object], value)


def _object_list(value: object, name: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"Invalid {name}: expected list")
    rows: list[dict[str, object]] = []
    for idx, item in enumerate(value):
        rows.append(_object_dict(item, f"{name}[{idx}]"))
    return rows


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Invalid {name}: expected list of strings")
    return cast(list[str], value)


def run(args: list[str]) -> None:
    """Print backend metadata catalog."""
    from retrain.backends.catalog import describe_backends_catalog

    fmt = "text"
    for arg in args:
        if arg == "--json":
            fmt = "json"
        elif arg.startswith("--"):
            print(f"Unknown backends flag: {arg}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"Unexpected argument for backends: {arg}", file=sys.stderr)
            sys.exit(1)

    payload = describe_backends_catalog()
    if fmt == "json":
        print(json.dumps(payload, indent=2))
        return

    print("Built-in backends:")
    for backend_item in _object_list(payload.get("builtins"), "backends.builtins"):
        name = str(backend_item.get("name", ""))
        dep = _object_dict(backend_item.get("dependency"), "backend.dependency")
        caps = _object_dict(backend_item.get("capabilities"), "backend.capabilities")
        print(f"  {name}")
        print(f"    dependency: {dep.get('import', '')} ({dep.get('hint', '')})")
        print(
            "    capabilities: "
            f"reports_sync_loss={caps.get('reports_sync_loss')}, "
            f"preserves_token_advantages={caps.get('preserves_token_advantages')}, "
            f"supports_checkpoint_resume={caps.get('supports_checkpoint_resume')}, "
            f"resume_runtime_dependent={caps.get('resume_runtime_dependent')}, "
            f"checkpoint_resume_mode={caps.get('checkpoint_resume_mode')}, "
            f"supports_echo_shared_forward={caps.get('supports_echo_shared_forward')}"
        )
        option_schema = _object_dict(
            backend_item.get("option_schema"),
            "backend.option_schema",
        )
        if option_schema:
            print("    options:")
            for key, raw_spec in sorted(option_schema.items()):
                spec = _object_dict(raw_spec, f"backend.option_schema.{key}")
                choices = spec.get("choices")
                choice_text = f" choices={choices}" if choices else ""
                print(
                    f"      {key}: type={spec.get('type')} "
                    f"default={spec.get('default')!r}{choice_text}"
                )
        else:
            print("    options: none")

    plugin = _object_dict(payload.get("plugin"), "backends.plugin")
    print("\nPlugin metadata hooks:")
    print(f"  dotted_path_supported: {plugin.get('dotted_path_supported')}")
    print(
        "  capability_hooks     : "
        f"{', '.join(_string_list(plugin.get('capability_hooks'), 'plugin.capability_hooks'))}"
    )
    print(
        "  option_schema_hooks  : "
        f"{', '.join(_string_list(plugin.get('option_schema_hooks'), 'plugin.option_schema_hooks'))}"
    )
    print(f"  schema_format        : {plugin.get('option_schema_format', '')}")
