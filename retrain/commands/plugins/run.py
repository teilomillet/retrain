"""`retrain plugins` command."""

from __future__ import annotations

import importlib
import json
import sys


def run(args: list[str]) -> None:
    """List built-in and discovered plugins."""
    from retrain.advantages import (
        get_builtin_algorithm_modes,
        get_builtin_advantage_modes,
        get_builtin_transform_modes,
    )
    from retrain.config import TrainConfig, load_config
    from retrain.plugins.resolve import discover_plugin_modules, get_plugin_runtime
    from retrain.registry.builtin import get_registry

    fmt = "text"
    config_path: str | None = None
    for arg in args:
        if arg == "--json":
            fmt = "json"
        elif arg.startswith("--"):
            print(f"Unknown plugins flag: {arg}", file=sys.stderr)
            sys.exit(1)
        else:
            config_path = arg

    if config_path:
        cfg = load_config(config_path)
    else:
        cfg = TrainConfig()
    runtime = get_plugin_runtime()
    discovered = discover_plugin_modules(cfg.plugins_search_paths)

    discovered_entries: list[dict[str, str]] = []
    for module_name in discovered:
        try:
            importlib.import_module(module_name)
            status = "ok"
        except Exception as exc:
            status = f"error: {exc.__class__.__name__}"
        discovered_entries.append({"module": module_name, "status": status})

    builtins: dict[str, list[str]] = {
        "algorithm_mode": get_builtin_algorithm_modes(),
        "advantage_mode": get_builtin_advantage_modes(),
        "transform_mode": get_builtin_transform_modes(),
        "backend": get_registry("backend").builtin_names,
        "inference_engine": get_registry("inference_engine").builtin_names,
        "reward": get_registry("reward").builtin_names,
        "planning_detector": get_registry("planning_detector").builtin_names,
        "data_source": get_registry("data_source").builtin_names,
        "backpressure": get_registry("backpressure").builtin_names,
    }

    payload: dict[str, object] = {
        "runtime": {
            "search_paths": list(runtime.search_paths),
            "strict": runtime.strict,
        },
        "builtins": builtins,
        "discovered": discovered_entries,
    }

    if fmt == "json":
        print(json.dumps(payload, indent=2))
        return

    print("Plugin runtime:")
    print(f"  search_paths: {', '.join(runtime.search_paths)}")
    print(f"  strict      : {runtime.strict}")
    print("\nBuilt-ins:")
    for key, values in builtins.items():
        print(f"  {key}: {', '.join(values)}")
    print("\nDiscovered modules:")
    if discovered_entries:
        for item in discovered_entries:
            print(f"  {item['module']}: {item['status']}")
    else:
        print("  (none)")
