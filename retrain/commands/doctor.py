"""`retrain doctor` command and dependency warnings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from retrain.commands.capability import capability_payload, capability_summary

if TYPE_CHECKING:
    from retrain.config import TrainConfig


def run() -> None:
    """Print dependency status for all known components."""
    from retrain.backends.catalog import get_builtin_backend_definitions
    from retrain.registry import check_environment, probe_backend_runtime

    print("retrain doctor — checking component dependencies\n")
    results = check_environment(config=None)
    all_ok = True
    for name, import_name, hint, available in results:
        status = "OK" if available else "MISSING"
        if not available:
            all_ok = False
        print(f"  {name:20s} {import_name:25s} {status}")
        if not available:
            print(f"  {'':20s} -> {hint}")

    print("\nBackend capability summary:")
    for backend_name in sorted(get_builtin_backend_definitions()):
        caps = capability_payload(backend_name, {})
        print(f"  {backend_name:20s} {capability_summary(caps)}")
    plugin_caps = capability_payload("myplugin.CustomBackend", {})
    print(f"  {'plugin/default':20s} {capability_summary(plugin_caps)}")

    print("\nRuntime probes:")
    for probe in probe_backend_runtime(config=None):
        print(
            f"  {probe.backend:20s} {probe.probe:20s} "
            f"{probe.status.upper():5s} {probe.detail}"
        )

    print()
    if all_ok:
        print("All optional dependencies are installed.")
    else:
        print("Some optional dependencies are missing (see above).")


def warn_missing(config: "TrainConfig") -> None:
    """Warn if the config references components whose deps are missing."""
    from retrain.registry import check_environment

    results = check_environment(config=config)
    for name, import_name, hint, available in results:
        if not available:
            print(
                f"WARNING: component '{name}' requires '{import_name}' "
                f"which is not installed.\n  -> {hint}"
            )
