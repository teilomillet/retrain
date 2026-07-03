"""`retrain trace` command."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def run(args: list[str]) -> None:
    """Pre-flight validation: build flow, trace with synthetic data."""
    from retrain.config import load_config
    from retrain.flow import build_flow

    config_path: str | None = None
    json_mode = False
    for arg in args:
        if arg == "--json":
            json_mode = True
        elif not arg.startswith("--") and config_path is None:
            config_path = arg
        else:
            print(f"Unknown argument: {arg}", file=sys.stderr)
            sys.exit(1)

    if config_path is not None and not Path(config_path).is_file():
        print(f"File not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(1)

    flow = build_flow(config, gpu=False)
    result = flow.trace()

    if json_mode:
        payload = {
            "ok": result.ok,
            "probe_cases_run": result.probe_cases_run,
            "probe_cases_passed": result.probe_cases_passed,
            "issues": [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "message": issue.message,
                }
                for issue in result.issues
            ],
            "flow": {
                "condition_label": flow.condition_label,
                "backend": config.backend,
                "backend_capability_source": flow.backend_capability_source,
                "needs_planning": flow.needs_planning,
                "uses_sepa_controller": flow.uses_sepa_controller,
                "preserves_token_advantages": flow.backend_capabilities.preserves_token_advantages,
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Condition: {flow.condition_label}")
        print(f"Backend:   {config.backend} (source: {flow.backend_capability_source})")
        print(
            f"Probes:    {result.probe_cases_passed}/{result.probe_cases_run} passed"
        )
        if result.issues:
            print()
            for issue in result.issues:
                tag = "[ERROR]" if issue.severity == "error" else "[WARN]"
                print(f"  {tag} [{issue.category}] {issue.message}")
        print()
        if result.ok:
            print("PASS")
        else:
            print("FAIL")

    sys.exit(0 if result.ok else 1)
