"""`retrain resume-check` command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from retrain.training.resume_check import ResumeCheckResult, check_resume_dir


def run(args: list[str]) -> None:
    """Check whether a checkpoint/log directory can be resumed."""
    fmt = "text"
    config_path = ""
    positional: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--json":
            fmt = "json"
        elif arg == "--config":
            i += 1
            if i >= len(args):
                print("--config requires a path", file=sys.stderr)
                sys.exit(1)
            config_path = args[i]
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
        elif arg.startswith("--"):
            print(f"Unknown resume-check flag: {arg}", file=sys.stderr)
            sys.exit(1)
        else:
            positional.append(arg)
        i += 1

    if len(positional) != 1:
        print(
            "Usage: retrain resume-check <log_dir_or_artifact_dir> "
            "[--config config.toml] [--json]",
            file=sys.stderr,
        )
        sys.exit(1)

    config = None
    if config_path:
        from retrain.config import load_config

        if not Path(config_path).is_file():
            print(f"Config not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        config = load_config(config_path)

    result = check_resume_dir(positional[0], config=config, config_path=config_path)
    if fmt == "json":
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(_format_text(result))
    if not result.ok:
        sys.exit(1)


def _format_text(result: ResumeCheckResult) -> str:
    lines = ["retrain resume-check"]
    lines.append(f"  path          : {result.path}")
    lines.append(f"  ok            : {'yes' if result.ok else 'no'}")
    lines.append(
        f"  trainer_state : {'ok' if result.trainer_state_valid else 'missing/invalid'}"
    )
    if result.trainer:
        lines.append(f"  trainer       : {result.trainer}")
    if result.config_source:
        config_detail = result.config_path or result.config_source
        lines.append(f"  config        : {config_detail}")
    if result.step is not None:
        step_text = f"saved={result.step} next={result.next_step}"
        if result.max_steps is not None:
            step_text += f" max_steps={result.max_steps}"
        lines.append(f"  step          : {step_text}")
    if result.checkpoint_path or result.resolved_checkpoint_path:
        lines.append(f"  checkpoint    : {result.checkpoint_path or '(none)'}")
        if (
            result.resolved_checkpoint_path
            and result.resolved_checkpoint_path != result.checkpoint_path
        ):
            lines.append(f"  resolved      : {result.resolved_checkpoint_path}")
        lines.append(f"  payload       : {result.checkpoint_payload}")
    if result.adapter_payload_ok is not None:
        adapter_status = "ok" if result.adapter_payload_ok else "missing"
        if result.adapter_payload_files:
            adapter_status += f" ({', '.join(result.adapter_payload_files)})"
        lines.append(f"  adapter       : {adapter_status}")
    if result.resume_mode:
        lines.append(f"  resume mode   : {result.resume_mode}")
    if result.resume_warning:
        lines.append(f"  resume warning: {result.resume_warning}")
    if result.sft_data:
        recoverable = "yes" if result.sft_data.get("recoverable") else "no"
        lines.append(f"  sft data      : recoverable={recoverable}")

    errors = [issue for issue in result.issues if issue.severity == "error"]
    warnings = [issue for issue in result.issues if issue.severity == "warning"]
    infos = [issue for issue in result.issues if issue.severity == "info"]
    for title, issues in (
        ("Errors", errors),
        ("Warnings", warnings),
        ("Notes", infos),
    ):
        if not issues:
            continue
        lines.append("")
        lines.append(f"{title}:")
        for issue in issues:
            lines.append(f"  - [{issue.code}] {issue.message}")
    return "\n".join(lines)
