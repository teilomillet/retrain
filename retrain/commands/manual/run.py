"""Manual command dispatch."""

from __future__ import annotations

import json
import sys

from retrain.commands.manual.sync import check_file, sync_file
from retrain.commands.manual.text import ManualPath, load_text
from retrain.commands.manual.topic import TOPIC_TO_SECTION, extract_section


def run(args: list[str], *, cli_name: str, manual_path: ManualPath) -> None:
    """Print manual text or JSON from the bundled editable file."""
    fmt = "text"
    topic: str | None = None
    show_path = False
    list_topics = False
    sync = False
    check = False
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--json":
            fmt = "json"
        elif arg == "--path":
            show_path = True
        elif arg == "--list-topics":
            list_topics = True
        elif arg == "--sync":
            sync = True
        elif arg == "--check":
            check = True
        elif arg in ("--format", "-f"):
            i += 1
            if i >= len(args):
                print("Flag --format requires a value: text|json", file=sys.stderr)
                sys.exit(1)
            fmt = args[i]
        elif arg.startswith("--format="):
            fmt = arg.split("=", 1)[1]
        elif arg in ("--topic", "-t"):
            i += 1
            if i >= len(args):
                print("Flag --topic requires a value.", file=sys.stderr)
                sys.exit(1)
            topic = args[i]
        elif arg.startswith("--topic="):
            topic = arg.split("=", 1)[1]
        else:
            print(f"Unknown man flag: {arg}", file=sys.stderr)
            sys.exit(1)
        i += 1

    if fmt not in ("text", "json", "troff", "html"):
        print(f"Unsupported format '{fmt}'. Use text|json|troff|html.", file=sys.stderr)
        sys.exit(1)

    if sync and check:
        print("Flags --sync and --check cannot be used together.", file=sys.stderr)
        sys.exit(1)

    path = manual_path()
    if sync:
        try:
            path, changed = sync_file(cli_name, manual_path)
        except ValueError as exc:
            print(f"Manual sync failed: {exc}", file=sys.stderr)
            sys.exit(1)
        status = "updated" if changed else "already up to date"
        print(f"{path} ({status})")
        return

    if check:
        try:
            path, up_to_date = check_file(cli_name, manual_path)
        except ValueError as exc:
            print(f"Manual check failed: {exc}", file=sys.stderr)
            sys.exit(1)
        if up_to_date:
            print(f"{path} (up to date)")
            return
        print(
            f"{path} (out of date). Run: {cli_name} man --sync",
            file=sys.stderr,
        )
        sys.exit(1)

    manual_text = load_text(cli_name, manual_path)

    if show_path:
        print(str(path))
        return

    if list_topics:
        for name in sorted(TOPIC_TO_SECTION):
            print(name)
        return

    section_name: str | None = None
    section_text: str | None = None
    if topic is not None:
        section_name = TOPIC_TO_SECTION.get(topic.lower(), topic.upper())
        section_text = extract_section(manual_text, section_name)
        if section_text is None:
            print(
                f"Unknown topic '{topic}'. Available: {sorted(TOPIC_TO_SECTION)}",
                file=sys.stderr,
            )
            sys.exit(1)

    if fmt in ("troff", "html"):
        from retrain.commands.manual.export import parse_manual, to_html, to_troff

        source = section_text if section_text is not None else manual_text
        sections = parse_manual(source)
        formatter = to_troff if fmt == "troff" else to_html
        print(formatter(sections).rstrip())
        return

    if fmt == "json":
        if section_text is None:
            payload = {
                "tool": cli_name,
                "path": str(path),
                "topics": sorted(TOPIC_TO_SECTION),
                "manual": manual_text,
            }
        else:
            payload = {
                "tool": cli_name,
                "path": str(path),
                "topic": topic,
                "section": section_name,
                "content": section_text,
            }
        print(json.dumps(payload, indent=2))
        return

    if section_text is None:
        print(manual_text.rstrip())
    else:
        print(section_text.rstrip())
