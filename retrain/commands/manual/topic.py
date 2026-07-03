"""Manual topics and section lookup."""

from __future__ import annotations

TOPIC_TO_SECTION = {
    "quickstart": "QUICKSTART",
    "environment": "ENVIRONMENT",
    "environments": "ENVIRONMENT",
    "troubleshooting": "TROUBLESHOOTING",
    "commands": "COMMANDS",
    "options": "OPTIONS",
    "configuration": "CONFIGURATION",
    "campaign": "CAMPAIGN MODE",
    "squeeze": "SQUEEZE MODE",
    "inference": "INFERENCE ENGINES",
    "validation": "VALIDATION",
    "diff": "DIFF MODE",
    "logging": "LOGGING",
    "examples": "EXAMPLES",
    "advantages": "ADVANTAGE PIPELINE",
    "metrics": "METRICS GUIDE",
    "benchmark": "BENCHMARKING",
    "benchmarking": "BENCHMARKING",
    "capacity": "CAPACITY PLANNING",
    "capacity-planning": "CAPACITY PLANNING",
    "architecture": "ARCHITECTURE",
    "files": "FILES",
    "plugins": "PLUGINS",
    "glossary": "GLOSSARY",
    "tree": "TREE MODE",
}


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or s != line:
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-()")
    return all(ch in allowed for ch in s) and any(ch.isalpha() for ch in s)


def extract_section(manual_text: str, heading: str) -> str | None:
    """Extract one heading section from manual text."""
    wanted = heading.strip().upper()
    lines = manual_text.splitlines()

    start = None
    for i, line in enumerate(lines):
        if is_heading(line) and line.strip().upper() == wanted:
            start = i
            break
    if start is None:
        return None

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if is_heading(lines[i]) and lines[i].strip().upper() != wanted:
            end = i
            break
    return "\n".join(lines[start:end]).rstrip() + "\n"
