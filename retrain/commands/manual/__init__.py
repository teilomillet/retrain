"""Manual command package."""

from __future__ import annotations

from retrain.commands.manual.render import (
    render_commands,
    render_environment,
    render_options,
    render_quickstart,
)
from retrain.commands.manual.run import run
from retrain.commands.manual.sync import check_file, replace_auto_block, sync_file
from retrain.commands.manual.text import load_text
from retrain.commands.manual.topic import (
    TOPIC_TO_SECTION,
    extract_section,
    is_heading,
)

__all__ = [
    "TOPIC_TO_SECTION",
    "check_file",
    "extract_section",
    "is_heading",
    "load_text",
    "render_commands",
    "render_environment",
    "render_options",
    "render_quickstart",
    "replace_auto_block",
    "run",
    "sync_file",
]
