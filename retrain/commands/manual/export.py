"""Export the retrain manual to troff (man page) and HTML formats."""

from __future__ import annotations

import html as html_mod
import re
from dataclasses import dataclass, field


@dataclass
class ManualSection:
    """One heading + body from the manual."""

    heading: str
    body_lines: list[str] = field(default_factory=list)


def _is_manual_heading(line: str) -> bool:
    """Detect ALL-CAPS section headings (no leading whitespace)."""
    s = line.strip()
    if not s or s != line:
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-()")
    return all(ch in allowed for ch in s) and any(ch.isalpha() for ch in s)


_AUTO_RE = re.compile(r"<<(?:END:)?AUTO:\w+>>")


def parse_manual(text: str) -> list[ManualSection]:
    """Split manual text into sections by heading, stripping auto markers."""
    sections: list[ManualSection] = []
    current: ManualSection | None = None

    for line in text.splitlines():
        if _AUTO_RE.fullmatch(line.strip()):
            continue
        if _is_manual_heading(line):
            current = ManualSection(heading=line.rstrip())
            sections.append(current)
        elif current is not None:
            current.body_lines.append(line.rstrip())

    return sections


def _escape_troff(text: str) -> str:
    """Escape characters that have special meaning in troff."""
    text = text.replace("\\", "\\\\")
    if text.startswith("."):
        text = "\\&" + text
    return text


def to_troff(sections: list[ManualSection]) -> str:
    """Render sections as a troff man(1) page."""
    lines: list[str] = ['.TH RETRAIN 1 "" "" "retrain manual"']

    for sec in sections:
        lines.append(f".SH {sec.heading}")
        in_block = False
        for body in sec.body_lines:
            stripped = body.lstrip()
            indented = len(body) - len(stripped) >= 8 if stripped else False

            if indented and not in_block:
                lines.append(".nf")
                in_block = True
            elif not indented and in_block:
                lines.append(".fi")
                in_block = False

            lines.append(_escape_troff(body))

        if in_block:
            lines.append(".fi")

    return "\n".join(lines) + "\n"


def _make_anchor(heading: str) -> str:
    """Convert a heading into a URL-safe HTML id."""
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def to_html(sections: list[ManualSection]) -> str:
    """Render sections as self-contained HTML with embedded CSS and nav TOC."""
    css = (
        "body{font-family:monospace;max-width:80ch;margin:2em auto;padding:0 1em}"
        "nav{margin-bottom:2em}"
        "nav a{margin-right:1em}"
        "h2{border-bottom:1px solid #ccc;padding-bottom:.3em}"
        "pre{background:#f5f5f5;padding:1em;overflow-x:auto}"
    )

    # Build TOC
    toc_items: list[str] = []
    for sec in sections:
        anchor = _make_anchor(sec.heading)
        toc_items.append(
            f'<a href="#{anchor}">{html_mod.escape(sec.heading)}</a>'
        )

    nav = "<nav>" + "\n".join(toc_items) + "</nav>"

    # Build body
    body_parts: list[str] = []
    for sec in sections:
        anchor = _make_anchor(sec.heading)
        body_parts.append(
            f'<h2 id="{anchor}">{html_mod.escape(sec.heading)}</h2>'
        )

        in_pre = False
        buf: list[str] = []

        def _flush_pre() -> None:
            nonlocal in_pre
            if in_pre and buf:
                body_parts.append("<pre>" + "\n".join(buf) + "</pre>")
                buf.clear()
                in_pre = False

        def _flush_text() -> None:
            if buf:
                body_parts.append("<p>" + "\n".join(buf) + "</p>")
                buf.clear()

        for body in sec.body_lines:
            stripped = body.lstrip()
            indented = len(body) - len(stripped) >= 8 if stripped else False

            if indented:
                if not in_pre:
                    _flush_text()
                    in_pre = True
                buf.append(html_mod.escape(body))
            else:
                if in_pre:
                    _flush_pre()
                if not stripped:
                    _flush_text()
                else:
                    buf.append(html_mod.escape(body))

        if in_pre:
            _flush_pre()
        else:
            _flush_text()

    body_html = "\n".join(body_parts)

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>RETRAIN(1)</title>\n"
        f"<style>{css}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{nav}\n"
        f"{body_html}\n"
        "</body>\n"
        "</html>\n"
    )
