"""Tests for retrain.man_export — manual parser and troff/HTML formatters."""

from retrain.man_export import (
    ManualSection,
    _escape_troff,
    _make_anchor,
    parse_manual,
    to_html,
    to_troff,
)


class TestParseManual:
    def test_basic_section_split(self):
        text = "RETRAIN(1)\n\nNAME\n    retrain - trainer\n\nCOMMANDS\n    run it\n"
        sections = parse_manual(text)
        assert len(sections) == 3
        assert sections[0].heading == "RETRAIN(1)"
        assert sections[1].heading == "NAME"
        assert "    retrain - trainer" in sections[1].body_lines
        assert sections[2].heading == "COMMANDS"

    def test_auto_block_markers_stripped(self):
        text = (
            "COMMANDS\n"
            "<<AUTO:COMMANDS>>\n"
            "    some content\n"
            "<<END:AUTO:COMMANDS>>\n"
        )
        sections = parse_manual(text)
        assert len(sections) == 1
        body = "\n".join(sections[0].body_lines)
        assert "<<AUTO:" not in body
        assert "<<END:AUTO:" not in body
        assert "some content" in body

    def test_empty_input(self):
        sections = parse_manual("")
        assert sections == []

    def test_preserves_indentation(self):
        text = "EXAMPLE\n        code_line()\n    normal line\n"
        sections = parse_manual(text)
        assert sections[0].body_lines[0] == "        code_line()"
        assert sections[0].body_lines[1] == "    normal line"

    def test_multiple_auto_blocks(self):
        text = (
            "SEC\n"
            "<<AUTO:FOO>>\nold\n<<END:AUTO:FOO>>\n"
            "    kept\n"
            "<<AUTO:BAR>>\nold2\n<<END:AUTO:BAR>>\n"
        )
        sections = parse_manual(text)
        body = "\n".join(sections[0].body_lines)
        assert "old" not in body or "kept" in body
        assert "kept" in body


class TestToTroff:
    def test_th_header(self):
        sections = [ManualSection(heading="NAME", body_lines=["    retrain"])]
        result = to_troff(sections)
        assert result.startswith(".TH RETRAIN 1")

    def test_sh_macros(self):
        sections = [
            ManualSection(heading="NAME", body_lines=["    retrain"]),
            ManualSection(heading="COMMANDS", body_lines=["    run it"]),
        ]
        result = to_troff(sections)
        assert ".SH NAME" in result
        assert ".SH COMMANDS" in result

    def test_code_blocks(self):
        sections = [
            ManualSection(
                heading="EXAMPLE",
                body_lines=[
                    "    description",
                    "        code_here()",
                    "        more_code()",
                    "    back to text",
                ],
            )
        ]
        result = to_troff(sections)
        assert ".nf" in result
        assert ".fi" in result

    def test_empty_sections(self):
        sections = [ManualSection(heading="EMPTY")]
        result = to_troff(sections)
        assert ".SH EMPTY" in result

    def test_escape_leading_dot(self):
        assert _escape_troff(".foo") == "\\&.foo"

    def test_escape_backslash(self):
        assert _escape_troff("a\\b") == "a\\\\b"


class TestToHtml:
    def test_doctype(self):
        sections = [ManualSection(heading="NAME", body_lines=["    retrain"])]
        result = to_html(sections)
        assert "<!DOCTYPE html>" in result

    def test_h2_tags(self):
        sections = [
            ManualSection(heading="NAME", body_lines=[]),
            ManualSection(heading="COMMANDS", body_lines=[]),
        ]
        result = to_html(sections)
        assert '<h2 id="name">NAME</h2>' in result
        assert '<h2 id="commands">COMMANDS</h2>' in result

    def test_nav_toc(self):
        sections = [
            ManualSection(heading="NAME", body_lines=[]),
            ManualSection(heading="COMMANDS", body_lines=[]),
        ]
        result = to_html(sections)
        assert "<nav>" in result
        assert 'href="#name"' in result
        assert 'href="#commands"' in result

    def test_css_embedded(self):
        sections = [ManualSection(heading="NAME", body_lines=[])]
        result = to_html(sections)
        assert "<style>" in result

    def test_pre_blocks(self):
        sections = [
            ManualSection(
                heading="EXAMPLE",
                body_lines=[
                    "    text",
                    "        code_line()",
                    "    text again",
                ],
            )
        ]
        result = to_html(sections)
        assert "<pre>" in result
        assert "</pre>" in result

    def test_html_escaping(self):
        sections = [
            ManualSection(heading="TEST", body_lines=["    x < y & z > w"])
        ]
        result = to_html(sections)
        assert "&lt;" in result
        assert "&amp;" in result
        assert "&gt;" in result

    def test_make_anchor(self):
        assert _make_anchor("CAMPAIGN MODE") == "campaign-mode"
        assert _make_anchor("SEE ALSO") == "see-also"
        assert _make_anchor("RETRAIN(1)") == "retrain-1"
