"""Tests for logging.mojo — JSONL builder and file logger.

Covers:
    - KeyValue construction + helper factories
    - JSON string escaping (quotes, backslash, control chars)
    - build_json_line: empty, single, multi-entry, string quoting
    - JsonlLogger: append to file and read back
"""

from testing import assert_true, assert_equal

from src.logging import (
    KeyValue,
    json_float,
    json_int,
    json_string,
    json_bool,
    build_json_line,
    JsonlLogger,
)


# ---------------------------------------------------------------------------
# KeyValue construction + factories
# ---------------------------------------------------------------------------


fn test_json_float_basic() raises:
    """json_float creates numeric (unquoted) entry."""
    var kv = json_float("loss", 0.5)
    assert_equal(kv.key, "loss")
    assert_equal(kv.is_string, False)
    # Value should contain "0.5" (exact format may vary)
    assert_true(len(kv.value) > 0, "Value should not be empty")


fn test_json_int_basic() raises:
    """json_int creates integer (unquoted) entry."""
    var kv = json_int("step", 42)
    assert_equal(kv.key, "step")
    assert_equal(kv.value, "42")
    assert_equal(kv.is_string, False)


fn test_json_string_basic() raises:
    """json_string creates quoted entry."""
    var kv = json_string("model", "gpt-4")
    assert_equal(kv.key, "model")
    assert_equal(kv.value, "gpt-4")
    assert_equal(kv.is_string, True)


fn test_json_bool_true() raises:
    """json_bool true -> "true"."""
    var kv = json_bool("enabled", True)
    assert_equal(kv.key, "enabled")
    assert_equal(kv.value, "true")
    assert_equal(kv.is_string, False)


fn test_json_bool_false() raises:
    """json_bool false -> "false"."""
    var kv = json_bool("enabled", False)
    assert_equal(kv.value, "false")


# ---------------------------------------------------------------------------
# build_json_line
# ---------------------------------------------------------------------------


fn test_build_json_line_empty() raises:
    """Empty entries -> {}."""
    var line = build_json_line(List[KeyValue]())
    assert_equal(line, "{}")


fn test_build_json_line_single_numeric() raises:
    """Single numeric entry."""
    var entries = List[KeyValue]()
    entries.append(json_int("step", 1))
    var line = build_json_line(entries)
    assert_equal(line, '{"step": 1}')


fn test_build_json_line_single_string() raises:
    """Single string entry (quoted)."""
    var entries = List[KeyValue]()
    entries.append(json_string("name", "test"))
    var line = build_json_line(entries)
    assert_equal(line, '{"name": "test"}')


fn test_build_json_line_multiple() raises:
    """Multiple entries in order."""
    var entries = List[KeyValue]()
    entries.append(json_int("step", 1))
    entries.append(json_float("loss", 0.5))
    entries.append(json_string("model", "gpt"))
    entries.append(json_bool("ok", True))
    var line = build_json_line(entries)
    # Verify it starts and ends correctly
    assert_true(line.startswith("{"), "Should start with {")
    assert_true(line.endswith("}"), "Should end with }")
    # Verify key entries are present
    assert_true('"step": 1' in line, "Should contain step")
    assert_true('"model": "gpt"' in line, "Should contain quoted model")
    assert_true('"ok": true' in line, "Should contain ok")


fn test_build_json_line_escapes_quotes() raises:
    """String values with quotes are escaped."""
    var entries = List[KeyValue]()
    entries.append(json_string("msg", 'say "hello"'))
    var line = build_json_line(entries)
    assert_true('\\"hello\\"' in line, "Quotes should be escaped")


fn test_build_json_line_escapes_backslash() raises:
    """Backslashes in values are escaped."""
    var entries = List[KeyValue]()
    entries.append(json_string("path", "a\\b"))
    var line = build_json_line(entries)
    assert_true("a\\\\b" in line, "Backslash should be escaped")


fn test_build_json_line_escapes_newline() raises:
    """Newlines in values become \\n."""
    var entries = List[KeyValue]()
    entries.append(json_string("text", "line1\nline2"))
    var line = build_json_line(entries)
    assert_true("line1\\nline2" in line, "Newline should be escaped as \\n")


fn test_build_json_line_key_escaped() raises:
    """Keys with special chars are also escaped."""
    var entries = List[KeyValue]()
    entries.append(json_int('ke"y', 1))
    var line = build_json_line(entries)
    assert_true('ke\\"y' in line, "Key quotes should be escaped")


# ---------------------------------------------------------------------------
# JsonlLogger (file I/O)
# ---------------------------------------------------------------------------


fn test_logger_creates_and_appends() raises:
    """Logger creates file and appends multiple lines."""
    var path = "/tmp/test_retrain_logger.jsonl"

    # Truncate file to start fresh
    var cleanup = open(path, "w")
    cleanup.close()

    var logger = JsonlLogger(path)

    # Write two entries
    var entries1 = List[KeyValue]()
    entries1.append(json_int("step", 0))
    entries1.append(json_float("loss", 1.5))
    logger.log(entries1)

    var entries2 = List[KeyValue]()
    entries2.append(json_int("step", 1))
    entries2.append(json_float("loss", 0.8))
    logger.log(entries2)

    # Read back and verify
    var f = open(path, "r")
    var content = f.read()
    f.close()

    # Should be two lines
    var lines = content.split("\n")
    # There may be a trailing empty line from the final \n
    var non_empty = 0
    for i in range(len(lines)):
        if len(String(lines[i]).strip()) > 0:
            non_empty += 1
    assert_equal(non_empty, 2)

    # Verify content
    assert_true('"step": 0' in content, "Should contain first entry")
    assert_true('"step": 1' in content, "Should contain second entry")


fn test_logger_raw() raises:
    """log_raw writes pre-formatted JSON."""
    var path = "/tmp/test_retrain_logger_raw.jsonl"
    # Truncate file to start fresh
    var cleanup = open(path, "w")
    cleanup.close()
    var logger = JsonlLogger(path)
    logger.log_raw('{"custom": true}')

    var f = open(path, "r")
    var content = f.read()
    f.close()

    assert_true('{"custom": true}' in content, "Raw line should be preserved")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


fn main() raises:
    # KeyValue factories
    test_json_float_basic()
    test_json_int_basic()
    test_json_string_basic()
    test_json_bool_true()
    test_json_bool_false()

    # build_json_line
    test_build_json_line_empty()
    test_build_json_line_single_numeric()
    test_build_json_line_single_string()
    test_build_json_line_multiple()
    test_build_json_line_escapes_quotes()
    test_build_json_line_escapes_backslash()
    test_build_json_line_escapes_newline()
    test_build_json_line_key_escaped()

    # JsonlLogger
    test_logger_creates_and_appends()
    test_logger_raw()

    print("All 15 logging tests passed!")
