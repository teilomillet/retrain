"""JSONL logger with native file I/O.

Builds JSON strings manually for flat objects (no external JSON library).
Each log entry is a single JSON object on one line, appended to a file.
"""


struct KeyValue(Copyable, Movable):
    """A single key-value pair for JSON logging."""

    var key: String
    var value: String
    var is_string: Bool  # True if value should be quoted in JSON

    fn __init__(out self, key: String, value: String, is_string: Bool = False):
        self.key = key
        self.value = value
        self.is_string = is_string


fn json_float(key: String, value: Float64) -> KeyValue:
    """Create a float entry for JSON logging."""
    return KeyValue(key, String(value))


fn json_int(key: String, value: Int) -> KeyValue:
    """Create an integer entry for JSON logging."""
    return KeyValue(key, String(value))


fn json_string(key: String, value: String) -> KeyValue:
    """Create a string entry for JSON logging."""
    return KeyValue(key, value, is_string=True)


fn json_bool(key: String, value: Bool) -> KeyValue:
    """Create a boolean entry for JSON logging."""
    if value:
        return KeyValue(key, "true")
    return KeyValue(key, "false")


fn _escape_json_string(s: String) -> String:
    """Escape special characters for JSON string values."""
    var result = String()
    var bytes = s.as_bytes()
    for i in range(len(s)):
        var b = bytes[i]
        if b == UInt8(ord('"')):
            result += '\\"'
        elif b == UInt8(ord("\\")):
            result += "\\\\"
        elif b == UInt8(ord("\n")):
            result += "\\n"
        elif b == UInt8(ord("\r")):
            result += "\\r"
        elif b == UInt8(ord("\t")):
            result += "\\t"
        else:
            result += String(s[i : i + 1])
    return result^


fn build_json_line(entries: List[KeyValue]) -> String:
    """Build a single JSON object string from key-value pairs.

    Args:
        entries: List of KeyValue pairs.

    Returns:
        JSON object as a single-line string (no trailing newline).
    """
    var result = String("{")
    for i in range(len(entries)):
        if i > 0:
            result += ", "
        result += '"' + _escape_json_string(entries[i].key) + '": '
        if entries[i].is_string:
            result += '"' + _escape_json_string(entries[i].value) + '"'
        else:
            result += entries[i].value
    result += "}"
    return result^


struct JsonlLogger:
    """Append-only JSONL file logger."""

    var file_path: String

    fn __init__(out self, file_path: String):
        self.file_path = file_path

    fn log(self, entries: List[KeyValue]) raises:
        """Write a JSON object as a single line to the log file.

        Args:
            entries: Key-value pairs to log.
        """
        var line = build_json_line(entries) + "\n"
        var f = open(self.file_path, "a")
        f.write_string(line)
        f.close()

    fn log_raw(self, json_line: String) raises:
        """Write a pre-formatted JSON line to the log file.

        Args:
            json_line: Complete JSON string (newline will be appended).
        """
        var f = open(self.file_path, "a")
        f.write_string(json_line + "\n")
        f.close()
