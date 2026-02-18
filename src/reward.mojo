"""Reward function trait and built-in implementations.

Users implement RewardFn to bring custom tasks. The framework calls
reward_fn.score(response, reference) for each completion — no forking
required.
"""


trait RewardFn(Movable):
    """Score a model response against a reference answer.

    Returns a scalar reward (typically 0.0 or 1.0 for binary correctness).
    """

    fn score(self, response: String, reference: String) -> Float64:
        ...


# ---------------------------------------------------------------------------
# Built-in: boxed math reward
# ---------------------------------------------------------------------------


fn extract_boxed(text: String) -> String:
    """Extract \\boxed{...} answer from MATH solution text."""
    var marker = "\\boxed{"
    var marker_len = len(marker)

    var idx = text.rfind(marker)
    if idx == -1:
        return String("")

    var start = idx + marker_len
    var depth = 1
    var pos = start
    var text_len = len(text)
    var bytes = text.as_bytes()
    while pos < text_len and depth > 0:
        if bytes[pos] == UInt8(ord("{")):
            depth += 1
        elif bytes[pos] == UInt8(ord("}")):
            depth -= 1
        pos += 1

    var extracted = String(text[start : pos - 1])
    return String(extracted.strip())


fn grade_answer(given: String, expected: String) -> Bool:
    """Simple string-match grading. Strips whitespace."""
    return String(given.strip()) == String(expected.strip())


fn get_reward(response: String, answer: String) -> Float64:
    """Binary correctness reward for MATH problems.

    Convenience wrapper — equivalent to BoxedMathReward().score().
    """
    return BoxedMathReward().score(response, answer)


struct BoxedMathReward(RewardFn):
    """Binary correctness reward: extract \\boxed{...} and string-match."""

    fn __init__(out self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass

    fn score(self, response: String, reference: String) -> Float64:
        var given = extract_boxed(response)
        if grade_answer(given, reference):
            return 1.0
        return 0.0
