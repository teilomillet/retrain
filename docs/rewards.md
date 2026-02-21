# Reward Functions

retrain supports four reward types, selected via `[reward] type` in your TOML config. All reward functions implement the same interface: `score(response, reference) -> float`.

## match (default)

Extracts the last `\boxed{...}` from the model's response and string-compares it against the reference answer. Returns `1.0` on exact match, `0.0` otherwise.

No extra dependencies. Fast and deterministic.

```toml
[reward]
type = "match"
```

## math

Symbolic math equivalence via `math_verify`. Parses both the model answer and reference into symbolic expressions, then checks mathematical equivalence. Handles equivalent forms like `\frac{1}{2}` vs `0.5`.

Uses `math_verify.parse()` and `math_verify.verify()` directly (bypassing the async wrapper for performance). Reference parses are cached since the same problem is scored `group_size` times.

Requires the `verifiers` library:

```bash
pip install verifiers
```

```toml
[reward]
type = "math"
```

## judge

LLM-based evaluation via `verifiers.JudgeRubric`. Sends the model's completion and the reference answer to an LLM judge, which returns a yes/no verdict.

```toml
[reward]
type = "judge"
judge_model = "gpt-4o-mini"
```

The `judge_model` field specifies which LLM to use as the judge. Defaults to `gpt-4o-mini` if not set.

Requires the `verifiers` library and an API key for the judge model.

## custom

Load a user-provided Python function as the reward. The function receives `(response: str, reference: str)` and returns a float.

```toml
[reward]
type = "custom"
custom_module = "my_package.rewards"
custom_function = "my_score"
```

The module is imported via `importlib.import_module()`. The function can be synchronous or async (async functions are run with `asyncio.run()`).

### Writing a custom reward

Create a Python module accessible on `PYTHONPATH`:

```python
# my_package/rewards.py

def my_score(response: str, reference: str) -> float:
    """Custom reward function.

    Args:
        response: The model's full completion text.
        reference: The ground-truth answer string.

    Returns:
        Float reward value (typically 0.0 or 1.0).
    """
    # Your scoring logic here
    extracted = extract_answer(response)
    return 1.0 if extracted == reference else 0.0
```

The function must accept exactly two string arguments and return a float. The `RewardFunction` protocol:

```python
class RewardFunction(Protocol):
    def score(self, response: str, reference: str) -> float: ...
```

## Choosing a reward type

| Type | Speed | Accuracy | Dependencies |
|------|-------|----------|-------------|
| `match` | Fastest | Exact string match only | None |
| `math` | Fast | Handles equivalent forms | `verifiers` |
| `judge` | Slow (API call) | Flexible, handles free-form | `verifiers` + API key |
| `custom` | Varies | You decide | Your module |

For MATH training, `match` is usually sufficient. Use `math` when you need equivalence checking (e.g., `\frac{1}{2}` matching `0.5`). Use `judge` for tasks where correctness can't be verified symbolically.
