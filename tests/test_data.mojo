"""Tests for data.mojo: DataSource trait, Example struct, MathDataSource.

Covers:
    - Example struct construction and field access
    - MathDataSource conformance (mock — no real dataset load)
    - InMemoryDataSource custom DataSource compiles and works
"""

from testing import assert_true, assert_equal

from src.data import DataSource, Example, MathDataSource


# ---------------------------------------------------------------------------
# Example struct
# ---------------------------------------------------------------------------


fn test_example_fields() raises:
    """Example stores prompt and reference correctly."""
    var ex = Example("What is 2+2?", "4")
    assert_equal(ex.prompt, "What is 2+2?")
    assert_equal(ex.reference, "4")


fn test_example_copy() raises:
    """Example is Copyable."""
    var ex = Example("prompt", "ref")
    var ex2 = ex.copy()
    assert_equal(ex2.prompt, "prompt")
    assert_equal(ex2.reference, "ref")


# ---------------------------------------------------------------------------
# Custom DataSource: InMemoryDataSource
# ---------------------------------------------------------------------------


struct InMemoryDataSource(DataSource):
    """In-memory dataset for testing — verifies trait conformance."""

    var examples: List[Example]

    fn __init__(out self):
        self.examples = List[Example]()

    fn __init__(out self, var examples: List[Example]):
        self.examples = examples^

    fn __moveinit__(out self, deinit take: Self):
        self.examples = take.examples^

    fn load(mut self) raises -> List[Example]:
        var out = List[Example]()
        for i in range(len(self.examples)):
            out.append(self.examples[i].copy())
        return out^


fn test_in_memory_data_source_empty() raises:
    """Empty InMemoryDataSource returns empty list."""
    var ds = InMemoryDataSource()
    var examples = ds.load()
    assert_equal(len(examples), 0)


fn test_in_memory_data_source_with_data() raises:
    """InMemoryDataSource returns stored examples."""
    var data = List[Example]()
    data.append(Example("p1", "r1"))
    data.append(Example("p2", "r2"))
    data.append(Example("p3", "r3"))

    var ds = InMemoryDataSource(data^)
    var examples = ds.load()
    assert_equal(len(examples), 3)
    assert_equal(examples[0].prompt, "p1")
    assert_equal(examples[0].reference, "r1")
    assert_equal(examples[2].prompt, "p3")
    assert_equal(examples[2].reference, "r3")


fn test_in_memory_data_source_reload() raises:
    """InMemoryDataSource can be loaded multiple times."""
    var data = List[Example]()
    data.append(Example("prompt", "ref"))

    var ds = InMemoryDataSource(data^)
    var first = ds.load()
    var second = ds.load()
    assert_equal(len(first), 1)
    assert_equal(len(second), 1)
    assert_equal(first[0].prompt, second[0].prompt)


# ---------------------------------------------------------------------------
# MathDataSource conformance (structural — no network call)
# ---------------------------------------------------------------------------


fn test_math_data_source_init() raises:
    """MathDataSource can be constructed with max_examples."""
    var ds = MathDataSource(max_examples=10)
    assert_equal(ds.max_examples, 10)


fn test_math_data_source_default() raises:
    """MathDataSource default: max_examples=0 (unlimited)."""
    var ds = MathDataSource()
    assert_equal(ds.max_examples, 0)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


fn main() raises:
    # Example struct
    test_example_fields()
    test_example_copy()

    # InMemoryDataSource
    test_in_memory_data_source_empty()
    test_in_memory_data_source_with_data()
    test_in_memory_data_source_reload()

    # MathDataSource conformance
    test_math_data_source_init()
    test_math_data_source_default()

    print("All 7 data tests passed!")
