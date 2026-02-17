"""Tests for pybridge utility functions (pure Mojo subset).

Covers:
    - vocab_lookup: token ID → string via pre-loaded table
"""

from testing import assert_true, assert_equal

from src.pybridge import vocab_lookup


# ---------------------------------------------------------------------------
# vocab_lookup
# ---------------------------------------------------------------------------


fn test_vocab_lookup_basic() raises:
    """Basic lookup from known table."""
    var table = List[String]()
    table.append("hello")
    table.append("world")
    table.append("foo")

    var ids = List[Int]()
    ids.append(0)
    ids.append(2)
    ids.append(1)

    var result = vocab_lookup(table, ids)
    assert_equal(len(result), 3)
    assert_equal(result[0], "hello")
    assert_equal(result[1], "foo")
    assert_equal(result[2], "world")


fn test_vocab_lookup_out_of_range() raises:
    """Out-of-range IDs produce empty strings."""
    var table = List[String]()
    table.append("a")
    table.append("b")

    var ids = List[Int]()
    ids.append(5)
    ids.append(100)

    var result = vocab_lookup(table, ids)
    assert_equal(len(result), 2)
    assert_equal(result[0], "")
    assert_equal(result[1], "")


fn test_vocab_lookup_negative_id() raises:
    """Negative IDs produce empty strings."""
    var table = List[String]()
    table.append("a")

    var ids = List[Int]()
    ids.append(-1)
    ids.append(-100)

    var result = vocab_lookup(table, ids)
    assert_equal(len(result), 2)
    assert_equal(result[0], "")
    assert_equal(result[1], "")


fn test_vocab_lookup_empty_table() raises:
    """Empty table -> all empty strings."""
    var table = List[String]()
    var ids = List[Int]()
    ids.append(0)
    ids.append(1)

    var result = vocab_lookup(table, ids)
    assert_equal(len(result), 2)
    assert_equal(result[0], "")
    assert_equal(result[1], "")


fn test_vocab_lookup_empty_ids() raises:
    """Empty IDs -> empty result."""
    var table = List[String]()
    table.append("a")

    var result = vocab_lookup(table, List[Int]())
    assert_equal(len(result), 0)


fn test_vocab_lookup_mixed_valid_invalid() raises:
    """Mix of valid and invalid IDs."""
    var table = List[String]()
    table.append("zero")
    table.append("one")
    table.append("two")

    var ids = List[Int]()
    ids.append(1)
    ids.append(99)
    ids.append(0)
    ids.append(-1)
    ids.append(2)

    var result = vocab_lookup(table, ids)
    assert_equal(len(result), 5)
    assert_equal(result[0], "one")
    assert_equal(result[1], "")
    assert_equal(result[2], "zero")
    assert_equal(result[3], "")
    assert_equal(result[4], "two")


fn test_vocab_lookup_duplicate_ids() raises:
    """Same ID repeated -> same string repeated."""
    var table = List[String]()
    table.append("token")

    var ids = List[Int]()
    ids.append(0)
    ids.append(0)
    ids.append(0)

    var result = vocab_lookup(table, ids)
    assert_equal(len(result), 3)
    for i in range(3):
        assert_equal(result[i], "token")


fn test_vocab_lookup_boundary_id() raises:
    """ID at exact boundary of table size."""
    var table = List[String]()
    table.append("a")
    table.append("b")
    table.append("c")

    var ids = List[Int]()
    ids.append(2)  # last valid
    ids.append(3)  # first invalid

    var result = vocab_lookup(table, ids)
    assert_equal(result[0], "c")
    assert_equal(result[1], "")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


fn main() raises:
    test_vocab_lookup_basic()
    test_vocab_lookup_out_of_range()
    test_vocab_lookup_negative_id()
    test_vocab_lookup_empty_table()
    test_vocab_lookup_empty_ids()
    test_vocab_lookup_mixed_valid_invalid()
    test_vocab_lookup_duplicate_ids()
    test_vocab_lookup_boundary_id()

    print("All 8 pybridge tests passed!")
