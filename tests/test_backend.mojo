"""Tests for the backend abstraction.

Covers:
    - MockBackend conforming to TrainingBackend trait
    - Generic function dispatch with MockBackend
    - SampleSequence struct operations
    - Method call tracking and contract verification
"""

from testing import assert_true, assert_equal, assert_almost_equal

from src.backend import TrainingBackend, SampleSequence


# ---------------------------------------------------------------------------
# MockBackend — test double implementing TrainingBackend
# ---------------------------------------------------------------------------


struct MockBackend(TrainingBackend):
    """Test backend that records method calls without side effects."""

    var checkpoint_count: Int
    var sample_count: Int
    var train_count: Int
    var save_names: List[String]
    var last_num_datums: Int
    var last_lr: Float64

    fn __init__(out self):
        self.checkpoint_count = 0
        self.sample_count = 0
        self.train_count = 0
        self.save_names = List[String]()
        self.last_num_datums = 0
        self.last_lr = 0.0

    fn checkpoint_for_sampling(mut self, name: String) raises:
        self.checkpoint_count += 1

    fn sample_batch(
        mut self,
        prompts: List[List[Int]],
        num_samples: Int,
        max_tokens: Int,
        temperature: Float64,
        top_p: Float64,
    ) raises -> List[List[SampleSequence]]:
        self.sample_count += 1
        var results = List[List[SampleSequence]]()
        for p in range(len(prompts)):
            var group = List[SampleSequence]()
            for s in range(num_samples):
                var tokens = List[Int]()
                tokens.append(100 + s)
                tokens.append(200 + s)
                var logprobs = List[Float64]()
                logprobs.append(-0.5)
                logprobs.append(-1.0)
                group.append(SampleSequence(tokens^, logprobs^))
            results.append(group^)
        return results^

    fn train_step(
        mut self,
        all_tokens: List[List[Int]],
        all_logprobs: List[List[Float64]],
        all_advantages: List[List[Float64]],
        lr: Float64,
        weight_decay: Float64,
    ) raises -> Float64:
        self.train_count += 1
        self.last_num_datums = len(all_tokens)
        self.last_lr = lr
        return 0.42

    fn save(mut self, name: String) raises:
        self.save_names.append(name)


# ---------------------------------------------------------------------------
# Generic function test — proves trait dispatch works
# ---------------------------------------------------------------------------


fn step_mock[B: TrainingBackend](mut backend: B) raises:
    """Call all backend methods through the generic interface."""
    backend.checkpoint_for_sampling("test_ckpt")

    var prompts = List[List[Int]]()
    var p = List[Int]()
    p.append(1)
    p.append(2)
    prompts.append(p^)
    var results = backend.sample_batch(prompts, 2, 100, 0.7, 0.95)

    var tokens = List[List[Int]]()
    var logprobs = List[List[Float64]]()
    var advantages = List[List[Float64]]()
    for i in range(len(results)):
        for j in range(len(results[i])):
            tokens.append(results[i][j].tokens.copy())
            logprobs.append(results[i][j].logprobs.copy())
            var adv = List[Float64](length=len(results[i][j].tokens), fill=1.0)
            advantages.append(adv^)

    _ = backend.train_step(tokens, logprobs, advantages, 1e-4, 0.0)
    backend.save("test_save")


# ---------------------------------------------------------------------------
# Tests: MockBackend + trait conformance
# ---------------------------------------------------------------------------


fn test_mock_conforms_to_trait() raises:
    """MockBackend works with a generic function."""
    var mock = MockBackend()
    step_mock(mock)
    assert_equal(mock.checkpoint_count, 1)
    assert_equal(mock.sample_count, 1)
    assert_equal(mock.train_count, 1)
    assert_equal(len(mock.save_names), 1)
    assert_equal(mock.save_names[0], "test_save")


fn test_mock_sample_returns_correct_shape() raises:
    """sample_batch returns correct number of prompts x samples."""
    var mock = MockBackend()
    var prompts = List[List[Int]]()
    for _ in range(3):
        var p = List[Int]()
        p.append(1)
        prompts.append(p^)
    var results = mock.sample_batch(prompts, 4, 100, 0.7, 0.95)
    assert_equal(len(results), 3)
    for i in range(3):
        assert_equal(len(results[i]), 4)
        for j in range(4):
            assert_equal(len(results[i][j].tokens), 2)
            assert_equal(len(results[i][j].logprobs), 2)


fn test_mock_train_records_datums() raises:
    """train_step records the number of datums and lr."""
    var mock = MockBackend()
    var tokens = List[List[Int]]()
    var logprobs = List[List[Float64]]()
    var advantages = List[List[Float64]]()
    for _ in range(5):
        var t = List[Int]()
        t.append(1)
        tokens.append(t^)
        var lp = List[Float64]()
        lp.append(-0.5)
        logprobs.append(lp^)
        var a = List[Float64]()
        a.append(1.0)
        advantages.append(a^)
    var loss = mock.train_step(tokens, logprobs, advantages, 3e-5, 0.01)
    assert_equal(mock.last_num_datums, 5)
    assert_almost_equal(mock.last_lr, 3e-5, atol=1e-10)
    assert_almost_equal(loss, 0.42, atol=1e-6)


fn test_mock_save_records_names() raises:
    """save records checkpoint names in order."""
    var mock = MockBackend()
    mock.save("step_1")
    mock.save("step_2")
    mock.save("final")
    assert_equal(len(mock.save_names), 3)
    assert_equal(mock.save_names[0], "step_1")
    assert_equal(mock.save_names[1], "step_2")
    assert_equal(mock.save_names[2], "final")


fn test_multiple_checkpoints() raises:
    """Multiple checkpoint calls are tracked."""
    var mock = MockBackend()
    mock.checkpoint_for_sampling("step_0")
    mock.checkpoint_for_sampling("step_1")
    mock.checkpoint_for_sampling("step_2")
    assert_equal(mock.checkpoint_count, 3)


# ---------------------------------------------------------------------------
# Tests: SampleSequence struct
# ---------------------------------------------------------------------------


fn test_sample_sequence_struct() raises:
    """SampleSequence stores tokens and logprobs correctly."""
    var tokens = List[Int]()
    tokens.append(10)
    tokens.append(20)
    tokens.append(30)
    var logprobs = List[Float64]()
    logprobs.append(-0.1)
    logprobs.append(-0.2)
    logprobs.append(-0.3)
    var seq = SampleSequence(tokens^, logprobs^)
    assert_equal(len(seq.tokens), 3)
    assert_equal(seq.tokens[0], 10)
    assert_equal(seq.tokens[2], 30)
    assert_almost_equal(seq.logprobs[1], -0.2, atol=1e-10)


fn test_sample_sequence_copy() raises:
    """SampleSequence is Copyable."""
    var tokens = List[Int]()
    tokens.append(1)
    var logprobs = List[Float64]()
    logprobs.append(-0.5)
    var seq1 = SampleSequence(tokens^, logprobs^)
    var seq2 = seq1.copy()
    assert_equal(seq2.tokens[0], 1)
    assert_almost_equal(seq2.logprobs[0], -0.5, atol=1e-10)


fn test_sample_sequence_in_list() raises:
    """SampleSequence works in nested lists (List[List[SampleSequence]])."""
    var outer = List[List[SampleSequence]]()
    for _ in range(2):
        var inner = List[SampleSequence]()
        for j in range(3):
            var t = List[Int]()
            t.append(j)
            var lp = List[Float64]()
            lp.append(Float64(-j))
            inner.append(SampleSequence(t^, lp^))
        outer.append(inner^)
    assert_equal(len(outer), 2)
    assert_equal(len(outer[0]), 3)
    assert_equal(outer[1][2].tokens[0], 2)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


fn main() raises:
    # Trait conformance
    test_mock_conforms_to_trait()
    test_mock_sample_returns_correct_shape()
    test_mock_train_records_datums()
    test_mock_save_records_names()
    test_multiple_checkpoints()

    # SampleSequence
    test_sample_sequence_struct()
    test_sample_sequence_copy()
    test_sample_sequence_in_list()

    print("All 8 backend tests passed!")
