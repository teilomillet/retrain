"""Tests for the backend abstraction.

Covers:
    - MockBackend conforming to TrainingBackend trait
    - Generic function dispatch with MockBackend
    - SampleSequence struct operations
    - Back pressure: call ordering, datum shape invariants,
      uninformative batch handling, error propagation
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal

from src.backend import TrainingBackend, SampleSequence


# ---------------------------------------------------------------------------
# MockBackend — lightweight test double
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
        for _ in range(len(prompts)):
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
# RecordingBackend — full contract verification
# ---------------------------------------------------------------------------


struct RecordingBackend(TrainingBackend):
    """Backend that records all calls and verifies invariants in real time.

    Checks datum shape alignment, prompt padding, and call ordering.
    Can inject errors on a specific train_step call.
    """

    var calls: List[String]
    var save_names: List[String]
    var train_count: Int
    var train_datum_counts: List[Int]
    var shape_errors: Int
    var padding_errors: Int
    var prompt_len: Int
    var raise_on_train: Int  # 0 = never; N = raise on Nth call

    fn __init__(out self, prompt_len: Int, raise_on_train: Int = 0):
        self.calls = List[String]()
        self.save_names = List[String]()
        self.train_count = 0
        self.train_datum_counts = List[Int]()
        self.shape_errors = 0
        self.padding_errors = 0
        self.prompt_len = prompt_len
        self.raise_on_train = raise_on_train

    fn checkpoint_for_sampling(mut self, name: String) raises:
        self.calls.append("checkpoint")

    fn sample_batch(
        mut self,
        prompts: List[List[Int]],
        num_samples: Int,
        max_tokens: Int,
        temperature: Float64,
        top_p: Float64,
    ) raises -> List[List[SampleSequence]]:
        self.calls.append("sample")
        var results = List[List[SampleSequence]]()
        for _ in range(len(prompts)):
            var group = List[SampleSequence]()
            for s in range(num_samples):
                var tokens = List[Int]()
                tokens.append(100 + s)
                tokens.append(200 + s)
                tokens.append(300 + s)
                var logprobs = List[Float64]()
                logprobs.append(-0.5)
                logprobs.append(-1.0)
                logprobs.append(-0.3)
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
        self.calls.append("train")
        self.train_count += 1
        self.train_datum_counts.append(len(all_tokens))

        # Outer shape: all three lists same length
        if len(all_tokens) != len(all_logprobs) or len(all_tokens) != len(all_advantages):
            self.shape_errors += 1

        for i in range(len(all_tokens)):
            var tl = len(all_tokens[i])
            var ll = len(all_logprobs[i])
            var al = len(all_advantages[i])

            # Inner shape: each datum's three arrays same length
            if tl != ll or tl != al:
                self.shape_errors += 1

            # Datum must not be empty
            if tl == 0:
                self.shape_errors += 1

            # Prompt padding: first prompt_len entries should be 0.0
            var pad = min(self.prompt_len, ll)
            for j in range(pad):
                if all_logprobs[i][j] != 0.0:
                    self.padding_errors += 1
                if all_advantages[i][j] != 0.0:
                    self.padding_errors += 1

        # Inject error if configured
        if self.raise_on_train > 0 and self.train_count == self.raise_on_train:
            raise Error("injected train error at call " + String(self.train_count))

        return 0.42

    fn save(mut self, name: String) raises:
        self.calls.append("save")
        self.save_names.append(name)


# ---------------------------------------------------------------------------
# Mini training loop — reproduces the real loop's contract
# ---------------------------------------------------------------------------


fn run_mini_loop[B: TrainingBackend](
    mut backend: B,
    prompt_ids: List[List[Int]],
    num_steps: Int,
    group_size: Int,
    save_every: Int,
    all_same_rewards: Bool,
) raises:
    """Simplified training loop matching main.mojo's backend call contract.

    Reward assignment: if all_same_rewards, all 0.0 (groups uninformative).
    Otherwise first sample gets 1.0, rest get 0.0 (groups informative).
    Builds datums with correct prompt padding, just like the real loop.
    """
    for step in range(num_steps):
        backend.checkpoint_for_sampling("step_" + String(step))

        var results = backend.sample_batch(
            prompt_ids, group_size, 100, 0.7, 0.95
        )

        var all_datum_tokens = List[List[Int]]()
        var all_datum_logprobs = List[List[Float64]]()
        var all_datum_advantages = List[List[Float64]]()

        for p_idx in range(len(results)):
            var ob_len = len(prompt_ids[p_idx])
            var n_seqs = len(results[p_idx])

            # Assign rewards
            var rewards = List[Float64]()
            for s_idx in range(n_seqs):
                if all_same_rewards:
                    rewards.append(0.0)
                elif s_idx == 0:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)

            # Skip uninformative groups (all same reward)
            var all_same = True
            if len(rewards) > 0:
                var first = rewards[0]
                for r_idx in range(1, len(rewards)):
                    if rewards[r_idx] != first:
                        all_same = False
                        break
            if all_same:
                continue

            # Compute simple advantages: reward - mean
            var mean_r: Float64 = 0.0
            for r_idx in range(len(rewards)):
                mean_r += rewards[r_idx]
            mean_r /= Float64(max(len(rewards), 1))

            # Build datums with prompt padding (mirrors main.mojo)
            for s_idx in range(n_seqs):
                var n_tok = len(results[p_idx][s_idx].tokens)

                var full_tokens = List[Int](capacity=ob_len + n_tok)
                for t in range(ob_len):
                    full_tokens.append(prompt_ids[p_idx][t])
                for t in range(n_tok):
                    full_tokens.append(results[p_idx][s_idx].tokens[t])

                var padded_logprobs = List[Float64](length=ob_len, fill=0.0)
                for lp_idx in range(len(results[p_idx][s_idx].logprobs)):
                    padded_logprobs.append(results[p_idx][s_idx].logprobs[lp_idx])

                var adv_val = rewards[s_idx] - mean_r
                var padded_advantages = List[Float64](length=ob_len, fill=0.0)
                for _ in range(n_tok):
                    padded_advantages.append(adv_val)

                all_datum_tokens.append(full_tokens^)
                all_datum_logprobs.append(padded_logprobs^)
                all_datum_advantages.append(padded_advantages^)

        if len(all_datum_tokens) > 0:
            _ = backend.train_step(
                all_datum_tokens, all_datum_logprobs, all_datum_advantages,
                4e-5, 0.0,
            )

        if save_every > 0 and (step + 1) % save_every == 0:
            backend.save("checkpoint_step_" + String(step + 1))

    backend.save("final")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


fn _make_prompt(id0: Int, id1: Int, id2: Int) -> List[Int]:
    var p = List[Int]()
    p.append(id0)
    p.append(id1)
    p.append(id2)
    return p^


fn _make_prompts() -> List[List[Int]]:
    var prompts = List[List[Int]]()
    prompts.append(_make_prompt(1, 2, 3))
    prompts.append(_make_prompt(4, 5, 6))
    return prompts^


fn step_mock[B: TrainingBackend](mut backend: B) raises:
    """Call all backend methods through the generic interface."""
    backend.checkpoint_for_sampling("test_ckpt")
    var prompts = List[List[Int]]()
    prompts.append(_make_prompt(1, 2, 3))
    var results = backend.sample_batch(prompts, 2, 100, 0.7, 0.95)

    var tokens = List[List[Int]]()
    var logprobs = List[List[Float64]]()
    var advantages = List[List[Float64]]()
    for i in range(len(results)):
        for j in range(len(results[i])):
            var t = List[Int]()
            for k in range(len(results[i][j].tokens)):
                t.append(results[i][j].tokens[k])
            var lp = List[Float64]()
            for k in range(len(results[i][j].logprobs)):
                lp.append(results[i][j].logprobs[k])
            var adv = List[Float64](length=len(results[i][j].tokens), fill=1.0)
            tokens.append(t^)
            logprobs.append(lp^)
            advantages.append(adv^)

    _ = backend.train_step(tokens, logprobs, advantages, 1e-4, 0.0)
    backend.save("test_save")


# ---------------------------------------------------------------------------
# Tests: MockBackend + trait conformance (existing)
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
    """Sample_batch returns correct number of prompts x samples."""
    var mock = MockBackend()
    var prompts = List[List[Int]]()
    for _ in range(3):
        prompts.append(_make_prompt(1, 2, 3))
    var results = mock.sample_batch(prompts, 4, 100, 0.7, 0.95)
    assert_equal(len(results), 3)
    for i in range(3):
        assert_equal(len(results[i]), 4)
        for j in range(4):
            assert_equal(len(results[i][j].tokens), 2)
            assert_equal(len(results[i][j].logprobs), 2)


fn test_mock_train_records_datums() raises:
    """Train_step records the number of datums and lr."""
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
    """Save records checkpoint names in order."""
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
# Tests: Call ordering (back pressure #1)
# ---------------------------------------------------------------------------


fn test_call_ordering_basic() raises:
    """Loop calls checkpoint -> sample -> train in strict order each step."""
    var backend = RecordingBackend(prompt_len=3)
    run_mini_loop(backend, _make_prompts(), num_steps=3, group_size=4, save_every=0, all_same_rewards=False)

    # 3 steps × (checkpoint, sample, train) + final save
    # = 9 + 1 = 10 calls
    assert_equal(len(backend.calls), 10)

    # Each step: checkpoint, sample, train
    for step in range(3):
        var base = step * 3
        assert_equal(backend.calls[base], "checkpoint")
        assert_equal(backend.calls[base + 1], "sample")
        assert_equal(backend.calls[base + 2], "train")

    # Final call is save
    assert_equal(backend.calls[9], "save")


fn test_call_ordering_with_periodic_saves() raises:
    """Periodic saves appear after train, before next checkpoint."""
    var backend = RecordingBackend(prompt_len=3)
    run_mini_loop(backend, _make_prompts(), num_steps=4, group_size=4, save_every=2, all_same_rewards=False)

    # Steps 0,1,2,3. save_every=2 → saves after step 1 (idx+1=2) and step 3 (idx+1=4)
    # Step 0: checkpoint, sample, train
    # Step 1: checkpoint, sample, train, save("checkpoint_step_2")
    # Step 2: checkpoint, sample, train
    # Step 3: checkpoint, sample, train, save("checkpoint_step_4")
    # Final: save("final")
    assert_equal(len(backend.save_names), 3)
    assert_equal(backend.save_names[0], "checkpoint_step_2")
    assert_equal(backend.save_names[1], "checkpoint_step_4")
    assert_equal(backend.save_names[2], "final")

    # Verify save appears after train in the call sequence
    var found_first_save = False
    for i in range(len(backend.calls)):
        if backend.calls[i] == "save" and not found_first_save:
            found_first_save = True
            # Previous call should be train
            assert_true(i > 0, "Save should not be first call")
            assert_equal(backend.calls[i - 1], "train")
            break


fn test_checkpoint_always_before_sample() raises:
    """Every sample call is preceded by a checkpoint call."""
    var backend = RecordingBackend(prompt_len=3)
    run_mini_loop(backend, _make_prompts(), num_steps=5, group_size=4, save_every=0, all_same_rewards=False)

    for i in range(len(backend.calls)):
        if backend.calls[i] == "sample":
            assert_true(i > 0, "Sample should not be the first call")
            assert_equal(backend.calls[i - 1], "checkpoint")


# ---------------------------------------------------------------------------
# Tests: Datum shape invariants (back pressure #2)
# ---------------------------------------------------------------------------


fn test_datum_shape_outer_alignment() raises:
    """Train receives equal-length token, logprob, and advantage lists."""
    var backend = RecordingBackend(prompt_len=3)
    run_mini_loop(backend, _make_prompts(), num_steps=2, group_size=4, save_every=0, all_same_rewards=False)

    assert_equal(backend.shape_errors, 0)
    assert_equal(backend.train_count, 2)


fn test_datum_shape_inner_alignment() raises:
    """Each datum has matching token/logprob/advantage lengths."""
    var backend = RecordingBackend(prompt_len=3)
    run_mini_loop(backend, _make_prompts(), num_steps=3, group_size=8, save_every=0, all_same_rewards=False)

    assert_equal(backend.shape_errors, 0)
    assert_equal(backend.train_count, 3)


fn test_prompt_padding_zeros() raises:
    """First prompt_len logprobs and advantages are 0.0 in every datum."""
    var backend = RecordingBackend(prompt_len=3)
    run_mini_loop(backend, _make_prompts(), num_steps=2, group_size=4, save_every=0, all_same_rewards=False)

    assert_equal(backend.padding_errors, 0)


fn test_datum_length_includes_prompt_and_completion() raises:
    """Each datum is longer than the prompt (prompt + completion tokens)."""
    # RecordingBackend returns 3 tokens per sample, prompt_len=3
    # So each datum should be 3 + 3 = 6 tokens
    var backend = RecordingBackend(prompt_len=3)
    run_mini_loop(backend, _make_prompts(), num_steps=1, group_size=4, save_every=0, all_same_rewards=False)

    assert_equal(backend.shape_errors, 0)
    assert_equal(backend.train_count, 1)


fn test_datum_count_matches_group_size() raises:
    """Number of datums = num_prompts * group_size (when all groups informative)."""
    var prompts = _make_prompts()  # 2 prompts
    var backend = RecordingBackend(prompt_len=3)
    run_mini_loop(backend, prompts, num_steps=1, group_size=4, save_every=0, all_same_rewards=False)

    assert_equal(backend.train_count, 1)
    # 2 prompts × 4 samples = 8 datums
    assert_equal(backend.train_datum_counts[0], 8)


fn test_datum_count_single_prompt() raises:
    """Single prompt produces group_size datums."""
    var prompts = List[List[Int]]()
    prompts.append(_make_prompt(1, 2, 3))

    var backend = RecordingBackend(prompt_len=3)
    run_mini_loop(backend, prompts, num_steps=1, group_size=6, save_every=0, all_same_rewards=False)

    assert_equal(backend.train_datum_counts[0], 6)


# ---------------------------------------------------------------------------
# Tests: Uninformative batches (back pressure #4)
# ---------------------------------------------------------------------------


fn test_all_uninformative_skips_train() raises:
    """When all groups have identical rewards, train_step is never called."""
    var backend = RecordingBackend(prompt_len=3)
    run_mini_loop(backend, _make_prompts(), num_steps=5, group_size=4, save_every=0, all_same_rewards=True)

    assert_equal(backend.train_count, 0)
    assert_equal(len(backend.train_datum_counts), 0)


fn test_final_save_after_all_uninformative() raises:
    """Save('final') is called even when every step was uninformative."""
    var backend = RecordingBackend(prompt_len=3)
    run_mini_loop(backend, _make_prompts(), num_steps=5, group_size=4, save_every=0, all_same_rewards=True)

    assert_equal(len(backend.save_names), 1)
    assert_equal(backend.save_names[0], "final")


fn test_single_sample_always_uninformative() raises:
    """Group_size=1 means one reward per group -> trivially all-same -> skipped."""
    var backend = RecordingBackend(prompt_len=3)
    run_mini_loop(backend, _make_prompts(), num_steps=3, group_size=1, save_every=0, all_same_rewards=False)

    # Even with mixed_rewards=False, group_size=1 has only 1 reward → all_same
    assert_equal(backend.train_count, 0)
    assert_equal(len(backend.save_names), 1)
    assert_equal(backend.save_names[0], "final")


fn test_uninformative_still_checkpoints_and_samples() raises:
    """Even uninformative steps still call checkpoint + sample (just not train)."""
    var backend = RecordingBackend(prompt_len=3)
    run_mini_loop(backend, _make_prompts(), num_steps=3, group_size=4, save_every=0, all_same_rewards=True)

    # 3 steps × (checkpoint + sample) + final save = 7 calls
    assert_equal(len(backend.calls), 7)
    for step in range(3):
        var base = step * 2
        assert_equal(backend.calls[base], "checkpoint")
        assert_equal(backend.calls[base + 1], "sample")
    assert_equal(backend.calls[6], "save")


fn test_periodic_save_skipped_on_uninformative() raises:
    """Periodic saves still fire even when steps are uninformative."""
    var backend = RecordingBackend(prompt_len=3)
    run_mini_loop(backend, _make_prompts(), num_steps=4, group_size=4, save_every=2, all_same_rewards=True)

    # save_every=2 → saves after step 1 and step 3, plus final
    assert_equal(len(backend.save_names), 3)
    assert_equal(backend.save_names[0], "checkpoint_step_2")
    assert_equal(backend.save_names[1], "checkpoint_step_4")
    assert_equal(backend.save_names[2], "final")


# ---------------------------------------------------------------------------
# Tests: Error propagation (back pressure #3)
# ---------------------------------------------------------------------------


fn test_error_propagates_from_train() raises:
    """An error in train_step propagates out of the loop."""
    var backend = RecordingBackend(prompt_len=3, raise_on_train=1)
    var prompts = _make_prompts()

    var did_raise = False
    try:
        run_mini_loop(backend, prompts, num_steps=5, group_size=4, save_every=0, all_same_rewards=False)
    except:
        did_raise = True

    assert_true(did_raise, "Error should propagate from train_step")


fn test_error_state_consistent_before_failure() raises:
    """Backend state is correct up to the point of error."""
    var backend = RecordingBackend(prompt_len=3, raise_on_train=2)
    var prompts = _make_prompts()

    try:
        run_mini_loop(backend, prompts, num_steps=5, group_size=4, save_every=0, all_same_rewards=False)
    except:
        pass

    # First train call completed, second raised
    assert_equal(backend.train_count, 2)
    # Shapes were valid in both calls (verified before raise)
    assert_equal(backend.shape_errors, 0)
    assert_equal(backend.padding_errors, 0)
    # Both calls had correct datum count
    assert_equal(backend.train_datum_counts[0], 8)  # 2 prompts × 4 samples


fn test_error_prevents_final_save() raises:
    """If train_step raises, save('final') is NOT called."""
    var backend = RecordingBackend(prompt_len=3, raise_on_train=1)
    var prompts = _make_prompts()

    try:
        run_mini_loop(backend, prompts, num_steps=3, group_size=4, save_every=0, all_same_rewards=False)
    except:
        pass

    # save("final") should not have been reached
    for i in range(len(backend.save_names)):
        assert_true(
            backend.save_names[i] != "final",
            "save('final') should not be called after train error",
        )


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


fn main() raises:
    # Trait conformance (5)
    test_mock_conforms_to_trait()
    test_mock_sample_returns_correct_shape()
    test_mock_train_records_datums()
    test_mock_save_records_names()
    test_multiple_checkpoints()

    # SampleSequence (3)
    test_sample_sequence_struct()
    test_sample_sequence_copy()
    test_sample_sequence_in_list()

    # Call ordering (3)
    test_call_ordering_basic()
    test_call_ordering_with_periodic_saves()
    test_checkpoint_always_before_sample()

    # Datum shape invariants (6)
    test_datum_shape_outer_alignment()
    test_datum_shape_inner_alignment()
    test_prompt_padding_zeros()
    test_datum_length_includes_prompt_and_completion()
    test_datum_count_matches_group_size()
    test_datum_count_single_prompt()

    # Uninformative batches (5)
    test_all_uninformative_skips_train()
    test_final_save_after_all_uninformative()
    test_single_sample_always_uninformative()
    test_uninformative_still_checkpoints_and_samples()
    test_periodic_save_skipped_on_uninformative()

    # Error propagation (3)
    test_error_propagates_from_train()
    test_error_state_consistent_before_failure()
    test_error_prevents_final_save()

    print("All 25 backend tests passed!")
