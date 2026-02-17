"""Backend abstraction for swappable training backends.

Defines the TrainingBackend trait and shared types. The training loop
calls the 4 trait methods (checkpoint, sample, train, save) and never
touches backend internals directly. Each backend is monomorphized at
compile time via Mojo generics — zero dispatch overhead.
"""


@fieldwise_init
struct SampleSequence(Copyable, Movable):
    """Extracted data from one sample sequence."""

    var tokens: List[Int]
    var logprobs: List[Float64]


trait TrainingBackend(Movable):
    """Abstract interface for training backends.

    Implementations provide the 4 I/O boundary points:
    checkpoint, sample, train, save.
    """

    fn checkpoint_for_sampling(mut self, name: String) raises:
        ...

    fn sample_batch(
        mut self,
        prompts: List[List[Int]],
        num_samples: Int,
        max_tokens: Int,
        temperature: Float64,
        top_p: Float64,
    ) raises -> List[List[SampleSequence]]:
        ...

    fn train_step(
        mut self,
        all_tokens: List[List[Int]],
        all_logprobs: List[List[Float64]],
        all_advantages: List[List[Float64]],
        lr: Float64,
        weight_decay: Float64,
    ) raises -> Float64:
        ...

    fn save(mut self, name: String) raises:
        ...
