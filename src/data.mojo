"""Data source trait and built-in implementations.

Users implement DataSource to bring custom datasets. Combined with
RewardFn, this is the minimum needed to run retrain on any task.
"""

from src.pybridge import load_math_dataset, MathExample


@fieldwise_init
struct Example(Copyable, Movable):
    """A single training example: prompt + reference answer."""

    var prompt: String
    var reference: String


trait DataSource(Movable):
    """Load a dataset of prompt/reference pairs.

    Called once at the start of training. The returned list is cycled
    through for the duration of the run.
    """

    fn load(mut self) raises -> List[Example]:
        ...


# ---------------------------------------------------------------------------
# Built-in: MATH dataset
# ---------------------------------------------------------------------------


struct MathDataSource(DataSource):
    """Loads hendrycks/MATH via EleutherAI mirror."""

    var max_examples: Int

    fn __init__(out self, max_examples: Int = 0):
        self.max_examples = max_examples

    fn __moveinit__(out self, deinit take: Self):
        self.max_examples = take.max_examples

    fn load(mut self) raises -> List[Example]:
        var raw = load_math_dataset(max_examples=self.max_examples)
        var out = List[Example]()
        for i in range(len(raw)):
            out.append(Example(raw[i].problem, raw[i].answer))
        return out^
