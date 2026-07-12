"""Embedding-based planning detector using sentence-transformers."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, cast

from retrain.planning.tokens import clean_fragment

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class _SentenceEmbeddingModel(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool,
        batch_size: int = 32,
    ) -> NDArray[np.float64]: ...


# Math-tuned anchor phrases discovered in the v2 diagnostic.
# These define the planning/execution centroids in embedding space.

_PLANNING_ANCHORS = [
    # Metacognitive / reflective
    "let me reconsider this approach",
    "this doesn't seem right, let me try again",
    "wait, I made an error in the previous step",
    "let me verify this result by checking",
    "going back to reconsider the strategy",
    # Strategic / planning
    "the key insight here is that",
    "notice that we can use the fact that",
    "a clever trick is to substitute",
    "we can simplify by observing that",
    "the idea is to use symmetry",
    "let's try a different approach",
    "instead of computing directly, we can",
    "we should consider what happens when",
    "to solve this, the strategy is",
    "the crucial observation is that",
    # Structural / organizing
    "step 1: understand the problem",
    "first, let's set up the equation",
    "now we need to find",
    "the goal is to compute",
    "our plan is to first find and then substitute",
]

_EXECUTION_ANCHORS = [
    "substituting x equals 2 into the equation",
    "expanding the left side gives",
    "simplifying the fraction yields",
    "therefore the answer is 42",
    "computing 3 plus 5 equals 8",
    "factoring out x from both terms",
    "taking the derivative of x squared gives 2x",
    "by the quadratic formula x equals",
    "multiplying both sides by 2",
    "solving for x we get x equals 7",
    "the discriminant is b squared minus 4ac",
    "so the sum is equal to 15",
    "plugging in the values we obtain",
    "cross-multiplying gives us",
    "collecting like terms on the left",
]


class SemanticPlanningDetector:
    """Embedding-based planning detector using sentence-transformers.

    Reconstructs ~12-word sentence windows from token fragments, embeds
    them in batch, classifies by centroid distance, and maps matches
    back to token-level masks.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.02,
    ) -> None:
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            raise ImportError(
                "Semantic planning detector requires sentence-transformers.\n"
                "Install it with:  pip install 'retrain[semantic]'"
            ) from exc

        import numpy as np

        self._np = np
        self._threshold = threshold

        sentence_transformer = cast(
            Callable[[str], _SentenceEmbeddingModel],
            getattr(sentence_transformers, "SentenceTransformer"),
        )
        self._model = sentence_transformer(model_name)

        # Pre-compute centroids
        plan_embs = self._model.encode(_PLANNING_ANCHORS, normalize_embeddings=True)
        exec_embs = self._model.encode(_EXECUTION_ANCHORS, normalize_embeddings=True)
        plan_centroid = np.mean(plan_embs, axis=0)
        plan_centroid /= np.linalg.norm(plan_centroid)
        exec_centroid = np.mean(exec_embs, axis=0)
        exec_centroid /= np.linalg.norm(exec_centroid)

        self._plan_centroid = plan_centroid
        self._exec_centroid = exec_centroid

    def detect(self, token_strs: list[str]) -> list[int]:
        n = len(token_strs)
        if n == 0:
            return []

        # 1. Reconstruct sentence windows from token fragments
        windows, window_spans = self._build_windows(token_strs)

        if not windows:
            return [0] * n

        # 2. Batch embed all windows
        embs = self._model.encode(windows, normalize_embeddings=True, batch_size=256)

        # 3. Classify each window by centroid distance
        plan_sims = embs @ self._plan_centroid
        exec_sims = embs @ self._exec_centroid

        # 4. Map classifications back to token mask
        mask = [0] * n
        for i in range(len(windows)):
            if plan_sims[i] > exec_sims[i] + self._threshold and plan_sims[i] > 0.25:
                start, end = window_spans[i]
                for j in range(start, end):
                    mask[j] = 1

        return mask

    @staticmethod
    def _build_windows(
        token_strs: list[str],
        target_words: int = 12,
        stride_words: int = 6,
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """Build overlapping sentence windows from token fragments.

        Returns (window_texts, window_spans) where each span is
        (start_token_idx, end_token_idx) exclusive.
        """
        # Clean all fragments and build a word-to-token index
        cleaned = [clean_fragment(t) for t in token_strs]

        # Accumulate into a flat word list with token provenance
        words: list[str] = []
        word_to_token: list[int] = []  # word_idx -> token_idx

        for tok_idx, frag in enumerate(cleaned):
            for w in frag.split():
                if w:
                    words.append(w)
                    word_to_token.append(tok_idx)

        if len(words) < 3:
            return [], []

        windows: list[str] = []
        spans: list[tuple[int, int]] = []

        wi = 0
        while wi < len(words):
            end_wi = min(wi + target_words, len(words))
            window_text = " ".join(words[wi:end_wi])

            # Token span: from the token of the first word to one past the
            # token of the last word
            tok_start = word_to_token[wi]
            tok_end = word_to_token[end_wi - 1] + 1

            windows.append(window_text)
            spans.append((tok_start, tok_end))

            wi += stride_words
            if end_wi >= len(words):
                break

        return windows, spans
