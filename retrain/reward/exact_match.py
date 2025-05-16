from typing import List, Any, Dict, Optional

from .reward import reward

@reward
def exact_match(
    prompts: List[Any],
    completions: List[str],
    answers: List[Any],
    match_score: float = 1.0,
    mismatch_score: float = 0.0,
    **kwargs: Any
) -> List[float]:
    """
    Computes a reward based on exact string match between completion and answer.

    Args:
        prompts: A list of prompts (ignored).
        completions: A list of strings, model generated responses.
        answers: A list of target answer strings.
        match_score (float): Score for a match. Defaults to 1.0.
        mismatch_score (float): Score for a mismatch. Defaults to 0.0.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        A list of float scores.

    Raises:
        ValueError: If the lengths of completions and answers lists do not match.
    """
    if len(completions) != len(answers):
        raise ValueError("Completions and answers lists must have the same length.")

    scores = []
    for completion, answer in zip(completions, answers):
        # Ensure both are strings for comparison, and strip whitespace
        comp_str = str(completion).strip()
        ans_str = str(answer).strip()
        
        if comp_str == ans_str:
            scores.append(match_score)
        else:
            scores.append(mismatch_score)
    return scores 