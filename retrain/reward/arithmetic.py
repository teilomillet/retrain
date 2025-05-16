import re
from typing import List, Any, Dict, Optional

from .reward import reward 



@reward 
def contains_number(
    completions: List[str],
    number_present_score: float = 1.0,
    number_absent_score: float = -1.0,
    **kwargs: Any
) -> List[float]:
    """
    Computes rewards based on whether the completion string contains a digit.
    Inspired by Unsloth's GRPO Reward Example #1 (Function 1).

    Args:
        completions: A list of strings, model generated responses.
        number_present_score (float): Score if a digit is detected. Defaults to 1.0.
        number_absent_score (float): Score if no digit is detected. Defaults to -1.0.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        A list of float scores.
    """
    digit_regex = re.compile(r'\d')
    scores = []
    for completion in completions:
        if digit_regex.search(str(completion)):
            scores.append(number_present_score)
        else:
            scores.append(number_absent_score)
    return scores

_NUMBER_EXTRACT_REGEX = re.compile(r'[-+]?\d*\.?\d+|[-+]?\d+')

def _extract_first_number(text: Any) -> Optional[float]:
    """Extracts the first number found in the text as a float."""
    match = _NUMBER_EXTRACT_REGEX.search(str(text))
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None

@reward 
def arithmetic_correctness(
    completions: List[str],
    answers: List[Any],
    correct_score: float = 3.0,
    incorrect_score: float = -3.0,
    **kwargs: Any
) -> List[float]:
    """
    Computes rewards based on matching the first extractable number in completion and answer.
    Inspired by Unsloth's GRPO Reward Example #1 (Function 2).

    Args:
        completions: A list of strings, model generated responses.
        answers: A list of target answers.
        correct_score (float): Score if extracted numbers match. Defaults to 3.0.
        incorrect_score (float): Score if extracted numbers don't match or extraction fails. Defaults to -3.0.
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
        comp_num = _extract_first_number(completion)
        ans_num = _extract_first_number(answer)

        # If both numbers are successfully extracted and are equal
        if comp_num is not None and ans_num is not None and comp_num == ans_num:
            scores.append(correct_score)
        else:
            # If extraction failed for either or numbers don't match
            scores.append(incorrect_score)
    return scores