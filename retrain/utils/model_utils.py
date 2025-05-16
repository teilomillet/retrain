import torch
from typing import Optional

# Assuming selective_log_softmax will be a local or imported utility.
# For now, let's define a placeholder if not readily available from TRL utils.
# Ideally: from trl.trainer.utils import selective_log_softmax
# Placeholder:
def selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes log softmax only for the logits corresponding to the labels.
    Args:
        logits: Tensor of shape (batch_size, sequence_length, vocab_size)
        labels: Tensor of shape (batch_size, sequence_length)
    Returns:
        Tensor of shape (batch_size, sequence_length) containing logprobs of actual tokens.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)


def calculate_completion_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor, # Shape: (batch_size, full_sequence_length) prompt + completion
    attention_mask: torch.Tensor, # Shape: (batch_size, full_sequence_length)
    completion_ids: torch.Tensor, # Shape: (batch_size, completion_length) - just the completion part
    temperature: float,
    logits_processing_batch_size: Optional[int] = None # For chunking if full sequence is too large for one pass
) -> torch.Tensor:
    """
    Calculates the log probabilities of the completion tokens given the full input sequence.

    Args:
        model: The policy model.
        input_ids: Tokenized input sequence (prompt + completion).
        attention_mask: Attention mask for the input_ids.
        completion_ids: Tokenized completion sequence (used to determine which logits to keep and for gathering logprobs).
        temperature: Temperature for scaling logits before log_softmax.
        logits_processing_batch_size: Optional batch size for processing logits to save memory.

    Returns:
        torch.Tensor: Log probabilities for each token in the completion. Shape: (batch_size, completion_length)
    """
    
    num_completion_tokens = completion_ids.size(1)
    if num_completion_tokens == 0:
        return torch.empty((input_ids.size(0), 0), device=input_ids.device, dtype=torch.float32)

    effective_batch_size = logits_processing_batch_size if logits_processing_batch_size is not None else input_ids.size(0)
    
    all_completion_logps = []

    # Ensure model is on the same device as inputs
    # model.to(input_ids.device) # Usually handled by accelerator or trainer setup

    was_training = model.training
    model.eval() # Set to eval mode for consistent logprob calculation

    with torch.no_grad():
        for i in range(0, input_ids.size(0), effective_batch_size):
            batch_input_ids = input_ids[i : i + effective_batch_size]
            batch_attention_mask = attention_mask[i : i + effective_batch_size]
            batch_completion_ids = completion_ids[i : i + effective_batch_size]

            # Model forward pass
            # The model call here should return an object with a 'logits' attribute.
            # TRL's GRPOTrainer passes `logits_to_keep + 1` to the model if the model supports it,
            # to optimize the forward pass. Here, we assume the model returns full logits.
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits # Shape: (batch_size, full_sequence_length, vocab_size)

            # We are interested in the logits that predicted the completion tokens.
            # The logits at position t are used to predict token t+1.
            # So, for completion_ids[t], we need logits[prompt_len + t -1].
            # The logit for the *first* completion token is at the end of the prompt part of the sequence.
            # Logits are for predicting the *next* token.
            # So, if prompt has P tokens, completion has C tokens:
            # input_ids: [p1..pP, c1..cC]
            # logits:    [l(p2)..l(pP), l(c1)..l(cC), l(next_after_cC)] - total P+C logits
            # We need logits that predicted c1...cC. These are:
            # l(c1) is at index P-1 in the original full sequence logits (if 0-indexed)
            # l(cC) is at index P+C-2 in the original full sequence logits
            
            # Extract logits corresponding to the positions *before* each completion token.
            # Start index for these logits in the full sequence logits:
            prompt_length = batch_input_ids.size(1) - batch_completion_ids.size(1)
            
            # Logits from index (prompt_length - 1) up to (prompt_length -1 + num_completion_tokens -1)
            # No, should be: logits for tokens that *are* the completion.
            # The logit at sequence position `j` is for predicting token `j+1`.
            # So, to get logprobs for `completion_ids`, we need `logits` from positions
            # corresponding to `prompt_tokens` and `completion_tokens[:-1]`.
            # More simply: TRL's `_get_per_token_logps` does:
            # `logits = logits[:, :-1, :]` (shape B, L-1, V) - removes logit for token after last input
            # `input_ids_batch = input_ids_batch[:, -logits_to_keep:]` (shape B, C) - takes completion_ids
            # `logits = logits[:, -logits_to_keep:]` (shape B, C, V) - takes logits for predicting completion tokens
            
            # Simpler approach: get all logits, then slice.
            # Logits corresponding to predictions of completion tokens are needed.
            # If full_seq_len = P + C, then logits has shape (batch, P+C, Vocab)
            # (using HF convention where logits[i] predicts token[i+1])
            # The logits that predicted completion_ids start at index `prompt_length` of input_ids,
            # so the relevant logits are from `prompt_length -1` up to `prompt_length + num_completion_tokens -1 -1`
            # if we take all input_ids.

            # Let's use TRL's slicing logic directly for clarity:
            # These are the logits used to predict the tokens input_ids[1:]
            relevant_logits = logits[:, :-1, :] # Shape: (batch_size, full_sequence_length - 1, vocab_size)
            
            # We only care about the part of these logits that predicted the completion_ids
            # The completion_ids are at the end of the input_ids sequence.
            completion_predicting_logits = relevant_logits[:, -num_completion_tokens:, :]
            # Shape: (batch_size, num_completion_tokens, vocab_size)

            if temperature == 0 or temperature == 1.0: # Avoid division by zero or one if not needed
                scaled_logits = completion_predicting_logits
            else:
                scaled_logits = completion_predicting_logits / temperature
            
            # Gather the logprobs for the actual completion tokens
            completion_logps = selective_log_softmax(scaled_logits, batch_completion_ids)
            all_completion_logps.append(completion_logps)

    if was_training:
        model.train() # Restore training mode

    return torch.cat(all_completion_logps, dim=0)

# Add __init__.py to utils if it's a new directory
# (No tool for this, assume it's handled or dir exists) 