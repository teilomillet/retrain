"""Optimizer-step implementations for the local backend.

One module per training objective: ``rl`` (importance-sampling policy
loss), ``sft`` (weighted cross-entropy, padded or per-microbatch padding),
and ``hybrid`` (RL + ECHO sharing one forward). ``shared`` holds the
scaffold they have in common.
"""
