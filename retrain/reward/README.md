# Reward System (`retrain.reward`)

This directory defines how we measure or "score" our AI model's performance. This score, or "reward," is crucial feedback that helps the model learn and improve, especially in Reinforcement Learning.

## Core Ideas

1.  **Defining Rewards (`reward.py`):**
    *   Individual reward functions assess a single model completion (e.g., one prompt, one response) and return a `float` score.
    *   These functions are registered using the `@reward` decorator, making them available by name (e.g., `get_reward_function("my_reward_name")`).
    *   `calculate_total_reward` can sum scores from all registered functions for a comprehensive evaluation of one instance, but this isn't typically the direct training signal.

2.  **Using Rewards in Training:**
    *   RL trainers (like GRPO) usually need rewards for batches of data.
    *   You define simple, single-instance reward logic.
    *   The system (e.g., via `create_grpo_batch_reward_func`) adapts your function to work with batches, often incorporating verifiers (see `retrain/verifier/README.md`).
    *   This setup is configured in the trainer (e.g., through `reward_config`).

3.  **Combining Multiple Reward Aspects:**
    *   For training based on several quality aspects (e.g., correctness, fluency), create a *composite* reward function.
    *   This function, also registered with `@reward`, calls other reward functions and combines their scores into a single output. This unified score then becomes the training signal.

4.  **Processing Full Interactions (`RewardCalculator` in `calculator.py`):**
    *   For tasks with multiple steps or turns (a "rollout"), the `RewardCalculator` processes the entire sequence.
    *   **Input:** `RawRolloutData` (full conversation, model actions, etc.).
    *   **Function:** Calculates detailed rewards for each step by applying configured step-level and rollout-level reward functions. It reconstructs prompts for each turn.
    *   **Output:** `ProcessedTrajectory` – a list of (prompt, completion, combined_reward) for each turn, ready for the trainer.
    *   This allows for sophisticated reward schemes in multi-turn agent training.

This system allows you to build modular reward components that can be flexibly combined for evaluation or to create focused training signals.

