# Trainers (`retrain.trainer`)

This directory contains the "engines" that manage the Reinforcement Learning (RL) training process in `retrain`. Trainers are responsible for taking a model and improving it based on its interactions within an environment.

## Core Ideas

1.  **`BaseTrainer` (the standard blueprint - in `trainer.py`)**:
    *   This is a fundamental template (an Abstract Base Class) that all specific trainer implementations must follow.
    *   It ensures that any RL algorithm can be consistently integrated and used by the main `retrain` system (e.g., by `retrain.run.py`).
    *   Key functions that all trainers will have include methods to start the main training loop (e.g., `async train()`) and potentially to perform a single update step.

2.  **Algorithm-Specific Trainers (the actual engines)**:
    *   For each different RL algorithm (like GRPO, RLOO), there's a specific trainer class that inherits from `BaseTrainer`.
    *   These trainers often act as "adapters" that connect `retrain`'s components (like our custom `Environment` and `RewardCalculator`) with established RL libraries (e.g., TRL - Transformer Reinforcement Learning).
    *   For example, `retrain.trainer.grpo.trl.GRPOTrainer` is a specialized trainer that uses TRL's GRPO algorithm.

## How Training Generally Works

A typical trainer in `retrain` follows a cycle like this:

1.  **Setup**: The trainer is initialized with everything it needs:
    *   The AI `model` to be trained.
    *   Configuration details for the specific RL `algorithm`.
    *   An `environment` for the model to interact with.
    *   A `reward_calculator` to figure out how well the model is doing.
    *   A source of `prompts` to start interactions.
    *   A `tokenizer` to process text for the model.

2.  **Training Loop (`async def train(...)`)**: The trainer then repeats the following steps:
    *   **Gather Experience (Rollout)**: The model interacts with the `environment` (e.g., has a conversation, tries to complete a task). This full interaction is called a "rollout," and it produces data like the conversation history, actions taken, and any immediate feedback from the environment.
    *   **Calculate Rewards**: The data from the rollout is given to the `reward_calculator`. This component analyzes the interaction and computes a more detailed reward signal that tells the model how well it performed during that rollout.
    *   **Update the Model**: The trainer uses this processed experience (including the calculated rewards and details of the model's actions) to update the AI `model`. The specifics of this step depend on the RL algorithm being used (e.g., GRPO, PPO), often handled by the underlying RL library.

## Design Goals

This structure aims to be flexible and clear:
*   The `Environment` handles the rules of interaction.
*   The `RewardCalculator` focuses on how to score performance.
*   The `Trainer` orchestrates the learning process and connects to RL algorithm libraries.

This separation makes it easier to experiment with different environments, reward strategies, or RL algorithms without having to rewrite everything.

For details on a specific trainer implementation, such as how `GRPOTrainerAdapter` uses TRL, you would look into its specific files (e.g., `retrain/trainer/grpo/trl.py`). The primary purpose of this README is to give a general understanding of what trainers do in `retrain`. 