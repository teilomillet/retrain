<div align="center">
<h1 style="font-family: Helvetica, Arial, sans-serif; font-weight: bold; font-size: 4em;">
Re:🏋️‍♂️
</h1>
</div>

```bash
uv add retrain
```

# Training AI Agents with Reinforcement Learning in `retrain`

The `retrain` library empowers you to teach AI models, particularly Large Language Models (LLMs), new behaviors using a technique called Reinforcement Learning (RL). This is especially useful for tasks where the model needs to interact, use tools, or follow complex instructions over several steps.

Imagine teaching a smart assistant to accomplish a goal. Instead of just giving it examples of right answers, we let it try things out in a simulated environment. It gets feedback (rewards or penalties) based on its actions, and it learns to get better over time.

## Core Workflow

Here's how the key pieces work together in `retrain` when training an agent:

1.  **The Environment (`SmolAgentEnv`): The Agent's Playground**
    *   This is where our AI agent lives and acts. Think of it as a chat interface or a task simulator.
    *   The agent starts with an initial instruction or prompt (e.g., "Plan a trip to Paris").
    *   The `SmolAgentEnv` defines what tools are available (e.g., a search engine, a calculator) and how the world responds to the agent's actions.
    *   **Rollout**: The agent interacts with the environment step-by-step. This whole interaction, from the first prompt to the final outcome, is called a "rollout."
        *   The agent "speaks" (generates text).
        *   If it tries to use a tool, the environment executes it (or simulates execution) and gives back a result (e.g., search results).
        *   If it gives a final answer, the episode might end.
    *   The `SmolAgentEnv` keeps track of the entire conversation and all actions taken during this rollout.

2.  **The Reward Calculator (`RewardCalculator`): The Judge of Performance**
    *   After a full rollout (the agent's attempt to complete the task) is finished, we need to evaluate how well it did.
    *   The `RewardCalculator` takes all the data from the rollout:
        *   The full conversation history.
        *   What actions the LLM took (text responses, tool calls).
        *   Any simple, immediate feedback the environment might have given at each step.
        *   Information about how likely the LLM was to say what it said (log probabilities).
    *   It then applies one or more "reward functions" to this data. These functions are the rules we define to score the agent's behavior. Examples:
        *   Did it successfully use a required tool?
        *   Was its final answer correct?
        *   Did it follow instructions?
        *   Was its reasoning clear?
    *   The `RewardCalculator` combines these scores into a single, final numerical reward for specific parts of the rollout (e.g., for each "turn" or for the entire rollout). This reward tells the training algorithm how good that particular sequence of actions was.

3.  **The Trainer (`GRPOTrainer` using TRL): The Teacher**
    *   The `Trainer` is responsible for taking the processed information from the `RewardCalculator` and actually teaching the LLM.
    *   It uses an RL algorithm called GRPO (Group Relative Policy Optimization), powered by the TRL (Transformer Reinforcement Learning) library.
    *   **Learning Loop**:
        1.  The `Trainer` asks the `SmolAgentEnv` to perform a rollout with the current LLM.
        2.  The `RawRolloutData` (including log probabilities of the LLM's own actions) is passed to the `RewardCalculator`.
        3.  The `RewardCalculator` returns the `ProcessedTrajectory`, which includes prompts, the LLM's actual completions, and the final calculated rewards.
        4.  The `Trainer` then takes this structured data (prompts, completions, rewards, and the original log probabilities) and feeds it to the TRL GRPO algorithm.
        5.  TRL's GRPO algorithm looks at the rewards and figures out how to adjust the LLM's internal "wiring" (its parameters) so that in the future, it's more likely to take actions that lead to higher rewards and less likely to take actions that lead to low or negative rewards.
    *   This process (rollout -> reward calculation -> training update) is repeated many times. With each iteration, the LLM gets slightly better at the task.

**In Simple Terms:**

The **Environment** is the game. The **LLM Agent** is the player. The **Reward Calculator** is the scorekeeper who looks at the whole game and decides how many points the player gets. The **Trainer** is the coach who uses those scores to help the player learn better strategies for the next game.

This setup allows `retrain` to teach complex, multi-step behaviors to LLMs by letting them learn from experience in a structured way.

## Getting Started: Running a Training Job

To start a training process with `retrain`, you need to:
1.  Define a configuration for your training run.
2.  Use the `retrain.run()` function to execute it.

### 1. Configuring a Training Run

Training runs are configured using a structured format that can be provided as a Python dictionary or, more commonly, loaded from a YAML file. The entire configuration schema is defined using Pydantic models located in [`retrain/config_models.py`](mdc:retrain/config_models.py). This file serves as the definitive source of truth for all available options and their expected data types.

Key sections in the configuration include:

*   `model`: Specifies the base model to be trained, how to load it (e.g., from Hugging Face), and any PEFT (Parameter-Efficient Fine-Tuning) adaptations.
*   `algorithm`: Defines the reinforcement learning algorithm to use (e.g., GRPO), the backend (e.g., TRL), and its specific `hyperparameters` (like learning rate, batch size, etc.).
*   `environment`: Configures the environment in which the agent will learn (e.g., `smolagent` for conversational tasks).
*   `prompt_source`: Sets up where the initial prompts or tasks for the agent will come from (e.g., a list of strings, a Hugging Face dataset).
*   `reward_setup`: Details how rewards are calculated, including which reward functions to use for step-level and rollout-level evaluations, their weights, and any associated verifiers.
*   Optional top-level settings like `seed`, `experiment_name`, and `logging_level` are also available.

### 2. Initiating a Training Run

The main entry point to start training is the `retrain.run()` asynchronous function. You'll typically load your configuration and pass it to this function.

**Example Python script (`main.py`):**

```python
import asyncio
from retrain import run
from retrain.config_models import TrainingConfig # For loading YAML
from pydantic import ValidationError

async def main():
    config_path = "config.yaml" # Path to your configuration file

    try:
        # Load configuration from YAML
        print(f"Loading configuration from: {config_path}")
        training_config_pydantic = TrainingConfig.from_yaml(config_path)
        
        # Convert Pydantic model to dictionary for retrain.run()
        config_dict = training_config_pydantic.model_dump()

        print("Starting training run...")
        # Ensure PyYAML is installed: pip install PyYAML
        # Ensure TRL and other dependencies are installed as per GRPOConfig requirements.
        results = await run(config=config_dict)
        print("Training run finished.")
        print(f"Results: {results}")

    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
    except ValidationError as e:
        print(f"Error: Configuration validation failed.\n{e}")
    except ImportError as e:
        print(f"ImportError: {e}. Ensure all dependencies like PyYAML or TRL are installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

Alternatively, you can construct the configuration dictionary directly in Python if preferred, though YAML is recommended for complex setups.

This updated structure should guide users effectively in setting up and launching their training jobs with `retrain`. 