# Environment (`retrain.environment`)

Think of this as the "world," "task," or "game" where your AI model learns and acts. It's the bridge between your model and the problem it's trying to solve.

## What Does an Environment Do?

An environment in `retrain` (usually by building on `retrain.environment.Environment`) handles the back-and-forth conversation with the AI model. Its main jobs are:

1.  **Getting Ready (`async setup()`):** Prepares anything the environment needs before starting, like loading tools or connecting to other systems.
2.  **Starting Fresh (`async reset()`):** Resets the task to its beginning state for a new attempt or episode, giving the model its first piece of information (`EnvironmentObservation`).
3.  **Taking a Turn (`async step(action: LLMAction)`):** When the model decides to do something (an `LLMAction` - like responding with text or using a tool), the environment processes it, updates the situation, and tells the model what happened next (`EnvironmentObservation`), if it earned a reward, and if the task is finished.
4.  **Keeping Track:** For ongoing tasks, the environment remembers the current situation (e.g., conversation history, game state).
5.  **Standard Communication:**
    *   **Model Actions (`LLMAction`):** A standard way for the model to say what it wants to do: either generate text or call a specific tool with certain inputs.
    *   **Environment Feedback (`EnvironmentObservation`):** A standard way for the environment to give information back to the model, like the initial setup, the result of a tool use, or an update on the current state.

## Giving Environments Tools

Environments can use "tools" (from `retrain.environment.tool`) to let the model do more, like search the web or run calculations. There are two main ways an environment gets its tools:

1.  **Pre-defined Tools:** You tell the environment about tools that are already known and registered in the system.
2.  **Dynamically Discovered Tools:** The environment can find tools from other places, like an external server, using "Tool Providers."

*(For more on how tools are made and registered, see the README in the `retrain/environment/tool/` directory.)*

## Types of Environments

*   **Simple (Stateless) Environments:** Used for simpler training where the model might just give one response to a prompt. Tools might not be necessary here.
*   **Interactive (Stateful) Environments:** For more complex tasks like multi-turn conversations or where the model needs to use tools. These environments keep track of the interaction over time. `SmolAgentEnv` is an example built for modern reinforcement learning, showing how to manage conversations and tool use over many steps.

## Rewards: Simple Feedback vs. Overall Score

When the model takes an action, the `step()` method gives an immediate, simple reward. For example, 0 for most actions, or a small penalty if it tries something invalid.

However, to figure out a more complete score for how well the model did (especially over a whole task), `retrain` uses a separate `RewardCalculator` (see `retrain/reward/README.md`). The environment gathers all the details of an interaction (a "rollout"), and this data is then passed to the `RewardCalculator` to determine the final, more nuanced rewards used for training the AI.

## Using Environments from Other Systems (Bridges)

If you want to use an environment from somewhere else (like a game or simulator) that doesn't speak `retrain`'s language, "adapter modules" (in `retrain.environment.bridge`) help translate between them. This keeps the main training code clean and able to work with different kinds of environments. It's still a work in progress.

## Running Full Tasks: `async def rollout(...)`

For many AI training methods (especially Reinforcement Learning), it's useful to run a whole task or "episode" from start to finish. The `Environment` has an important method for this: `async def rollout(...)`.

*   **Purpose:** It manages the entire interaction between the AI model and the environment for one full episode, starting with an initial prompt and continuing until the task is done or a limit is reached.
*   **How it works:** It repeatedly asks the model for an action, processes that action using its `step()` method, and collects all the information about what happened.
*   **What it Gathers (`RawRolloutData`):** This method collects a set of data about the entire rollout, including the full conversation, all actions taken by the model, any immediate rewards, observations at each step, and the probabilities of the text the model generated. This complete package of data is then used by the training process.

In short, the `retrain.environment` part of the project provides the stage for AI models to act and learn, supporting everything from simple text generation to complex, tool-using interactions.