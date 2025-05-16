# Verifier System (`retrain.verifier`)

**Purpose:**
The Verifier system in `retrain` provides a way to perform automated, prerequisite checks on an AI model's output *before* detailed reward functions are evaluated. This ensures responses meet basic quality standards or formatting rules, making the reward process more robust and efficient.

**Key Functionality:**

1.  **Defining Verifiers:**
    *   A verifier is a Python function that typically takes the `prompt`, `completion` (the AI's output), and `example` data (for context). It performs a specific check on the `completion` and returns `True` if the check passes, or `False` if it fails.

    ```python
    # Example in a custom verifier file (e.g., my_project/my_verifiers.py)
    # This file needs to be imported by the system, e.g., in retrain/verifier/__init__.py
    from retrain.verifier import verifier
    from typing import Dict, Any

    @verifier # Auto-registers with the function name: "has_greeting"
    def has_greeting(prompt: str, completion: str, example: Dict[str, Any]) -> bool:
        """Checks if the completion starts with a greeting."""
        return completion.lower().startswith("hello")
    ```

2.  **Registering Verifiers:**
    *   The `@verifier` decorator is used to register the verifier function with the system.
    *   `@verifier`: Registers the function using its own name (e.g., `has_greeting`).
    *   `@verifier(name="custom_check_name")`: Registers the function with a specific custom name.

3.  **Linking Verifiers to Reward Functions:**
    *   Verifiers are associated with specific reward functions in the main training configuration file (usually a JSON or Python dict), within the `reward_setup.functions` section.
    *   For each reward function, you can list the `verifiers` (by their registered names) that must pass.
    *   A `verifier_penalty` (e.g., `0.0`) is specified. This value will be used as the score for the reward function if any of its associated verifiers fail.

    ```json
    // Snippet from a training configuration file:
    "reward_setup": {
        "functions": {
            "score_politeness": {
                 "verifiers": ["has_greeting"],       // Must pass the "has_greeting" verifier
                 "verifier_penalty": 0.0          // Score if "has_greeting" fails
             },
             // ... other reward functions ...
        }
    }
    ```

4.  **Execution Logic During Reward Calculation:**
    *   When a reward function (e.g., `score_politeness`) is about to be evaluated, the system first checks for any linked verifiers (e.g., `has_greeting`).
    *   These verifier functions are executed on the AI's `completion`.
    *   If **any** of the associated verifiers return `False`:
        *   The main logic of that reward function (e.g., `score_politeness`) is **skipped**.
        *   The reward function is assigned the configured `verifier_penalty` (e.g., `0.0`).
    *   If **all** associated verifiers return `True`:
        *   The main logic of the reward function is executed as normal to calculate its score.
    *   This conditional execution is managed by the `retrain.verifier.apply_verifiers_to_reward` utility, based on your configuration.

**Benefits:**

*   **Targeted Pre-Checks:** Apply specific basic checks only where relevant for a particular reward signal.
*   **Efficiency:** Avoid running computationally expensive reward function logic on outputs that fail fundamental criteria.
*   **Modularity & Reusability:** Simple verifier functions can be written once and reused as prerequisites for multiple different reward functions.
*   **Robustness:** Ensures that reward signals are more meaningful by first filtering out clearly inadequate responses.

