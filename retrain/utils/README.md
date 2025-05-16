# Utilities (`retrain.utils`)

This directory contains shared helper functions, common tools, and utility modules that support various parts of the `retrain` library. The goal is to provide reusable components for common tasks.

## Key Utilities

Below are some of the primary utilities found in this directory:

-   **`parser/` (sub-directory)**:
    *   **Purpose**: Provides tools for understanding (parsing) structured text that AI models output, and for formatting data back into text that models can read. This is crucial for interactions where the model needs to use tools or provide answers in a specific format.
    *   **Key Components**: Includes a base template for creating new parsers (`BaseParser`) and a specific parser for `smol-agents` style XML-like output (`SmolAgentParser`).
    *   Refer to the [`parser/README.md`](./parser/README.md) for more details.

-   **`prompt_source.py`**:
    *   **Purpose**: Manages the supply of initial prompts or questions that are used to start an interaction with the AI model during training or evaluation. It can handle various ways of sourcing these prompts, such as from a file or a dataset.

-   **`logging_utils.py`**:
    *   **Purpose**: Offers helper functions to set up and manage logging consistently across the `retrain` project. This ensures that log messages are formatted and handled in a standard way, aiding in debugging and monitoring.

-   **`model_utils.py`**:
    *   **Purpose**: Contains general helper functions related to AI models. These are typically smaller, reusable functions that might assist with model loading, configuration checking, or other common model-related tasks that don't fit into the main `retrain.model` abstraction.

## General Guidelines

-   Utilities in this directory should ideally be broadly applicable within the `retrain` project.
-   If a utility becomes complex or has several related components, it's good practice to organize it into its own sub-directory (like `parser/`) with its own README. 