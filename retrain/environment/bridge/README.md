# `retrain.environment.bridge`

**Purpose**: This directory houses adapters that allow `retrain` to use external environment sources (like game benchmarks or simulators) which may not natively conform to `retrain`'s internal asynchronous `Environment` interface.

## Key Concepts

- Each subdirectory within `bridge` typically corresponds to a specific external source (e.g., `gg_bench` for environments from the GG-Bench project).
- These bridges are responsible for the "translation" layer, handling tasks such as:
    - Loading the external environment code or instance.
    - Applying any necessary or standard wrappers provided by the external source itself (e.g., for game logic, timeouts).
    - Converting the external environment's potentially synchronous API to `retrain`'s asynchronous API (as defined in `retrain.environment.Environment`).
    - Mapping action and observation types between `retrain`'s standardized types (e.g., `LLMAction`, `EnvironmentObservation`) and those used by the external environment, if they differ.

This modular approach allows `retrain`'s core training logic to remain agnostic to the specifics of diverse external training scenarios. It promotes cleaner code and makes it easier to integrate new types of environments in the future. 