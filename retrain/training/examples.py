"""Training example loading across data-source and environment providers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from retrain.config import TrainConfig
from retrain.data.source import Example


class DataSource(Protocol):
    def load(self) -> list[Example]:
        """Load training examples."""
        ...


@dataclass(frozen=True)
class LoadedTrainingExamples:
    examples: list[Example]
    environment: object | None = None


def load_training_examples(
    config: TrainConfig,
    *,
    data_source_factory: Callable[[str, TrainConfig], DataSource],
    verifiers_environment_loader: Callable[[TrainConfig], object],
    verifiers_examples_loader: Callable[[object, TrainConfig], list[Example]],
) -> LoadedTrainingExamples:
    """Load examples from the configured source and optional environment."""
    print("Loading dataset...")
    environment = None
    if config.environment_provider == "verifiers":
        environment = verifiers_environment_loader(config)
        examples = verifiers_examples_loader(environment, config)
        print(
            f"Loaded {len(examples)} examples from verifiers env "
            f"'{config.environment_id}'"
        )
    elif config.environment_provider == "openenv":
        from retrain.environments.openenv import (
            examples_from_environment as load_examples_from_openenv,
            load_environment as load_openenv_environment,
        )

        environment = load_openenv_environment(config)
        examples = load_examples_from_openenv(environment, config)
        print(
            f"Loaded {len(examples)} seed examples from OpenEnv server "
            f"'{config.environment_id}'"
        )
    else:
        examples = data_source_factory(config.data_source, config).load()

    if environment is not None:
        if config.data_source != "math":
            print("NOTE: [data].source is ignored when [environment].provider is set.")
        if config.reward_type != "match":
            print(
                "NOTE: [reward] settings are ignored when [environment].provider "
                "is set; the environment rubric is used."
            )
    return LoadedTrainingExamples(examples=examples, environment=environment)
