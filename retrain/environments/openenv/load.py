"""Build an OpenEnv environment and its seed dataset from TrainConfig.

TOML surface:

    [environment]
    provider = "openenv"
    id = "http://localhost:8765"           # base URL of a running server
    args = { renderer = "pkg.mod.render_prompt" }

Seeds are the dataset mechanism (as in verifiers' OpenEnv integration):
example *i* carries ``info.seed = config.seed + i`` and its prompt is the
rendered ``reset(seed)`` observation, fetched over one connection at load
time so training prompts match exactly what rollouts will see.
"""

from __future__ import annotations

import asyncio
from typing import cast

from retrain.data.source import Example
from retrain.environments import load as env_load
from retrain.environments.openenv.client import (
    OpenEnvClient,
    StepResult,
    fetch_action_schema,
)
from retrain.environments.openenv.environment import OpenEnvEnvironment
from retrain.environments.openenv.render import (
    Renderer,
    default_renderer,
    render_messages,
    resolve_renderer,
)

# Matches verifiers' OpenEnvEnv num_train_examples default; applies only when
# [data] max_examples is unset (0), since seeds are generated, not enumerated.
DEFAULT_NUM_EXAMPLES = 100

_KNOWN_ARGS = frozenset({"renderer", "message_timeout_s"})


def load_environment(config: object) -> OpenEnvEnvironment:
    """Construct an OpenEnvEnvironment from ``[environment]`` config."""
    base_url = str(getattr(config, "environment_id"))
    args = env_load.parse_args(getattr(config, "environment_args"))
    unknown = set(args) - _KNOWN_ARGS
    if unknown:
        raise ValueError(
            f"Unknown [environment] args for provider 'openenv': "
            f"{sorted(unknown)}. Known: {sorted(_KNOWN_ARGS)}."
        )

    renderer: Renderer = default_renderer
    renderer_target = args.get("renderer")
    if renderer_target:
        renderer = resolve_renderer(str(renderer_target))

    message_timeout_s = float(cast(float, args.get("message_timeout_s", 300.0)))
    action_schema = fetch_action_schema(base_url)
    return OpenEnvEnvironment(
        base_url,
        action_schema=action_schema,
        renderer=renderer,
        message_timeout_s=message_timeout_s,
    )


def examples_from_environment(
    env: OpenEnvEnvironment,
    config: object,
) -> list[Example]:
    """Render one Example per seed from the env's reset observations."""
    max_examples = int(getattr(config, "max_examples"))
    count = max_examples if max_examples > 0 else DEFAULT_NUM_EXAMPLES
    seed_value = int(getattr(config, "seed"))
    base_seed = seed_value if seed_value >= 0 else 0
    seeds = [base_seed + i for i in range(count)]

    observations = asyncio.run(_reset_observations(env, seeds))
    examples: list[Example] = []
    for index, (seed, result) in enumerate(zip(seeds, observations, strict=True)):
        prompt = render_messages(
            env.renderer,
            result.observation,
            context="reset",
            action_schema=env.action_schema,
            seed=seed,
        )
        examples.append(
            Example(
                prompt=prompt,
                reference="",
                task="openenv",
                info={"seed": seed},
                example_id=index,
            )
        )
    return examples


async def _reset_observations(
    env: OpenEnvEnvironment,
    seeds: list[int],
) -> list[StepResult]:
    """Fetch reset observations for all seeds over a single connection."""
    client = cast(OpenEnvClient, env.client_factory(env.base_url))
    try:
        await client.connect()
        return [await client.reset(seed=seed) for seed in seeds]
    finally:
        await client.close()
