"""Build an OpenEnv environment and its seed dataset from TrainConfig.

TOML surface:

    [environment]
    provider = "openenv"
    id = "http://localhost:8765"           # base URL of a running server
    args = { renderer = "pkg.mod.render_prompt" }

Seeds are the dataset mechanism (as in verifiers' OpenEnv integration):
example *i* carries ``info.seed = config.seed + i`` and its prompt is the
rendered reset observation, fetched over one connection at load time with the
same positive ``max_turns`` used by live rollouts. Optional provenance guards
also pin each live seed to the task identity observed during this preload.
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
from retrain.environments.openenv.provenance import (
    ResetProvenance,
    ResetProvenanceGuard,
    parse_expected_task_ids,
    parse_expected_task_source,
)
from retrain.environments.openenv.render import (
    Renderer,
    default_renderer,
    render_messages,
    resolve_renderer,
)
from retrain.types import JSONObject

# Matches verifiers' OpenEnvEnv num_train_examples default; applies only when
# [data] max_examples is unset (0), since seeds are generated, not enumerated.
DEFAULT_NUM_EXAMPLES = 100

_KNOWN_ARGS = frozenset(
    {
        "expected_task_ids",
        "expected_task_source",
        "message_timeout_s",
        "renderer",
    }
)


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
    provenance_guard = ResetProvenanceGuard(
        expected_task_source=parse_expected_task_source(
            args.get("expected_task_source")
        ),
        expected_task_ids=parse_expected_task_ids(args.get("expected_task_ids")),
    )
    configured_max_turns = int(getattr(config, "environment_max_turns", -1))
    reset_max_turns = configured_max_turns if configured_max_turns > 0 else None
    action_schema = fetch_action_schema(base_url)
    return OpenEnvEnvironment(
        base_url,
        action_schema=action_schema,
        renderer=renderer,
        message_timeout_s=message_timeout_s,
        provenance_guard=provenance_guard,
        reset_max_turns=reset_max_turns,
    )


def examples_from_environment(
    env: OpenEnvEnvironment,
    config: object,
) -> list[Example]:
    """Render one Example per seed from the env's reset observations."""
    max_examples = int(getattr(config, "max_examples"))
    count = max_examples if max_examples > 0 else DEFAULT_NUM_EXAMPLES
    expected_task_ids = env.provenance_guard.expected_task_ids
    if expected_task_ids is not None and count < len(expected_task_ids):
        raise ValueError(
            "[data] max_examples must be at least the number of "
            f"expected_task_ids ({len(expected_task_ids)}), got {count}."
        )
    seed_value = int(getattr(config, "seed"))
    base_seed = seed_value if seed_value >= 0 else 0
    seeds = [base_seed + i for i in range(count)]

    observations = asyncio.run(_reset_observations(env, seeds))
    provenances = _validate_preload_provenance(env, seeds, observations)
    examples: list[Example] = []
    rows = zip(seeds, observations, provenances, strict=True)
    for index, (seed, result, provenance) in enumerate(rows):
        prompt = render_messages(
            env.renderer,
            result.observation,
            context="reset",
            action_schema=env.action_schema,
            seed=seed,
        )
        info: JSONObject = {"seed": seed}
        if provenance is not None:
            if provenance.task_id is not None:
                info["openenv_task_id"] = provenance.task_id
            if provenance.task_source is not None:
                info["openenv_task_source"] = provenance.task_source
        examples.append(
            Example(
                prompt=prompt,
                reference="",
                task="openenv",
                info=info,
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
        return [await client.reset(**env.reset_kwargs(seed)) for seed in seeds]
    finally:
        await client.close()


def _validate_preload_provenance(
    env: OpenEnvEnvironment,
    seeds: list[int],
    observations: list[StepResult],
) -> list[ResetProvenance | None]:
    guard = env.provenance_guard
    if not guard.enabled:
        return [None] * len(observations)
    provenances = [
        guard.validate(
            result.observation,
            context=f"OpenEnv preload reset for seed {seed}",
        )
        for seed, result in zip(seeds, observations, strict=True)
    ]
    guard.validate_exact_task_set(provenances, context="OpenEnv preload resets")
    return provenances
