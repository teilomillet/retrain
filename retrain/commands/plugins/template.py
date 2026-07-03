"""Plugin scaffold templates."""

from __future__ import annotations


def render(kind: str, fn_name: str) -> tuple[str, str]:
    """Return module content and a TOML snippet for a plugin kind."""
    if kind == "transform":
        return (
            (
                "from retrain import TransformOutput\n\n"
                f"def {fn_name}(ctx):\n"
                "    \"\"\"ctx contains episode_advantages/logprobs/planning masks/params.\"\"\"\n"
                "    token_advs = []\n"
                "    for i, logprobs in enumerate(ctx.logprobs_G):\n"
                "        adv = ctx.episode_advantages[i]\n"
                "        token_advs.append([adv for _ in logprobs])\n"
                "    return TransformOutput(\n"
                "        token_advs=token_advs,\n"
                "        has_stats=False,\n"
                "        needs_planning=False,\n"
                "        uses_sepa_controller=False,\n"
                "    )\n"
            ),
            (
                "[algorithm]\n"
                f'transform_mode = "plugins.{fn_name}.{fn_name}"\n'
                "\n[algorithm.transform_params]\n"
                "scale = 1.0\n"
            ),
        )
    if kind == "advantage":
        return (
            (
                f"def {fn_name}(rewards, params=None):\n"
                "    \"\"\"Return one advantage per reward.\"\"\"\n"
                "    if not rewards:\n"
                "        return []\n"
                "    scale = float((params or {}).get('scale', 1.0))\n"
                "    mean_r = sum(rewards) / len(rewards)\n"
                "    return [scale * (r - mean_r) for r in rewards]\n"
            ),
            (
                "[algorithm]\n"
                f'advantage_mode = "plugins.{fn_name}.{fn_name}"\n'
                "\n[algorithm.advantage_params]\n"
                "scale = 2.0\n"
            ),
        )
    if kind == "algorithm":
        return (
            (
                "from retrain import AlgorithmOutput\n\n"
                f"def {fn_name}(ctx):\n"
                "    \"\"\"Full algorithm hook: return token-level advantages directly.\"\"\"\n"
                "    token_advs = []\n"
                "    for rewards_idx, logprobs in enumerate(ctx.logprobs_G):\n"
                "        reward = ctx.rewards_G[rewards_idx]\n"
                "        token_advs.append([reward for _ in logprobs])\n"
                "    return AlgorithmOutput(token_advs=token_advs, has_stats=False)\n"
            ),
            (
                "[algorithm]\n"
                f'algorithm_mode = "plugins.{fn_name}.{fn_name}"\n'
                "\n[algorithm.params]\n"
                "alpha = 0.1\n"
            ),
        )
    if kind == "reward":
        return (
            (
                f"class {fn_name.title().replace('_', '')}Reward:\n"
                "    def score(self, response: str, reference: str) -> float:\n"
                "        return float(response.strip() == reference.strip())\n\n"
                f"def {fn_name}(config):\n"
                f"    return {fn_name.title().replace('_', '')}Reward()\n"
            ),
            (
                "[reward]\n"
                'type = "custom"\n'
                f'custom_module = "plugins.{fn_name}"\n'
                f'custom_function = "{fn_name}"\n'
            ),
        )

    if kind == "trainer":
        return (
            (
                f"def {fn_name}(config):\n"
                '    """Run training. Return adapter path or None.\n\n'
                "    config: retrain.config.TrainConfig\n"
                "    Expected output:\n"
                "      - metrics.jsonl in config.log_dir (for retrain status)\n"
                "      - adapter at config.adapter_path (for downstream use)\n"
                '    """\n'
                "    raise NotImplementedError\n"
            ),
            (
                "[training]\n"
                f'trainer = "plugins.{fn_name}.{fn_name}"\n'
            ),
        )

    generic = (
        f"def {fn_name}(config):\n"
        f"    raise NotImplementedError(\"Implement {kind} plugin contract here.\")\n"
    )
    return (
        generic,
        f"# Use dotted plugin path: plugins.{fn_name}.{fn_name}\n",
    )
