"""Manual auto-block renderers."""

from __future__ import annotations


def render_commands(cli_name: str) -> list[str]:
    return [
        f"    {cli_name} [config.toml]",
        "        Run one training job.",
        "        If config.toml is omitted, uses ./retrain.toml when present.",
        "",
        f"    {cli_name} campaign.toml",
        "        Runs campaign mode when [campaign] exists in TOML.",
        "        Generates conditions x seeds matrix of training runs.",
        "        Set parallel = true in [campaign] for concurrent execution.",
        "        max_workers limits concurrent subprocess count.",
        "",
        f"    {cli_name} squeeze.toml",
        "        Runs squeeze mode when [squeeze] exists in TOML.",
        "        Analyzes LoRA rank via SVD and optionally compresses.",
        "",
        f"    {cli_name} doctor",
        "        Checks optional dependencies for configured components.",
        "",
        f"    {cli_name} backends [--json]",
        "        Prints backend metadata (capabilities, deps, option schema).",
        "        --json            machine-readable JSON output.",
        "",
        f"    {cli_name} migrate-config <config.toml> [--check|--write|--output PATH] [--backup] [--stdin|--stdout] [--json]",
        "        Migrates legacy [backend] prime_rl_* keys into [backend.options].",
        "        Default mode previews a unified diff and does not write files.",
        "        --check           exits 1 when migration is required.",
        "        --write           writes migration in place.",
        "        --output PATH     writes migrated config to a new path.",
        "        --backup          writes <config>.bak before in-place write.",
        "        --stdin           reads source TOML from stdin.",
        "        --stdout          prints migrated TOML instead of a diff.",
        "        --json            machine-readable report.",
        "",
        f"    {cli_name} init [--template NAME] [--list] [--interactive]",
        "        Writes a starter config in the current directory.",
        "        Templates: default, quickstart, experiment, campaign.",
        "        --template NAME   selects a template (default: default).",
        "        --list            shows available templates.",
        "        --interactive/-i  guided setup with prompts.",
        "",
        f"    {cli_name} init-plugin --kind KIND --name NAME [--output-dir DIR] [--with-test]",
        "        Scaffolds a student-friendly plugin module.",
        "        Kinds: transform, advantage, algorithm, reward, planning, data,",
        "               backend, inference, backpressure.",
        "        --with-test       also generates a smoke test template.",
        "",
        f"    {cli_name} plugins [--json] [config.toml]",
        "        Lists built-ins + discovered plugin modules.",
        "        If config.toml is provided, uses its [plugins] search_paths.",
        "",
        f"    {cli_name} status [logdir] [--json] [--all] [--watch]",
        "        Scans log directories for run and campaign status.",
        "        Defaults to ./logs when no path is given.",
        "        Shows only active campaigns by default (last 24h).",
        "        --json            machine-readable JSON output.",
        "        --all             show all campaigns including old dead/done.",
        "        --active          show only active campaigns (default).",
        "        --watch           refresh every 5 seconds (Ctrl-C to stop).",
        "",
        f"    {cli_name} top [logdir]",
        "        Live dashboard: alias for status --watch --active.",
        "",
        f"    {cli_name} explain [config.toml] [--json]",
        "        Dry-run preview: shows what a config would do.",
        "        Works for single runs, campaigns, and squeeze configs.",
        "        --json            machine-readable JSON output.",
        "",
        f"    {cli_name} diff <run_a> <run_b> [--json]",
        "        Compares metrics between two training runs.",
        f"    {cli_name} diff <campaign_dir> <cond_a> <cond_b> [--json]",
        "        Compares two conditions in a campaign (averaged across seeds).",
        "        --json            machine-readable JSON output.",
        "",
        f"    {cli_name} benchmark <config.toml> [--repeat N] [--output-dir DIR] [--json]",
        "        Runs one or more benchmark repetitions with isolated log/adapters.",
        "        Disables wandb by default to reduce measurement noise.",
        f"    {cli_name} benchmark <run_dir|suite_dir> [--json]",
        "        Summarizes an existing run or benchmark suite directory.",
        "        --repeat N        number of repeated executions (config mode only).",
        "        --output-dir DIR  benchmark suite root (config mode only).",
        "        --json            machine-readable JSON output.",
        "",
        f"    {cli_name} trace [config.toml] [--json]",
        "        Pre-flight validation: build flow, trace with synthetic data.",
        "        Catches incompatible modes, missing data, and unsupported",
        "        uncertainty kinds before committing GPU time.",
        "        --json            machine-readable JSON output.",
        "",
        f"    {cli_name} man",
        "        Shows this manual.",
        "        --topic <name>    prints one section.",
        "        --path            prints the manual file path.",
        "        --list-topics     lists supported topic names.",
        "        --sync            refreshes auto-generated manual blocks.",
        "        --check           exits non-zero if auto blocks are stale.",
        "        --json            outputs JSON (full manual or single topic).",
    ]


def render_options() -> list[str]:
    from retrain.config import _CLI_FLAG_MAP

    # Keep only canonical flags for readability; skip alias here and add below.
    canonical = sorted(k for k in _CLI_FLAG_MAP if k != "--resume")
    lines = [
        "    TrainConfig fields are exposed as --kebab-case CLI flags.",
        "    Flags override TOML values.",
        "",
        "    Common examples:",
        "        retrain config.toml --seed 42 --max-steps 50",
        "        retrain config.toml --lr 1e-4 --batch-size 16",
        "        retrain config.toml --advantage-mode grpo",
        "        retrain config.toml --advantage-param scale=2.0 --transform-param cap=0.1",
        "        retrain config.toml --inference-engine vllm --inference-url http://localhost:8000",
        "        retrain config.toml --backend prime_rl --backend-opt transport=zmq --backend-opt zmq_port=7777",
        "",
        "    All flags (sorted):",
    ]
    for flag in canonical:
        lines.append(f"        {flag}")

    lines.extend(
        [
            "",
            "    Special flags:",
            "        --backend-opt K=V    backend-specific option override (repeatable).",
            "        --algorithm-param K=V algorithm plugin params (repeatable).",
            "        --advantage-param K=V advantage plugin params (repeatable).",
            "        --transform-param K=V transform plugin params (repeatable).",
            "        --resume VALUE    alias for --resume-from VALUE",
            "",
            "    Unknown flags produce an error with close-match suggestions.",
        ]
    )
    return lines


def render_quickstart(cli_name: str) -> list[str]:
    return [
        "    cp retrain.toml my_run.toml",
        f"    {cli_name} my_run.toml",
        f"    {cli_name} my_run.toml --seed 42 --max-steps 50",
    ]


def render_environment(cli_name: str) -> list[str]:
    from retrain.environments.verifiers import _FALLBACK_TRAINING_ENVS

    lines = [
        f"    {cli_name} uses verifiers environments for RLVR training data.",
        "    Set [environment].provider = \"verifiers\" and specify a Hub ID.",
        "",
        "    Trainable verifiers examples:",
    ]
    for env_id in _FALLBACK_TRAINING_ENVS:
        lines.append(f"        {env_id}")
    lines.extend(
        [
            "",
            "    Caveat:",
            "        Some hub environments are eval-only and do not expose training",
            f"        datasets. In that case {cli_name} fails fast with actionable guidance.",
        ]
    )
    return lines
