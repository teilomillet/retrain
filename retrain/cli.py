"""Single entry point for retrain.

Usage:
    retrain                  # loads retrain.toml from cwd
    retrain config.toml      # single training run
    retrain campaign.toml    # campaign (if TOML has [campaign] section)
    retrain backends         # list backend capabilities/schema metadata
    retrain init             # generate a starter retrain.toml
    retrain doctor           # check installed dependencies for all components
    retrain migrate-config config.toml   # migrate legacy backend keys
    retrain man              # human/agent-friendly manual
    retrain --seed 42 --lr 1e-4   # override config values from CLI

A TOML with a [campaign] section runs multiple conditions × seeds.
A TOML without it runs a single training job. Same command either way.
"""

from __future__ import annotations

import difflib
import json
import os
import sys
import tomllib
from pathlib import Path

from retrain.commands import manual as manual_command
from retrain.commands.backends.capability import payload as capability_payload
from retrain.commands.backends.capability import summary as capability_summary
from retrain.commands.backends.run import run as run_backends
from retrain.commands.doctor.run import run as run_doctor
from retrain.commands.doctor.warn import warn_missing
from retrain.commands.init.run import run as run_init
from retrain.commands.name import resolve as resolve_cli_name
from retrain.commands.plugins.run import run as run_plugins
from retrain.commands.plugins.scaffold import run as run_init_plugin
from retrain.commands.help import print_help
from retrain.commands.status.run import run as run_status
from retrain.commands.status.top import run as run_top


def _manual_path() -> Path:
    """Location of the editable bundled manual file."""
    return Path(__file__).with_name("retrain.man")


def _load_dotenv() -> None:
    """Load .env file if present. Sets vars into os.environ."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        eq = line.find("=")
        if eq == -1:
            continue
        key = line[:eq].strip()
        val = line[eq + 1 :].strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        os.environ[key] = val
    print("Loaded .env", file=sys.stderr)


def _explain_single(config_path: str | None, fmt: str) -> None:
    """Explain what a single training run would do."""
    import warnings

    from retrain.config import load_config
    from retrain.registry import check_environment

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = load_config(config_path)

    if config.trainer == "sft":
        condition = "sft"
        datums_per_step = config.sft_batch_size if config.sft_batch_size > 0 else config.batch_size
    else:
        condition = (
            config.algorithm_mode
            if config.algorithm_mode
            else f"{config.advantage_mode}+{config.transform_mode}"
        )
        datums_per_step = config.batch_size * config.group_size
    total_datums = datums_per_step * config.max_steps
    sft_loss_fn = config.sft_loss_fn
    if sft_loss_fn == "auto":
        sft_loss_fn = "cross_entropy" if config.trainer == "sft" else "importance_sampling"
    lora_alpha = config.lora_alpha if config.lora_alpha else config.lora_rank * 2
    data_info = config.data_source
    if config.environment_provider:
        data_info = f"{config.environment_provider}:{config.environment_id}"
    if config.trainer == "sft":
        data_info = config.sft_data_path
    backend_capabilities = capability_payload(
        config.backend,
        config.backend_options,
    )

    info: dict = {
        "mode": "single",
        "config": config_path or "retrain.toml",
        "model": config.model,
        "trainer": config.trainer,
        "backend": config.backend,
        "backend_options": dict(config.backend_options),
        "backend_capabilities": backend_capabilities,
        "condition": condition,
        "algorithm_mode": config.algorithm_mode,
        "advantage_mode": config.advantage_mode,
        "transform_mode": config.transform_mode,
        "max_steps": config.max_steps,
        "batch_size": config.batch_size,
        "group_size": config.group_size,
        "datums_per_step": datums_per_step,
        "total_datums": total_datums,
        "sft_warmup_steps": config.sft_warmup_steps,
        "sft_data_path": config.sft_data_path,
        "sft_batch_size": config.sft_batch_size,
        "sft_max_tokens": config.sft_max_tokens,
        "sft_loss_fn": sft_loss_fn,
        "sft_batch_order": config.sft_batch_order,
        "sft_length_bucket_size": config.sft_length_bucket_size,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "lr": config.lr,
        "seed": config.seed,
        "lora_rank": config.lora_rank,
        "lora_alpha": lora_alpha,
        "data": data_info,
        "reward_type": config.reward_type,
        "log_dir": config.log_dir,
        "adapter_path": config.adapter_path,
        "wandb_project": config.wandb_project or "(disabled)",
        "warnings": [str(w.message) for w in caught],
    }

    # Dependency warnings
    dep_warnings = []
    results = check_environment(config=config)
    for name, import_name, hint, available in results:
        if not available:
            dep_warnings.append(f"{name} requires {import_name} ({hint})")
    if dep_warnings:
        info["dep_warnings"] = dep_warnings

    if fmt == "json":
        print(json.dumps(info, indent=2))
        return

    print(f"retrain explain — dry-run preview")
    print(f"  config        : {info['config']}")
    print(f"  model         : {config.model}")
    print(f"  trainer       : {config.trainer}")
    print(f"  backend       : {config.backend}")
    print(f"  backend caps  : {capability_summary(backend_capabilities)}")
    if not backend_capabilities["reports_sync_loss"]:
        print("  note          : loss is reported as placeholder by backend design")
    print(f"  condition     : {condition}")
    print(f"  steps         : {config.max_steps}")
    print(f"  batch_size    : {config.batch_size}")
    if config.trainer == "sft":
        print(f"  sft_batch     : {datums_per_step}")
    else:
        print(f"  group_size    : {config.group_size}")
    print(f"  datums/step   : {datums_per_step}")
    print(f"  total datums  : {total_datums}")
    print(f"  max_tokens    : {config.max_tokens}")
    print(f"  temperature   : {config.temperature}")
    print(f"  lr            : {config.lr}")
    if config.trainer == "sft" or config.sft_warmup_steps > 0:
        print(
            "  sft           : "
            f"steps={config.max_steps if config.trainer == 'sft' else config.sft_warmup_steps} "
            f"batch={config.sft_batch_size or '(default)'} "
            f"loss={sft_loss_fn} "
            f"order={config.sft_batch_order}"
        )
        if config.sft_data_path:
            print(f"  sft_data      : {config.sft_data_path}")
    print(f"  seed          : {config.seed}")
    print(f"  lora          : rank={config.lora_rank} alpha={lora_alpha}")
    print(f"  data          : {data_info}")
    print(f"  reward        : {config.reward_type}")
    print(f"  log_dir       : {config.log_dir}")
    print(f"  adapter_path  : {config.adapter_path}")
    print(f"  wandb         : {info['wandb_project']}")
    if caught:
        print("\nWarnings:")
        for w in caught:
            print(f"  - {w.message}")
    if dep_warnings:
        print("\nMissing dependencies:")
        for dw in dep_warnings:
            print(f"  - {dw}")


def _explain_campaign(config_path: str, fmt: str) -> None:
    """Explain what a campaign would do."""
    from retrain.campaign import DEFAULT_SEEDS, _parse_campaign_conditions

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    campaign = data.get("campaign", {})
    seeds = campaign.get("seeds", DEFAULT_SEEDS)
    max_steps = campaign.get("max_steps", 500)
    raw_conditions = campaign.get("conditions", None)
    conditions = _parse_campaign_conditions(raw_conditions, config_path)
    condition_labels = [c.label for c in conditions]
    total_runs = len(conditions) * len(seeds)
    backend_sec = data.get("backend", {})
    backend_name = "local"
    backend_options: dict[str, object] = {}
    if isinstance(backend_sec, dict):
        backend_name = str(backend_sec.get("backend", "local") or "local")
        raw_options = backend_sec.get("options", {})
        if isinstance(raw_options, dict):
            backend_options = dict(raw_options)
    backend_capabilities = capability_payload(
        backend_name,
        backend_options,
    )

    info = {
        "mode": "campaign",
        "config": config_path,
        "backend": backend_name,
        "backend_capabilities": backend_capabilities,
        "conditions": condition_labels,
        "seeds": seeds,
        "max_steps": max_steps,
        "total_runs": total_runs,
    }

    if fmt == "json":
        print(json.dumps(info, indent=2))
        return

    print(f"retrain explain — campaign dry-run preview")
    print(f"  config        : {config_path}")
    print(f"  backend       : {backend_name}")
    print(f"  backend caps  : {capability_summary(backend_capabilities)}")
    print(f"  conditions    : {', '.join(condition_labels)}")
    print(f"  seeds         : {seeds}")
    print(f"  max_steps     : {max_steps}")
    print(f"  total runs    : {total_runs}")


def _explain_squeeze(config_path: str, fmt: str) -> None:
    """Explain what a squeeze run would do."""
    from retrain.config import load_squeeze_config

    cfg = load_squeeze_config(config_path)

    info = {
        "mode": "squeeze",
        "config": config_path,
        "adapter_path": cfg.adapter_path,
        "source_rank": cfg.source_rank,
        "min_variance_retention": cfg.min_variance_retention,
    }
    if cfg.output_path:
        info["output_path"] = cfg.output_path
    if cfg.compress_to > 0:
        info["compress_to"] = cfg.compress_to

    if fmt == "json":
        print(json.dumps(info, indent=2))
        return

    print(f"retrain explain — squeeze dry-run preview")
    print(f"  config                : {config_path}")
    print(f"  adapter_path          : {cfg.adapter_path}")
    print(f"  source_rank           : {cfg.source_rank}")
    print(f"  min_variance_retention: {cfg.min_variance_retention}")
    if cfg.output_path:
        print(f"  output_path           : {cfg.output_path}")
    if cfg.compress_to > 0:
        print(f"  compress_to           : {cfg.compress_to}")


def _run_diff(args: list[str]) -> None:
    """Compare two runs or campaign conditions."""
    from retrain.diff import diff_conditions, diff_runs, format_diff

    fmt = "text"
    positional: list[str] = []
    for arg in args:
        if arg == "--json":
            fmt = "json"
        elif arg.startswith("--"):
            print(f"Unknown diff flag: {arg}", file=sys.stderr)
            sys.exit(1)
        else:
            positional.append(arg)

    if len(positional) == 2:
        dir_a, dir_b = Path(positional[0]), Path(positional[1])
        try:
            result = diff_runs(dir_a, dir_b)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
    elif len(positional) == 3:
        campaign_dir = Path(positional[0])
        cond_a, cond_b = positional[1], positional[2]
        try:
            result = diff_conditions(campaign_dir, cond_a, cond_b)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
    else:
        print("Usage:", file=sys.stderr)
        print("  retrain diff <run_a> <run_b>", file=sys.stderr)
        print("  retrain diff <campaign_dir> <cond_a> <cond_b>", file=sys.stderr)
        sys.exit(1)

    if fmt == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_diff(result))


def _run_benchmark(args: list[str]) -> None:
    """Run or summarize a benchmark suite."""
    from retrain.benchmark import (
        default_benchmark_output_dir,
        format_run_summary,
        format_suite_summary,
        run_benchmark_suite,
        summarize_run,
        summarize_suite,
    )
    from retrain.config import load_config, parse_cli_overrides
    from retrain.registry import get_registry

    fmt = "text"
    repeats = 1
    output_dir: str | None = None
    passthrough: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--json":
            fmt = "json"
        elif arg.startswith("--repeat="):
            repeats = int(arg.split("=", 1)[1])
        elif arg == "--repeat":
            i += 1
            if i >= len(args):
                print("Flag --repeat requires a value.", file=sys.stderr)
                sys.exit(1)
            repeats = int(args[i])
        elif arg.startswith("--output-dir="):
            output_dir = arg.split("=", 1)[1]
        elif arg == "--output-dir":
            i += 1
            if i >= len(args):
                print("Flag --output-dir requires a value.", file=sys.stderr)
                sys.exit(1)
            output_dir = args[i]
        else:
            passthrough.append(arg)
        i += 1

    config_path, overrides = parse_cli_overrides(passthrough)
    if config_path is None:
        print("Usage:", file=sys.stderr)
        print(
            "  retrain benchmark <config.toml> [--repeat N] [--output-dir DIR] [--json]",
            file=sys.stderr,
        )
        print(
            "  retrain benchmark <run_dir|suite_dir> [--json]",
            file=sys.stderr,
        )
        sys.exit(1)

    target = Path(config_path)
    if target.is_dir():
        try:
            if (target / "metrics.jsonl").is_file():
                run_summary = summarize_run(target)
                if fmt == "json":
                    print(json.dumps(run_summary.to_dict(), indent=2))
                else:
                    print(format_run_summary(run_summary))
                return
            suite_summary = summarize_suite(target)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        if fmt == "json":
            print(json.dumps(suite_summary.to_dict(), indent=2))
        else:
            print(format_suite_summary(suite_summary))
        return

    if not target.is_file():
        print(f"File not found: {target}", file=sys.stderr)
        sys.exit(1)

    config = load_config(str(target), overrides=overrides)
    warn_missing(config)
    suite_dir = Path(output_dir) if output_dir else default_benchmark_output_dir(
        str(target),
        config,
    )
    suite_summary = run_benchmark_suite(
        config,
        repeats=repeats,
        output_dir=suite_dir,
        runner_factory=lambda cfg: get_registry("trainer").create(cfg.trainer, cfg),
        disable_wandb=True,
    )
    if fmt == "json":
        print(json.dumps(suite_summary.to_dict(), indent=2))
    else:
        print(format_suite_summary(suite_summary))


def _run_migrate_config(args: list[str]) -> None:
    """Migrate legacy backend config keys to [backend.options] format."""
    from retrain.config import migrate_legacy_backend_keys_toml_text

    check_only = False
    write_in_place = False
    backup = False
    stdin_mode = False
    stdout_mode = False
    json_mode = False
    output_path: str | None = None
    positional: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--check":
            check_only = True
        elif arg == "--write":
            write_in_place = True
        elif arg == "--backup":
            backup = True
        elif arg == "--stdin":
            stdin_mode = True
        elif arg == "--stdout":
            stdout_mode = True
        elif arg == "--json":
            json_mode = True
        elif arg in ("--output", "-o"):
            i += 1
            if i >= len(args):
                print("Flag --output requires a path.", file=sys.stderr)
                sys.exit(1)
            output_path = args[i]
        elif arg.startswith("--output="):
            output_path = arg.split("=", 1)[1]
        elif arg.startswith("--"):
            print(f"Unknown migrate-config flag: {arg}", file=sys.stderr)
            sys.exit(1)
        else:
            positional.append(arg)
        i += 1

    if stdin_mode and positional:
        print("Use either a config path or --stdin, not both.", file=sys.stderr)
        sys.exit(1)
    if not stdin_mode and len(positional) != 1:
        print(
            "Usage: retrain migrate-config <config.toml> "
            "[--check|--write|--output PATH] [--backup] [--stdin|--stdout] [--json]",
            file=sys.stderr,
        )
        sys.exit(1)
    if check_only and (write_in_place or output_path or stdout_mode or backup):
        print(
            "Flag --check cannot be combined with --write, --output, --stdout, or --backup.",
            file=sys.stderr,
        )
        sys.exit(1)
    if stdin_mode and write_in_place:
        print("Flag --write requires a file path input (cannot be used with --stdin).", file=sys.stderr)
        sys.exit(1)
    if write_in_place and output_path:
        print("Use either --write or --output, not both.", file=sys.stderr)
        sys.exit(1)
    if stdout_mode and (write_in_place or output_path):
        print("Flag --stdout cannot be combined with --write or --output.", file=sys.stderr)
        sys.exit(1)
    if backup and not write_in_place:
        print("Flag --backup can only be used with --write.", file=sys.stderr)
        sys.exit(1)

    config_path: Path | None = None
    source_label = "<stdin>"
    if stdin_mode:
        original_text = sys.stdin.read()
        if not original_text:
            print("No TOML content received on stdin.", file=sys.stderr)
            sys.exit(1)
    else:
        config_path = Path(positional[0])
        if not config_path.is_file():
            print(f"File not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        source_label = str(config_path)
        original_text = config_path.read_text()

    try:
        migrated = migrate_legacy_backend_keys_toml_text(original_text)
    except tomllib.TOMLDecodeError as exc:
        print(f"Invalid TOML in {source_label}: {exc}", file=sys.stderr)
        sys.exit(1)

    needs_migration = bool(migrated.legacy_keys)
    diff_text = "\n".join(
        difflib.unified_diff(
            original_text.splitlines(),
            migrated.output_text.splitlines(),
            fromfile=source_label,
            tofile=f"{source_label}.migrated",
            lineterm="",
        )
    )

    mode = "preview"
    if check_only:
        mode = "check"
    elif write_in_place:
        mode = "write"
    elif output_path:
        mode = "output"
    elif stdout_mode:
        mode = "stdout"

    payload: dict[str, object] = {
        "config": source_label,
        "mode": mode,
        "needs_migration": needs_migration,
        "changed": migrated.changed,
        "legacy_keys": list(migrated.legacy_keys),
        "merged_backend_options": migrated.merged_backend_options,
        "diff": diff_text,
        "written": False,
        "output_path": None,
        "backup_path": None,
    }

    if check_only:
        if json_mode:
            print(json.dumps(payload, indent=2))
        elif needs_migration:
            keys = ", ".join(migrated.legacy_keys)
            print(f"Migration required in {source_label} (legacy keys: {keys}).")
        else:
            print(f"No migration needed: {source_label}")
        if needs_migration:
            sys.exit(1)
        return

    if write_in_place:
        assert config_path is not None
        if backup:
            backup_path = Path(str(config_path) + ".bak")
            backup_path.write_text(original_text)
            payload["backup_path"] = str(backup_path)
        config_path.write_text(migrated.output_text)
        payload["written"] = True
        payload["output_path"] = str(config_path)
    elif output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(migrated.output_text)
        payload["written"] = True
        payload["output_path"] = str(out_path)

    if json_mode:
        print(json.dumps(payload, indent=2))
        return

    if stdout_mode:
        print(migrated.output_text, end="")
        return

    if payload["written"]:
        if needs_migration:
            print(f"Migrated config written to {payload['output_path']}")
        else:
            print(f"No migration required. Wrote unchanged config to {payload['output_path']}")
        if payload["backup_path"]:
            print(f"Backup written to {payload['backup_path']}")
        return

    if needs_migration:
        print(diff_text)
    else:
        print(f"No migration needed: {source_label}")


def _run_explain(args: list[str]) -> None:
    """Dry-run: show what a config would do without running it."""
    fmt = "text"
    config_path: str | None = None
    for arg in args:
        if arg == "--json":
            fmt = "json"
        elif arg.startswith("--"):
            print(f"Unknown explain flag: {arg}", file=sys.stderr)
            sys.exit(1)
        else:
            config_path = arg

    # Resolve config path
    if config_path is None:
        if Path("retrain.toml").is_file():
            config_path = "retrain.toml"
        else:
            print("No config file specified and no retrain.toml in cwd.")
            sys.exit(1)

    if not Path(config_path).is_file():
        print(f"File not found: {config_path}")
        sys.exit(1)

    # Route by config type
    from retrain.config import config_kind

    kind = config_kind(config_path)
    if kind == "campaign":
        _explain_campaign(config_path, fmt)
    elif kind == "squeeze":
        _explain_squeeze(config_path, fmt)
    else:
        _explain_single(config_path, fmt)


def _run_trace(args: list[str]) -> None:
    """Pre-flight validation: build flow, trace with synthetic data."""
    from retrain.config import load_config
    from retrain.flow import build_flow

    config_path: str | None = None
    json_mode = False
    for arg in args:
        if arg == "--json":
            json_mode = True
        elif not arg.startswith("--") and config_path is None:
            config_path = arg
        else:
            print(f"Unknown argument: {arg}", file=sys.stderr)
            sys.exit(1)

    if config_path is not None and not Path(config_path).is_file():
        print(f"File not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(1)

    flow = build_flow(config, gpu=False)
    result = flow.trace()

    if json_mode:
        payload = {
            "ok": result.ok,
            "probe_cases_run": result.probe_cases_run,
            "probe_cases_passed": result.probe_cases_passed,
            "issues": [
                {
                    "severity": i.severity,
                    "category": i.category,
                    "message": i.message,
                }
                for i in result.issues
            ],
            "flow": {
                "condition_label": flow.condition_label,
                "backend": config.backend,
                "backend_capability_source": flow.backend_capability_source,
                "needs_planning": flow.needs_planning,
                "uses_sepa_controller": flow.uses_sepa_controller,
                "preserves_token_advantages": flow.backend_capabilities.preserves_token_advantages,
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Condition: {flow.condition_label}")
        print(f"Backend:   {config.backend} (source: {flow.backend_capability_source})")
        print(
            f"Probes:    {result.probe_cases_passed}/{result.probe_cases_run} passed"
        )
        if result.issues:
            print()
            for issue in result.issues:
                tag = "[ERROR]" if issue.severity == "error" else "[WARN]"
                print(f"  {tag} [{issue.category}] {issue.message}")
        print()
        if result.ok:
            print("PASS")
        else:
            print("FAIL")

    sys.exit(0 if result.ok else 1)


def _run_tree(args: list[str]) -> None:
    """Experiment tech-tree commands."""
    from retrain.tree import (
        format_next,
        format_show,
        format_tree,
        format_tree_json,
        load_tree,
        reset_node,
    )

    usage = (
        "Usage:\n"
        "  retrain tree [tree.toml] [--json]          — view tree\n"
        "  retrain tree next [tree.toml]              — show pending nodes\n"
        "  retrain tree show <node> [tree.toml] [--json] — node details\n"
        "  retrain tree run <node> [tree.toml]        — launch node's campaign\n"
        '  retrain tree note <node> "text" [tree.toml] — add annotation\n'
        "  retrain tree eval [tree.toml]              — evaluate success conditions\n"
        "  retrain tree reset <node> [tree.toml]      — reset node to pending\n"
    )

    if not args or args[0] in ("-h", "--help"):
        print(usage)
        return

    # Extract --json flag
    json_flag = "--json" in args
    if json_flag:
        args = [a for a in args if a != "--json"]

    # Determine subcommand vs bare tree path
    subcommands = {"next", "run", "note", "eval", "show", "reset"}
    if args and args[0] in subcommands:
        subcmd = args[0]
        rest = args[1:]
    else:
        # Default: view
        subcmd = "view"
        rest = args

    # --- view ---
    if subcmd == "view":
        tree_path = rest[0] if rest else "tree.toml"
        tree = load_tree(tree_path)
        if json_flag:
            print(json.dumps(format_tree_json(tree), indent=2))
        else:
            print(format_tree(tree))
        return

    # --- next ---
    if subcmd == "next":
        tree_path = rest[0] if rest else "tree.toml"
        tree = load_tree(tree_path)
        print(format_next(tree))
        return

    # --- show <node> [tree.toml] ---
    if subcmd == "show":
        if not rest:
            print("Error: 'show' requires a node id.")
            print(usage)
            sys.exit(1)
        node_id = rest[0]
        tree_path = rest[1] if len(rest) > 1 else "tree.toml"
        tree = load_tree(tree_path)
        try:
            if json_flag:
                data = format_tree_json(tree)
                node_data = [n for n in data["nodes"] if n["id"] == node_id]
                if not node_data:
                    raise KeyError(node_id)
                print(json.dumps(node_data[0], indent=2))
            else:
                print(format_show(tree, node_id))
        except KeyError:
            print(f"Error: unknown node {node_id!r}")
            sys.exit(1)
        return

    # --- run <node> [tree.toml] ---
    if subcmd == "run":
        if not rest:
            print("Error: 'run' requires a node id.")
            print(usage)
            sys.exit(1)
        node_id = rest[0]
        tree_path = rest[1] if len(rest) > 1 else "tree.toml"
        tree = load_tree(tree_path)
        _tree_run_node(tree, node_id)
        return

    # --- note <node> "text" [tree.toml] ---
    if subcmd == "note":
        if len(rest) < 2:
            print("Error: 'note' requires a node id and text.")
            print(usage)
            sys.exit(1)
        node_id = rest[0]
        text = rest[1]
        tree_path = rest[2] if len(rest) > 2 else "tree.toml"
        tree = load_tree(tree_path)
        _tree_add_note(tree, node_id, text)
        return

    # --- eval [tree.toml] ---
    if subcmd == "eval":
        tree_path = rest[0] if rest else "tree.toml"
        tree = load_tree(tree_path)
        _tree_eval(tree)
        return

    # --- reset <node> [tree.toml] ---
    if subcmd == "reset":
        if not rest:
            print("Error: 'reset' requires a node id.")
            print(usage)
            sys.exit(1)
        node_id = rest[0]
        tree_path = rest[1] if len(rest) > 1 else "tree.toml"
        tree = load_tree(tree_path)
        reset_node(tree, node_id)
        print(f"Node {node_id!r} reset to pending.")
        return

    print(f"Unknown tree subcommand: {subcmd}")
    print(usage)
    sys.exit(1)


def _tree_run_node(tree, node_id: str) -> None:
    """Launch a node's campaign and record state."""
    from datetime import datetime, timezone

    from retrain.campaign import run_campaign
    from retrain.tree import save_state

    if node_id not in tree.node_map:
        print(f"Error: unknown node {node_id!r}")
        sys.exit(1)

    node = tree.node_map[node_id]
    from retrain.tree import NodeState

    ns = tree.state.nodes.setdefault(node_id, NodeState())

    ns.status = "running"
    ns.started_at = datetime.now(timezone.utc).isoformat()
    save_state(tree)

    print(f"Running node {node_id!r}: {node.campaign}")
    campaign_dir = run_campaign(node.campaign)

    ns.campaign_dir = campaign_dir
    ns.status = "done"
    ns.completed_at = datetime.now(timezone.utc).isoformat()
    save_state(tree)

    print(f"Node {node_id!r} campaign completed. Dir: {campaign_dir}")
    print("Run 'retrain tree eval' to evaluate success conditions.")


def _tree_add_note(tree, node_id: str, text: str) -> None:
    """Add an annotation to a node."""
    from datetime import datetime, timezone

    from retrain.tree import Annotation, NodeState, save_state

    if node_id not in tree.node_map:
        print(f"Error: unknown node {node_id!r}")
        sys.exit(1)

    ns = tree.state.nodes.setdefault(node_id, NodeState())
    ns.annotations.append(
        Annotation(text=text, at=datetime.now(timezone.utc).isoformat())
    )
    save_state(tree)
    print(f"Note added to {node_id!r}.")


def _tree_eval(tree) -> None:
    """Evaluate success conditions for all done nodes without outcomes."""
    from retrain.tree import evaluate_node

    evaluated = 0
    for node in tree.nodes:
        ns = tree.state.nodes.get(node.id)
        if ns and ns.status == "done" and not ns.outcome:
            if node.success_condition is None:
                print(f"  {node.id}: no success condition defined")
                continue
            try:
                outcome = evaluate_node(tree, node.id)
                result_str = ""
                if ns.result:
                    parts = [f"{k}={v}" for k, v in ns.result.items()]
                    result_str = f" ({', '.join(parts)})"
                print(f"  {node.id}: {outcome}{result_str}")
                evaluated += 1
            except Exception as e:
                print(f"  {node.id}: error — {e}")

    if evaluated == 0:
        print("No nodes to evaluate.")


def main() -> None:
    """Single entry point: retrain config.toml"""
    _load_dotenv()

    args = sys.argv[1:]
    cli_name = resolve_cli_name()

    if args and args[0] in ("-h", "--help", "help"):
        print_help(cli_name)
        sys.exit(0)

    if args and args[0] in ("man", "manual"):
        manual_command.run(args[1:], cli_name=cli_name, manual_path=_manual_path)
        sys.exit(0)

    if args and args[0] == "backends":
        run_backends(args[1:])
        sys.exit(0)

    if args and args[0] == "doctor":
        run_doctor()
        sys.exit(0)

    if args and args[0] == "migrate-config":
        _run_migrate_config(args[1:])
        sys.exit(0)

    if args and args[0] == "init":
        run_init(args=args[1:], cli_name=cli_name)
        sys.exit(0)

    if args and args[0] == "init-plugin":
        run_init_plugin(args=args[1:], cli_name=cli_name)
        sys.exit(0)

    if args and args[0] == "plugins":
        run_plugins(args[1:])
        sys.exit(0)

    if args and args[0] == "status":
        run_status(args[1:])
        sys.exit(0)

    if args and args[0] == "top":
        run_top(args[1:])
        sys.exit(0)

    if args and args[0] == "explain":
        _run_explain(args[1:])
        sys.exit(0)

    if args and args[0] == "diff":
        _run_diff(args[1:])
        sys.exit(0)

    if args and args[0] == "benchmark":
        _run_benchmark(args[1:])
        sys.exit(0)

    if args and args[0] == "trace":
        _run_trace(args[1:])
        sys.exit(0)

    if args and args[0] == "tree":
        _run_tree(args[1:])
        sys.exit(0)

    # Parse CLI overrides
    from retrain.config import parse_cli_overrides

    config_path, overrides = parse_cli_overrides(args)

    # Resolve config path
    if config_path is None:
        if Path("retrain.toml").is_file():
            config_path = "retrain.toml"
        elif not overrides:
            if sys.stdin.isatty():
                answer = input("No retrain.toml found. Create one now? [Y/n]: ").strip().lower()
                if answer in ("", "y", "yes"):
                    run_init(cli_name=cli_name)
                    sys.exit(0)
            print("No retrain.toml found. Create one with:")
            print(f"  {cli_name} init")
            print("Or pass a path:")
            print(f"  {cli_name} path/to/config.toml")
            print("Manual:")
            print(f"  {cli_name} man")
            sys.exit(1)
        # else: overrides-only mode, use defaults

    if config_path is not None and not Path(config_path).is_file():
        print(f"File not found: {config_path}")
        sys.exit(1)

    # Route: campaign | squeeze | single run
    # Campaign/squeeze only when a TOML file is provided (CLI overrides don't apply)
    if config_path is not None:
        from retrain.config import config_kind

        kind = config_kind(config_path)
        if kind == "campaign":
            from retrain.campaign import run_campaign
            run_campaign(config_path)
            return
        if kind == "squeeze":
            from retrain.squeeze import run_squeeze
            run_squeeze(config_path)
            return
    from retrain.config import load_config
    from retrain.registry import get_registry

    config = load_config(config_path, overrides=overrides)
    warn_missing(config)
    meta_dir = Path(config.log_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "run_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "trainer": config.trainer,
                "run_id": meta_dir.name or "run",
                "status": "running",
            }
        )
    )
    runner = get_registry("trainer").create(config.trainer, config)
    result = runner.run(config)
    meta: dict[str, object] = {"trainer": config.trainer}
    meta.update(result.to_dict())
    meta_path.write_text(json.dumps(meta))
    if not result.ok:
        print(
            f"Training failed ({result.failure_status}): {result.error_message}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
