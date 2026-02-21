"""Entry point for the retrain CLI.

Usage:
    retrain              # loads retrain.toml from cwd
    retrain config.toml  # loads specified TOML file
    retrain help         # prints config reference

Override any config field from the command line:
    retrain --seed 42 --lr 1e-4 --wandb-project my-run
    retrain config.toml --batch-size 4 --advantage-mode grpo
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


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
        # Strip surrounding quotes
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        os.environ[key] = val
    print("Loaded .env")


# CLI flag -> TrainConfig field name
_OVERRIDE_FLAGS: dict[str, str] = {
    "--advantage-mode": "advantage_mode",
    "--transform-mode": "transform_mode",
    "--seed": "seed",
    "--lr": "lr",
    "--batch-size": "batch_size",
    "--group-size": "group_size",
    "--max-steps": "max_steps",
    "--max-tokens": "max_tokens",
    "--temperature": "temperature",
    "--top-p": "top_p",
    "--weight-decay": "weight_decay",
    "--lora-rank": "lora_rank",
    "--reward-type": "reward_type",
    "--backend": "backend",
    "--log-dir": "log_dir",
    "--wandb-project": "wandb_project",
    "--wandb-entity": "wandb_entity",
    "--wandb-group": "wandb_group",
    "--wandb-tags": "wandb_tags",
    "--wandb-run-name": "wandb_run_name",
}

_HELP_TEXT = """\
Usage: retrain [config.toml | help] [--flag value ...]

Train on MATH with textpolicy advantages. Config is TOML-only.
If no path is given, loads retrain.toml from the current directory.
CLI flags override TOML values.

Example retrain.toml:

  [backend]
  backend = "local"          # local | tinker
  devices = "gpu:0"          # e.g. gpu:0,gpu:1

  [model]
  model = "Qwen/Qwen3-4B-Instruct-2507"
  lora_rank = 32

  [algorithm]
  advantage_mode = "maxrl"   # grpo | maxrl
  transform_mode = "gtpo_sepa"  # none | gtpo | gtpo_hicra | gtpo_sepa

  [training]
  max_steps = 500
  batch_size = 8
  group_size = 16
  max_tokens = 2048
  temperature = 0.7
  lr = 4e-5
  save_every = 20

  [inference]
  engine = "pytorch"         # pytorch | max | vllm | sglang | trtllm | openai
  attention_kernel = "default"  # default | flash | triton | tk | cutlass
  dtype = "auto"             # auto | bf16 | fp8 | fp4
  kv_cache_dtype = "auto"    # auto | bf16 | fp8 | int8
  prefix_caching = true      # share prompt KV across group completions

  [sepa]
  steps = 500
  schedule = "linear"        # linear | auto
  delay_steps = 50

  [logging]
  log_dir = "logs/train"
  # wandb_project = "my-project"

  [reward]
  type = "match"             # match | math | judge | custom
  # judge_model = "gpt-4o-mini"   # only for type = "judge"
  # custom_module = "my_pkg.rewards"  # only for type = "custom"
  # custom_function = "my_score"      # only for type = "custom"

  [backpressure]
  enabled = true
  warmup_steps = 10

  [resume]
  from = "logs/train"       # resume from trainer_state.json in this dir

CLI override flags (override any TOML value):

  --advantage-mode MODE    grpo | maxrl
  --transform-mode MODE    none | gtpo | gtpo_hicra | gtpo_sepa
  --seed N                 RNG seed (-1 = no seed)
  --lr FLOAT               learning rate
  --batch-size N           batch size
  --group-size N           group size
  --max-steps N            number of training steps
  --lora-rank N            LoRA rank
  --log-dir PATH           log directory
  --wandb-project NAME     wandb project
  --wandb-entity NAME      wandb entity / team
  --wandb-group NAME       wandb group (for grouping runs)
  --wandb-tags TAGS        comma-separated wandb tags
  --wandb-run-name NAME    wandb run name
  --resume PATH            resume from checkpoint dir

Examples:

  retrain --seed 42 --lr 1e-4
  retrain config.toml --batch-size 4 --wandb-project sepa-deep
  retrain --advantage-mode grpo --transform-mode none --seed 101
"""


def main() -> None:
    """CLI entry point."""
    _load_dotenv()

    args = sys.argv[1:]

    if args and args[0] in ("-h", "--help", "help"):
        print(_HELP_TEXT)
        sys.exit(0)

    from retrain.config import _FIELD_TYPES, load_config
    from retrain.trainer import train

    # Parse --flag value pairs and positional args
    resume_from = ""
    overrides: dict[str, str] = {}
    positional: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        # --resume (special: not a TrainConfig field)
        if arg == "--resume" and i + 1 < len(args):
            resume_from = args[i + 1]
            i += 2
            continue
        if arg.startswith("--resume="):
            resume_from = arg.split("=", 1)[1]
            i += 1
            continue
        # --flag=value form
        if "=" in arg and arg.startswith("--"):
            flag, val = arg.split("=", 1)
            if flag in _OVERRIDE_FLAGS:
                overrides[flag] = val
                i += 1
                continue
        # --flag value form
        if arg in _OVERRIDE_FLAGS:
            if i + 1 < len(args):
                overrides[arg] = args[i + 1]
                i += 2
                continue
            else:
                print(f"Error: {arg} requires a value")
                sys.exit(1)
        # Positional (config file path)
        if not arg.startswith("--"):
            positional.append(arg)
        else:
            print(f"Error: unknown flag '{arg}'")
            print("Run 'retrain help' for available flags.")
            sys.exit(1)
        i += 1

    config_path = positional[0] if positional else None

    if config_path is None and not Path("retrain.toml").is_file():
        # Allow running with only CLI overrides (no TOML needed)
        if not overrides:
            print("Error: no config file found.")
            print("Place a retrain.toml in the current directory, or pass a path:")
            print("  retrain path/to/config.toml")
            print()
            print("Run 'retrain help' for a config reference.")
            sys.exit(1)

    config = load_config(config_path)

    # Apply CLI overrides with type coercion
    for flag, raw_val in overrides.items():
        field_name = _OVERRIDE_FLAGS[flag]
        ftype = _FIELD_TYPES[field_name]
        if ftype is bool:
            setattr(config, field_name, raw_val.lower() in ("true", "1", "yes"))
        elif ftype is int:
            setattr(config, field_name, int(raw_val))
        elif ftype is float:
            setattr(config, field_name, float(raw_val))
        else:
            setattr(config, field_name, raw_val)

    if resume_from:
        config.resume_from = resume_from
    train(config)


if __name__ == "__main__":
    main()
