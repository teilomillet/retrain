"""Entry point for the retrain CLI.

Usage:
    retrain              # loads retrain.toml from cwd
    retrain config.toml  # loads specified TOML file
    retrain help         # prints config reference
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


_HELP_TEXT = """\
Usage: retrain [config.toml | help]

Train on MATH with textpolicy advantages. Config is TOML-only.
If no path is given, loads retrain.toml from the current directory.

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
"""


def main() -> None:
    """CLI entry point."""
    _load_dotenv()

    args = sys.argv[1:]

    if args and args[0] in ("-h", "--help", "help"):
        print(_HELP_TEXT)
        sys.exit(0)

    from retrain.config import load_config
    from retrain.trainer import train

    # Parse --resume flag
    resume_from = ""
    remaining: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--resume" and i + 1 < len(args):
            resume_from = args[i + 1]
            i += 2
        elif args[i].startswith("--resume="):
            resume_from = args[i].split("=", 1)[1]
            i += 1
        else:
            remaining.append(args[i])
            i += 1

    config_path = remaining[0] if remaining else None

    if config_path is None and not Path("retrain.toml").is_file():
        print("Error: no config file found.")
        print("Place a retrain.toml in the current directory, or pass a path:")
        print("  retrain path/to/config.toml")
        print()
        print("Run 'retrain help' for a config reference.")
        sys.exit(1)

    config = load_config(config_path)
    if resume_from:
        config.resume_from = resume_from
    train(config)


if __name__ == "__main__":
    main()
