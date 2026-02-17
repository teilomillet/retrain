"""TrainConfig struct and CLI argument parser.

Ports argparse from textpolicy/tinker/train_math.py into native Mojo.
"""

from sys import argv


@fieldwise_init
struct TrainConfig(Copyable, Movable, Writable):
    """All training hyperparameters, matching Python's argparse defaults."""

    # Algorithm selection (composable)
    var advantage_mode: String
    var transform_mode: String

    # Model & Tinker
    var model: String
    var base_url: String  # empty string means None/production default
    var lora_rank: Int

    # Training
    var max_steps: Int
    var batch_size: Int
    var group_size: Int
    var max_tokens: Int
    var temperature: Float64
    var lr: Float64
    var weight_decay: Float64
    var max_examples: Int  # 0 means no limit
    var save_every: Int

    # Algorithm hyperparameters
    var gtpo_beta: Float64
    var hicra_alpha: Float64

    # SEPA
    var sepa_steps: Int
    var sepa_schedule: String
    var sepa_delay_steps: Int
    var sepa_correct_rate_gate: Float64

    # Strategic grams (JSON string, empty = use defaults)
    var strategic_grams: String

    # Logging
    var log_dir: String
    var wandb_project: String  # empty = disabled
    var wandb_run_name: String  # empty = auto

    fn __init__(out self):
        self.advantage_mode = "maxrl"
        self.transform_mode = "gtpo_sepa"
        self.model = "Qwen/Qwen3-4B-Instruct-2507"
        self.base_url = ""
        self.lora_rank = 32
        self.max_steps = 500
        self.batch_size = 8
        self.group_size = 16
        self.max_tokens = 2048
        self.temperature = 0.7
        self.lr = 4e-5
        self.weight_decay = 0.0
        self.max_examples = 0
        self.save_every = 20
        self.gtpo_beta = 0.1
        self.hicra_alpha = 0.2
        self.sepa_steps = 500
        self.sepa_schedule = "linear"
        self.sepa_delay_steps = 50
        self.sepa_correct_rate_gate = 0.1
        self.strategic_grams = ""
        self.log_dir = "logs/tinker_math"
        self.wandb_project = ""
        self.wandb_run_name = ""

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "TrainConfig(",
            "advantage_mode=", self.advantage_mode,
            ", transform_mode=", self.transform_mode,
            ", model=", self.model,
            ", batch_size=", self.batch_size,
            ", group_size=", self.group_size,
            ", max_steps=", self.max_steps,
            ")",
        )


fn print_usage():
    print("Usage: retrain-tinker [OPTIONS]")
    print()
    print("Train on MATH with textpolicy advantages via Tinker.")
    print()
    print("Algorithm selection (composable):")
    print("  --advantage-mode MODE    grpo | maxrl (default: maxrl)")
    print("  --transform-mode MODE    none | gtpo | gtpo_hicra | gtpo_sepa (default: gtpo_sepa)")
    print("  --algorithm ALG          Legacy: grpo = grpo+none, full = maxrl+gtpo_sepa")
    print()
    print("Model & Tinker:")
    print("  --model MODEL            HuggingFace model ID (default: Qwen/Qwen3-4B-Instruct-2507)")
    print("  --base-url URL           Tinker service URL (default: production)")
    print("  --lora-rank N            LoRA rank (default: 32)")
    print()
    print("Training:")
    print("  --max-steps N            Training steps (default: 500)")
    print("  --batch-size N           Prompts per step (default: 8)")
    print("  --group-size N           Completions per prompt (default: 16)")
    print("  --max-tokens N           Max completion tokens (default: 2048)")
    print("  --temperature F          Sampling temperature (default: 0.7)")
    print("  --lr F                   Learning rate (default: 4e-5)")
    print("  --weight-decay F         Weight decay (default: 0.0)")
    print("  --max-examples N         Limit dataset size (default: unlimited)")
    print("  --save-every N           Checkpoint every N steps (default: 20)")
    print()
    print("Algorithm hyperparameters:")
    print("  --gtpo-beta F            GTPO entropy weight (default: 0.1)")
    print("  --hicra-alpha F          HICRA amplification (default: 0.2)")
    print()
    print("SEPA:")
    print("  --sepa-steps N           Linear ramp steps (default: 500)")
    print("  --sepa-schedule MODE     linear | auto (default: linear)")
    print("  --sepa-delay-steps N     Delay before ramp (default: 50)")
    print("  --sepa-correct-rate-gate F  Min correct rate (default: 0.1)")
    print()
    print("Logging:")
    print("  --log-dir DIR            Log directory (default: logs/tinker_math)")
    print("  --wandb-project NAME     Enable wandb logging")
    print("  --wandb-run-name NAME    Wandb run name")
    print("  --strategic-grams JSON   JSON list of strategic gram phrases")
    print()
    print("  --help                   Show this help message")


fn parse_args() raises -> TrainConfig:
    """Parse CLI arguments into a TrainConfig.

    Matches Python's argparse behavior including legacy --algorithm resolution.
    """
    var config = TrainConfig()
    var args = argv()
    var n = len(args)
    var i = 1  # skip program name

    var has_algorithm = False
    var algorithm_value = String("")
    var has_advantage_mode = False
    var has_transform_mode = False

    while i < n:
        var arg = String(args[i])

        if arg == "--help" or arg == "-h":
            print_usage()
            raise Error("help requested")

        if i + 1 >= n:
            raise Error("Missing value for argument: " + arg)

        var val = String(args[i + 1])
        i += 2

        if arg == "--advantage-mode":
            if val != "grpo" and val != "maxrl":
                raise Error("--advantage-mode must be 'grpo' or 'maxrl', got: " + val)
            config.advantage_mode = val
            has_advantage_mode = True
        elif arg == "--transform-mode":
            if val != "none" and val != "gtpo" and val != "gtpo_hicra" and val != "gtpo_sepa":
                raise Error(
                    "--transform-mode must be none|gtpo|gtpo_hicra|gtpo_sepa, got: " + val
                )
            config.transform_mode = val
            has_transform_mode = True
        elif arg == "--algorithm":
            if val != "grpo" and val != "full":
                raise Error("--algorithm must be 'grpo' or 'full', got: " + val)
            has_algorithm = True
            algorithm_value = val
        elif arg == "--model":
            config.model = val
        elif arg == "--base-url":
            config.base_url = val
        elif arg == "--lora-rank":
            config.lora_rank = Int(val)
        elif arg == "--max-steps":
            config.max_steps = Int(val)
        elif arg == "--batch-size":
            config.batch_size = Int(val)
        elif arg == "--group-size":
            config.group_size = Int(val)
        elif arg == "--max-tokens":
            config.max_tokens = Int(val)
        elif arg == "--temperature":
            config.temperature = Float64(val)
        elif arg == "--lr":
            config.lr = Float64(val)
        elif arg == "--weight-decay":
            config.weight_decay = Float64(val)
        elif arg == "--max-examples":
            config.max_examples = Int(val)
        elif arg == "--save-every":
            config.save_every = Int(val)
        elif arg == "--gtpo-beta":
            config.gtpo_beta = Float64(val)
        elif arg == "--hicra-alpha":
            config.hicra_alpha = Float64(val)
        elif arg == "--sepa-steps":
            config.sepa_steps = Int(val)
        elif arg == "--sepa-schedule":
            if val != "linear" and val != "auto":
                raise Error("--sepa-schedule must be 'linear' or 'auto', got: " + val)
            config.sepa_schedule = val
        elif arg == "--sepa-delay-steps":
            config.sepa_delay_steps = Int(val)
        elif arg == "--sepa-correct-rate-gate":
            config.sepa_correct_rate_gate = Float64(val)
        elif arg == "--strategic-grams":
            config.strategic_grams = val
        elif arg == "--log-dir":
            config.log_dir = val
        elif arg == "--wandb-project":
            config.wandb_project = val
        elif arg == "--wandb-run-name":
            config.wandb_run_name = val
        else:
            raise Error("Unknown argument: " + arg)

    # Resolve legacy --algorithm flag
    if has_algorithm:
        if has_advantage_mode or has_transform_mode:
            raise Error(
                "--algorithm cannot be used with --advantage-mode/--transform-mode"
            )
        if algorithm_value == "grpo":
            config.advantage_mode = "grpo"
            config.transform_mode = "none"
        else:  # "full"
            config.advantage_mode = "maxrl"
            config.transform_mode = "gtpo_sepa"

    return config^
