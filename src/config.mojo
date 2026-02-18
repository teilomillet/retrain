"""TrainConfig struct and CLI argument parser.

Ports argparse from textpolicy/tinker/train_math.py into native Mojo.
Supports --config PATH for TOML-based configuration (CLI args override).
"""

from python import Python, PythonObject
from sys import argv


@fieldwise_init
struct TrainConfig(Copyable, Movable, Writable):
    """All training hyperparameters, matching Python's argparse defaults."""

    # Algorithm selection (composable)
    var advantage_mode: String
    var transform_mode: String

    # Backend selection
    var backend: String  # "tinker" or "max"
    var devices: String  # e.g. "gpu:0,gpu:1"
    var adapter_path: String  # LoRA adapter exchange directory

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

    # Back pressure
    var bp_enabled: Bool
    var bp_warmup_steps: Int
    var bp_ema_decay: Float64
    var bp_throttle_margin: Float64
    var bp_increase_margin: Float64
    var bp_min_batch_size: Int
    var bp_max_batch_size: Int
    var bp_peak_gflops: Float64
    var bp_peak_bw_gb_s: Float64

    # Logging
    var log_dir: String
    var wandb_project: String  # empty = disabled
    var wandb_run_name: String  # empty = auto

    fn __init__(out self):
        self.advantage_mode = "maxrl"
        self.transform_mode = "gtpo_sepa"
        self.backend = "tinker"
        self.devices = "gpu:0"
        self.adapter_path = "/tmp/retrain_adapter"
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
        self.bp_enabled = False
        self.bp_warmup_steps = 10
        self.bp_ema_decay = 0.9
        self.bp_throttle_margin = 0.85
        self.bp_increase_margin = 0.5
        self.bp_min_batch_size = 1
        self.bp_max_batch_size = 64
        self.bp_peak_gflops = 0.0
        self.bp_peak_bw_gb_s = 0.0
        self.log_dir = "logs/tinker_math"
        self.wandb_project = ""
        self.wandb_run_name = ""

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "TrainConfig(",
            "advantage_mode=", self.advantage_mode,
            ", transform_mode=", self.transform_mode,
            ", backend=", self.backend,
            ", model=", self.model,
            ", batch_size=", self.batch_size,
            ", group_size=", self.group_size,
            ", max_steps=", self.max_steps,
            ", bp_enabled=", self.bp_enabled,
            ")",
        )


fn print_usage():
    print("Usage: retrain-tinker [OPTIONS]")
    print()
    print("Train on MATH with textpolicy advantages via Tinker.")
    print()
    print("Backend:")
    print("  --backend MODE           tinker | max (default: tinker)")
    print("  --devices SPECS          Device list, e.g. gpu:0,gpu:1 (default: gpu:0)")
    print("  --adapter-path PATH      LoRA adapter exchange directory (default: /tmp/retrain_adapter)")
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
    print("Back pressure (USL+Roofline):")
    print("  --bp-enabled BOOL        Enable adaptive back pressure (default: false)")
    print("  --bp-warmup-steps N      Warmup before fitting (default: 10)")
    print("  --bp-ema-decay F         EMA decay for throughput (default: 0.9)")
    print("  --bp-throttle-margin F   Throttle when p > p*×margin (default: 0.85)")
    print("  --bp-increase-margin F   Increase when p < p*×margin (default: 0.5)")
    print("  --bp-min-batch-size N    Minimum batch size (default: 1)")
    print("  --bp-max-batch-size N    Maximum batch size (default: 64)")
    print("  --bp-peak-gflops F       Peak GFLOPS for roofline (default: 0=auto)")
    print("  --bp-peak-bw-gb-s F      Peak bandwidth GB/s for roofline (default: 0=auto)")
    print()
    print("Config file:")
    print("  --config PATH            TOML config file (CLI args override)")
    print()
    print("  --help                   Show this help message")


fn _py_str(obj: PythonObject, key: String) raises -> String:
    """Get a string value from a Python dict, or empty string if missing."""
    if key in obj:
        return String(obj[key])
    return String("")


fn _py_int(obj: PythonObject, key: String, default: Int) raises -> Int:
    """Get an int value from a Python dict, or default if missing."""
    if key in obj:
        return Int(String(obj[key]))
    return default


fn _py_float(obj: PythonObject, key: String, default: Float64) raises -> Float64:
    """Get a float value from a Python dict, or default if missing."""
    if key in obj:
        return Float64(String(obj[key]))
    return default


fn _apply_toml(mut config: TrainConfig, path: String) raises:
    """Load a TOML config file and apply its values to config.

    Uses Python's tomllib (3.11+) for parsing. TOML sections map to
    config fields:
        [algorithm]   -> advantage_mode, transform_mode
        [backend]     -> backend, devices, adapter_path
        [model]       -> model, base_url, lora_rank
        [training]    -> max_steps, batch_size, group_size, etc.
        [gtpo]        -> gtpo_beta
        [hicra]       -> hicra_alpha
        [sepa]        -> sepa_steps, sepa_schedule, etc.
        [logging]     -> log_dir, wandb_project, wandb_run_name, strategic_grams
    """
    var tomllib = Python.import_module("tomllib")
    var builtins = Python.import_module("builtins")
    var f = builtins.open(path, "rb")
    var data = tomllib.load(f)
    f.close()

    # [algorithm]
    if "algorithm" in data:
        var sec = data["algorithm"]
        var v = _py_str(sec, "advantage_mode")
        if len(v) > 0:
            config.advantage_mode = v
        v = _py_str(sec, "transform_mode")
        if len(v) > 0:
            config.transform_mode = v

    # [backend]
    if "backend" in data:
        var sec = data["backend"]
        var v = _py_str(sec, "backend")
        if len(v) > 0:
            config.backend = v
        v = _py_str(sec, "devices")
        if len(v) > 0:
            config.devices = v
        v = _py_str(sec, "adapter_path")
        if len(v) > 0:
            config.adapter_path = v

    # [model]
    if "model" in data:
        var sec = data["model"]
        var v = _py_str(sec, "model")
        if len(v) > 0:
            config.model = v
        v = _py_str(sec, "base_url")
        if len(v) > 0:
            config.base_url = v
        config.lora_rank = _py_int(sec, "lora_rank", config.lora_rank)

    # [training]
    if "training" in data:
        var sec = data["training"]
        config.max_steps = _py_int(sec, "max_steps", config.max_steps)
        config.batch_size = _py_int(sec, "batch_size", config.batch_size)
        config.group_size = _py_int(sec, "group_size", config.group_size)
        config.max_tokens = _py_int(sec, "max_tokens", config.max_tokens)
        config.temperature = _py_float(sec, "temperature", config.temperature)
        config.lr = _py_float(sec, "lr", config.lr)
        config.weight_decay = _py_float(sec, "weight_decay", config.weight_decay)
        config.max_examples = _py_int(sec, "max_examples", config.max_examples)
        config.save_every = _py_int(sec, "save_every", config.save_every)

    # [gtpo]
    if "gtpo" in data:
        var sec = data["gtpo"]
        config.gtpo_beta = _py_float(sec, "beta", config.gtpo_beta)

    # [hicra]
    if "hicra" in data:
        var sec = data["hicra"]
        config.hicra_alpha = _py_float(sec, "alpha", config.hicra_alpha)

    # [sepa]
    if "sepa" in data:
        var sec = data["sepa"]
        config.sepa_steps = _py_int(sec, "steps", config.sepa_steps)
        var v = _py_str(sec, "schedule")
        if len(v) > 0:
            config.sepa_schedule = v
        config.sepa_delay_steps = _py_int(sec, "delay_steps", config.sepa_delay_steps)
        config.sepa_correct_rate_gate = _py_float(sec, "correct_rate_gate", config.sepa_correct_rate_gate)

    # [backpressure]
    if "backpressure" in data:
        var sec = data["backpressure"]
        if "enabled" in sec:
            config.bp_enabled = String(sec["enabled"]).lower() == "true"
        config.bp_warmup_steps = _py_int(sec, "warmup_steps", config.bp_warmup_steps)
        config.bp_ema_decay = _py_float(sec, "ema_decay", config.bp_ema_decay)
        config.bp_throttle_margin = _py_float(sec, "throttle_margin", config.bp_throttle_margin)
        config.bp_increase_margin = _py_float(sec, "increase_margin", config.bp_increase_margin)
        config.bp_min_batch_size = _py_int(sec, "min_batch_size", config.bp_min_batch_size)
        config.bp_max_batch_size = _py_int(sec, "max_batch_size", config.bp_max_batch_size)
        config.bp_peak_gflops = _py_float(sec, "peak_gflops", config.bp_peak_gflops)
        config.bp_peak_bw_gb_s = _py_float(sec, "peak_bw_gb_s", config.bp_peak_bw_gb_s)

    # [logging]
    if "logging" in data:
        var sec = data["logging"]
        var v = _py_str(sec, "log_dir")
        if len(v) > 0:
            config.log_dir = v
        v = _py_str(sec, "wandb_project")
        if len(v) > 0:
            config.wandb_project = v
        v = _py_str(sec, "wandb_run_name")
        if len(v) > 0:
            config.wandb_run_name = v
        v = _py_str(sec, "strategic_grams")
        if len(v) > 0:
            config.strategic_grams = v


fn parse_args() raises -> TrainConfig:
    """Parse CLI arguments into a TrainConfig.

    If --config PATH is given, TOML values are loaded first, then CLI
    arguments override. Matches Python's argparse behavior including
    legacy --algorithm resolution.
    """
    var config = TrainConfig()
    var args = argv()
    var n = len(args)
    var i = 1  # skip program name

    # First pass: find --config and apply TOML before CLI overrides
    var ci = 1
    while ci < n:
        if String(args[ci]) == "--config" and ci + 1 < n:
            _apply_toml(config, String(args[ci + 1]))
            break
        ci += 1

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

        if arg == "--backend":
            if val != "tinker" and val != "max":
                raise Error("--backend must be 'tinker' or 'max', got: " + val)
            config.backend = val
        elif arg == "--devices":
            config.devices = val
        elif arg == "--adapter-path":
            config.adapter_path = val
        elif arg == "--advantage-mode":
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
        elif arg == "--bp-enabled":
            config.bp_enabled = val.lower() == "true" or val == "1"
        elif arg == "--bp-warmup-steps":
            config.bp_warmup_steps = Int(val)
        elif arg == "--bp-ema-decay":
            config.bp_ema_decay = Float64(val)
        elif arg == "--bp-throttle-margin":
            config.bp_throttle_margin = Float64(val)
        elif arg == "--bp-increase-margin":
            config.bp_increase_margin = Float64(val)
        elif arg == "--bp-min-batch-size":
            config.bp_min_batch_size = Int(val)
        elif arg == "--bp-max-batch-size":
            config.bp_max_batch_size = Int(val)
        elif arg == "--bp-peak-gflops":
            config.bp_peak_gflops = Float64(val)
        elif arg == "--bp-peak-bw-gb-s":
            config.bp_peak_bw_gb_s = Float64(val)
        elif arg == "--strategic-grams":
            config.strategic_grams = val
        elif arg == "--log-dir":
            config.log_dir = val
        elif arg == "--wandb-project":
            config.wandb_project = val
        elif arg == "--wandb-run-name":
            config.wandb_run_name = val
        elif arg == "--config":
            pass  # Already handled in first pass
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
