"""TrainConfig struct and TOML config loader.

Loads training configuration from a TOML file. If argv[1] is provided,
it is used as the TOML path; otherwise retrain.toml in cwd is loaded.
"""

from python import Python, PythonObject
from sys import argv, exit


@fieldwise_init
struct TrainConfig(Copyable, Movable, Writable):
    """All training hyperparameters, matching Python's argparse defaults."""

    # Algorithm selection (composable)
    var advantage_mode: String
    var transform_mode: String

    # Backend selection
    var backend: String  # "tinker" or "local"
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

    # Inference engine
    var inference_engine: String  # "pytorch" | "max" | "vllm" | "sglang" | "openai"
    var inference_url: String  # Server URL for server-based engines (empty = default)

    # Logging
    var log_dir: String
    var wandb_project: String  # empty = disabled
    var wandb_run_name: String  # empty = auto

    fn __init__(out self):
        self.advantage_mode = "maxrl"
        self.transform_mode = "gtpo_sepa"
        self.backend = "local"
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
        self.inference_engine = "pytorch"
        self.inference_url = ""
        self.bp_enabled = False
        self.bp_warmup_steps = 10
        self.bp_ema_decay = 0.9
        self.bp_throttle_margin = 0.85
        self.bp_increase_margin = 0.5
        self.bp_min_batch_size = 1
        self.bp_max_batch_size = 64
        self.bp_peak_gflops = 0.0
        self.bp_peak_bw_gb_s = 0.0
        self.log_dir = "logs/train"
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
    print("Usage: retrain-tinker [config.toml]")
    print()
    print("Train on MATH with textpolicy advantages. Config is TOML-only.")
    print("If no path is given, loads retrain.toml from the current directory.")
    print()
    print("Example retrain.toml:")
    print()
    print("  [backend]")
    print("  backend = \"local\"          # local | tinker")
    print("  devices = \"gpu:0\"          # e.g. gpu:0,gpu:1")
    print()
    print("  [model]")
    print("  model = \"Qwen/Qwen3-4B-Instruct-2507\"")
    print("  lora_rank = 32")
    print()
    print("  [algorithm]")
    print("  advantage_mode = \"maxrl\"   # grpo | maxrl")
    print("  transform_mode = \"gtpo_sepa\"  # none | gtpo | gtpo_hicra | gtpo_sepa")
    print()
    print("  [training]")
    print("  max_steps = 500")
    print("  batch_size = 8")
    print("  group_size = 16")
    print("  max_tokens = 2048")
    print("  temperature = 0.7")
    print("  lr = 4e-5")
    print("  save_every = 20")
    print()
    print("  [inference]")
    print("  engine = \"pytorch\"         # pytorch | max | vllm | sglang | openai")
    print()
    print("  [sepa]")
    print("  steps = 500")
    print("  schedule = \"linear\"        # linear | auto")
    print("  delay_steps = 50")
    print()
    print("  [logging]")
    print("  log_dir = \"logs/train\"")
    print("  # wandb_project = \"my-project\"")
    print()
    print("  [backpressure]")
    print("  enabled = true")
    print("  warmup_steps = 10")


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

    # [inference]
    if "inference" in data:
        var sec = data["inference"]
        var v = _py_str(sec, "engine")
        if len(v) > 0:
            config.inference_engine = v
        v = _py_str(sec, "url")
        if len(v) > 0:
            config.inference_url = v

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
    """Load config from a TOML file.

    If argv[1] exists and isn't a help flag, treat it as a TOML path.
    Otherwise, look for retrain.toml in cwd. If neither found, print
    help and exit.
    """
    var config = TrainConfig()
    var args = argv()
    var n = len(args)

    if n > 1:
        var arg1 = String(args[1])
        if arg1 == "-h" or arg1 == "--help" or arg1 == "help":
            print_usage()
            exit(0)
        # argv[1] is a TOML path
        _apply_toml(config, arg1)
        return config^

    # No argument: try retrain.toml in cwd
    var os = Python.import_module("os")
    if os.path.isfile("retrain.toml"):
        _apply_toml(config, "retrain.toml")
        return config^

    # No config found
    print("Error: no config file found.")
    print("Place a retrain.toml in the current directory, or pass a path:")
    print("  retrain-tinker path/to/config.toml")
    print()
    print("Run 'retrain-tinker help' for a config reference.")
    exit(1)
    return config^
