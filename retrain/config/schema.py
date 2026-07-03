"""TrainConfig schema."""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

from retrain.config.constants import _DEFAULT_ADAPTER_PATH


@dataclass
class TrainConfig:
    """All training hyperparameters."""

    # Algorithm selection
    algorithm_mode: str = ""
    advantage_mode: str = "maxrl"
    transform_mode: str = "gtpo_sepa"
    algorithm_params: dict[str, object] = field(default_factory=dict)
    advantage_params: dict[str, object] = field(default_factory=dict)
    transform_params: dict[str, object] = field(default_factory=dict)

    # Backend selection
    backend: str = "local"
    devices: str = "gpu:0"
    adapter_path: str = _DEFAULT_ADAPTER_PATH
    backend_options: dict[str, object] = field(default_factory=dict)

    # Training runner
    trainer: str = "retrain"       # "retrain", "command", or dotted plugin path
    trainer_command: str = ""      # shell command template for trainer = "command"

    # Model
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    base_url: str = ""
    lora_rank: int = 32

    # Training
    seed: int = -1
    max_steps: int = 500
    batch_size: int = 8
    group_size: int = 16
    max_tokens: int = 10240
    temperature: float = 0.7
    top_p: float = 0.95
    lr: float = 4e-5
    weight_decay: float = 0.0
    clip_eps: float = 0.0        # 0 = disabled (no PPO-style ratio clipping)
    clip_eps_high: float = 0.0   # 0 = symmetric (uses clip_eps for upper bound)
    policy_loss_mode: str = "standard"  # "standard", "kl_cov", or "clip_cov"
    kl_cov_percent: float = 0.2  # percent of valid policy tokens receiving KL-Cov
    kl_cov_coef: float = 1.0
    clip_cov_ratio: float = 0.0002
    clip_cov_min: float = 1.0
    clip_cov_max: float = 5.0
    adv_clip_max: float = 0.0    # 0 = disabled; caps token advantages to [-max, +max]
    batch_advantage_norm: bool = False  # REINFORCE++: normalize advantages across full batch
    max_examples: int = 0
    save_every: int = 20

    # Optimizer
    optim_beta1: float = 0.9
    optim_beta2: float = 0.95
    optim_eps: float = 1e-8
    grad_clip_norm: float = 0.0  # 0 = disabled; max global gradient norm
    clip_ratio_c: float = 0.0   # 0 = disabled; dual-clip lower bound for negative advs

    # LoRA
    lora_alpha: int = 0  # 0 = auto = rank * 2
    lora_dropout: float = 0.0

    # Algorithm hyperparameters
    gtpo_beta: float = 0.1
    hicra_alpha: float = 0.2
    uncertainty_kind: str = "surprisal"
    surprisal_mask_rho: float = 0.0

    # SEPA
    sepa_steps: int = 500
    sepa_schedule: str = "linear"
    sepa_delay_steps: int = 50
    sepa_correct_rate_gate: float = 0.1

    # Strategic grams (JSON string, empty = use defaults)
    strategic_grams: str = ""

    # Planning detector
    planning_detector: str = "regex"
    planning_model: str = "all-MiniLM-L6-v2"
    planning_threshold: float = 0.02

    # Back pressure
    bp_enabled: bool = False
    bp_warmup_steps: int = 10
    bp_ema_decay: float = 0.9
    bp_throttle_margin: float = 0.85
    bp_increase_margin: float = 0.5
    bp_min_batch_size: int = 1
    bp_max_batch_size: int = 64
    bp_peak_gflops: float = 0.0
    bp_peak_bw_gb_s: float = 0.0

    # Inference engine
    inference_engine: str = "pytorch"
    inference_url: str = ""
    attention_kernel: str = "default"
    inference_dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    prefix_caching: bool = True

    # Data source
    data_source: str = "math"

    # Environment bridge (optional; e.g. verifiers envs from Prime Intellect Hub)
    environment_provider: str = ""
    environment_id: str = ""
    environment_args: str = ""
    environment_max_turns: int = -1
    environment_auto_install: bool = False
    environment_rollout_env_workers: int = 1
    environment_rollout_buffer_size: int = 0

    # Reward / verifier
    reward_type: str = "match"
    reward_judge_model: str = ""
    reward_custom_module: str = ""
    reward_custom_function: str = "score"

    # Tinker throttle (cross-process concurrency limiter)
    tinker_throttle_dir: str = ""
    tinker_max_concurrent: int = 4

    # SFT warmup (supervised fine-tuning before RL)
    sft_warmup_steps: int = 0
    sft_data_path: str = ""  # JSONL messages, prompt/completion, or text rows
    sft_batch_size: int = 0  # 0 = trainer default
    sft_max_tokens: int = 0  # 0 = trainer default
    sft_lr: float = 0.0      # 0 = use main lr; separate LR for SFT phase
    sft_loss_fn: str = "auto"  # "auto", "importance_sampling", or "cross_entropy"
    sft_batch_order: str = "shuffle"  # "shuffle", "length", "length_desc", "length_bucket"
    sft_length_bucket_size: int = 0  # 0 = full tokenized traversal for length_bucket

    # ECHO: same-rollout supervised world-modeling on environment/tool tokens.
    echo_enabled: bool = False
    echo_weight: float = 0.05
    echo_loss_fn: str = "cross_entropy"
    echo_max_tokens_per_step: int = 2048
    echo_max_token_ratio: float = 0.5
    echo_entropy_floor: float = 0.01
    echo_min_prompt_overlap: float = 0.5

    # TL-GRPO (Turn-Level GRPO with branching)
    tl_grpo: bool = False
    tl_grpo_branch_mode: str = "action_space"  # "action_space" or "llm"
    tl_grpo_branch_size: int = 4
    tl_grpo_lookahead_steps: int = 2
    tl_grpo_ema_decay: float = 0.9
    tl_grpo_ema_init: float = 0.5

    # Resume
    resume_from: str = ""

    # Logging
    log_dir: str = "logs/train"
    wandb_project: str = ""
    wandb_run_name: str = ""
    wandb_entity: str = ""
    wandb_group: str = ""
    wandb_tags: str = ""
    log_generations: bool = True
    generation_log_samples_per_prompt: int = 1
    generation_top_surprisal_limit: int = 0

    # Plugin loading
    plugins_search_paths: list[str] = field(default_factory=lambda: ["plugins"])
    plugins_strict: bool = True


    def __post_init__(self) -> None:
        from retrain.config.validate import validate_train_config

        validate_train_config(self)

    @property
    def post_process_params(self) -> dict[str, object]:
        """Build the params dict passed to TransformSpec.post_process hooks.

        Collects algorithm hyperparameters that post-process hooks may need.
        Hooks pick the keys they care about; unknown keys are ignored.
        """
        params = dict(self.transform_params)
        params.setdefault("uncertainty_kind", self.uncertainty_kind)
        params.setdefault("surprisal_mask_rho", self.surprisal_mask_rho)
        params.setdefault("entropy_mask_rho", self.surprisal_mask_rho)
        return params

    @property
    def effective_advantage_params(self) -> dict[str, object]:
        """Params passed to advantage plugins/callables."""
        params = dict(self.advantage_params)
        params.setdefault("algorithm_mode", self.algorithm_mode)
        params.setdefault("advantage_mode", self.advantage_mode)
        params.setdefault("transform_mode", self.transform_mode)
        return params

    @property
    def effective_algorithm_params(self) -> dict[str, object]:
        """Params passed to algorithm plugins/callables."""
        params = dict(self.algorithm_params)
        params.setdefault("advantage_mode", self.advantage_mode)
        params.setdefault("transform_mode", self.transform_mode)
        params.setdefault("advantage_params", dict(self.advantage_params))
        raw_transform_params = params.get("transform_params")
        if isinstance(raw_transform_params, dict):
            transform_params = dict(raw_transform_params)
        elif isinstance(raw_transform_params, typing.Mapping):
            transform_params = dict(raw_transform_params)
        else:
            transform_params = dict(self.transform_params)
        transform_params.setdefault("uncertainty_kind", self.uncertainty_kind)
        params["transform_params"] = transform_params
        params.setdefault("uncertainty_kind", self.uncertainty_kind)
        params.setdefault("surprisal_mask_rho", self.surprisal_mask_rho)
        params.setdefault("entropy_mask_rho", self.surprisal_mask_rho)
        params.setdefault("gtpo_beta", self.gtpo_beta)
        params.setdefault("hicra_alpha", self.hicra_alpha)
        return params
