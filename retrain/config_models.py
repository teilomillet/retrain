from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, validator, Field

print("[config_models.py] START OF FILE") # DEBUG

# Forward declaration for PEFTConfig if it becomes complex, for now dict is fine
PEFTConfig = Dict[str, Any] 

class ModelConfig(BaseModel):
    """Configuration for the model loading and PEFT adaptation."""
    name_or_path: str = Field(..., description="The name or path of the base model to load.")
    loader: Literal["huggingface", "unsloth"] = Field("huggingface", description="The model loader to use.")
    peft_config: Optional[PEFTConfig] = Field(None, description="Configuration for PEFT adaptation.")
    # Add other model-specific fields like 'trust_remote_code', 'torch_dtype', etc. as needed
    trust_remote_code: Optional[bool] = Field(None, description="Whether to trust remote code when loading the model.")
    torch_dtype: Optional[str] = Field(None, description="Optional torch dtype for model loading (e.g., 'float16', 'bfloat16').")

print("[config_models.py] After ModelConfig") # DEBUG

class AlgorithmConfig(BaseModel):
    """Configuration for the training algorithm."""
    name: str = Field(..., description="Name of the algorithm (e.g., 'grpo', 'rloo').")
    backend: str = Field(..., description="Backend implementation for the algorithm (e.g., 'trl').")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm-specific hyperparameters. For TRL-based trainers like GRPO, these are passed to the TRL *Config (e.g., GRPOConfig), which often inherits from transformers.TrainingArguments. Refer to TRL/Transformers documentation for available options (e.g., learning_rate, report_to, wandb_project).")
    peft_config: Optional[PEFTConfig] = Field(None, description="PEFT configuration, can also be specified under ModelConfig.")

    # Wandb specific configurations - these will be merged into hyperparameters for TRL
    report_to: Optional[List[str]] = Field(None, description="A list of integrations to report results to (e.g., ['wandb']). If 'wandb' is included, other wandb_ fields will be used.")
    wandb_project: Optional[str] = Field(None, description="The W&B project name to use. If not set, TRL might use a default or require WANDB_PROJECT env var.")
    wandb_entity: Optional[str] = Field(None, description="The W&B entity (user or team) to use. If not set, TRL might use a default or require WANDB_ENTITY env var.")
    wandb_run_name: Optional[str] = Field(None, description="The W&B run name. If not set, TRL/W&B will generate one.")

    @validator('name')
    def name_must_be_supported(cls, v):
        # Example: This list would grow as more algorithms are supported
        # Or, this could integrate with a registry of available algorithms
        supported_algos = ["grpo", "rloo"] # Placeholder
        if v.lower() not in supported_algos:
            raise ValueError(f"Unsupported algorithm name: '{v}'. Supported: {supported_algos}")
        return v.lower()

    @validator('backend')
    def backend_must_be_supported(cls, v, values):
        algo_name = values.get('name')
        # Example: TRL is currently the main backend for GRPO
        if algo_name == "grpo" and v.lower() != "trl":
            raise ValueError(f"Unsupported backend '{v}' for algorithm '{algo_name}'. Expected 'trl'.")
        # Add more backend checks as needed
        return v.lower()

print("[config_models.py] After AlgorithmConfig") # DEBUG

class EnvironmentConfig(BaseModel):
    """Configuration for the training environment."""
    type: str = Field(..., description="Type of the environment (e.g., 'smol_agent').")
    # Add common environment parameters or allow a flexible dict for env-specific settings
    env_specific_config: Dict[str, Any] = Field(default_factory=dict, description="Environment-specific settings.")

    @validator('type')
    def type_must_be_supported(cls, v):
        # This could also integrate with an environment registry
        supported_envs = ["smol_agent", "fastmcp_env"] # Added "fastmcp_env"
        if v.lower() not in supported_envs:
            raise ValueError(f"Unsupported environment type: '{v}'. Supported: {supported_envs}")
        return v.lower()

print("[config_models.py] After EnvironmentConfig") # DEBUG

class PromptSourceConfig(BaseModel):
    """Configuration for the prompt source."""
    type: str = Field(..., description="Type of prompt source (e.g., 'list_provider', 'hf_dataset').")
    # Flexible config for different prompt source types
    source_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration specific to the prompt source type.")
    
    # Example: Add a validator if 'list_provider' requires a 'prompts' key in source_config
    @validator('source_config', always=True)
    def check_list_provider_prompts(cls, v, values):
        if values.get('type') == 'list_provider' and 'prompts' not in v:
            raise ValueError("'prompts' key is required in 'source_config' for 'list_provider' type.")
        return v

print("[config_models.py] Before RewardFunctionConfig") # DEBUG

class RewardFunctionConfig(BaseModel):
    """Configuration for a single reward function, including verifiers."""
    weight: float = Field(1.0, description="Weight of this reward function.")
    # Parameters specific to the reward function can be added here or via a flexible dict
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the reward function.")
    verifiers: Optional[List[str]] = Field(None, description="List of verifier names for this reward.")
    verifier_penalty: float = Field(0.0, description="Penalty to apply if a verifier fails.")
    distribution_strategy: Optional[str] = Field(None, description="Strategy for distributing rollout-level rewards (e.g., 'last_step', 'all_steps_average'). Defaults to None, calculator might use 'last_step'.")
    # Example: "pass_through_if_invalid": False -> if a verifier makes the data invalid, this reward isn't calculated.

print("[config_models.py] After RewardFunctionConfig") # DEBUG

class RewardSetupConfig(BaseModel):
    """Configuration for the RewardCalculator."""
    step_reward_configs: Dict[str, RewardFunctionConfig] = Field(default_factory=dict, description="Configurations for step-level reward functions. Key is the registered reward name.")
    rollout_reward_configs: Dict[str, RewardFunctionConfig] = Field(default_factory=dict, description="Configurations for rollout-level reward functions. Key is the registered reward name.")
    # Global verifier settings if any, or tokenizer path if needed directly by RewardCalculator

print("[config_models.py] After RewardSetupConfig") # DEBUG

class TrainingConfig(BaseModel):
    """Root configuration model for a training run."""
    model: ModelConfig
    environment: EnvironmentConfig
    algorithm: AlgorithmConfig
    prompt_source: PromptSourceConfig
    reward_setup: RewardSetupConfig = Field(default_factory=RewardSetupConfig, description="Setup for reward calculation, including step and rollout rewards.")

    # Optional top-level configurations for managing the training run
    experiment_name: Optional[str] = Field(None, description="Optional name for the experiment run, useful for tracking.")
    seed: Optional[int] = Field(None, description="Optional random seed for improving reproducibility across runs.")
    logging_level: str = Field("INFO", description="Logging level for the training run (e.g., DEBUG, INFO, WARNING, ERROR).")

    @classmethod
    def from_yaml(cls, file_path: str) -> 'TrainingConfig':
        """Loads training configuration from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load configuration from YAML. Please install it: pip install PyYAML")
        
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            raise ValueError(f"YAML file '{file_path}' is empty or invalid.")
        return cls(**config_data) # Pydantic will validate here

print("[config_models.py] After TrainingConfig class definition (including inner Config class)") # DEBUG

# These debug prints and try-except must be at the top level (zero indent)
print("[config_models.py] END OF FILE (before final try-except)") # DEBUG

try:
    # This block is just to confirm the module itself loads without Python syntax errors up to this point.
    # The real test is the import in calculator.py
    print("[config_models.py] Module level: Attempting to confirm RewardFunctionConfig is in globals()") # DEBUG
    if 'RewardFunctionConfig' in globals():
        print("[config_models.py] Module level: RewardFunctionConfig IS in globals().") # DEBUG
    else:
        print("[config_models.py] Module level: RewardFunctionConfig IS NOT in globals(). THIS IS THE PROBLEM IF IT HAPPENS!") # DEBUG
except Exception as e:
    print(f"[config_models.py] Error at module level check: {e}") # DEBUG 