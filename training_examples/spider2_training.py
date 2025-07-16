#!/usr/bin/env python3
"""
Spider2 Training Script using MCP Alchemy Integration

This script demonstrates how to train a language model on the Spider2 benchmark
using the retrain framework with MCP Alchemy for database interaction.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add the retrain package to the path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from retrain.config_models import (
    TrainingConfig, ModelConfig, EnvironmentConfig, 
    PromptSourceConfig, RewardSetupConfig, AlgorithmConfig,
    RewardFunctionConfig
)
from retrain.environment.env_spider2 import Spider2Env
from retrain.run import run_async_training
from loguru import logger

def create_spider2_training_config(
    model_name: str = "Qwen/Qwen3-0.6B",
    spider2_type: str = "lite",
    mcp_server_config: Optional[Dict[str, Any]] = None,
    max_steps: int = 20,
    num_iterations: int = 100,
    batch_size: int = 4,  # Back to 4 to be divisible by num_generations=2
    learning_rate: float = 1e-5,
    max_length: int = 512,
    output_dir: str = "trainer_output",
    logging_dir: Optional[str] = None
) -> TrainingConfig:
    """Create a training configuration for Spider2."""
    
    # Model configuration
    model_config = ModelConfig(
        name_or_path=model_name,
        loader="huggingface",
        peft_config=None,
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    # Environment configuration
    env_config = EnvironmentConfig(
        type="spider2",
        env_specific_config={
            "spider2_data_path": f"Spider2/spider2-{spider2_type}/spider2-{spider2_type}.jsonl",
            "spider2_type": spider2_type,
            "mcp_server_config": mcp_server_config,
            "max_steps": max_steps,
            "enable_external_knowledge": True
        }
    )
    
    # Prompt source configuration (NEW)
    # The Spider2Env generates its own detailed prompt, so we use a simple template here.
    prompt_source_config = PromptSourceConfig(
        type="environment",
        source_config={}
    )

    # Reward configuration
    reward_setup_config = RewardSetupConfig(
        step_reward_configs={
            "format_compliance": RewardFunctionConfig(
                weight=3.0,  # High weight to heavily penalize format violations
                params={},
                verifiers=[],
                verifier_penalty=0.0,
                distribution_strategy="every_step"
            ),
            "xml_structure_enforcement": RewardFunctionConfig(
                weight=2.0,  # Strong enforcement of XML structure
                params={},
                verifiers=[],
                verifier_penalty=0.0,
                distribution_strategy="every_step"
            ),
            "sql_quality": RewardFunctionConfig(
                weight=1.0,
                params={"syntax_check": True, "semantic_check": True, "execution_check": True},
                verifiers=["sql_syntax", "sql_semantics"],
                verifier_penalty=0.5,
                distribution_strategy="last_step"
            ),
            "tool_usage": RewardFunctionConfig(
                weight=0.5,
                params={"encourage_exploration": True, "reward_successful_execution": True},
                verifiers=[],
                verifier_penalty=0.0,
                distribution_strategy="last_step"
            )
        },
        rollout_reward_configs={
            "final_answer_quality": RewardFunctionConfig(
                weight=2.0,
                params={"accuracy_threshold": 0.8, "completeness_check": True},
                verifiers=["answer_accuracy"],
                verifier_penalty=1.0,
                distribution_strategy="last_step"
            )
        }
    )
    
    # Algorithm configuration (GRPO with TRL backend)
    algorithm_config = AlgorithmConfig(
        name="grpo",
        backend="trl",
        hyperparameters={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_length": max_length,
            "num_iterations": num_iterations,
            "gradient_accumulation_steps": 4,  # Back to 4 for better gradient stability
            "warmup_steps": 5,  # Reduced warmup for faster convergence
            "logging_steps": 2,  # More frequent logging
            "save_steps": 25,  # More frequent saves
            "eval_steps": 15,  # More frequent evaluation
            "evaluation_strategy": "no",
            "save_strategy": "steps",
            "load_best_model_at_end": False,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
            "bf16": True,
            "tf32": False,
            "gradient_checkpointing": True,
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "seed": 42,
            # Memory efficiency settings
            "dataloader_num_workers": 0,  # Reduce workers to save memory
            "max_prompt_length": 256,  # Reduce prompt length
            "max_completion_length": 256,  # Reduce completion length
            # GRPO-specific settings
            "num_generations": 2,  # Minimum required for GRPO (must divide effective batch size)
            "temperature": 0.8,  # Generation temperature
            "top_p": 0.9,  # Top-p sampling
            # Enable completion logging to WandB
            "log_completions": True,
            "num_completions_to_print": 4,  # Reduced from 8 to save memory
            "wandb_log_unique_prompts": False
        },
        peft_config=None,
        report_to=["wandb"],  # Enable WandB logging - will auto-login
        wandb_project="spider2-training",
        wandb_entity=None,
        wandb_run_name=f"spider2-{spider2_type}-grpo"
    )
    
    # Main training configuration
    training_config = TrainingConfig(
        algorithm=algorithm_config,
        environment=env_config,
        model=model_config,
        prompt_source=prompt_source_config,
        reward_setup=reward_setup_config,
        num_episodes=num_iterations,
        batch_size=batch_size,
        seed=42,
        logging_level="DEBUG",
        output_dir=output_dir,
        logging_dir=logging_dir or str(Path(output_dir) / "logs"),
        experiment_name=f"spider2_training_{spider2_type}",
        run_name=f"run_{model_name.replace('/', '_')}_{spider2_type}"
    )
    
    return training_config

async def setup_mcp_alchemy_server():
    """Setup and start MCP Alchemy server for database access."""
    logger.info("Setting up MCP Alchemy server...")
    
    # This would typically involve starting the MCP Alchemy server
    # For now, we'll assume it's running externally
    # In a real setup, you might start it as a subprocess here
    
    # Example configuration for different Spider2 databases
    mcp_configs = {
        "lite": {
            "bigquery": "postgresql://user:pass@localhost/bigquery_spider2",
            "snowflake": "snowflake://user:pass@account/database/schema",
            "sqlite": "sqlite:///spider2_lite.db"
        },
        "snow": {
            "snowflake": "snowflake://user:pass@account/database/schema"
        },
        "dbt": {
            "duckdb": "duckdb:///spider2_dbt.db"
        }
    }
    
    return mcp_configs

async def main():
    """Main training function."""
    logger.info("Starting Spider2 training with MCP Alchemy integration")
    
    # Setup MCP configurations
    mcp_configs = await setup_mcp_alchemy_server()
    
    # Training configurations for different Spider2 variants
    training_configs = {}
    
    # Spider2-Lite training
    if os.path.exists("Spider2/spider2-lite/spider2-lite.jsonl"):
        # Define the stdio transport configuration for MCP Alchemy
        lite_mcp_config = {
            "command": "uvx",
            "args": ["mcp-alchemy"],
            "env": {
                "DB_URL": "sqlite:///Spider2/spider2-lite/resource/databases/spider2-localdb/chinook.sqlite"
            }
        }
        
        training_configs["lite"] = create_spider2_training_config(
            model_name="Qwen/Qwen3-0.6B",
            spider2_type="lite",
            mcp_server_config=lite_mcp_config,  # Use stdio transport config
            max_steps=20,
            num_iterations=100,  # Reduced back since we're using larger batches
            batch_size=4,  # Back to 4 for GRPO compatibility
            learning_rate=1e-5,  # Back to original LR
            output_dir="trainer_output/spider2-lite",
            logging_dir="trainer_output/spider2-lite/logs"
        )
        logger.info("Created Spider2-Lite training configuration")
    
    # Spider2-Snow training disabled for now
    
    # Spider2-DBT training
    if os.path.exists("Spider2/spider2-dbt/spider2-dbt.jsonl"):
        # Define the stdio transport configuration for DBT
        dbt_mcp_config = {
            "command": "uvx",
            "args": ["mcp-alchemy"],
            "env": {
                "DB_URL": "duckdb:///Spider2/spider2-dbt/spider2_dbt.db"
            }
        }
        
        training_configs["dbt"] = create_spider2_training_config(
            model_name="Qwen/Qwen3-0.6B",
            spider2_type="dbt",
            mcp_server_config=dbt_mcp_config,  # Use stdio transport config
            max_steps=30,
            num_iterations=150,  # Adjusted for larger batches
            batch_size=4,  # Back to 4 for GRPO compatibility
            learning_rate=2e-5,  # Adjusted LR
            output_dir="trainer_output/spider2-dbt",
            logging_dir="trainer_output/spider2-dbt/logs"
        )
        logger.info("Created Spider2-DBT training configuration")
    
    if not training_configs:
        logger.error("No Spider2 data files found. Please ensure Spider2 data is available.")
        return
    
    # Run training for each configuration
    for spider2_type, config in training_configs.items():
        logger.info(f"Starting training for Spider2-{spider2_type}")
        
        try:
            # Create output directories
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(config.logging_dir, exist_ok=True)
            
            # Save configuration
            config_path = Path(config.output_dir) / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(config.model_dump(), f, indent=2)
            
            # Run training
            await run_async_training(config)
            
            logger.info(f"Training completed for Spider2-{spider2_type}")
            
        except Exception as e:
            import traceback
            logger.error(f"Training failed for Spider2-{spider2_type}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            continue

def create_mcp_alchemy_config():
    """Create MCP Alchemy configuration for Claude Desktop."""
    config = {
        "mcpServers": {
            "spider2_lite_bigquery": {
                "command": "uvx",
                "args": [
                    "--from", "mcp-alchemy==2025.7.9.172934",
                    "--with", "psycopg2-binary",
                    "--refresh-package", "mcp-alchemy",
                    "mcp-alchemy"
                ],
                "env": {
                    "DB_URL": "postgresql://user:password@localhost/bigquery_spider2"
                }
            },
            "spider2_lite_snowflake": {
                "command": "uvx",
                "args": [
                    "--from", "mcp-alchemy==2025.7.9.172934",
                    "--with", "snowflake-connector-python",
                    "--refresh-package", "mcp-alchemy",
                    "mcp-alchemy"
                ],
                "env": {
                    "DB_URL": "snowflake://user:password@account/database/schema"
                }
            },
            "spider2_lite_sqlite": {
                "command": "uvx",
                "args": [
                    "--from", "mcp-alchemy==2025.7.9.172934",
                    "--refresh-package", "mcp-alchemy",
                    "mcp-alchemy"
                ],
                "env": {
                    "DB_URL": "sqlite:///spider2_lite.db"
                }
            },
            "spider2_snow": {
                "command": "uvx",
                "args": [
                    "--from", "mcp-alchemy==2025.7.9.172934",
                    "--with", "snowflake-connector-python",
                    "--refresh-package", "mcp-alchemy",
                    "mcp-alchemy"
                ],
                "env": {
                    "DB_URL": "snowflake://user:password@account/database/schema"
                }
            },
            "spider2_dbt": {
                "command": "uvx",
                "args": [
                    "--from", "mcp-alchemy==2025.7.9.172934",
                    "--with", "duckdb",
                    "--refresh-package", "mcp-alchemy",
                    "mcp-alchemy"
                ],
                "env": {
                    "DB_URL": "duckdb:///spider2_dbt.db"
                }
            }
        }
    }
    
    return config

if __name__ == "__main__":
    # Create MCP Alchemy configuration
    mcp_config = create_mcp_alchemy_config()
    
    # Save MCP configuration
    with open("mcp_alchemy_config.json", "w") as f:
        json.dump(mcp_config, f, indent=2)
    
    logger.info("MCP Alchemy configuration saved to mcp_alchemy_config.json")
    logger.info("Add this configuration to your Claude Desktop config file")
    
    # Run training
    asyncio.run(main()) 