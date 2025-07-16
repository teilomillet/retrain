#!/usr/bin/env python3
"""
Spider2 Training Setup Script

This script helps set up the environment for training on Spider2 benchmark
using MCP Alchemy for database interaction.
"""

import os
import json
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    import importlib.metadata
    
    required_packages = [
        "fastmcp",
        "sqlalchemy",
        "psycopg2-binary",
        "snowflake-connector-python",
        "duckdb"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def setup_directories():
    """Create necessary directories."""
    directories = [
        "trainer_output",
        "trainer_output/spider2-lite",
        "trainer_output/spider2-snow", 
        "trainer_output/spider2-dbt",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def create_database_configs():
    """Create database configuration templates."""
    
    # SQLite configuration (easiest to start with)
    sqlite_config = {
        "mcpServers": {
            "spider2_sqlite": {
                "command": "uvx",
                "args": [
                    "--from", "mcp-alchemy==2025.7.9.172934",
                    "--refresh-package", "mcp-alchemy",
                    "mcp-alchemy"
                ],
                "env": {
                    "DB_URL": "sqlite:///spider2_lite.db"
                }
            }
        }
    }
    
    # PostgreSQL configuration
    postgres_config = {
        "mcpServers": {
            "spider2_postgres": {
                "command": "uvx",
                "args": [
                    "--from", "mcp-alchemy==2025.7.9.172934",
                    "--with", "psycopg2-binary",
                    "--refresh-package", "mcp-alchemy",
                    "mcp-alchemy"
                ],
                "env": {
                    "DB_URL": "postgresql://user:password@localhost/spider2_db"
                }
            }
        }
    }
    
    # Snowflake configuration
    snowflake_config = {
        "mcpServers": {
            "spider2_snowflake": {
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
            }
        }
    }
    
    # Save configurations
    configs = {
        "sqlite": sqlite_config,
        "postgres": postgres_config,
        "snowflake": snowflake_config
    }
    
    for db_type, config in configs.items():
        config_file = f"mcp_alchemy_{db_type}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created {config_file}")
    
    return configs

def create_training_configs():
    """Create training configuration templates."""
    
    # Basic training config for SQLite (easiest to start)
    basic_config = {
        "model": {
            "name_or_path": "microsoft/DialoGPT-medium",
            "loader": "huggingface",
            "peft_config": None,
            "torch_dtype": "auto",
            "trust_remote_code": True
        },
        "environment": {
            "type": "spider2",
            "env_specific_config": {
                "spider2_data_path": "Spider2/spider2-lite/spider2-lite.jsonl",
                "spider2_type": "lite",
                "mcp_server_url": "http://localhost:8008",
                "max_steps": 20,
                "max_query_length": 1000,
                "enable_external_knowledge": True,
                "evaluation_mode": False
            }
        },
        "prompt_source": {
            "type": "spider2",
            "source_config": {
                "data_path": "Spider2/spider2-lite/spider2-lite.jsonl"
            }
        },
        "reward_setup": {
            "step_reward_configs": {
                "sql_quality": {
                    "weight": 1.0,
                    "params": {
                        "syntax_check": True,
                        "semantic_check": True,
                        "execution_check": True
                    },
                    "verifiers": ["sql_syntax", "sql_semantics"],
                    "verifier_penalty": 0.5,
                    "distribution_strategy": "last_step"
                }
            },
            "rollout_reward_configs": {
                "final_answer_quality": {
                    "weight": 2.0,
                    "params": {
                        "accuracy_threshold": 0.8,
                        "completeness_check": True
                    },
                    "verifiers": ["answer_accuracy"],
                    "verifier_penalty": 1.0,
                    "distribution_strategy": "last_step"
                }
            }
        },
        "algorithm": {
            "name": "grpo",
            "backend": "trl",
            "hyperparameters": {
                "learning_rate": 1e-5,
                "batch_size": 4,
                "max_length": 512,
                "num_iterations": 100,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 10,
                "logging_steps": 5,
                "save_steps": 50,
                "eval_steps": 25,
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "remove_unused_columns": False,
                "dataloader_pin_memory": False,
                "bf16": True,
                "tf32": True,
                "gradient_checkpointing": True,
                "optim": "adamw_torch",
                "lr_scheduler_type": "cosine",
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "seed": 42
            },
            "peft_config": None,
            "report_to": [],
            "wandb_project": "spider2-training",
            "wandb_entity": None,
            "wandb_run_name": "spider2-lite-grpo"
        },
        "experiment_name": "spider2-lite-training",
        "seed": 42,
        "logging_level": "INFO"
    }
    
    # Save training config
    with open("spider2_training_config.json", 'w') as f:
        json.dump(basic_config, f, indent=2)
    print("Created spider2_training_config.json")
    
    return basic_config

def create_startup_script():
    """Create a startup script for easy training."""
    
    script_content = '''#!/usr/bin/env python3
"""
Quick start script for Spider2 training
"""

import asyncio
import sys
from pathlib import Path

# Add retrain to path
sys.path.append(str(Path(__file__).parent.parent))

from training_examples.spider2_training import main

if __name__ == "__main__":
    print("Starting Spider2 training...")
    asyncio.run(main())
'''
    
    with open("start_spider2_training.py", 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod("start_spider2_training.py", 0o755)
    print("Created start_spider2_training.py")

def print_setup_instructions():
    """Print setup instructions for the user."""
    
    instructions = """
=== Spider2 Training Setup Complete ===

Next steps:

1. Configure your database connection:
   - Edit the MCP Alchemy config files (mcp_alchemy_*.json)
   - Update the DB_URL with your actual database credentials

2. Start the MCP Alchemy server:
   - For SQLite: The config is ready to use
   - For PostgreSQL/Snowflake: Update credentials and start server

3. Configure training:
   - Edit spider2_training_config.json if needed
   - Set your model path and hyperparameters

4. Run training:
   python start_spider2_training.py

Or run the full training script:
   python training_examples/spider2_training.py

=== Database Setup Notes ===

SQLite (Recommended for testing):
- Easiest to set up
- No external database required
- Good for development and testing

PostgreSQL:
- Requires PostgreSQL server
- Update connection string in config
- Install psycopg2-binary

Snowflake:
- Requires Snowflake account
- Update connection string with your credentials
- Install snowflake-connector-python

=== Troubleshooting ===

1. If you get import errors:
   pip install fastmcp sqlalchemy psycopg2-binary snowflake-connector-python duckdb

2. If MCP server fails to start:
   - Check your database credentials
   - Ensure database is accessible
   - Check firewall settings

3. If training fails:
   - Check model path exists
   - Verify Spider2 data files are present
   - Check GPU memory if using large models
"""
    
    print(instructions)

def main():
    """Main setup function."""
    print("Setting up Spider2 training environment...")
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        return
    
    # Setup directories
    setup_directories()
    
    # Create configurations
    create_database_configs()
    create_training_configs()
    
    # Create startup script
    create_startup_script()
    
    # Print instructions
    print_setup_instructions()

if __name__ == "__main__":
    main() 