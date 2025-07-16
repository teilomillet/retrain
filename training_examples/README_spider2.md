# Spider2 Training with MCP Alchemy

This guide shows how to train language models on the Spider2 benchmark using the retrain framework with MCP Alchemy for database interaction.

## Overview

Spider2 is a challenging text-to-SQL benchmark that tests language models on real-world enterprise database scenarios. This setup integrates:

- **Spider2 Benchmark**: 547 complex text-to-SQL questions across multiple databases
- **MCP Alchemy**: Database interaction tools via Model Context Protocol
- **Retrain Framework**: Reinforcement learning training with GRPO algorithm
- **Multiple Database Support**: SQLite, PostgreSQL, Snowflake, and DuckDB

## Quick Start

### 1. Setup Environment

```bash
# Run the setup script
uv run training_examples/setup_spider2_training.py

# Install dependencies if needed
uv add fastmcp sqlalchemy psycopg2-binary snowflake-connector-python duckdb
```

### 2. Configure Database

Choose your database setup:

**SQLite (Recommended for testing):**
```json
{
  "mcpServers": {
    "spider2_sqlite": {
      "command": "uvx",
      "args": ["--from", "mcp-alchemy==2025.7.9.172934", "--refresh-package", "mcp-alchemy", "mcp-alchemy"],
      "env": {
        "DB_URL": "sqlite:///spider2_lite.db"
      }
    }
  }
}
```

**PostgreSQL:**
```json
{
  "mcpServers": {
    "spider2_postgres": {
      "command": "uvx",
      "args": ["--from", "mcp-alchemy==2025.7.9.172934", "--with", "psycopg2-binary", "--refresh-package", "mcp-alchemy", "mcp-alchemy"],
      "env": {
        "DB_URL": "postgresql://user:password@localhost/spider2_db"
      }
    }
  }
}
```

### 3. Start Training

```bash
# Quick start
python start_spider2_training.py

# Or run the full script
python training_examples/spider2_training.py
```

## Architecture

### Components

1. **Spider2Env**: Custom environment that loads Spider2 questions and manages database interactions
2. **Spider2DatabaseTool**: MCP Alchemy tool for database operations
3. **Spider2MCPProvider**: Tool provider that discovers and registers database tools
4. **Training Pipeline**: GRPO training with TRL backend

### Data Flow

```
Spider2 Questions → Spider2Env → MCP Alchemy → Database
                     ↓
                LLM generates SQL
                     ↓
                Tool execution
                     ↓
                Reward calculation
                     ↓
                Model update (GRPO)
```

## Configuration

### Training Configuration

The training configuration is defined in `spider2_training_config.json`:

```json
{
  "model": {
    "name_or_path": "microsoft/DialoGPT-medium",
    "loader": "huggingface",
    "peft_config": null,
    "torch_dtype": "auto",
    "trust_remote_code": true
  },
  "environment": {
    "type": "spider2",
    "env_specific_config": {
      "spider2_data_path": "Spider2/spider2-lite/spider2-lite.jsonl",
      "spider2_type": "lite",
      "mcp_server_url": "http://localhost:8000",
      "max_steps": 20,
      "max_query_length": 1000,
      "enable_external_knowledge": true,
      "evaluation_mode": false
    }
  },
  "algorithm": {
    "name": "grpo",
    "backend": "trl",
    "hyperparameters": {
      "learning_rate": 1e-5,
      "batch_size": 4,
      "max_length": 512,
      "num_iterations": 100
    }
  }
}
```

### Spider2 Variants

The setup supports three Spider2 variants:

1. **Spider2-Lite**: 547 questions across BigQuery, Snowflake, and SQLite
2. **Spider2-Snow**: 547 questions on Snowflake only
3. **Spider2-DBT**: 68 questions on DuckDB with DBT

## Database Setup

### SQLite (Development)

```bash
# Create SQLite database
sqlite3 spider2_lite.db
```

### PostgreSQL

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb spider2_db
sudo -u postgres psql -c "CREATE USER spider2_user WITH PASSWORD 'your_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE spider2_db TO spider2_user;"
```

### Snowflake

1. Sign up for Snowflake account
2. Create database and schema
3. Update connection string in config
4. Install snowflake-connector-python

## Training Process

### 1. Environment Reset

For each training episode:
- Randomly select a Spider2 question
- Load database schema via MCP Alchemy
- Present question to the model

### 2. Model Interaction

The model can:
- Explore database schema using tools
- Execute SQL queries
- Generate final answers

### 3. Reward Calculation

Rewards are based on:
- **SQL Quality**: Syntax and semantic correctness
- **Tool Usage**: Effective database exploration
- **Answer Accuracy**: Final answer correctness

### 4. Model Update

GRPO algorithm updates the model based on:
- Step-level rewards for tool usage
- Rollout-level rewards for final answers
- Verifier penalties for incorrect SQL

## Monitoring

### Logs

Training logs are saved to:
- `trainer_output/spider2-{type}/logs/`
- `trainer_output/spider2-{type}/checkpoint-{step}/`

### Metrics

Track these metrics:
- `eval_loss`: Training loss
- `sql_accuracy`: SQL generation accuracy
- `tool_usage_rate`: Database tool usage
- `episode_length`: Average episode length

### Weights & Biases

Enable W&B logging by setting:
```bash
export WANDB_API_KEY=your_api_key
```

## Advanced Configuration

### Custom Models

Replace the model in config:
```json
{
  "model": {
    "name_or_path": "your-model-path",
    "loader": "huggingface",
    "peft_config": {
      "lora_alpha": 16,
      "lora_dropout": 0.1,
      "r": 64,
      "target_modules": ["q_proj", "v_proj"]
    }
  }
}
```

### Custom Rewards

Add custom reward functions:
```json
{
  "reward_setup": {
    "step_reward_configs": {
      "custom_reward": {
        "weight": 1.0,
        "params": {"custom_param": "value"},
        "verifiers": ["custom_verifier"],
        "verifier_penalty": 0.5,
        "distribution_strategy": "last_step"
      }
    }
  }
}
```

### Hyperparameter Tuning

Key hyperparameters to tune:
- `learning_rate`: Start with 1e-5
- `batch_size`: Adjust based on GPU memory
- `max_steps`: Longer episodes for complex questions
- `num_iterations`: More iterations for better convergence

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install fastmcp sqlalchemy psycopg2-binary
   ```

2. **Database Connection Failed**
   - Check credentials in config
   - Verify database is running
   - Test connection manually

3. **Out of Memory**
   - Reduce batch_size
   - Use smaller model
   - Enable gradient checkpointing

4. **Training Not Converging**
   - Increase learning rate
   - Add more reward functions
   - Check reward scaling

### Debug Mode

Enable debug logging:
```json
{
  "logging_level": "DEBUG"
}
```

## Performance Tips

1. **Start Small**: Use SQLite and small model for initial testing
2. **Gradual Scaling**: Increase model size and database complexity gradually
3. **Reward Engineering**: Carefully tune reward functions for your use case
4. **Data Quality**: Ensure Spider2 data is properly loaded
5. **Hardware**: Use GPU with sufficient memory for larger models

## Evaluation

### Metrics

Evaluate on:
- **SQL Execution Accuracy**: Does generated SQL run correctly?
- **Answer Accuracy**: Does the final answer match ground truth?
- **Tool Usage Efficiency**: How effectively does the model explore the database?

### Benchmarking

Compare against:
- Baseline models (GPT-4, Claude)
- Previous Spider2 submissions
- Your own model variants

## Contributing

To extend this setup:

1. **New Database**: Add database driver and config
2. **New Reward**: Implement custom reward function
3. **New Model**: Add model loader support
4. **New Algorithm**: Implement new training algorithm

## References

- [Spider2 Paper](https://arxiv.org/abs/2411.07763)
- [MCP Alchemy](https://github.com/runekaagaard/mcp-alchemy)
- [Retrain Framework](../README.md)
- [TRL Documentation](https://huggingface.co/docs/trl)

## License

This setup follows the licenses of the underlying components:
- Spider2: MIT License
- MCP Alchemy: Mozilla Public License 2.0
- Retrain: Project-specific license 