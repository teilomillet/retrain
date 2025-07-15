# MCP Training with Retrain

Simple training setup to teach Qwen3-0.6B to use MCP tools effectively.

## Quick Start

```bash
# Install retrain
pip install retrain

# Start MCP server (in one terminal)
python training_examples/mcp_server.py

# Run training (in another terminal)
python training_examples/mcp_training.py
```

## Files

- **`mcp_training.py`** - Main training script with reward functions
- **`mcp_config.yaml`** - Training configuration
- **`mcp_server.py`** - MCP server with training tools

## What It Teaches

The model learns to:
- Discover and explore available MCP tools
- Use appropriate tools for tasks
- Provide correct tool parameters
- Complete multi-step tasks
- Search for tools when needed

## MCP Tools

- `get_server_time(timezone)` - Get current server time
- `perform_operation(operation, operand1, operand2)` - Arithmetic operations
- `list_server_capabilities()` - List available tools
- `search_tools(query)` - Search for tools by keyword

## Monitoring

Training logs to Weights & Biases project `qwen3_mcp_training`.

## Customization

Edit `mcp_config.yaml` to modify training parameters, or add new reward functions to `mcp_training.py`. 