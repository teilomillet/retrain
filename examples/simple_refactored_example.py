"""
Simple Refactored Example

This example demonstrates the refactored retrain architecture using the basic run() function
which handles Ray configuration automatically for different platforms.
"""

import asyncio
import sys
import os
from pathlib import Path
import yaml
from typing import Dict, Any
from loguru import logger

# Set Ray environment for macOS
os.environ["RAY_ENABLE_MAC_LARGE_OBJECT_STORE"] = "1"

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrain import run
from retrain.reward import reward

@reward(name="simple_calculator_reward")
def simple_calculator_reward(prompt: str, completion: str, config_params: Dict[str, Any], **kwargs) -> float:
    """
    Simple reward function for calculator examples that leverages the refactored architecture.
    """
    step_info = kwargs.get("step_info", {})
    
    # Check for parsing errors (enhanced error handling from refactor)
    if step_info.get("parsing_error_in_text_response"):
        logger.debug(f"Parsing error detected: {step_info['parsing_error_in_text_response']}")
        return -1.0
    
    # Check for expected answer
    expected = str(config_params.get("expected_answer", ""))
    if expected and expected in completion:
        logger.info(f"Found expected answer: {expected}")
        return 2.0
    
    # Reward for mathematical operations
    if any(op in completion for op in ["=", "+", "-", "*", "/"]):
        logger.debug("Mathematical operations found")
        return 0.5
    
    return 0.0

async def main():
    """
    Main function using the simple run() interface that leverages the refactored architecture.
    """
    logger.info("=== Simple Refactored Retrain Example ===")
    
    # Create a simple configuration that showcases the refactored features
    config = {
        "experiment_name": "simple_refactored_test",
        "seed": 42,
        "logging_level": "INFO",
        
        # Model config - hardware detection will optimize this
        "model": {
            "name_or_path": "Qwen/Qwen3-0.6B",
            "loader": "huggingface",
            "torch_dtype": "auto"  # Hardware detector will choose optimal
        },
        
        # Algorithm config - uses the new unified GRPO implementation
        "algorithm": {
            "name": "grpo",
            "backend": "trl",
            "hyperparameters": {
                "learning_rate": 0.00001,
                "num_iterations": 2,  # Keep small for demo
                "logging_steps": 1,
                "beta": 0.01,
                "max_prompt_length": 128,
                "max_completion_length": 256,
                "num_generations": 2,
                "per_device_train_batch_size": 1,  # Small for demo
                "gradient_accumulation_steps": 1,
                "temperature": 0.7,
                "top_p": 0.9
            }
        },
        
        # Environment config - uses the refactored environment system
        "environment": {
            "type": "smol_agent",
            "env_specific_config": {
                "max_turns": 3,
                "max_tokens_per_llm_turn": 256,
                "tools": {
                    "registry_keys": ["simple_calculator_tool"]
                }
            }
        },
        
        # Prompt source
        "prompt_source": {
            "type": "list",
            "source_config": {
                "prompts": [
                    "Calculate 15 + 27 using the calculator tool.",
                    "What is 8 * 9? Use the calculator to verify.",
                ]
            }
        },
        
        # Reward setup using the new reward system
        "reward_setup": {
            "step_reward_configs": {
                "simple_calculator_reward": {
                    "weight": 1.0,
                    "params": {
                        "expected_answer": "42"  # Will be different for each problem
                    }
                }
            },
            "rollout_reward_configs": {}
        }
    }
    
    logger.info("Configuration created, starting training...")
    logger.info("This will use:")
    logger.info("  - Hardware detection for optimal resource allocation")
    logger.info("  - Unified GRPO implementation (no hardware-specific actors)")
    logger.info("  - DataBuffer coordination at the manager level")
    logger.info("  - Ray-first distributed architecture")
    
    try:
        # Use the simple run() function which internally uses ReManager
        results = await run(config=config)
        
        logger.success("Training completed successfully!")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())