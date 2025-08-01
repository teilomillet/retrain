"""
Working macOS Example with Ray Configuration Fix

This example demonstrates the refactored retrain architecture with proper macOS Ray configuration.
It manually sets the Ray object store size to avoid the macOS limitation.
"""

import asyncio
import sys
import os
from pathlib import Path
import yaml
from typing import Dict, Any
from loguru import logger

# Set Ray environment for macOS BEFORE any Ray imports
os.environ["RAY_ENABLE_MAC_LARGE_OBJECT_STORE"] = "1"

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import Ray early to initialize with proper config
import ray

from retrain import run
from retrain.reward import reward

@reward(name="calculator_accuracy_reward")
def calculator_accuracy_reward(prompt: str, completion: str, config_params: Dict[str, Any], **kwargs) -> float:
    """
    Reward function for calculator accuracy that works with the refactored architecture.
    """
    step_info = kwargs.get("step_info", {})
    
    # Check for parsing errors (enhanced from refactor)
    if step_info.get("parsing_error_in_text_response"):
        logger.debug("Parsing error detected, applying penalty")
        return -1.0
    
    # Check if expected answer is present
    expected_answers = config_params.get("expected_answers", [])
    for expected in expected_answers:
        if str(expected) in completion:
            logger.info(f"Found expected answer: {expected}")
            return 2.0
    
    # Reward for mathematical operations and tool usage
    reward_score = 0.0
    
    if "=" in completion:
        reward_score += 0.3
        logger.debug("Found equation in completion")
    
    if any(op in completion for op in ["+", "-", "*", "/"]):
        reward_score += 0.2
        logger.debug("Found mathematical operators")
    
    if "calculator" in completion.lower() or "tool" in completion.lower():
        reward_score += 0.5
        logger.debug("Found tool usage attempt")
    
    return reward_score

async def main():
    """
    Main function using proper Ray configuration for macOS.
    """
    logger.info("=== Working macOS Example with Refactored Architecture ===")
    
    # Initialize Ray manually with macOS-friendly settings
    if not ray.is_initialized():
        logger.info("Initializing Ray with macOS-optimized configuration...")
        ray.init(
            object_store_memory=2_000_000_000,  # 2GB limit for macOS
            log_to_driver=True,
            local_mode=False  # Use distributed mode to showcase the refactor
        )
        logger.success("Ray initialized successfully")
    
    # Configuration showcasing the refactored architecture
    config = {
        "experiment_name": "working_macos_refactor_demo",
        "seed": 42,
        "logging_level": "INFO",
        
        # Model config - hardware detection optimizes this automatically
        "model": {
            "name_or_path": "Qwen/Qwen3-0.6B",
            "loader": "huggingface",
            "torch_dtype": "auto"  # Hardware detector chooses optimal
        },
        
        # Algorithm using the unified GRPO implementation
        "algorithm": {
            "name": "grpo",
            "backend": "trl",
            "hyperparameters": {
                "learning_rate": 0.00001,
                "num_iterations": 2,  # Small for demo
                "logging_steps": 1,
                "beta": 0.01,
                "max_prompt_length": 128,
                "max_completion_length": 256,
                "num_generations": 2,
                "per_device_train_batch_size": 1,  # Small for macOS
                "gradient_accumulation_steps": 1,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        },
        
        # Environment using the refactored environment system
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
        
        # Diverse prompts to test the system
        "prompt_source": {
            "type": "list",
            "source_config": {
                "prompts": [
                    "Calculate 15 + 27 using the calculator tool. What is the result?",
                    "What is 8 * 9? Please use the calculator to verify your answer.",
                    "I need to know 144 / 12. Can you calculate this for me?",
                ]
            }
        },
        
        # Reward setup leveraging the new reward system
        "reward_setup": {
            "step_reward_configs": {
                "calculator_accuracy_reward": {
                    "weight": 1.0,
                    "params": {
                        "expected_answers": ["42", "72", "12"]  # Answers for the prompts
                    }
                }
            },
            "rollout_reward_configs": {}
        }
    }
    
    logger.info("Starting training with refactored architecture features:")
    logger.info("  ✓ Pre-configured Ray for macOS compatibility")
    logger.info("  ✓ Unified GRPO implementation (no hardware-specific actors)")
    logger.info("  ✓ ReManager coordination with hardware detection")
    logger.info("  ✓ DataBuffer integration at manager level")
    logger.info("  ✓ Enhanced reward functions with step_info")
    
    try:
        # Use the clean run() interface that leverages ReManager internally
        results = await run(config=config)
        
        logger.success("Training completed successfully!")
        logger.info("Key refactor benefits demonstrated:")
        logger.info("  • Automatic hardware detection and optimization")
        logger.info("  • Simplified configuration with intelligent defaults")
        logger.info("  • Distributed actor coordination via ReManager")
        logger.info("  • Enhanced reward system with better context")
        
        if isinstance(results, dict):
            logger.info(f"Training metrics: {results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean shutdown
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())