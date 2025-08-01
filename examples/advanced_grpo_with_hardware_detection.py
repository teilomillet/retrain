"""
Advanced GRPO Example with Hardware Detection and ReManager

This example demonstrates the refactored retrain architecture:
- ReManager for intelligent hardware detection and resource allocation
- Automatic actor factory optimization
- DataBuffer integration at the manager level
- Ray-first distributed training
"""

import asyncio
import sys
from pathlib import Path
import yaml
from typing import Dict, Any
from loguru import logger

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrain.manager import ReManager # type: ignore
from retrain.config_models import TrainingConfig # type: ignore
from retrain.reward import reward # type: ignore

# Custom reward function leveraging step_info from the new architecture
@reward(name="advanced_math_reward")
def advanced_math_reward(prompt: str, completion: str, config_params: Dict[str, Any], **kwargs) -> float:
    """
    Advanced reward function for mathematical problem solving.
    Leverages the enhanced step_info from the refactored architecture.
    """
    step_info = kwargs.get("step_info", {})
    
    # Check for parsing errors (enhanced error handling from refactor)
    parsing_error_penalty = config_params.get("parsing_error_penalty", -2.0)
    if step_info.get("parsing_error_in_text_response"):
        logger.warning(f"Parsing error detected: {step_info['parsing_error_in_text_response']}")
        return parsing_error_penalty
    
    # Check for correct final answer
    expected_answer = str(config_params.get("expected_answer", ""))
    if expected_answer in completion:
        logger.info(f"Correct answer found: {expected_answer}")
        return 2.0
    
    # Reward partial progress (tool usage, intermediate steps)
    partial_reward = 0.0
    
    # Reward for showing work
    if "=" in completion and any(op in completion for op in ["+", "-", "*", "/"]):
        partial_reward += 0.5
        logger.debug("Partial reward for showing mathematical work")
    
    # Reward for tool usage
    if "tool" in completion.lower() or "calculate" in completion.lower():
        partial_reward += 0.3
        logger.debug("Partial reward for attempting tool usage")
    
    return partial_reward

async def main():
    """
    Main function demonstrating the new ReManager-based training flow.
    """
    logger.info("=== Advanced GRPO Training with ReManager ===")
    
    # Load configuration
    config_path = Path(__file__).parent / "advanced_grpo_config.yaml"
    logger.info(f"Loading configuration from: {config_path}")
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Validate configuration using the new config models
        config = TrainingConfig(**config_dict)
        logger.success("Configuration loaded and validated successfully")
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Initialize ReManager (new centralized orchestrator)
    logger.info("Initializing ReManager with hardware detection...")
    manager = ReManager(config)
    
    try:
        # Initialize the manager (detects hardware, creates actors, etc.)
        await manager.initialize()
        logger.success("ReManager initialized successfully")
        
        # Log hardware detection results
        hardware_info = manager.hardware_detector.capabilities
        logger.info(f"Detected platform: {hardware_info['platform']['system']} ({hardware_info['platform']['machine']})")
        logger.info(f"Deployment type: {manager.hardware_detector.recommendations['deployment_type']}")
        
        # Start training using the new distributed architecture
        logger.info("Starting training with ReManager...")
        training_results = await manager.train()  # Changed from run_training() to train()
        
        logger.success("Training completed successfully!")
        logger.info(f"Final metrics: {training_results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean shutdown
        await manager.shutdown()
        logger.info("ReManager shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())