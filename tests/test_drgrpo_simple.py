#!/usr/bin/env python3
"""
Simple test script for DRGRPO initialization.
"""

import asyncio
import ray
import logging
from pathlib import Path
import sys

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrain.config_models import TrainingConfig

async def test_drgrpo():
    """Test DRGRPO initialization."""
    print("Testing DRGRPO initialization...")
    
    # Initialize Ray with 2GB object store memory limit for macOS compatibility
    if not ray.is_initialized():
        print("Initializing Ray...")
        ray.init(log_to_driver=True, logging_level=logging.INFO, object_store_memory=2_000_000_000)
        print("Ray initialized successfully")
    
    # Create a minimal config
    config_dict = {
        "model": {"name_or_path": "Qwen/Qwen3-0.6B", "loader": "huggingface"},
        "algorithm": {
            "name": "drgrpo",
            "backend": "retrain",
            "hyperparameters": {"learning_rate": 0.00001}
        },
        "environment": {"type": "smol_agent", "env_specific_config": {}},
        "prompt_source": {"type": "list", "source_config": {"prompts": ["test"]}},
        "reward_setup": {"step_reward_configs": {}, "rollout_reward_configs": {}},
        "experiment_name": "test_drgrpo"
    }
    
    config = TrainingConfig(**config_dict)
    print(f"Config created: {config.experiment_name}")
    
    # Test DRGRPO import
    try:
        from retrain.trainer.grpo.drgrpo import DRGRPO
        print("DRGRPO imported successfully")
        
        # Create DRGRPO actor
        drgrpo_actor = DRGRPO.remote(config=config)  # type: ignore
        print("DRGRPO actor created successfully")
        
        # Test initialization
        await ray.get(drgrpo_actor.initialize.remote())
        print("DRGRPO actor initialized successfully")
        
        # Test health check
        health = await ray.get(drgrpo_actor.health_check.remote())
        print(f"DRGRPO health check: {health}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Shutdown Ray
    if ray.is_initialized():
        ray.shutdown()
        print("Ray shutdown")

if __name__ == "__main__":
    asyncio.run(test_drgrpo()) 