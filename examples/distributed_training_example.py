"""
Distributed Training Example with Ray Actors

This example demonstrates the distributed training capabilities of the refactored retrain:
- Multiple actor groups working in parallel
- DataBuffer coordination across actors
- Hardware-optimized resource allocation
- Performance monitoring and optimization
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

from retrain.manager import ReManager
from retrain.config_models import TrainingConfig
from retrain.reward import reward
from retrain.hardware import HardwareDetector

@reward(name="sql_accuracy_reward")
def sql_accuracy_reward(prompt: str, completion: str, config_params: Dict[str, Any], **kwargs) -> float:
    """
    Reward function for SQL query generation accuracy.
    Uses the enhanced step_info from distributed actors.
    """
    step_info = kwargs.get("step_info", {})
    
    # Check for SQL execution results from verifier actors
    sql_result = step_info.get("sql_execution_result")
    if sql_result:
        if sql_result.get("success", False):
            logger.info("SQL query executed successfully")
            return 3.0
        else:
            logger.warning(f"SQL execution failed: {sql_result.get('error', 'Unknown error')}")
            return -1.0
    
    # Reward for SQL syntax elements
    sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY"]
    found_keywords = sum(1 for keyword in sql_keywords if keyword in completion.upper())
    keyword_reward = found_keywords * 0.2
    
    # Penalize obvious errors
    if "ERROR" in completion.upper() or "INVALID" in completion.upper():
        return -2.0
    
    return keyword_reward

async def main():
    """
    Main function demonstrating distributed training with multiple actor groups.
    """
    logger.info("=== Distributed Training with Ray Actors ===")
    
    # First, let's examine the hardware capabilities
    hardware_detector = HardwareDetector()
    capabilities = hardware_detector.capabilities
    recommendations = hardware_detector.recommendations
    
    logger.info("Hardware Detection Results:")
    logger.info(f"  Platform: {capabilities['platform']['name']}")
    logger.info(f"  CPU Cores: {capabilities['cpu']['cores']}")
    logger.info(f"  Memory: {capabilities['memory']['total_gb']:.1f} GB")
    logger.info(f"  GPU Available: {capabilities['device']['cuda_available']}")
    logger.info(f"  Deployment Type: {recommendations['deployment_type']}")
    logger.info(f"  Recommended Actors: {recommendations['max_concurrent_actors']}")
    
    # Load configuration
    config_path = Path(__file__).parent / "distributed_training_config.yaml"
    logger.info(f"Loading configuration from: {config_path}")
    
    if not config_path.exists():
        logger.info("Creating distributed training configuration...")
        await create_distributed_config(config_path, recommendations)
        return
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = TrainingConfig(**config_dict)
        logger.success("Configuration loaded and validated")
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Initialize ReManager for distributed training
    logger.info("Initializing ReManager for distributed training...")
    manager = ReManager(config)
    
    try:
        # Initialize with hardware-optimized actor allocation
        await manager.initialize()
        logger.success("ReManager initialized with distributed actors")
        
        # Display actor group information
        logger.info("Active Actor Groups:")
        for group_name, group_info in manager.actor_groups.items():
            logger.info(f"  {group_name}: {len(group_info) if isinstance(group_info, list) else 1} actors")
        
        # Monitor training with performance metrics
        logger.info("Starting distributed training with monitoring...")
        training_results = await run_monitored_training(manager)
        
        logger.success("Distributed training completed!")
        logger.info(f"Training results: {training_results}")
        
    except Exception as e:
        logger.error(f"Distributed training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await manager.shutdown()
        logger.info("All actors shut down successfully")

async def run_monitored_training(manager: ReManager) -> Dict[str, Any]:
    """
    Run training with performance monitoring and optimization.
    """
    training_metrics = {
        "episodes_completed": 0,
        "total_rollouts": 0,
        "avg_episode_time": 0.0,
        "actor_utilization": {},
        "databuffer_stats": {}
    }
    
    try:
        # Start the main training loop
        results = await manager.run_training()
        
        # Collect final performance metrics
        if manager.databuffer:
            databuffer_stats = await manager.databuffer.get_buffer_statistics.remote()
            training_metrics["databuffer_stats"] = databuffer_stats
        
        training_metrics.update(results)
        return training_metrics
        
    except Exception as e:
        logger.error(f"Error during monitored training: {e}")
        return training_metrics

async def create_distributed_config(config_path: Path, recommendations: Dict[str, Any]):
    """Create a configuration optimized for distributed training."""
    
    # Adjust batch sizes based on hardware recommendations
    batch_size = max(1, recommendations.get('max_concurrent_actors', 2) // 2)
    
    config_content = f"""# Distributed Training Configuration
experiment_name: "distributed_sql_training"
seed: 42
logging_level: "INFO"

# Model optimized for distributed training
model:
  name_or_path: "Qwen/Qwen3-0.6B"
  loader: "huggingface"
  torch_dtype: "auto"

# Algorithm with distributed-friendly hyperparameters
algorithm:
  name: "grpo"
  backend: "trl"
  
  report_to: ["wandb"]
  wandb_project: "retrain_distributed_sql"
  wandb_run_name: "multi_actor_training"
  
  hyperparameters:
    learning_rate: 0.00003
    num_iterations: 5
    logging_steps: 1
    beta: 0.02
    
    # Optimized for distributed training
    max_prompt_length: 512
    max_completion_length: 256
    num_generations: 2
    
    # Hardware-optimized batch sizes
    per_device_train_batch_size: {batch_size}
    gradient_accumulation_steps: 2
    
    temperature: 0.6
    top_p: 0.95

# Environment for SQL generation
environment:
  type: "spider2"
  env_specific_config:
    max_turns: 3
    max_tokens_per_llm_turn: 256
    database_schema_path: null  # Will use default schemas

# Multiple prompts for distributed processing
prompt_source:
  type: "list"
  source_config:
    prompts:
      - "Write a SQL query to find all customers who made purchases in the last 30 days"
      - "Create a query to calculate the average order value by customer segment"
      - "Generate SQL to find the top 10 products by revenue in each category"
      - "Write a query to identify customers with no purchases in the last 6 months"
      - "Create SQL to calculate monthly sales trends for the past year"
      - "Generate a query to find duplicate customer records based on email"

# Reward system optimized for SQL accuracy
reward_setup:
  step_reward_configs:
    sql_accuracy_reward:
      weight: 2.5
      params:
        syntax_weight: 1.0
        execution_weight: 2.0
      verifiers: ["sql_syntax_verifier"]
      verifier_penalty: -1.5
  rollout_reward_configs: {{}}
"""
    
    config_path.write_text(config_content)
    logger.info(f"Distributed training configuration created at: {config_path}")
    logger.info("Configuration optimized for your hardware capabilities")
    logger.info("Run the script again to start distributed training")

if __name__ == "__main__":
    asyncio.run(main())