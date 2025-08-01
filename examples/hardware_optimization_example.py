"""
Hardware Optimization Example

This example demonstrates the hardware detection and optimization capabilities
of the refactored retrain system:
- Automatic hardware detection and capability assessment
- Dynamic actor configuration based on resources
- Memory optimization and placement group management
- Performance monitoring and resource utilization
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any
from loguru import logger

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrain.hardware import HardwareDetector, ActorFactory, Optimizer
from retrain.manager import ReManager
from retrain.config_models import TrainingConfig

async def main():
    """
    Demonstrate comprehensive hardware optimization features.
    """
    logger.info("=== Hardware Optimization Demonstration ===")
    
    # Step 1: Hardware Detection
    logger.info("Step 1: Detecting hardware capabilities...")
    detector = HardwareDetector()
    
    # Display detailed hardware information
    capabilities = detector.capabilities
    recommendations = detector.recommendations
    
    logger.info("Hardware Detection Results:")
    logger.info("=" * 50)
    
    # Platform information
    platform_info = capabilities['platform']
    logger.info(f"Platform: {platform_info['name']} ({platform_info['version']})")
    logger.info(f"Architecture: {platform_info['architecture']}")
    logger.info(f"macOS: {platform_info['is_macos']}")
    
    # CPU information
    cpu_info = capabilities['cpu']
    logger.info(f"CPU Cores: Physical={cpu_info['physical_cores']}, Logical={cpu_info['cores']}")
    logger.info(f"CPU Frequency: {cpu_info['frequency_ghz']:.2f} GHz")
    
    # Memory information
    memory_info = capabilities['memory']
    logger.info(f"Memory: {memory_info['total_gb']:.1f} GB total, {memory_info['available_gb']:.1f} GB available")
    
    # Device information
    device_info = capabilities['device']
    logger.info(f"CUDA Available: {device_info['cuda_available']}")
    logger.info(f"MPS Available: {device_info['mps_available']}")
    logger.info(f"Recommended Device: {device_info['recommended']}")
    
    if device_info['cuda_available']:
        logger.info(f"GPU Count: {device_info['gpu_count']}")
        logger.info(f"GPU Memory: {device_info['gpu_memory_gb']:.1f} GB")
    
    # Recommendations
    logger.info("\nOptimization Recommendations:")
    logger.info("=" * 50)
    logger.info(f"Deployment Type: {recommendations['deployment_type']}")
    logger.info(f"Max Concurrent Actors: {recommendations['max_concurrent_actors']}")
    logger.info(f"Recommended Batch Size: {recommendations['batch_size_recommendation']}")
    logger.info(f"Memory Optimization: {recommendations['memory_optimization_level']}")
    
    # Step 2: Actor Factory Demonstration
    logger.info("\nStep 2: Testing Actor Factory optimization...")
    factory = ActorFactory(detector)
    
    # Test different actor configurations
    mock_config = type('MockConfig', (), {})()
    
    logger.info("Actor Resource Configurations:")
    for actor_type in ['trainer', 'inference', 'reward', 'verifier']:
        actor_config = detector.get_actor_config(actor_type)
        logger.info(f"  {actor_type.capitalize()}: {actor_config}")
    
    # Step 3: Hardware Optimization
    logger.info("\nStep 3: Applying hardware optimizations...")
    optimizer = Optimizer(detector)
    
    # Test memory optimization
    memory_config = optimizer.optimize_memory_usage()
    logger.info(f"Memory Optimization Config: {memory_config}")
    
    # Test batch size optimization
    batch_config = optimizer.optimize_batch_sizes(model_size_mb=1200)  # Qwen3-0.6B approx size
    logger.info(f"Batch Size Optimization: {batch_config}")
    
    # Step 4: Integration with ReManager
    logger.info("\nStep 4: Testing ReManager integration...")
    
    # Create a minimal config for testing
    test_config = create_test_config(recommendations)
    
    try:
        config = TrainingConfig(**test_config)
        manager = ReManager(config)
        
        # Test initialization (without actual training)
        logger.info("Testing ReManager initialization...")
        await manager.initialize()
        
        # Display created actor groups
        logger.info("Created Actor Groups:")
        for group_name, group in manager.actor_groups.items():
            if isinstance(group, list):
                logger.info(f"  {group_name}: {len(group)} actors")
            else:
                logger.info(f"  {group_name}: 1 actor")
        
        # Test resource utilization monitoring
        if hasattr(manager, 'monitor_resources'):
            resource_stats = await manager.monitor_resources()
            logger.info(f"Resource Utilization: {resource_stats}")
        
        await manager.shutdown()
        logger.success("ReManager test completed successfully")
        
    except Exception as e:
        logger.error(f"ReManager test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Performance Recommendations
    logger.info("\nStep 5: Performance optimization recommendations...")
    display_optimization_tips(capabilities, recommendations)

def create_test_config(recommendations: Dict[str, Any]) -> Dict[str, Any]:
    """Create a minimal test configuration optimized for detected hardware."""
    
    return {
        "experiment_name": "hardware_optimization_test",
        "seed": 42,
        "logging_level": "INFO",
        
        "model": {
            "name_or_path": "Qwen/Qwen3-0.6B",
            "loader": "huggingface",
            "torch_dtype": "auto"
        },
        
        "algorithm": {
            "name": "grpo",
            "backend": "trl",
            "hyperparameters": {
                "learning_rate": 0.00001,
                "num_iterations": 1,
                "per_device_train_batch_size": recommendations.get('batch_size_recommendation', 1),
                "gradient_accumulation_steps": 1,
                "max_prompt_length": 128,
                "max_completion_length": 256,
                "num_generations": 2
            }
        },
        
        "environment": {
            "type": "smol_agent",
            "env_specific_config": {
                "max_turns": 2,
                "max_tokens_per_llm_turn": 256,
                "tools": {
                    "registry_keys": ["simple_calculator_tool"]
                }
            }
        },
        
        "prompt_source": {
            "type": "list",
            "source_config": {
                "prompts": ["Calculate 2 + 2 using the calculator tool."]
            }
        },
        
        "reward_setup": {
            "step_reward_configs": {
                "exact_match": {
                    "weight": 1.0,
                    "params": {"expected_value": "4"}
                }
            },
            "rollout_reward_configs": {}
        }
    }

def display_optimization_tips(capabilities: Dict[str, Any], recommendations: Dict[str, Any]):
    """Display hardware-specific optimization tips."""
    
    logger.info("Hardware-Specific Optimization Tips:")
    logger.info("=" * 50)
    
    deployment_type = recommendations['deployment_type']
    
    if deployment_type == 'development':
        logger.info("💻 Development Setup Detected:")
        logger.info("  - Use smaller models (0.6B-1.5B parameters) for faster iteration")
        logger.info("  - Enable MPS acceleration on macOS for better performance")
        logger.info("  - Consider gradient checkpointing to reduce memory usage")
        logger.info("  - Use lower batch sizes (1-2) to fit in available memory")
        
    elif deployment_type == 'production':
        logger.info("🚀 Production Setup Detected:")
        logger.info("  - Leverage multiple GPUs with data parallelism")
        logger.info("  - Use larger batch sizes for better GPU utilization")
        logger.info("  - Enable mixed precision training (fp16/bf16)")
        logger.info("  - Consider model sharding for very large models")
        
    else:
        logger.info("⚡ Standard Setup Detected:")
        logger.info("  - Balance batch size with available memory")
        logger.info("  - Use gradient accumulation if memory is limited")
        logger.info("  - Monitor GPU utilization for optimization opportunities")
    
    # Memory-specific tips
    memory_gb = capabilities['memory']['available_gb']
    if memory_gb < 8:
        logger.warning("⚠️  Low Memory Detected:")
        logger.warning("  - Use gradient checkpointing")
        logger.warning("  - Consider model quantization (4-bit/8-bit)")
        logger.warning("  - Use very small batch sizes (1)")
        logger.warning("  - Enable CPU offloading for large models")
    
    elif memory_gb < 16:
        logger.info("📊 Moderate Memory Available:")
        logger.info("  - Batch sizes of 2-4 should work well")
        logger.info("  - Consider gradient checkpointing for larger models")
        logger.info("  - Monitor memory usage during training")
    
    else:
        logger.info("💾 Abundant Memory Available:")
        logger.info("  - Can use larger batch sizes (4-8+)")
        logger.info("  - Can train larger models without optimization")
        logger.info("  - Consider increasing sequence lengths")

if __name__ == "__main__":
    asyncio.run(main())