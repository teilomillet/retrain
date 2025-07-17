"""
Example usage of the Retrain hardware detection and management system.

This demonstrates how the hardware detection automatically configures
Ray actors and provides recommendations for different platforms.
"""

import asyncio
import logging
from typing import Any
from ..hardware import HardwareDetector, ActorFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_hardware_detection():
    """Demonstrate hardware detection and automatic configuration."""
    
    print("🔍 Retrain Hardware Detection Demo")
    print("=" * 50)
    
    # Step 1: Detect hardware automatically
    detector = HardwareDetector()
    
    # Print comprehensive hardware summary
    detector.print_summary()
    
    # Step 2: Create actor factory with optimized configuration
    factory = ActorFactory(detector)
    
    # Print planned actor deployment
    factory.print_actor_plan()
    
    # Step 3: Show recommended configuration
    print("\n⚙️  Recommended Configuration:")
    print("-" * 30)
    
    recommendations = detector.recommendations
    
    print(f"Backend: {recommendations['backend']}")
    print(f"Model Size Class: {recommendations['model_size_class']}")
    print(f"Deployment Type: {recommendations['deployment_type']}")
    
    print("\nRecommended Models:")
    for model in recommendations['model_recommendations']:
        print(f"  • {model}")
        
    print("\nTraining Configuration:")
    training_config = recommendations['training_config']
    for key, value in training_config.items():
        print(f"  • {key}: {value}")
        
    # Step 4: Show Ray initialization configuration
    print("\n🚀 Ray Initialization:")
    ray_config = factory.get_ray_init_config()
    for key, value in ray_config.items():
        if key == 'object_store_memory':
            print(f"  • {key}: {value / 1e9:.1f}GB")
        else:
            print(f"  • {key}: {value}")
            
    # Step 5: Show performance optimization tips
    if 'performance_tips' in recommendations:
        print("\n💡 Performance Tips:")
        for tip in recommendations['performance_tips']:
            print(f"  • {tip}")
    
    return detector, factory


def create_sample_config() -> Any:
    """Create a sample training configuration."""
    # This would normally be loaded from a config file
    from types import SimpleNamespace
    
    config = SimpleNamespace()
    config.experiment_name = "hardware_detection_demo"
    config.num_episodes = 10
    config.batch_size = 4
    config.output_dir = "./demo_output"
    
    # Model configuration
    config.model = SimpleNamespace()
    config.model.name_or_path = "gpt2"
    config.model.trust_remote_code = True
    
    # Algorithm configuration
    config.algorithm = SimpleNamespace()
    config.algorithm.name = "grpo"
    config.algorithm.backend = "auto"  # Will be set by hardware detection
    config.algorithm.learning_rate = 1e-5
    
    # Environment configuration
    config.environment = SimpleNamespace()
    config.environment.type = "smol_agent"
    config.environment.env_specific_config = {}
    
    return config


async def demo_smart_manager():
    """Demonstrate the smart manager with hardware detection."""
    
    print("\n🏗️  Smart Manager Demo")
    print("=" * 30)
    
    # Create sample configuration
    config = create_sample_config()
    
    # The manager automatically detects hardware and configures everything
    from ..manager.manager import ReManager
    
    try:
        manager = ReManager(config)
        
        # Print the deployment plan
        manager.print_deployment_plan()
        
        # Show training status
        status = manager.get_training_status()
        print("\n📊 Training Status:")
        for key, value in status.items():
            if key not in ['recent_metrics']:  # Skip complex nested data
                print(f"  • {key}: {value}")
                
        print(f"\nManager successfully configured for {status['deployment_type']} deployment")
        print(f"    Backend: {status['backend']}")
        print(f"    Hardware: {status['hardware_summary']}")
        
    except Exception as e:
        print(f"Manager demo failed: {e}")
        print("   This is expected if Ray actors aren't fully implemented yet")


def show_platform_comparisons():
    """Show how configuration differs across platforms."""
    
    print("\n🔄 Platform Comparison")
    print("=" * 30)
    
    # This would show different configurations for different platforms
    platforms = [
        ("macOS (Apple Silicon)", "development", "transformers", "small"),
        ("Linux (Single GPU)", "small_scale", "mbridge", "medium"), 
        ("Linux (Multi-GPU)", "production", "mbridge", "large"),
        ("CPU-only", "cpu_only", "transformers", "small")
    ]
    
    for platform, deployment, backend, model_size in platforms:
        print(f"\n{platform}:")
        print(f"  • Deployment: {deployment}")
        print(f"  • Backend: {backend}")
        print(f"  • Model Size: {model_size}")
        
        if deployment == "development":
            print("  • Benefits: Fast iteration, stable development")
        elif deployment == "production":
            print("  • Benefits: Maximum performance, large model support")
        elif deployment == "small_scale":
            print("  • Benefits: Good performance, single GPU efficiency")
        else:
            print("  • Benefits: Universal compatibility, no GPU required")


async def main():
    """Run the complete hardware detection demo."""
    
    # Detect hardware and show recommendations
    detector, factory = await demo_hardware_detection()
    
    # Show platform comparisons
    show_platform_comparisons()
    
    # Demonstrate smart manager (may fail due to missing implementations)
    await demo_smart_manager()
    
    print("\n🎯 Summary:")
    print(f"  • Hardware: {detector.capabilities['summary']}")
    print(f"  • Recommended Backend: {detector.recommendations['backend']}")
    print(f"  • Deployment Type: {detector.recommendations['deployment_type']}")
    print(f"  • Ray Benefits: Parallel processing even on {detector.capabilities['device']['primary_device']}")
    
    print("\n✨ Next Steps:")
    print("  1. Use detected configuration for training")
    print("  2. Ray provides parallelization benefits on any hardware")
    print("  3. Same codebase scales from Mac development to GPU clusters")
    print("  4. Hardware-optimized actor allocation maximizes throughput")


if __name__ == "__main__":
    asyncio.run(main()) 