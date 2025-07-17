#!/usr/bin/env python3
"""
Example: Using Slime Backend with retrain Bridge System

This script demonstrates how to use retrain's new Slime backend integration
with the bridge system that connects retrain's Environment and RewardCalculator
to Slime's distributed training framework.
"""

import asyncio
import os
from pathlib import Path

# retrain imports
from retrain.config_models import TrainingConfig
from retrain.run import run_async_training


async def main():
    """
    Main example function showing Slime backend usage with bridges.
    """
    print("🚀 Starting Slime Backend with Bridge System Example")
    print("=" * 60)
    
    # Create configuration using Slime backend
    config_dict = {
        'experiment_name': "slime_bridge_example",
        'seed': 42,
        'logging_level': "INFO",
        
        'model': {
            'name_or_path': "Qwen/Qwen3-0.6B",
            'loader': "huggingface",
            'trust_remote_code': True,
            'torch_dtype': "bfloat16"
        },
        
        'environment': {
            'type': "fastmcp_env",
            'env_specific_config': {
                'server_url': "http://127.0.0.1:8765/mcp",
                'initial_prompt_template': "Hello! I need help with this task:",
                'max_steps': 5
            }
        },
        
        'algorithm': {
            'name': "grpo",
            'backend': "slime",  # 🔥 Using Slime backend!
            'hyperparameters': {
                # Core RL parameters
                'learning_rate': 1e-5,
                'num_iterations': 3,  # Small for example
                'rollout_batch_size': 2,
                'global_batch_size': 4,
                
                # Slime-specific parameters with bridge integration
                'slime_actor_num_nodes': 1,
                'slime_actor_num_gpus_per_node': 2,
                'slime_rollout_num_gpus': 2,
                'slime_colocate': True,  # Enable for small setups
                
                # Bridge-specific parameters
                'slime_use_retrain_environment': True,  # Use retrain Environment via bridge
                'slime_use_retrain_rewards': True,      # Use retrain RewardCalculator via bridge
                
                # Generation parameters
                'slime_rollout_temperature': 0.8,
                'slime_rollout_top_p': 0.9,
                'slime_rollout_max_response_len': 256,
                
                # System parameters  
                'slime_bf16': True,
                'slime_save_interval': 2
            }
        },
        
        'prompt_source': {
            'type': "environment",
            'source_config': {}
        },
        
        'reward_setup': {
            'step_reward_configs': {
                'task_completion': {
                    'weight': 1.0,
                    'params': {
                        'success_bonus': 10.0,
                        'failure_penalty': -2.0
                    }
                }
            },
            'rollout_reward_configs': {}
        },
        
        # Training parameters
        'num_episodes': 10,
        'batch_size': 2,
        'output_dir': "slime_bridge_output",
        'logging_dir': "slime_bridge_output/logs"
    }
    
    print("📋 Configuration Summary:")
    print(f"   - Algorithm: {config_dict['algorithm']['name']} with {config_dict['algorithm']['backend']} backend")
    print(f"   - Model: {config_dict['model']['name_or_path']}")
    print(f"   - Environment: {config_dict['environment']['type']}")
    print(f"   - Bridge Integration: Environment + RewardCalculator")
    print(f"   - Distributed: {config_dict['algorithm']['hyperparameters']['slime_actor_num_nodes']} nodes, {config_dict['algorithm']['hyperparameters']['slime_actor_num_gpus_per_node']} GPUs/node")
    print()
    
    try:
        # Create and validate configuration
        print("⚙️  Creating training configuration...")
        config = TrainingConfig(**config_dict)
        print(f"✅ Configuration validated successfully!")
        print(f"   - Backend: {config.algorithm.backend}")
        print(f"   - Learning rate: {config.algorithm.hyperparameters.get('learning_rate')}")
        print()
        
        # Demonstrate bridge functionality  
        print("🌉 Bridge System Features:")
        print("   - DataFormatBridge: Converts RawRolloutData ↔ Slime Sample objects")
        print("   - EnvironmentBridge: Adapts retrain Environment to Slime rollout generation")
        print("   - RewardBridge: Integrates retrain RewardCalculator with Slime rewards")
        print("   - RolloutBridge: Coordinates all bridges for seamless integration")
        print()
        
        print("🔥 Key Integration Benefits:")
        print("   1. **Simple User Experience**: Just change backend: 'slime'")
        print("   2. **Distributed Scaling**: Automatic Ray cluster + GPU allocation") 
        print("   3. **Component Reuse**: Keep your retrain Environment & RewardCalculator")
        print("   4. **Smart Defaults**: Auto-detects system resources and configures appropriately")
        print("   5. **Flexible Override**: Full access to Slime's 994 parameters via slime_ prefix")
        print()
        
        # Show configuration differences
        print("📊 TRL vs Slime Backend Comparison:")
        print("   TRL Backend:")
        print("     - Single-process training")
        print("     - HuggingFace model format")
        print("     - In-process generation")
        print("     - Limited scalability")
        print()
        print("   Slime Backend (with bridges):")
        print("     - Ray-based distributed training")
        print("     - Megatron model format (auto-converted)")
        print("     - SGLang server-based generation")
        print("     - Highly scalable multi-node support")
        print("     - retrain components via bridge system")
        print()
        
        print("🧪 Example Workflow:")
        print("   1. User configures algorithm.backend = 'slime'")
        print("   2. SlimeTrainerAdapter detects Slime backend")
        print("   3. Bridge system automatically created:")
        print("      - Environment → Slime rollout generation")
        print("      - RewardCalculator → Slime reward system")
        print("   4. Ray cluster auto-initialized with smart GPU allocation")
        print("   5. Training proceeds with retrain logic on Slime infrastructure")
        print()
        
        # Note about actual training
        print("⚠️  Note: This example validates configuration and shows integration.")
        print("   To run actual training, ensure:")
        print("   - Ray is installed: pip install 'ray[default]'")
        print("   - Slime is installed: pip install -e ./slime")
        print("   - Sufficient GPU resources for distributed training")
        print()
        
        # Show configuration file equivalent
        config_file_path = Path("examples/slime_bridge_config.yaml")
        print(f"💾 Equivalent configuration saved to: {config_file_path}")
        
        # Create equivalent YAML config
        import yaml
        with open(config_file_path, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
        
        print("   You can use this with: python retrain/run.py --config examples/slime_bridge_config.yaml")
        print()
        
        print("✨ Integration Complete!")
        print("   The Slime backend with bridge system is ready for distributed RL training!")
        print("   Users get Slime's performance with retrain's simplicity! 🚀")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 