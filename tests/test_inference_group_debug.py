#!/usr/bin/env python3
"""
Debug InferenceGroup creation step by step.
"""
import ray
import asyncio
import sys
import yaml
from pathlib import Path

async def test_inference_group_debug():
    """Debug InferenceGroup creation step by step."""
    print("=== InferenceGroup Debug Test ===")
    
    try:
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                num_cpus=4, 
                num_gpus=0, 
                local_mode=False,
                log_to_driver=False,
                runtime_env={"working_dir": None}
            )
        print("✓ Ray initialized")
        
        # Load config
        from retrain.config_models import TrainingConfig
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
        print("✓ Config loaded")
        
        # Create databuffer
        from retrain.manager.databuffer import ReDataBuffer
        databuffer = ReDataBuffer.remote(config)
        await databuffer.initialize.remote()
        print("✓ DataBuffer created and initialized")
        
        # Create InferenceGroup manually step by step
        print("\nCreating InferenceGroup manually...")
        from retrain.manager.inference_group import InferenceGroup
        
        # Create the actor
        print("Creating InferenceGroup actor...")
        inference_group = InferenceGroup.remote(config, databuffer, 1)
        print("✓ InferenceGroup actor created")
        
        # Test if we can call basic methods
        print("Testing basic method call...")
        status = await inference_group.get_group_status.remote()
        print(f"✓ Basic method call works: {status}")
        
        # Now try to initialize
        print("Calling initialize...")
        await inference_group.initialize.remote()
        print("✓ InferenceGroup initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run inference group debug test."""
    print("InferenceGroup Debug Test")
    print("=" * 50)
    
    result = await test_inference_group_debug()
    
    print("\n" + "=" * 50)
    print(f"OVERALL RESULT: {'SUCCESS' if result else 'FAILED'}")
    
    ray.shutdown()
    return result

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)