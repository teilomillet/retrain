#!/usr/bin/env python3
"""
Test individual actors without groups to isolate the hanging issue.
"""
import ray
import asyncio
import sys
import yaml
from pathlib import Path

async def test_individual_actors():
    """Test individual actor creation and initialization."""
    print("=== Individual Actors Test ===")
    
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
        
        # Test DataBuffer actor directly
        print("\nTesting DataBuffer...")
        from retrain.manager.databuffer import ReDataBuffer
        databuffer = ReDataBuffer.remote(config)
        print("✓ DataBuffer created")
        
        # Initialize databuffer
        print("Initializing DataBuffer...")
        await databuffer.initialize.remote()
        print("✓ DataBuffer initialized")
        
        # Test MacOSInferenceActor directly
        print("\nTesting MacOSInferenceActor...")
        from retrain.inference.macos import MacOSInferenceActor
        inference_actor = MacOSInferenceActor.remote(config, databuffer)
        print("✓ MacOSInferenceActor created")
        
        # Initialize inference actor
        print("Initializing MacOSInferenceActor...")
        await inference_actor.initialize.remote()
        print("✓ MacOSInferenceActor initialized")
        
        # Test ReTrainer directly
        print("\nTesting ReTrainer...")
        from retrain.trainer.trainer import ReTrainer
        trainer_actor = ReTrainer.remote(config, databuffer)
        print("✓ ReTrainer created")
        
        # Initialize trainer actor
        print("Initializing ReTrainer...")
        await trainer_actor.initialize.remote()
        print("✓ ReTrainer initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run individual actors test."""
    print("Individual Actors Test")
    print("=" * 50)
    
    result = await test_individual_actors()
    
    print("\n" + "=" * 50)
    print(f"OVERALL RESULT: {'SUCCESS' if result else 'FAILED'}")
    
    ray.shutdown()
    return result

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)