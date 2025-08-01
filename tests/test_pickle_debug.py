#!/usr/bin/env python3
"""
Minimal test to debug Ray pickle error with ReReward actor.
"""
import ray
import asyncio
import yaml
from pathlib import Path

# Initialize Ray first
ray.init(num_cpus=2, num_gpus=0, local_mode=False)

async def test_minimal_actor_creation():
    """Test creating a minimal ReReward actor to identify pickle issue."""
    print("=== Testing Minimal Actor Creation ===")
    
    try:
        # Load minimal config
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        from retrain.config_models import TrainingConfig 
        config = TrainingConfig(**config_dict)
        
        print("✓ Config loaded successfully")
        
        # Create a dummy databuffer reference (just None for now)
        databuffer = None
        
        print("Testing direct actor creation...")
        
        # Try to create the actor directly
        from retrain.reward.reward import ReReward
        print("✓ ReReward imported successfully")
        
        # Test actor creation
        print("Creating ReReward actor...")
        reward_actor = ReReward.remote(config, databuffer)
        print("✓ ReReward actor created successfully!")
        
        # Test initialization
        print("Initializing actor...")
        await reward_actor.initialize.remote()
        print("✓ Actor initialized successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_with_actor_factory():
    """Test creating actor through ActorFactory."""
    print("\n=== Testing Actor Factory Creation ===")
    
    try:
        # Load config
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        from retrain.config_models import TrainingConfig 
        config = TrainingConfig(**config_dict)
        
        # Create ActorFactory
        from retrain.hardware import HardwareDetector, ActorFactory
        hardware_detector = HardwareDetector()
        actor_factory = ActorFactory(hardware_detector)
        
        print("✓ ActorFactory created")
        
        # Create databuffer first
        databuffer = actor_factory.create_databuffer_actor(config)
        print("✓ DataBuffer created")
        
        # Try to create reward actor through factory
        print("Creating ReReward actor through factory...")
        reward_actor = actor_factory.create_reward_actor(config, databuffer)
        print("✓ ReReward actor created through factory!")
        
        # Test initialization
        print("Initializing actor...")
        await reward_actor.initialize.remote()
        print("✓ Actor initialized successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("Ray Pickle Debug Test")
    print("=" * 50)
    
    # Test 1: Direct actor creation
    success1 = await test_minimal_actor_creation()
    
    # Test 2: Actor factory creation
    success2 = await test_with_actor_factory()
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"Direct actor creation: {'SUCCESS' if success1 else 'FAILED'}")
    print(f"ActorFactory creation: {'SUCCESS' if success2 else 'FAILED'}")
    
    ray.shutdown()

if __name__ == "__main__":
    asyncio.run(main())