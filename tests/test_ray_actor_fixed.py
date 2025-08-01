#!/usr/bin/env python3
"""
Test Ray actor creation after fixing the import issues.
"""
import ray
import asyncio
import sys
import yaml
from pathlib import Path

async def test_simple_ray_actor():
    """Test simple Ray actor creation with config."""
    print("=== Simple Ray Actor Test ===")
    
    try:
        # Initialize Ray
        if not ray.is_initialized():
            print("Initializing Ray...")
            ray.init(num_cpus=2, num_gpus=0, local_mode=False, log_to_driver=False)
        
        print("✓ Ray initialized")
        
        @ray.remote
        class SimpleConfigActor:
            def __init__(self, config):
                # Clean logger setup
                try:
                    from loguru import logger
                    logger.remove()
                    logger.add(sys.stderr, level="INFO")
                except ImportError:
                    pass
                self.config = config
            
            def get_config_algorithm(self):
                return self.config.algorithm.name
        
        # Load config
        from retrain.config_models import TrainingConfig
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
        
        print("Creating simple actor...")
        actor = SimpleConfigActor.remote(config)
        result = await actor.get_config_algorithm.remote()
        print(f"✓ Simple actor works: algorithm = {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_reward_actor_creation():
    """Test creating actual ReReward actor."""
    print("\n=== ReReward Actor Test ===")
    
    try:
        # Load config
        from retrain.config_models import TrainingConfig
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
        
        print("Importing ReReward...")
        from retrain.reward.reward import ReReward
        print("✓ ReReward imported")
        
        print("Creating ReReward actor...")
        # Create with None databuffer for now
        reward_actor = ReReward.remote(config, None)
        print("✓ ReReward actor created successfully!")
        
        print("Initializing actor...")
        await reward_actor.initialize.remote()
        print("✓ ReReward actor initialized!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_hardware_detector_factory():
    """Test creating actors through HardwareDetector and ActorFactory."""
    print("\n=== Hardware Detector + ActorFactory Test ===")
    
    try:
        # Load config
        from retrain.config_models import TrainingConfig
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
        
        print("Creating HardwareDetector...")
        from retrain.hardware import HardwareDetector, ActorFactory
        hardware_detector = HardwareDetector()
        print("✓ HardwareDetector created")
        
        print("Creating ActorFactory...")
        actor_factory = ActorFactory(hardware_detector)
        print("✓ ActorFactory created")
        
        print("Creating DataBuffer actor...")
        databuffer = actor_factory.create_databuffer_actor(config)
        print("✓ DataBuffer actor created")
        
        print("Creating ReReward actor through factory...")
        reward_actor = actor_factory.create_reward_actor(config, databuffer)
        print("✓ ReReward actor created through factory!")
        
        print("Initializing reward actor...")
        await reward_actor.initialize.remote()
        print("✓ ReReward actor initialized through factory!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("Ray Actor Test (Fixed)")
    print("=" * 50)
    
    results = []
    
    # Test 1: Simple Ray actor
    results.append(await test_simple_ray_actor())
    
    # Test 2: ReReward actor (only if test 1 passed)
    if results[-1]:
        results.append(await test_reward_actor_creation())
    else:
        results.append(False)
    
    # Test 3: ActorFactory (only if test 2 passed)
    if results[-1]:
        results.append(await test_hardware_detector_factory())
    else:
        results.append(False)
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    test_names = ["Simple Ray actor", "ReReward actor", "ActorFactory"]
    for i, (name, result) in enumerate(zip(test_names, results)):
        print(f"{name}: {'SUCCESS' if result else 'FAILED'}")
    
    ray.shutdown()
    return all(results)

if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nOVERALL: {'SUCCESS' if result else 'FAILED'}")