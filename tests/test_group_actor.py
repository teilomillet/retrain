#!/usr/bin/env python3
"""
Test creating actors from within a Ray actor (like RewardGroup does).
"""
import ray
import asyncio
import sys
import yaml
from pathlib import Path

@ray.remote
class TestGroup:
    """Simple group actor to test creating workers from within."""
    
    def __init__(self, config, databuffer):
        self.config = config
        self.databuffer = databuffer
        self.workers = []
    
    async def create_reward_worker(self):
        """Test creating reward worker from within group actor."""
        try:
            print("TestGroup: Importing ReReward...")
            from retrain.reward.reward import ReReward
            print("TestGroup: ReReward imported")
            
            print("TestGroup: Creating ReReward actor...")
            worker = ReReward.remote(self.config, self.databuffer)
            print("TestGroup: ReReward actor created!")
            
            print("TestGroup: Initializing worker...")
            await worker.initialize.remote()
            print("TestGroup: Worker initialized!")
            
            self.workers.append(worker)
            return True
            
        except Exception as e:
            print(f"TestGroup: Error creating reward worker: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def create_verifier_worker(self):
        """Test creating verifier worker from within group actor."""
        try:
            print("TestGroup: Importing ReVerifier...")
            from retrain.verifier.verifier import ReVerifier
            print("TestGroup: ReVerifier imported")
            
            print("TestGroup: Creating ReVerifier actor...")
            worker = ReVerifier.remote(self.config, self.databuffer)
            print("TestGroup: ReVerifier actor created!")
            
            print("TestGroup: Initializing worker...")
            await worker.initialize.remote()
            print("TestGroup: Worker initialized!")
            
            self.workers.append(worker)
            return True
            
        except Exception as e:
            print(f"TestGroup: Error creating verifier worker: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def create_via_factory(self):
        """Test creating via ActorFactory from within group."""
        try:
            print("TestGroup: Creating ActorFactory...")
            from retrain.hardware import HardwareDetector, ActorFactory
            hardware_detector = HardwareDetector()  
            actor_factory = ActorFactory(hardware_detector)
            print("TestGroup: ActorFactory created")
            
            print("TestGroup: Creating reward worker via factory...")
            worker = actor_factory.create_reward_actor(self.config, self.databuffer)
            await worker.initialize.remote()
            print("TestGroup: Reward worker via factory created!")
            
            self.workers.append(worker)
            return True
            
        except Exception as e:
            print(f"TestGroup: Error creating via factory: {e}")
            import traceback
            traceback.print_exc()
            return False

async def test_group_actor():
    """Test creating workers from within a group actor."""
    print("=== Group Actor Test ===")
    
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
        
        # Create test group actor
        print("Creating TestGroup actor...")
        group = TestGroup.remote(config, None)
        print("✓ TestGroup created")
        
        # Test 1: Direct creation
        print("\nTest 1: Direct reward worker creation...")
        result1 = await group.create_reward_worker.remote()
        print(f"Result 1: {'SUCCESS' if result1 else 'FAILED'}")
        
        # Test 2: Direct verifier creation  
        print("\nTest 2: Direct verifier worker creation...")
        result2 = await group.create_verifier_worker.remote()
        print(f"Result 2: {'SUCCESS' if result2 else 'FAILED'}")
        
        # Test 3: Via factory
        print("\nTest 3: Via ActorFactory...")
        result3 = await group.create_via_factory.remote()
        print(f"Result 3: {'SUCCESS' if result3 else 'FAILED'}")
        
        return result1 and result2 and result3
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run group actor test."""
    print("Group Actor Creation Test")
    print("=" * 50)
    
    result = await test_group_actor()
    
    print("\n" + "=" * 50)
    print(f"OVERALL RESULT: {'SUCCESS' if result else 'FAILED'}")
    
    ray.shutdown()
    return result

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)