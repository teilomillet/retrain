#!/usr/bin/env python3
"""
Test creating actors with completely clean imports.
"""
import ray
import asyncio
import sys
import yaml
from pathlib import Path

@ray.remote
class CleanTestGroup:
    """Group actor with clean logger setup."""
    
    def __init__(self, config, databuffer):
        # Clean all loguru handlers before doing anything
        try:
            from loguru import logger
            logger.remove()  # Remove all handlers
            logger.add(sys.stderr, level="INFO")  # Add clean handler
        except ImportError:
            pass
            
        self.config = config
        self.databuffer = databuffer
        self.workers = []
    
    async def create_reward_worker_clean(self):
        """Test creating reward worker with clean import."""
        try:
            # Clean import - don't inherit any module state
            print("CleanTestGroup: Cleaning logger before import...")
            try:
                from loguru import logger
                logger.remove()  # Clean slate
                logger.add(sys.stderr, level="INFO")
            except ImportError:
                pass
            
            print("CleanTestGroup: Importing ReReward with clean state...")
            from retrain.reward.reward import ReReward
            print("CleanTestGroup: ReReward imported")
            
            print("CleanTestGroup: Creating ReReward actor...")
            worker = ReReward.remote(self.config, self.databuffer)
            print("CleanTestGroup: ReReward actor created!")
            
            print("CleanTestGroup: Initializing worker...")
            await worker.initialize.remote()
            print("CleanTestGroup: Worker initialized!")
            
            self.workers.append(worker)
            return True
            
        except Exception as e:
            print(f"CleanTestGroup: Error: {e}")
            import traceback
            traceback.print_exc()
            return False

async def test_clean_group():
    """Test with clean group actor."""
    print("=== Clean Group Test ===")
    
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
        
        # Create clean test group
        print("Creating CleanTestGroup...")
        group = CleanTestGroup.remote(config, None)
        print("✓ CleanTestGroup created")
        
        # Test clean creation
        print("\nTesting clean reward worker creation...")
        result = await group.create_reward_worker_clean.remote()
        print(f"Result: {'SUCCESS' if result else 'FAILED'}")
        
        return result
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run clean group test."""
    print("Clean Group Actor Test")
    print("=" * 50)
    
    result = await test_clean_group()
    
    print("\n" + "=" * 50)
    print(f"OVERALL RESULT: {'SUCCESS' if result else 'FAILED'}")
    
    ray.shutdown()
    return result

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)