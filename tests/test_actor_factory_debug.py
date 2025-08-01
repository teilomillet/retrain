#!/usr/bin/env python3
"""
Debug ActorFactory step by step.
"""
import ray
import asyncio
import yaml
from pathlib import Path

async def test_actor_factory_debug():
    """Debug ActorFactory creation step by step."""
    print("=== ActorFactory Debug Test ===")
    
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
        
        # Create hardware detector and factory
        print("Creating hardware detector...")
        from retrain.hardware import HardwareDetector, ActorFactory
        hardware_detector = HardwareDetector()
        print("✓ Hardware detector created")
        
        print("Creating actor factory...")
        actor_factory = ActorFactory(hardware_detector)
        print("✓ Actor factory created")
        
        # Create databuffer using factory
        print("Creating databuffer via factory...")
        databuffer = actor_factory.create_databuffer_actor(config)
        await databuffer.initialize.remote() # type: ignore
        print("✓ DataBuffer created and initialized via factory")
        
        # Test inference group creation via factory
        print("Creating InferenceGroup via factory...")
        inference_group = actor_factory.create_inference_group(config, databuffer, 1)
        print("✓ InferenceGroup created via factory")
        
        print("Initializing InferenceGroup...")
        await inference_group.initialize.remote() # type: ignore
        print("✓ InferenceGroup initialized via factory")
        
        # Test trainer group creation via factory  
        print("Creating TrainerGroup via factory...")
        trainer_group = actor_factory.create_trainer_group(config, databuffer, 1)
        print("✓ TrainerGroup created via factory")
        
        print("Initializing TrainerGroup...")
        await trainer_group.initialize.remote() # type: ignore
        print("✓ TrainerGroup initialized via factory")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run actor factory debug test."""
    print("ActorFactory Debug Test")
    print("=" * 50)
    
    result = await test_actor_factory_debug()
    
    print("\n" + "=" * 50)
    print(f"OVERALL RESULT: {'SUCCESS' if result else 'FAILED'}")
    
    ray.shutdown()
    return result

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)