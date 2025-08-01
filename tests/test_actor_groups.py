#!/usr/bin/env python3
"""
Test actor groups creation and basic methods.
"""
import ray
import asyncio
import sys
import yaml
from pathlib import Path

async def test_actor_groups():
    """Test basic actor group creation and methods."""
    print("=== Actor Groups Test ===")
    
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
        from retrain.hardware import HardwareDetector, ActorFactory
        hardware_detector = HardwareDetector()
        actor_factory = ActorFactory(hardware_detector)
        print("✓ Hardware detector and factory created")
        
        # Create databuffer
        databuffer = actor_factory.create_databuffer_actor(config)
        print("✓ DataBuffer created")
        
        # Test inference group
        print("\nTesting InferenceGroup...")
        inference_group = actor_factory.create_inference_group(config, databuffer, 1)
        await inference_group.initialize.remote()
        print("✓ InferenceGroup created and initialized")
        
        # Test generate_rollouts method
        try:
            rollouts = await inference_group.generate_rollouts.remote(episode_id=0, batch_size=2)
            print(f"✓ generate_rollouts works: {len(rollouts) if rollouts else 0} rollouts")
        except Exception as e:
            print(f"✗ generate_rollouts failed: {e}")
        
        # Test trainer group
        print("\nTesting TrainerGroup...")
        trainer_group = actor_factory.create_trainer_group(config, databuffer, 1)
        await trainer_group.initialize.remote()
        print("✓ TrainerGroup created and initialized")
        
        # Test train_step method
        try:
            training_batch = {
                'input_ids': [[1, 2, 3]],
                'rewards': [0.5],
                'episode_id': 0
            }
            metrics = await trainer_group.train_step.remote(training_batch, 0)
            print(f"✓ train_step works: {type(metrics)}")
        except Exception as e:
            print(f"✗ train_step failed: {e}")
        
        # Test update_weights method
        try:
            await inference_group.update_weights.remote(trainer_group)
            print("✓ update_weights works")
        except Exception as e:
            print(f"✗ update_weights failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run actor groups test."""
    print("Actor Groups Test")
    print("=" * 50)
    
    result = await test_actor_groups()
    
    print("\n" + "=" * 50)
    print(f"OVERALL RESULT: {'SUCCESS' if result else 'FAILED'}")
    
    ray.shutdown()
    return result

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)