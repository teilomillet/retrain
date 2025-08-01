#!/usr/bin/env python3
"""
Test ReVerifier actor creation specifically.
"""
import ray
import asyncio
import yaml
from pathlib import Path

async def test_verifier_actor():
    """Test creating ReVerifier actor directly."""
    print("=== ReVerifier Actor Test ===")
    
    try:
        # Initialize Ray with fixed settings
        if not ray.is_initialized():
            ray.init(
                num_cpus=2, 
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
        
        # Import and create ReVerifier actor
        print("Importing ReVerifier...")
        from retrain.verifier.verifier import ReVerifier
        print("✓ ReVerifier imported")
        
        print("Creating ReVerifier actor...")
        verifier_actor = ReVerifier.remote(config, None)  # type: ignore
        print("✓ ReVerifier actor created!")
        
        print("Initializing ReVerifier actor...")
        await verifier_actor.initialize.remote()
        print("✓ ReVerifier actor initialized!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_verifier_via_factory():
    """Test creating ReVerifier via ActorFactory."""
    print("\n=== ReVerifier via ActorFactory Test ===")
    
    try:
        # Load config
        from retrain.config_models import TrainingConfig
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
        
        print("Creating ActorFactory...")
        from retrain.hardware import HardwareDetector, ActorFactory
        hardware_detector = HardwareDetector()
        actor_factory = ActorFactory(hardware_detector)
        print("✓ ActorFactory created")
        
        print("Creating ReVerifier via factory...")
        verifier_actor = actor_factory.create_verifier_actor(config, None) # type: ignore
        print("✓ ReVerifier created via factory!")
        
        print("Initializing verifier actor...")
        await verifier_actor.initialize.remote() # type: ignore
        print("✓ ReVerifier initialized via factory!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run verifier tests."""
    print("ReVerifier Actor Test")
    print("=" * 50)
    
    result1 = await test_verifier_actor()
    result2 = await test_verifier_via_factory() if result1 else False
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"Direct ReVerifier: {'SUCCESS' if result1 else 'FAILED'}")
    print(f"ActorFactory ReVerifier: {'SUCCESS' if result2 else 'FAILED'}")
    
    if result1 and result2:
        print("\n🎉 ReVerifier actor creation works!")
    else:
        print("\n❌ ReVerifier tests failed")
    
    ray.shutdown()
    return result1 and result2

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)