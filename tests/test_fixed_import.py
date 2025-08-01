#!/usr/bin/env python3
"""
Test if config_models import works after fixing package init.
"""
import sys

def test_config_import_and_pickle():
    """Test importing config_models and pickle after fixing package init."""
    print("=== Fixed Config Import Test ===")
    
    try:
        print("1. Testing config_models import...")
        sys.stdout.flush()
        from retrain.config_models import TrainingConfig
        print("✓ TrainingConfig imported successfully")
        
        print("2. Testing config creation...")
        import yaml
        from pathlib import Path
        
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = TrainingConfig(**config_dict)
        print("✓ TrainingConfig created successfully")
        
        print("3. Testing pickle...")
        import pickle
        pickled = pickle.dumps(config)
        unpickled = pickle.loads(pickled)
        print("✓ Config pickled and unpickled successfully")
        
        print("4. Testing Ray with config...")
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=2, num_gpus=0, local_mode=False, log_to_driver=False)
        
        # Test putting config in Ray object store
        config_ref = ray.put(config)
        retrieved_config = ray.get(config_ref)
        print("✓ Config stored and retrieved from Ray object store")
        
        ray.shutdown()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        if 'ray' in locals():
            ray.shutdown()
        return False

def test_actor_creation():
    """Test basic Ray actor creation."""
    print("\n=== Actor Creation Test ===")
    
    try:
        import ray
        import sys
        
        @ray.remote
        class TestActor:
            def __init__(self, config):
                # Clean logger setup
                try:
                    from loguru import logger
                    logger.remove()
                    logger.add(sys.stderr, level="INFO")
                except ImportError:
                    pass
                self.config = config
            
            def get_config_type(self):
                return type(self.config).__name__
        
        if not ray.is_initialized():
            ray.init(num_cpus=2, num_gpus=0, local_mode=False, log_to_driver=False)
        
        # Load config
        import yaml
        from pathlib import Path
        from retrain.config_models import TrainingConfig
        
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
        
        print("Creating test actor...")
        actor = TestActor.remote(config)
        result = ray.get(actor.get_config_type.remote())
        print(f"✓ Test actor created and executed: {result}")
        
        ray.shutdown()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        if 'ray' in locals():
            ray.shutdown()
        return False

def main():
    """Run tests."""
    print("Fixed Import Test")
    print("=" * 40)
    
    result1 = test_config_import_and_pickle()
    result2 = test_actor_creation() if result1 else False
    
    print("\n" + "=" * 40)
    print("RESULTS:")
    print(f"Config import & pickle: {'SUCCESS' if result1 else 'FAILED'}")
    print(f"Actor creation: {'SUCCESS' if result2 else 'FAILED'}")

if __name__ == "__main__":
    main()