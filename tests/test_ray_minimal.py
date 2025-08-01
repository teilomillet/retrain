#!/usr/bin/env python3
"""
Minimal Ray test to see if Ray itself works.
"""
import ray
import pickle

def test_basic_ray():
    """Test basic Ray functionality."""
    print("=== Basic Ray Test ===")
    
    try:
        print("1. Initializing Ray...")
        if not ray.is_initialized():
            ray.init(num_cpus=1, num_gpus=0, local_mode=True, log_to_driver=False)
        print("✓ Ray initialized")
        
        @ray.remote
        class SimpleActor:
            def __init__(self, value):
                self.value = value
            
            def get_value(self):
                return self.value
        
        print("2. Creating simple actor...")
        actor = SimpleActor.remote("test")
        print("✓ Simple actor created")
        
        print("3. Calling actor method...")
        result = ray.get(actor.get_value.remote())
        print(f"✓ Actor method result: {result}")
        
        ray.shutdown()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        if ray.is_initialized():
            ray.shutdown()
        return False

def test_config_pickle_only():
    """Test config pickle without Ray."""
    print("\n=== Config Pickle Only ===")
    
    try:
        import yaml
        from pathlib import Path
        from retrain.config_models import TrainingConfig
        
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
        
        print("Testing config pickle...")
        pickled = pickle.dumps(config)
        unpickled = pickle.loads(pickled)
        print("✓ Config pickled successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Minimal Ray Test")
    print("=" * 30)
    
    # Test basic Ray first
    ray_result = test_basic_ray()
    
    # Test config pickle
    config_result = test_config_pickle_only()
    
    print("\n" + "=" * 30)
    print("RESULTS:")
    print(f"Basic Ray: {'SUCCESS' if ray_result else 'FAILED'}")  
    print(f"Config pickle: {'SUCCESS' if config_result else 'FAILED'}")