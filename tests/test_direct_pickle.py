#!/usr/bin/env python3
"""
Direct test of Ray actor creation without complex imports.
"""
import ray
import pickle
import sys

# Simple test actor without complex dependencies
@ray.remote
class SimpleTestActor:
    def __init__(self, value):
        # Clean logger setup like we did for ReReward
        try:
            from loguru import logger
            logger.remove()
            logger.add(sys.stderr, level="INFO")
        except ImportError:
            pass
        
        self.value = value
    
    def get_value(self):
        return self.value

def test_simple_actor():
    """Test simple actor creation."""
    print("=== Testing Simple Actor ===")
    
    try:
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_cpus=2, num_gpus=0, local_mode=False)
        
        print("Ray initialized")
        
        # Create simple actor
        actor = SimpleTestActor.remote("test_value")
        print("✓ Simple actor created")
        
        # Test method call
        result = ray.get(actor.get_value.remote())
        print(f"✓ Actor method call successful: {result}")
        
        ray.shutdown()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        ray.shutdown()
        return False

def test_config_only():
    """Test loading config without any ray imports."""
    print("=== Testing Config Only ===")
    
    try:
        import yaml
        from pathlib import Path
        
        # Load config dict only
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        print("✓ Config dict loaded")
        
        # Test pickle config dict
        pickled = pickle.dumps(config_dict)
        unpickled = pickle.loads(pickled)
        print("✓ Config dict pickled/unpickled successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Direct Pickle Test")
    print("=" * 30)
    
    # Test 1: Config only
    result1 = test_config_only()
    
    # Test 2: Simple actor
    result2 = test_simple_actor()
    
    print("\nRESULTS:")
    print(f"Config test: {'SUCCESS' if result1 else 'FAILED'}")
    print(f"Simple actor: {'SUCCESS' if result2 else 'FAILED'}")