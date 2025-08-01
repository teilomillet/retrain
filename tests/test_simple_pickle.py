#!/usr/bin/env python3
"""
Simple pickle test to identify what's causing the serialization issue.
"""
import pickle
import yaml
from pathlib import Path

def test_config_pickle():
    """Test if TrainingConfig can be pickled."""
    print("=== Testing TrainingConfig Pickle ===")
    
    try:
        # Load config
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        from retrain.config_models import TrainingConfig 
        config = TrainingConfig(**config_dict)
        
        print("✓ Config loaded successfully")
        
        # Test pickle
        print("Testing pickle serialization...")
        pickled_config = pickle.dumps(config)
        print("✓ Config pickled successfully!")
        
        # Test unpickle
        unpickled_config = pickle.loads(pickled_config)
        print("✓ Config unpickled successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Config pickle error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_actor_imports():
    """Test if importing actors causes pickle issues."""
    print("\n=== Testing Actor Imports ===")
    
    try:
        print("Importing ReReward...")
        from retrain.reward.reward import ReReward
        print("✓ ReReward imported")
        
        print("Importing ReVerifier...")
        from retrain.verifier.verifier import ReVerifier
        print("✓ ReVerifier imported")
        
        print("Importing ReDataBuffer...")
        from retrain.manager.databuffer import ReDataBuffer
        print("✓ ReDataBuffer imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_actor_class_pickle():
    """Test if the actor classes themselves can be pickled."""
    print("\n=== Testing Actor Class Pickle ===")
    
    try:
        from retrain.reward.reward import ReReward
        from retrain.verifier.verifier import ReVerifier
        
        # Test pickling the classes themselves
        print("Pickling ReReward class...")
        pickle.dumps(ReReward)
        print("✓ ReReward class pickled successfully!")
        
        print("Pickling ReVerifier class...")
        pickle.dumps(ReVerifier)
        print("✓ ReVerifier class pickled successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Actor class pickle error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logger_pickle():
    """Test if logger objects cause pickle issues."""
    print("\n=== Testing Logger Pickle ===")
    
    try:
        print("Testing loguru logger...")
        from loguru import logger
        
        # Try to pickle the logger
        try:
            pickle.dumps(logger)
            print("✓ Logger pickled successfully!")
        except Exception as e:
            print(f"✗ Logger pickle failed: {e}")
            
            # Try to pickle after removing handlers
            print("Trying with clean logger...")
            logger.remove()
            pickle.dumps(logger)
            print("✓ Clean logger pickled successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Logger test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Simple Pickle Debug Test")
    print("=" * 50)
    
    tests = [
        test_config_pickle,
        test_actor_imports,
        test_actor_class_pickle,
        test_logger_pickle
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    for i, result in enumerate(results):
        test_name = tests[i].__name__
        print(f"{test_name}: {'SUCCESS' if result else 'FAILED'}")

if __name__ == "__main__":
    main()