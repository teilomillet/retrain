#!/usr/bin/env python3
"""
Test pickle without Ray at all - just test the core objects.
"""
import pickle
import sys

def test_basic_pickle():
    """Test basic pickle functionality."""
    print("=== Testing Basic Pickle ===")
    
    try:
        # Test basic objects
        test_dict = {"key": "value", "number": 42}
        pickled = pickle.dumps(test_dict)
        unpickled = pickle.loads(pickled)
        print("✓ Basic dict pickle works")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic pickle error: {e}")
        return False

def test_loguru_pickle():
    """Test loguru logger pickle."""
    print("\n=== Testing Loguru Pickle ===")
    
    try:
        from loguru import logger
        print("✓ Loguru imported")
        
        # Test pickle before cleaning
        try:
            pickle.dumps(logger)
            print("✓ Original logger pickles successfully")
        except Exception as e:
            print(f"✗ Original logger pickle failed: {e}")
            
            # Try cleaning and re-adding
            logger.remove()
            logger.add(sys.stderr, level="INFO")
            
            try:
                pickle.dumps(logger)
                print("✓ Clean logger pickles successfully")
            except Exception as e2:
                print(f"✗ Clean logger pickle failed: {e2}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Loguru test error: {e}")
        return False

def test_config_creation():
    """Test TrainingConfig creation and pickle."""
    print("\n=== Testing TrainingConfig ===")
    
    try:
        import yaml
        from pathlib import Path
        
        # Load raw config
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        print("✓ Raw config loaded")
        
        # Test pickle raw config
        pickle.dumps(config_dict)
        print("✓ Raw config pickles")
        
        # Try to create TrainingConfig
        print("Creating TrainingConfig...")
        from retrain.config_models import TrainingConfig
        config = TrainingConfig(**config_dict)
        print("✓ TrainingConfig created")
        
        # Test pickle TrainingConfig
        pickle.dumps(config)
        print("✓ TrainingConfig pickles")
        
        return True
        
    except Exception as e:
        print(f"✗ Config error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run tests."""
    print("No-Ray Pickle Test")
    print("=" * 30)
    
    tests = [
        test_basic_pickle,
        test_loguru_pickle,
        test_config_creation,
    ]
    
    for test in tests:
        test()
        
    print("\nDone!")

if __name__ == "__main__":
    main()