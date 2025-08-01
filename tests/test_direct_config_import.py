#!/usr/bin/env python3
"""
Test importing config_models directly without triggering package __init__.
"""
import sys
import importlib.util

def test_direct_config_import():
    """Import config_models directly without package initialization."""
    print("=== Direct Config Import Test ===")
    
    try:
        print("1. Loading config_models module directly...")
        sys.stdout.flush()
        
        # Load the module directly without going through package __init__.py
        spec = importlib.util.spec_from_file_location(
            "config_models", 
            "/Users/teilo.millet/Code/retrain/retrain/config_models.py"
        )
        config_models = importlib.util.module_from_spec(spec)
        
        print("2. Executing config_models module...")
        sys.stdout.flush()
        spec.loader.exec_module(config_models)
        print("✓ config_models loaded directly")
        
        print("3. Testing TrainingConfig creation...")
        import yaml
        from pathlib import Path
        
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Use the directly loaded TrainingConfig
        config = config_models.TrainingConfig(**config_dict)
        print("✓ TrainingConfig created successfully")
        
        print("4. Testing pickle...")
        import pickle
        pickle.dumps(config)
        print("✓ Config pickled successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bypass_package_init():
    """Test importing without triggering retrain package __init__."""
    print("\n=== Bypass Package Init Test ===")
    
    try:
        print("1. Adding retrain to sys.modules to prevent __init__.py execution...")
        
        # Create a dummy module to prevent package __init__.py from running
        import types
        retrain_module = types.ModuleType('retrain')
        retrain_module.__path__ = ['/Users/teilo.millet/Code/retrain/retrain']
        sys.modules['retrain'] = retrain_module
        
        print("2. Now importing retrain.config_models...")
        sys.stdout.flush()
        from retrain.config_models import TrainingConfig
        print("✓ TrainingConfig imported without package init")
        
        print("3. Testing config creation...")
        import yaml
        from pathlib import Path
        
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = TrainingConfig(**config_dict)
        print("✓ TrainingConfig created")
        
        print("4. Testing pickle...")
        import pickle
        pickle.dumps(config)
        print("✓ Config pickled successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run tests."""
    print("Direct Config Import Test")
    print("=" * 40)
    
    # Test 1: Direct module loading
    result1 = test_direct_config_import()
    
    # Test 2: Bypass package init (only if test1 succeeded)
    result2 = False
    if result1:
        result2 = test_bypass_package_init()
    
    print("\n" + "=" * 40)
    print("RESULTS:")
    print(f"Direct import: {'SUCCESS' if result1 else 'FAILED'}")
    print(f"Bypass package init: {'SUCCESS' if result2 else 'FAILED'}")

if __name__ == "__main__":
    main()