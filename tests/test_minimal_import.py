#!/usr/bin/env python3
"""
Minimal import test to find exactly where the hang occurs.
"""
import sys

def test_step_by_step_imports():
    """Test imports one by one to find where it hangs."""
    print("=== Step by Step Import Test ===")
    
    try:
        print("1. Testing basic imports...")
        import yaml
        import pickle
        from pathlib import Path
        print("✓ Basic imports successful")
        
        print("2. Testing pydantic import...")
        from pydantic import BaseModel
        print("✓ Pydantic import successful")
        
        print("3. Testing config_models import...")
        sys.stdout.flush()  # Force flush before potentially hanging import
        from retrain.config_models import TrainingConfig
        print("✓ config_models import successful")
        
        print("4. Testing config creation...")
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
        print("✓ Config creation successful")
        
        print("5. Testing config pickle...")
        pickle.dumps(config)
        print("✓ Config pickle successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Error at step: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_module_imports():
    """Test importing individual modules."""
    print("\n=== Individual Module Import Test ===")
    
    modules_to_test = [
        "retrain",
        "retrain.config_models", 
        "retrain.hardware",
        "retrain.hardware.detector",
        "retrain.reward",
        "retrain.reward.reward",
        "retrain.verifier",
        "retrain.verifier.verifier",
    ]
    
    results = {}
    
    for module_name in modules_to_test:
        try:
            print(f"Importing {module_name}...")
            sys.stdout.flush()
            __import__(module_name)
            print(f"✓ {module_name} imported successfully")
            results[module_name] = True
        except Exception as e:
            print(f"✗ {module_name} failed: {e}")
            results[module_name] = False
            # Don't break, continue testing other modules
    
    return results

def main():
    """Run tests."""
    print("Minimal Import Debug Test")
    print("=" * 40)
    
    # Test 1: Step by step
    print("Running step-by-step test...")
    sys.stdout.flush()
    result1 = test_step_by_step_imports()
    
    # Test 2: Individual modules (only if step 1 didn't hang)
    if result1:
        print("\nRunning individual module test...")
        sys.stdout.flush()
        results2 = test_individual_module_imports()
        
        print("\n" + "=" * 40)
        print("INDIVIDUAL MODULE RESULTS:")
        for module, success in results2.items():
            print(f"{module}: {'SUCCESS' if success else 'FAILED'}")
    
    print(f"\nStep-by-step test: {'SUCCESS' if result1 else 'FAILED'}")

if __name__ == "__main__":
    main()