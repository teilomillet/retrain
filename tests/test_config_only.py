#!/usr/bin/env python3
"""
Test only config import and pickle without Ray.
"""
import sys

def test_config_only():
    """Test importing and pickling config only."""
    print("=== Config Only Test ===")
    
    try:
        print("1. Importing TrainingConfig...")
        sys.stdout.flush()
        from retrain.config_models import TrainingConfig
        print("✓ TrainingConfig imported")
        
        print("2. Loading config from YAML...")
        import yaml
        from pathlib import Path
        
        config_path = Path("examples/advanced_grpo_config.yaml")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = TrainingConfig(**config_dict)
        print("✓ TrainingConfig created")
        
        print("3. Testing pickle...")
        import pickle
        pickled = pickle.dumps(config)
        unpickled = pickle.loads(pickled)
        print("✓ Config pickled successfully")
        
        print("4. Validating unpickled config...")
        assert unpickled.algorithm.name == config.algorithm.name
        assert len(unpickled.reward_setup.step_reward_configs) == len(config.reward_setup.step_reward_configs)
        print("✓ Unpickled config is valid")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Config Only Test")
    print("=" * 30)
    
    result = test_config_only()
    print(f"\nRESULT: {'SUCCESS' if result else 'FAILED'}")