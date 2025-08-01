#!/usr/bin/env python3
"""
Test Ray with clean environment settings.
"""
import ray
import tempfile
import os

def test_ray_in_temp_dir():
    """Test Ray initialization in a temporary directory."""
    print("=== Ray in Temp Dir Test ===")
    
    try:
        # Change to temp directory
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            print(f"Changed to temp dir: {temp_dir}")
            
            print("Initializing Ray in temp dir...")
            if not ray.is_initialized():
                ray.init(num_cpus=1, num_gpus=0, local_mode=True, log_to_driver=False)
            print("✓ Ray initialized")
            
            @ray.remote
            class TestActor:
                def __init__(self):
                    self.value = "temp_test"
                
                def get_value(self):
                    return self.value
            
            print("Creating actor...")
            actor = TestActor.remote()
            result = ray.get(actor.get_value.remote())
            print(f"✓ Actor result: {result}")
            
            ray.shutdown()
            os.chdir(original_cwd)
            return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        if ray.is_initialized():
            ray.shutdown()
        os.chdir(original_cwd)
        return False

def test_ray_with_explicit_runtime_env():
    """Test Ray with explicit runtime_env to prevent auto-detection."""
    print("\n=== Ray with Explicit Runtime Env ===")
    
    try:
        print("Initializing Ray with explicit runtime_env...")
        if not ray.is_initialized():
            ray.init(
                num_cpus=1, 
                num_gpus=0, 
                local_mode=True, 
                log_to_driver=False,
                runtime_env={
                    "working_dir": None,  # Explicitly disable working_dir upload
                }
            )
        print("✓ Ray initialized with explicit runtime_env")
        
        @ray.remote
        class TestActor:
            def __init__(self):
                self.value = "explicit_env_test"
            
            def get_value(self):
                return self.value
        
        print("Creating actor...")
        actor = TestActor.remote()
        result = ray.get(actor.get_value.remote())
        print(f"✓ Actor result: {result}")
        
        ray.shutdown()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        if ray.is_initialized():
            ray.shutdown()
        return False

if __name__ == "__main__":
    print("Ray Clean Test")
    print("=" * 30)
    
    # Test 1: Ray in temp directory
    result1 = test_ray_in_temp_dir()
    
    # Test 2: Ray with explicit runtime_env
    result2 = test_ray_with_explicit_runtime_env()
    
    print("\n" + "=" * 30)
    print("RESULTS:")
    print(f"Ray in temp dir: {'SUCCESS' if result1 else 'FAILED'}")
    print(f"Ray with explicit env: {'SUCCESS' if result2 else 'FAILED'}")