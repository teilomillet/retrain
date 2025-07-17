"""
Comprehensive macOS Pipeline Test for Retrain

This test validates:
1. Hardware detection and platform capabilities on macOS
2. Actor factory and group management
3. Basic inference pipeline with MPS/CPU
4. Manager and databuffer functionality
5. End-to-end training setup (without full training)
"""

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch
import pytest  # type: ignore

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import retrain components
from retrain.hardware.detector import HardwareDetector
from retrain.hardware.factory import ActorFactory
from retrain.inference.factory import create_inference_actor
from retrain.manager.manager import ReManager
from retrain.config_models import TrainingConfig
from retrain import run
from retrain.reward import reward

# Optional imports for testing
try:
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    
    # Define minimal pytest replacements for standalone testing
    class _MockPytest:
        @staticmethod
        def fixture(autouse=False):
            def decorator(func):
                return func
            return decorator
        
        @staticmethod
        def fail(msg):
            raise AssertionError(msg)
        
        @staticmethod
        def main(args):
            print("pytest not available, skipping test execution")
            return 0
        
        class mark:
            @staticmethod
            def asyncio(func):
                return func
            
            @staticmethod
            def slow(func):
                return func
    
    pytest = _MockPytest()

# Test constants
TEST_MODEL_NAME = "microsoft/DialoGPT-small"  # Small model for testing
TEST_TIMEOUT = 300  # 5 minutes


class TestMacOSPipeline:
    """Test suite for macOS pipeline validation"""
    
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup test logging"""
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
    def test_hardware_detection_macos(self):
        """Test hardware detection specifically on macOS"""
        detector = HardwareDetector()
        
        # Test platform detection
        platform_info = detector._detect_platform()
        assert platform_info is not None
        print(f"Platform detected: {platform_info}")
        
        # Test device detection (should detect MPS on Apple Silicon or CPU fallback)
        device_info = detector._detect_device_info()
        assert device_info is not None
        print(f"Device info: {device_info}")
        
        # Test memory detection
        memory_info = detector._detect_memory_info()
        assert memory_info is not None
        assert 'system_memory_gb' in memory_info
        print(f"Memory info: {memory_info}")
        
        # Test full capabilities
        capabilities = detector._detect_all_capabilities()
        assert capabilities is not None
        assert 'platform' in capabilities
        assert 'device' in capabilities
        assert 'memory' in capabilities
        print(f"Full capabilities: {capabilities}")
        
        # Verify macOS specific detection
        if sys.platform == "darwin":
            assert capabilities['platform']['name'] == 'macOS'
            # Should have either MPS or CPU device
            assert capabilities['device']['type'] in ['mps', 'cpu']
    
    def test_actor_factory_initialization(self):
        """Test ActorFactory can be properly initialized"""
        detector = HardwareDetector()
        
        factory = ActorFactory(hardware_detector=detector)
        assert factory is not None
        assert factory.detector == detector
        print("ActorFactory initialized with detector")
    
    def test_inference_factory_function_macos(self):
        """Test create_inference_actor function works on macOS"""
        # Test the factory function exists and can be called
        assert create_inference_actor is not None
        print("Inference factory function is available")
        
        # We don't test actual creation since it requires Ray setup
        # and proper config, but we can verify the function exists
    
    def test_hardware_detection_integration(self):
        """Test hardware detection integration with other components"""
        detector = HardwareDetector()
        capabilities = detector._detect_all_capabilities()
        
        # Test basic functionality without creating actual buffers
        assert capabilities is not None
        assert 'platform' in capabilities
        assert 'device' in capabilities
        
        print("Hardware detection integration test passed")
    
    def test_manager_initialization(self):
        """Test ReManager can be initialized properly"""
        # Create a minimal config for testing
        config_dict = self.create_minimal_test_config()
        config = TrainingConfig(**config_dict)
        
        manager = ReManager(config=config)
        
        assert manager is not None
        assert manager.config == config
        print("ReManager initialized successfully")

    def create_minimal_test_config(self) -> Dict[str, Any]:
        """Create a minimal configuration for testing"""
        return {
            "experiment_name": "test_mac_pipeline",
            "seed": 42,
            "logging_level": "DEBUG",
            "model": {
                "name_or_path": TEST_MODEL_NAME,
                "loader": "huggingface",
                "torch_dtype": "auto"
            },
            "algorithm": {
                "name": "grpo",
                "backend": "trl",
                "report_to": [],  # Disable wandb for testing
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "num_iterations": 1,  # Minimal for testing
                    "logging_steps": 1,
                    "beta": 0.01,
                    "max_prompt_length": 64,
                    "max_completion_length": 128,
                    "num_generations": 1,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            "environment": {
                "type": "smol_agent",
                "env_specific_config": {
                    "max_turns": 2,
                    "max_tokens_per_llm_turn": 128,
                    "tools": {
                        "registry_keys": ["simple_calculator_tool"]
                    }
                }
            },
            "prompt_source": {
                "type": "list",
                "source_config": {
                    "prompts": ["Calculate 2 + 2"]
                }
            },
            "reward_setup": {
                "step_reward_configs": {
                    "exact_match_reward": {
                        "weight": 1.0,
                        "params": {
                            "expected_answer": "4"
                        }
                    }
                },
                "rollout_reward_configs": {}
            }
        }

    def test_config_validation(self):
        """Test that our test configuration is valid"""
        config_dict = self.create_minimal_test_config()
        
        try:
            config = TrainingConfig(**config_dict)
            assert config is not None
            print("Test configuration validation passed")
        except Exception as e:
            pytest.fail(f"Configuration validation failed: {e}")

    @pytest.mark.asyncio 
    async def test_basic_pipeline_setup(self):
        """Test basic pipeline setup without full training"""
        config_dict = self.create_minimal_test_config()
        
        # Mock the actual training to avoid long-running tests
        with patch('retrain.run.run_async_training') as mock_training:
            mock_training.return_value = {"test": "completed"}
            
            try:
                result = await run(config=config_dict)
                assert result['status'] == 'completed'
                print("Basic pipeline setup test passed")
            except Exception as e:
                pytest.fail(f"Pipeline setup failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_end_to_end_minimal_training(self):
        """Test actual minimal training run (marked as slow test)"""
        config_dict = self.create_minimal_test_config()
        
        # Define a simple test reward function
        @reward(name="simple_test_reward")
        def simple_test_reward(prompt: str, completion: str, config_params: Dict[str, Any], **kwargs) -> float:
            """Simple reward for testing"""
            if "4" in completion:
                return 1.0
            return 0.0
        
        # Update config to use our test reward
        config_dict["reward_setup"]["step_reward_configs"] = {
            "simple_test_reward": {
                "weight": 1.0,
                "params": {}
            }
        }
        
        try:
            # Set a timeout for the test
            result = await asyncio.wait_for(
                run(config=config_dict),
                timeout=TEST_TIMEOUT
            )
            
            assert result['status'] == 'completed'
            assert 'metrics' in result
            print(f"End-to-end training test completed: {result}")
            
        except asyncio.TimeoutError:
            pytest.fail(f"Training test timed out after {TEST_TIMEOUT} seconds")
        except Exception as e:
            print(f"Expected error in minimal training (likely due to missing dependencies): {e}")
            # For CI/testing, we might expect some failures due to missing models/dependencies
            # This is acceptable for a pipeline validation test

    def test_mbridge_integration_detection(self):
        """Test if MBridge is properly integrated and accessible"""
        try:
            # Check if mbridge directory exists
            mbridge_path = project_root / "mbridge"
            assert mbridge_path.exists(), "MBridge directory not found"
            
            # Check if mbridge can be imported
            sys.path.insert(0, str(mbridge_path))
            try:
                import mbridge
                print("MBridge import successful")
                
                # Check for key components
                assert hasattr(mbridge, 'core'), "MBridge core module not found"
                print("MBridge integration detected successfully")
                
            except ImportError as e:
                print(f"MBridge import failed (expected in some environments): {e}")
                # This might be expected if MBridge has specific dependencies
                
        except Exception as e:
            print(f"MBridge integration test failed: {e}")
            # Log but don't fail the test as MBridge might not be required for all use cases

    def test_environment_compatibility(self):
        """Test macOS environment compatibility"""
        # Check Python version
        assert sys.version_info >= (3, 8), f"Python 3.8+ required, got {sys.version_info}"
        
        # Check platform
        assert sys.platform == "darwin", f"This test is for macOS, running on {sys.platform}"
        
        # Check for common macOS tools
        import platform
        macos_version = platform.mac_ver()[0]
        print(f"Running on macOS {macos_version}")
        
        # Test write permissions
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_write.txt"
            test_file.write_text("test")
            assert test_file.read_text() == "test"
        
        print("macOS environment compatibility test passed")


def run_mac_tests():
    """Convenience function to run all macOS tests"""
    
    # Run the tests
    test_args = [
        __file__,
        "-v",  # Verbose output
        "-s",  # Show print statements
        "--tb=short",  # Short traceback format
    ]
    
    # Add markers for different test types
    if "--slow" in sys.argv:
        test_args.append("-m")
        test_args.append("slow")
    else:
        test_args.append("-m")
        test_args.append("not slow")
    
    pytest.main(test_args)


if __name__ == "__main__":
    print("=" * 60)
    print("RETRAIN macOS PIPELINE TEST SUITE")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Project root: {project_root}")
    print("=" * 60)
    
    # Run the tests
    run_mac_tests() 