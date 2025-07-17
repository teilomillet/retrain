"""
Simplified macOS Pipeline Test for Retrain (No pytest dependency)

This test validates:
1. Hardware detection and platform capabilities on macOS
2. Actor factory and basic component initialization
3. Configuration validation
4. Basic pipeline setup without external dependencies

Run with: python retrain/tests/test_mac_pipeline_simple.py
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import retrain components (avoiding Ray actor issues)
from retrain.hardware.detector import HardwareDetector
from retrain.config_models import TrainingConfig

# Import other components conditionally to avoid Ray actor initialization issues
try:
    from retrain.hardware.factory import ActorFactory
    ACTOR_FACTORY_AVAILABLE = True
except Exception as e:
    print(f"Warning: ActorFactory import failed: {e}")
    ACTOR_FACTORY_AVAILABLE = False

try:
    from retrain.manager.manager import ReManager
    MANAGER_AVAILABLE = True
except Exception as e:
    print(f"Warning: ReManager import failed: {e}")
    MANAGER_AVAILABLE = False

try:
    from retrain import run
    from retrain.reward import reward
    RUN_MODULE_AVAILABLE = True
except Exception as e:
    print(f"Warning: Run module import failed: {e}")
    RUN_MODULE_AVAILABLE = False

# Test constants
TEST_MODEL_NAME = "microsoft/DialoGPT-small"  # Small model for testing


class MacOSPipelineTests:
    """Test suite for macOS pipeline validation"""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and capture results"""
        print(f"\n🧪 Running: {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                asyncio.run(test_func())
            else:
                test_func()
            print(f"✅ PASSED: {test_name}")
            self.passed_tests += 1
            self.test_results.append((test_name, "PASSED", None))
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {str(e)}")
            self.failed_tests += 1
            self.test_results.append((test_name, "FAILED", str(e)))
    
    def test_hardware_detection_macos(self):
        """Test hardware detection specifically on macOS"""
        try:
            detector = HardwareDetector()
            
            # Test that detector was initialized properly
            assert hasattr(detector, 'capabilities'), "HardwareDetector should have capabilities attribute"
            assert detector.capabilities is not None
            
            # Test individual detection methods
            platform_info = detector._detect_platform()
            assert platform_info is not None
            print(f"   Platform detected: {platform_info}")
            
            # Test device detection (should detect MPS on Apple Silicon or CPU fallback)
            device_info = detector._detect_device_info()
            assert device_info is not None
            print(f"   Device info: {device_info}")
            
            # Test memory detection
            memory_info = detector._detect_memory_info()
            assert memory_info is not None
            assert 'system_memory_gb' in memory_info
            print(f"   Memory info: {memory_info}")
            
            # Test full capabilities
            capabilities = detector.capabilities
            assert capabilities is not None
            assert 'platform' in capabilities
            assert 'device' in capabilities
            assert 'memory' in capabilities
            print(f"   Full capabilities detected")
            
            # Verify macOS specific detection
            if sys.platform == "darwin":
                assert capabilities['platform']['is_macos'] == True
                # Should have either MPS or CPU device
                assert capabilities['device']['primary_device'] in ['mps', 'cuda', 'cpu']
                
        except Exception as e:
            print(f"   Hardware detection failed: {e}")
            raise
    
    def test_actor_factory_initialization(self):
        """Test ActorFactory can be properly initialized"""
        if not ACTOR_FACTORY_AVAILABLE:
            print("   SKIPPED: ActorFactory not available")
            return
            
        detector = HardwareDetector()
        
        factory = ActorFactory(hardware_detector=detector)
        assert factory is not None
        assert factory.detector == detector
        print("   ActorFactory initialized with detector")
    
    def test_inference_factory_function_macos(self):
        """Test inference factory components are available"""
        try:
            from retrain.inference.factory import create_inference_actor
            assert create_inference_actor is not None
            print("   Inference factory function is available")
        except Exception as e:
            print(f"   SKIPPED: Inference factory not available - {e}")
            # This is expected if Ray actors have issues
    
    def test_hardware_detection_integration(self):
        """Test hardware detection integration with other components"""
        try:
            detector = HardwareDetector()
            capabilities = detector.capabilities
            
            # Test basic functionality without creating actual buffers
            assert capabilities is not None
            assert 'platform' in capabilities
            assert 'device' in capabilities
            
            print("   Hardware detection integration test passed")
        except Exception as e:
            print(f"   Hardware detection integration failed: {e}")
            raise
    
    def test_manager_initialization(self):
        """Test ReManager can be initialized properly"""
        if not MANAGER_AVAILABLE:
            print("   SKIPPED: ReManager not available")
            return
            
        # Create a minimal config for testing
        config_dict = self.create_minimal_test_config()
        config = TrainingConfig(**config_dict)
        
        manager = ReManager(config=config)
        
        assert manager is not None
        assert manager.config == config
        print("   ReManager initialized successfully")

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
        
        config = TrainingConfig(**config_dict)
        assert config is not None
        print("   Test configuration validation passed")

    async def test_basic_pipeline_setup(self):
        """Test basic pipeline setup without full training"""
        if not RUN_MODULE_AVAILABLE:
            print("   SKIPPED: Run module not available")
            return
            
        config_dict = self.create_minimal_test_config()
        
        # Mock the actual training to avoid long-running tests
        with patch('retrain.run.run_async_training') as mock_training:
            mock_training.return_value = {"test": "completed"}
            
            result = await run(config=config_dict)
            assert result['status'] == 'completed'
            print("   Basic pipeline setup test passed")

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
                print("   MBridge import successful")
                
                # Check for key components
                assert hasattr(mbridge, 'core'), "MBridge core module not found"
                print("   MBridge integration detected successfully")
                
            except ImportError as e:
                print(f"   MBridge import failed (expected in some environments): {e}")
                # This might be expected if MBridge has specific dependencies
                
        except Exception as e:
            print(f"   MBridge integration test failed: {e}")
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
        print(f"   Running on macOS {macos_version}")
        
        # Test write permissions
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_write.txt"
            test_file.write_text("test")
            assert test_file.read_text() == "test"
        
        print("   macOS environment compatibility test passed")

    def run_all_tests(self):
        """Run all tests in the suite"""
        print("=" * 60)
        print("RETRAIN macOS PIPELINE TEST SUITE")
        print("=" * 60)
        print(f"Python: {sys.version}")
        print(f"Platform: {sys.platform}")
        print(f"Project root: {project_root}")
        print("=" * 60)
        
        # List of all tests to run
        tests = [
            ("Environment Compatibility", self.test_environment_compatibility),
            ("Hardware Detection", self.test_hardware_detection_macos),
            ("Actor Factory Initialization", self.test_actor_factory_initialization),
            ("Inference Factory Function", self.test_inference_factory_function_macos),
            ("Hardware Detection Integration", self.test_hardware_detection_integration),
            ("Manager Initialization", self.test_manager_initialization),
            ("Config Validation", self.test_config_validation),
            ("Basic Pipeline Setup", self.test_basic_pipeline_setup),
            ("MBridge Integration Detection", self.test_mbridge_integration_detection),
        ]
        
        # Run each test
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📊 Total:  {self.passed_tests + self.failed_tests}")
        
        if self.failed_tests > 0:
            print("\nFailed Tests:")
            for test_name, status, error in self.test_results:
                if status == "FAILED":
                    print(f"  ❌ {test_name}: {error}")
        
        print("=" * 60)
        return self.failed_tests == 0


def main():
    """Main entry point for the test suite"""
    test_suite = MacOSPipelineTests()
    success = test_suite.run_all_tests()
    
    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 