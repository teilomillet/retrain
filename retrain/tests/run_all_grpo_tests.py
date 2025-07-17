"""
Complete GRPO Test Suite Runner

Runs all GRPO and Dr. GRPO tests in sequence to provide comprehensive validation:
1. Base GRPO algorithm tests
2. Dr. GRPO bias fix tests
3. Comprehensive integration tests

Provides a single command to validate the entire GRPO implementation.
"""

import sys
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_test_file(test_file: str, description: str) -> bool:
    """Run a test file and return success status."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    test_path = project_root / "tests" / test_file
    
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        return False
    
    start_time = time.time()
    
    try:
        # Run the test
        result = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Check result
        if result.returncode == 0:
            print(f"\n✅ {description} PASSED (took {elapsed_time:.1f}s)")
            return True
        else:
            print(f"\n❌ {description} FAILED (exit code {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ {description} TIMED OUT (after 2 minutes)")
        return False
    except Exception as e:
        print(f"\n💥 {description} ERROR: {e}")
        return False


def main():
    """Run all GRPO tests in sequence."""
    print("🚀 COMPLETE GRPO TEST SUITE")
    print("=" * 60)
    print("Testing the complete GRPO implementation including:")
    print("• Base GRPO algorithm functionality")
    print("• Dr. GRPO bias fixes (GRPO Done Right)")
    print("• Hardware detection and integration")
    print("• Mathematical correctness validation")
    print("• End-to-end workflow testing")
    
    # Define test sequence
    tests = [
        ("test_grpo_simple.py", "Base GRPO Algorithm Tests"),
        ("test_drgrpo_correct.py", "Dr. GRPO Bias Fix Tests"),
        ("test_grpo_comprehensive.py", "Comprehensive Integration Tests"),
        ("test_grpo_ray_actors.py", "Ray Actor Distributed Tests"),
        ("test_grpo_performance_comparison.py", "Performance Comparison Tests"),
    ]
    
    # Track results
    results = []
    start_total = time.time()
    
    # Run each test
    for test_file, description in tests:
        success = run_test_file(test_file, description)
        results.append((description, success))
    
    total_time = time.time() - start_total
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎯 FINAL TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"📊 Results: {passed}/{total} test suites passed")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    
    # Detailed results
    print("\n📋 Detailed Results:")
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {description}")
    
    # Overall assessment
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ GRPO IMPLEMENTATION STATUS:")
        print("   • Base GRPO algorithm: WORKING CORRECTLY")
        print("   • Dr. GRPO bias fixes: IMPLEMENTED CORRECTLY")
        print("   • Hardware detection: FUNCTIONAL")
        print("   • Mathematical properties: VALIDATED")
        print("   • Integration: COMPLETE")
        
        print("\n🚀 READY FOR PRODUCTION:")
        print("   • Use GRPO.remote(config) for standard GRPO")
        print("   • Use DRGRPO.remote(config) for Dr. GRPO (recommended)")
        print("   • Both algorithms work with existing infrastructure")
        print("   • Hardware detection handles macOS, CPU, and CUDA automatically")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   • Dr. GRPO is recommended for production training")
        print("   • It removes length and std normalization biases")
        print("   • Provides better token efficiency and unbiased optimization")
        print("   • Backwards compatible with existing GRPO workflows")
        
        return 0
    else:
        print(f"\n❌ {total - passed} TEST SUITE(S) FAILED")
        print("\nPlease review the error messages above and fix the issues.")
        print("Each test suite validates a critical component of the GRPO implementation.")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 