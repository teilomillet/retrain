"""
Performance Comparison: Base GRPO vs Ray Actor GRPO

This test suite compares the performance characteristics of:
- BaseGRPO vs GRPO (Ray actor)
- BaseDRGRPO vs DRGRPO (Ray actor)

Measures:
- Initialization time overhead
- Operation execution time
- Memory usage patterns
- Concurrent execution benefits
- Ray communication overhead

Helps determine when to use base implementations vs Ray actors.
"""

import asyncio
import time
import psutil
from typing import Dict, Any
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Try importing Ray and implementations
try:
    import ray
    from retrain.trainer.grpo import GRPO, DRGRPO
    from retrain.trainer.grpo.grpo import BaseGRPO
    from retrain.trainer.grpo.drgrpo import BaseDRGRPO
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    GRPO = None  # type: ignore
    DRGRPO = None  # type: ignore
    BaseGRPO = None  # type: ignore
    BaseDRGRPO = None  # type: ignore


def create_test_config():
    """Create a mock configuration for testing."""
    config = MagicMock()
    
    # Model configuration
    config.model = MagicMock()
    config.model.name_or_path = "test_model"
    config.model.max_seq_length = 512
    config.model.load_in_4bit = False
    config.model.trust_remote_code = False
    
    # Algorithm configuration
    config.algorithm = MagicMock()
    config.algorithm.clip_range = 0.2
    config.algorithm.value_clip_range = 0.2
    config.algorithm.entropy_coef = 0.01
    config.algorithm.value_coef = 0.5
    config.algorithm.gamma = 0.99
    config.algorithm.gae_lambda = 0.95
    config.algorithm.hyperparameters = {'learning_rate': 1e-5}
    
    return config


def measure_memory_usage() -> Dict[str, float]:
    """Measure current memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'cpu_percent': process.cpu_percent()
    }


class PerformanceComparison:
    """Main performance comparison class."""
    
    def __init__(self):
        self.results = {
            'base_grpo': {},
            'ray_grpo': {},
            'base_drgrpo': {},
            'ray_drgrpo': {}
        }
        
    def run_all_comparisons(self) -> Dict[str, Any]:
        """Run all performance comparisons."""
        print("🔬 GRPO PERFORMANCE COMPARISON SUITE")
        print("=" * 60)
        
        if not RAY_AVAILABLE:
            print("❌ Ray not available, skipping performance comparisons")
            return self.results
            
        # Initialize Ray for tests
        if not ray.is_initialized():
            ray.init(
                num_cpus=4,
                num_gpus=0,
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False,
                object_store_memory=2_000_000_000,  # 2GB limit for macOS compatibility
                runtime_env={
                    "env_vars": {"RAY_DISABLE_IMPORT_WARNING": "1"},
                    "working_dir": ".",
                    "excludes": [".git"]
                },
                _skip_env_hook=True  # Skip uv runtime environment detection
            )
        
        try:
            # Test initialization performance
            self.test_initialization_performance()
            
            # Test operation performance
            self.test_operation_performance()
            
            # Test concurrent execution benefits
            self.test_concurrent_execution()
            
            # Test memory usage patterns
            self.test_memory_usage()
            
            # Generate summary
            self.print_performance_summary()
            
        finally:
            if ray.is_initialized():
                ray.shutdown()
                
        return self.results
        
    def test_initialization_performance(self):
        """Compare initialization time between base and Ray versions."""
        print("\n🚀 Testing Initialization Performance")
        print("-" * 40)
        
        config = create_test_config()
        
        # Test Base GRPO initialization
        print("Testing Base GRPO initialization...")
        start_time = time.time()
        base_grpo = BaseGRPO(config)  # type: ignore
        base_init_time = time.time() - start_time
        print(f"✓ Base GRPO init: {base_init_time:.3f}s")
        
        # Test Ray GRPO initialization
        print("Testing Ray GRPO initialization...")
        start_time = time.time()
        ray_grpo = GRPO.remote(config)  # type: ignore
        ray_init_time = time.time() - start_time
        print(f"✓ Ray GRPO init: {ray_init_time:.3f}s")
        
        # Test Base Dr. GRPO initialization
        print("Testing Base Dr. GRPO initialization...")
        start_time = time.time()
        base_drgrpo = BaseDRGRPO(config)  # type: ignore
        base_drgrpo_init_time = time.time() - start_time
        print(f"✓ Base Dr. GRPO init: {base_drgrpo_init_time:.3f}s")
        
        # Test Ray Dr. GRPO initialization
        print("Testing Ray Dr. GRPO initialization...")
        start_time = time.time()
        ray_drgrpo = DRGRPO.remote(config)  # type: ignore
        ray_drgrpo_init_time = time.time() - start_time
        print(f"✓ Ray Dr. GRPO init: {ray_drgrpo_init_time:.3f}s")
        
        # Store results
        self.results['base_grpo']['init_time'] = base_init_time
        self.results['ray_grpo']['init_time'] = ray_init_time
        self.results['base_drgrpo']['init_time'] = base_drgrpo_init_time
        self.results['ray_drgrpo']['init_time'] = ray_drgrpo_init_time
        
        # Helper to compute percentage safely
        def pct(delta: float, base: float) -> str:
            if base < 1e-3:  # <1 ms → percentage is meaningless
                return "n/a"
            return f"{(delta / base * 100):+.1f}%"

        # Calculate absolute deltas and safe overhead percentages
        grpo_delta = ray_init_time - base_init_time
        drgrpo_delta = ray_drgrpo_init_time - base_drgrpo_init_time

        grpo_overhead = pct(grpo_delta, base_init_time)
        drgrpo_overhead = pct(drgrpo_delta, base_drgrpo_init_time)

        print("\n📊 Initialization Overhead:")
        print(f"   Ray GRPO:  +{grpo_delta*1000:.2f} ms  ({grpo_overhead})")
        print(f"   Ray Dr. GRPO:  +{drgrpo_delta*1000:.2f} ms  ({drgrpo_overhead})")
        
        # Clean up
        ray.kill(ray_grpo)  # type: ignore
        ray.kill(ray_drgrpo)  # type: ignore
        
    def test_operation_performance(self):
        """Compare operation performance between base and Ray versions."""
        print("\n⚡ Testing Operation Performance")
        print("-" * 40)
        
        config = create_test_config()
        
        # Create instances
        base_grpo = BaseGRPO(config)          # type: ignore
        base_drgrpo = BaseDRGRPO(config)      # type: ignore

        # Spin up Ray actors **once** and warm-up
        ray_grpo = GRPO.remote(config)        # type: ignore
        ray_drgrpo = DRGRPO.remote(config)    # type: ignore

        # Save for reuse in other tests
        self.ray_actor_pool = [ray_grpo, ray_drgrpo]

        # NOTE: We intentionally skip `initialize.remote()` to avoid heavy
        # model loading (which pulls in optional deps like matplotlib).
        # For simple health-check timing this is sufficient.

        try:
            print("Testing health check operations...")

            # Warm-up once so imports / lazy inits are paid before timing
            _ = asyncio.run(base_grpo.health_check())
            _ = asyncio.run(base_drgrpo.health_check())
            ray.get([ray_grpo.health_check.remote(), ray_drgrpo.health_check.remote()]) # type: ignore

            # Base GRPO health checks (1000 calls)
            start = time.perf_counter()
            for _ in range(1000):
                _ = asyncio.run(base_grpo.health_check())
            base_health_time = (time.perf_counter() - start) / 1000
            print(f"✓ Base GRPO health check: {base_health_time*1000:.2f}ms")

            # Ray GRPO health checks (1000 calls on one actor)
            start = time.perf_counter()
            futs = [ray_grpo.health_check.remote() for _ in range(1000)] # type: ignore
            ray.get(futs) # type: ignore
            ray_health_time = (time.perf_counter() - start) / 1000
            print(f"✓ Ray GRPO  health check: {ray_health_time*1000:.2f}ms")

            # Base Dr. GRPO health checks (1000 calls)
            start = time.perf_counter()
            for _ in range(1000):
                _ = asyncio.run(base_drgrpo.health_check())
            base_drgrpo_health_time = (time.perf_counter() - start) / 1000
            print(f"✓ Base Dr. GRPO health check: {base_drgrpo_health_time*1000:.2f}ms")

            # Ray Dr. GRPO health checks (1000 calls on one actor)
            start = time.perf_counter()
            futs = [ray_drgrpo.health_check.remote() for _ in range(1000)] # type: ignore
            ray.get(futs)
            ray_drgrpo_health_time = (time.perf_counter() - start) / 1000
            print(f"✓ Ray Dr. GRPO health check: {ray_drgrpo_health_time*1000:.2f}ms")
            
            # Store results
            self.results['base_grpo']['health_check_time'] = base_health_time
            self.results['ray_grpo']['health_check_time'] = ray_health_time
            self.results['base_drgrpo']['health_check_time'] = base_drgrpo_health_time
            self.results['ray_drgrpo']['health_check_time'] = ray_drgrpo_health_time
            
            # Calculate communication overhead
            grpo_comm_overhead = (ray_health_time - base_health_time) / base_health_time * 100
            drgrpo_comm_overhead = (ray_drgrpo_health_time - base_drgrpo_health_time) / base_drgrpo_health_time * 100
            
            print("\n📊 Communication Overhead:")
            print(f"   Ray GRPO: {grpo_comm_overhead:+.1f}%")
            print(f"   Ray Dr. GRPO: {drgrpo_comm_overhead:+.1f}%")
            
        finally:
            # Defer actor kill to global cleanup
            if not hasattr(self, "_actors_to_cleanup"):
                self._actors_to_cleanup = []  # type: ignore
            self._actors_to_cleanup.extend([ray_grpo, ray_drgrpo])  # type: ignore
            
    def test_concurrent_execution(self):
        """Test concurrent execution benefits of Ray actors."""
        print("\n🔄 Testing Concurrent Execution Benefits")
        print("-" * 40)
        
        config = create_test_config()
        
        # Sequential execution: reuse one base instance, ask 1 000 health-checks serially
        base_seq = BaseGRPO(config)  # type: ignore
        start_time = time.time()
        for _ in range(1000):
            _ = asyncio.run(base_seq.health_check())
        sequential_time = time.time() - start_time
        print(f"✓ Sequential base GRPO (1000 calls): {sequential_time:.3f}s")

        # Concurrent execution: reuse already created actors when available to avoid packaging overhead
        ray_actors = getattr(self, "ray_actor_pool", None)
        if ray_actors is None:
            ray_actors = [GRPO.remote(config) for _ in range(4)]  # type: ignore
        start_time = time.time()
        futures = [ray_actors[i % len(ray_actors)].health_check.remote() for i in range(1000)]  # type: ignore
        _ = ray.get(futures)
        concurrent_time = time.time() - start_time
        print(f"✓ Concurrent Ray GRPO (4 actors, 1000 calls): {concurrent_time:.3f}s")
        
        # Remember for cleanup later
        if not hasattr(self, "_actors_to_cleanup"):
            self._actors_to_cleanup = []  # type: ignore
        self._actors_to_cleanup.extend(ray_actors)  # type: ignore
            
        # Calculate speedup
        speedup = sequential_time / concurrent_time
        efficiency = speedup / 4 * 100  # 4 is the number of parallel tasks
        
        print("\n📊 Concurrent Execution Results:")
        print(f"   Sequential time: {sequential_time:.3f}s")
        print(f"   Concurrent time: {concurrent_time:.3f}s")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Parallel efficiency: {efficiency:.1f}%")
        
        self.results['concurrency'] = {
            'sequential_time': sequential_time,
            'concurrent_time': concurrent_time,
            'speedup': speedup,
            'efficiency': efficiency
        }
        
    def test_memory_usage(self):
        """Compare memory usage patterns."""
        print("\n💾 Testing Memory Usage Patterns")
        print("-" * 40)
        
        config = create_test_config()
        
        # Measure baseline memory
        baseline_memory = measure_memory_usage()
        print(f"Baseline memory: {baseline_memory['rss_mb']:.1f} MB")
        
        # Test base implementation memory usage
        print("Testing base implementation memory...")
        base_grpo = BaseGRPO(config)  # type: ignore
        base_memory = measure_memory_usage()
        base_memory_usage = base_memory['rss_mb'] - baseline_memory['rss_mb']
        print(f"✓ Base GRPO memory usage: +{base_memory_usage:.1f} MB")
        
        # Test Ray actor memory usage
        print("Testing Ray actor memory...")
        ray_grpo = GRPO.remote(config)  # type: ignore
        time.sleep(1)  # Allow actor to fully initialize
        ray_memory = measure_memory_usage()
        ray_memory_usage = ray_memory['rss_mb'] - baseline_memory['rss_mb']
        print(f"✓ Ray GRPO memory usage: +{ray_memory_usage:.1f} MB")
        
        # Test multiple actors
        print("Testing multiple Ray actors...")
        ray_actors = [GRPO.remote(config) for _ in range(3)]  # type: ignore
        time.sleep(2)  # Allow actors to initialize
        multi_ray_memory = measure_memory_usage()
        multi_ray_memory_usage = multi_ray_memory['rss_mb'] - baseline_memory['rss_mb']
        print(f"✓ 4 Ray GRPO actors memory usage: +{multi_ray_memory_usage:.1f} MB")
        
        # Calculate memory efficiency
        per_actor_memory = (multi_ray_memory_usage - ray_memory_usage) / 3

        # Guard against zero or near-zero base usage to avoid divide-by-zero
        eps = 1e-3  # 1 kB
        effective_base = max(base_memory_usage, eps)
        memory_overhead = (ray_memory_usage - base_memory_usage) / effective_base * 100
        
        print(f"\n📊 Memory Usage Analysis:")
        delta_mb = ray_memory_usage - base_memory_usage
        if base_memory_usage < 1:
            overhead_str = f"+{delta_mb:.1f} MB (abs)"
        else:
            overhead_str = f"{(delta_mb / base_memory_usage * 100):+.1f}%"

        print(f"   Base GRPO: {base_memory_usage:.1f} MB")
        print(f"   Single Ray actor: {ray_memory_usage:.1f} MB")
        print(f"   Per additional actor: {per_actor_memory:.1f} MB")
        print(f"   Ray overhead: {overhead_str}")
        
        # Store results
        self.results['memory'] = {
            'base_usage_mb': base_memory_usage,
            'ray_single_usage_mb': ray_memory_usage,
            'ray_multi_usage_mb': multi_ray_memory_usage,
            'per_actor_mb': per_actor_memory,
            'overhead_percent': memory_overhead
        }
        
        # Clean up
        ray.kill(ray_grpo)  # type: ignore
        for actor in ray_actors:
            ray.kill(actor)  # type: ignore
            
    def print_performance_summary(self):
        """Print comprehensive performance summary."""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE COMPARISON SUMMARY")
        print("=" * 60)
        
        # Initialization comparison
        print("\n🚀 INITIALIZATION PERFORMANCE:")
        base_grpo_init = self.results['base_grpo']['init_time']
        ray_grpo_init = self.results['ray_grpo']['init_time']
        base_drgrpo_init = self.results['base_drgrpo']['init_time']
        ray_drgrpo_init = self.results['ray_drgrpo']['init_time']
        
        print(f"   Base GRPO:      {base_grpo_init:.3f}s")
        print(f"   Ray GRPO:       {ray_grpo_init:.3f}s ({ray_grpo_init/base_grpo_init:.1f}x)")
        print(f"   Base Dr. GRPO:  {base_drgrpo_init:.3f}s")
        print(f"   Ray Dr. GRPO:   {ray_drgrpo_init:.3f}s ({ray_drgrpo_init/base_drgrpo_init:.1f}x)")
        
        # Operation performance
        print("\n⚡ OPERATION PERFORMANCE (avg per call):")
        base_health = self.results['base_grpo']['health_check_time'] * 1000
        ray_health = self.results['ray_grpo']['health_check_time'] * 1000
        base_drgrpo_health = self.results['base_drgrpo']['health_check_time'] * 1000
        ray_drgrpo_health = self.results['ray_drgrpo']['health_check_time'] * 1000
        
        print(f"   Base GRPO:      {base_health:.1f}ms")
        print(f"   Ray GRPO:       {ray_health:.1f}ms ({ray_health/base_health:.1f}x)")
        print(f"   Base Dr. GRPO:  {base_drgrpo_health:.1f}ms")
        print(f"   Ray Dr. GRPO:   {ray_drgrpo_health:.1f}ms ({ray_drgrpo_health/base_drgrpo_health:.1f}x)")
        
        # Concurrency benefits
        if 'concurrency' in self.results:
            conc = self.results['concurrency']
            print(f"\n🔄 CONCURRENCY BENEFITS:")
            print(f"   Sequential (4 tasks):  {conc['sequential_time']:.3f}s")
            print(f"   Concurrent (4 actors): {conc['concurrent_time']:.3f}s")
            print(f"   Speedup:               {conc['speedup']:.2f}x")
            print(f"   Parallel efficiency:   {conc['efficiency']:.1f}%")
        
        # Memory usage
        if 'memory' in self.results:
            mem = self.results['memory']
            print(f"\n💾 MEMORY USAGE:")
            print(f"   Base implementation:   {mem['base_usage_mb']:.1f} MB")
            print(f"   Single Ray actor:      {mem['ray_single_usage_mb']:.1f} MB")
            print(f"   Ray overhead:          {mem['overhead_percent']:+.1f}%")
            print(f"   Per additional actor:  {mem['per_actor_mb']:.1f} MB")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if 'concurrency' in self.results and self.results['concurrency']['speedup'] > 1.5:
            print("   ✅ Ray actors provide significant speedup for concurrent workloads")
        else:
            print("   ⚠️  Ray actors have overhead - consider for concurrent workloads only")
            
        if 'memory' in self.results and self.results['memory']['overhead_percent'] < 50:
            print("   ✅ Ray memory overhead is reasonable")
        else:
            print("   ⚠️  Ray has significant memory overhead")
            
        print("\n🎯 USAGE GUIDELINES:")
        print("   • Use base implementations for:")
        print("     - Single-threaded workloads")
        print("     - Memory-constrained environments")
        print("     - Development and testing")
        print("   • Use Ray actors for:")
        print("     - Distributed training")
        print("     - Concurrent inference")
        print("     - Production clusters")
        print("     - Fault tolerance needs")

        # Clean-up any Ray actors we kept alive
        if hasattr(self, "_actors_to_cleanup"):
            for a in self._actors_to_cleanup:  # type: ignore
                try:
                    ray.kill(a)  # type: ignore
                except Exception:
                    pass


def run_performance_tests():
    """Main function to run all performance tests."""
    comparison = PerformanceComparison()
    results = comparison.run_all_comparisons()
    return results


if __name__ == "__main__":
    if RAY_AVAILABLE:
        results = run_performance_tests()
    else:
        print("❌ Ray not available, cannot run performance comparison tests")
        print("Install Ray with: pip install ray") 