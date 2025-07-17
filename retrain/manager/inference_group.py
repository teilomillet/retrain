# manager/inference_group.py

import asyncio
import logging
import time
import random
from typing import Dict, List, Any, Optional
import ray
from ray.util.placement_group import PlacementGroup

logger = logging.getLogger(__name__)

@ray.remote(num_cpus=1, num_gpus=0)
class InferenceGroup:
    """
    Resilient Inference Actor Group Manager.
    
    Manages multiple inference actor instances with:
    1. Load balancing across multiple inference workers
    2. Mixed hardware support (CPU + GPU workers)
    3. Automatic health monitoring and restart
    4. Parallel rollout generation
    5. Fault tolerance and recovery
    6. macOS CPU-only fallback support
    """
    
    def __init__(self, config: Any, databuffer: ray.ObjectRef, num_workers: int = 2, 
                 placement_group: Optional[PlacementGroup] = None):
        """
        Initialize InferenceGroup with configuration and worker management.
        
        Args:
            config: Training configuration object
            databuffer: Reference to the ReDataBuffer actor
            num_workers: Number of inference worker instances to maintain
            placement_group: Optional Ray placement group for resource allocation
        """
        self.config = config
        self.databuffer = databuffer
        self.num_workers = num_workers
        self.placement_group = placement_group
        
        # Worker management
        self.inference_workers: List[ray.ObjectRef] = []
        self.worker_types: List[str] = []  # Track worker types (cuda, macos, cpu)
        self.worker_health: Dict[str, Dict[str, Any]] = {}
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        
        # Load balancing (following Slime pattern)
        self.current_worker_index = 0
        self.pending_rollouts: Dict[str, ray.ObjectRef] = {}
        self.worker_load: Dict[str, int] = {}  # Track load per worker
        
        # Health monitoring
        self.health_check_interval = 30.0  # seconds
        self.max_restart_attempts = 3
        self.restart_counts: Dict[str, int] = {}
        
        # Platform optimization
        self.platform_optimized = False
        self.mixed_hardware = False
        self.is_initialized = False
        
        # Async coordination (following Slime pattern)
        self.pending_operations: Dict[str, List[ray.ObjectRef]] = {}
        
        logger.info(f"InferenceGroup initialized with {num_workers} workers")
        
    async def initialize(self) -> None:
        """Initialize inference workers with resilience following Slime async pattern."""
        logger.info("Initializing InferenceGroup workers...")
        
        try:
            # Detect platform and hardware for optimization
            await self._detect_hardware_capabilities()
            
            # Create mixed worker pool based on available hardware
            await self._create_mixed_worker_pool()
            
            # Initialize all workers in parallel (Slime async pattern)
            init_refs = await self._async_init_workers()
            await asyncio.gather(*[ref for ref in init_refs])
            
            # Start health monitoring
            asyncio.create_task(self._health_monitoring_loop())
            
            self.is_initialized = True
            logger.info(f"InferenceGroup initialization complete with {len(self.inference_workers)} workers")
            
        except Exception as e:
            logger.error(f"Failed to initialize InferenceGroup: {e}")
            raise
            
    async def _detect_hardware_capabilities(self) -> None:
        """Detect available hardware and plan worker distribution."""
        try:
            import platform
            import torch
            
            system = platform.system()
            has_cuda = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
            
            if system == "Darwin":  # macOS
                logger.info("Detected macOS - using CPU-only inference workers")
                self.num_workers = min(self.num_workers, 2)  # Conservative for macOS
                self.platform_optimized = True
                self.mixed_hardware = False
            elif has_cuda:
                logger.info("Detected CUDA - enabling mixed CPU/GPU inference workers")
                self.mixed_hardware = True
                # Plan: 70% GPU workers, 30% CPU workers for load balancing
                self.gpu_workers = max(1, int(self.num_workers * 0.7))
                self.cpu_workers = self.num_workers - self.gpu_workers
            else:
                logger.info("Detected CPU-only Linux - using CPU inference workers")
                self.mixed_hardware = False
                
        except Exception as e:
            logger.warning(f"Hardware detection failed: {e}")
            self.mixed_hardware = False
            
    async def _create_mixed_worker_pool(self) -> None:
        """Create mixed pool of inference workers based on available hardware."""
        # Import inference factories
        from retrain.inference.macos import MacOSInferenceActor
        from retrain.inference.cuda import CUDAInferenceActor
        from retrain.inference.cpu import CPUInferenceActor
        
        for worker_id in range(self.num_workers):
            try:
                worker_name = f"inference_worker_{worker_id}"
                
                # Determine worker type based on hardware and load balancing
                if self.platform_optimized:
                    # macOS: CPU-only workers
                    worker_type = "macos"
                    worker = MacOSInferenceActor.remote(self.config, self.databuffer)  # type: ignore
                elif self.mixed_hardware:
                    # Mixed hardware: distribute GPU and CPU workers
                    if worker_id < self.gpu_workers:
                        worker_type = "cuda"
                        worker = CUDAInferenceActor.remote(self.config, self.databuffer)  # type: ignore
                    else:
                        worker_type = "cpu"
                        worker = CPUInferenceActor.remote(self.config, self.databuffer)  # type: ignore
                else:
                    # CPU-only Linux
                    worker_type = "cpu"
                    worker = CPUInferenceActor.remote(self.config, self.databuffer)  # type: ignore
                
                self.inference_workers.append(worker)
                self.worker_types.append(worker_type)
                self.worker_health[worker_name] = {
                    'status': 'healthy',
                    'last_check': time.time(),
                    'start_time': time.time(),
                    'restart_count': 0,
                    'worker_type': worker_type
                }
                self.restart_counts[worker_name] = 0
                self.worker_load[worker_name] = 0
                
                logger.info(f"Created {worker_type} inference worker: {worker_name}")
                
            except Exception as e:
                logger.error(f"Failed to create inference worker {worker_id}: {e}")
                
        if not self.inference_workers:
            raise RuntimeError("Failed to create any inference workers")
            
        logger.info(f"Mixed worker pool created: {len(self.inference_workers)} workers "
                   f"({self.worker_types.count('cuda')} GPU, {self.worker_types.count('cpu')} CPU, "
                   f"{self.worker_types.count('macos')} macOS)")
            
    async def _async_init_workers(self) -> List[ray.ObjectRef]:
        """Initialize all workers in parallel using Slime async pattern."""
        init_refs = []
        for worker in self.inference_workers:
            init_refs.append(worker.initialize.remote())  # type: ignore
        return init_refs
        
    async def _health_monitoring_loop(self) -> None:
        """Continuous health monitoring and restart loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_worker_health()
                await self._restart_unhealthy_workers()
                
            except Exception as e:
                logger.error(f"Inference health monitoring error: {e}")
                
    async def _check_worker_health(self) -> None:
        """Check health of all inference workers."""
        health_futures = {}
        
        for i, worker in enumerate(self.inference_workers):
            if worker is not None:
                worker_name = f"inference_worker_{i}"
                try:
                    health_futures[worker_name] = worker.health_check.remote()  # type: ignore
                except Exception as e:
                    logger.warning(f"Failed to start health check for {worker_name}: {e}")
                    self.worker_health[worker_name]['status'] = 'unhealthy'
                    
        # Collect health results
        for worker_name, future in health_futures.items():
            try:
                health_result = await asyncio.wait_for(future, timeout=10.0)
                self.worker_health[worker_name].update({
                    'status': 'healthy',
                    'last_check': time.time(),
                    'last_health_data': health_result
                })
            except asyncio.TimeoutError:
                logger.warning(f"Inference health check timeout for {worker_name}")
                self.worker_health[worker_name]['status'] = 'timeout'
            except Exception as e:
                logger.warning(f"Inference health check failed for {worker_name}: {e}")
                self.worker_health[worker_name]['status'] = 'unhealthy'
                
    async def _restart_unhealthy_workers(self) -> None:
        """Restart inference workers that are unhealthy."""
        for i, worker in enumerate(self.inference_workers):
            worker_name = f"inference_worker_{i}"
            health = self.worker_health.get(worker_name, {})
            
            if health.get('status') in ['unhealthy', 'timeout']:
                restart_count = self.restart_counts.get(worker_name, 0)
                
                if restart_count < self.max_restart_attempts:
                    logger.warning(f"Restarting unhealthy inference worker: {worker_name} (attempt {restart_count + 1})")
                    
                    try:
                        # Kill the old worker
                        if worker is not None:
                            ray.kill(worker)
                            
                        # Create new worker of the same type
                        await self._create_single_worker(i, self.worker_types[i])
                        
                        self.restart_counts[worker_name] = restart_count + 1
                        logger.info(f"Successfully restarted inference {worker_name}")
                        
                    except Exception as e:
                        logger.error(f"Failed to restart inference {worker_name}: {e}")
                        self.worker_health[worker_name]['status'] = 'failed'
                else:
                    logger.error(f"Inference {worker_name} exceeded max restart attempts ({self.max_restart_attempts})")
                    self.worker_health[worker_name]['status'] = 'failed'
                    
    async def _create_single_worker(self, worker_id: int, worker_type: str) -> None:
        """Create a single inference worker to replace a failed one."""
        from retrain.inference.macos import MacOSInferenceActor
        from retrain.inference.cuda import CUDAInferenceActor
        from retrain.inference.cpu import CPUInferenceActor
        
        worker_name = f"inference_worker_{worker_id}"
        
        # Create replacement worker of the same type
        if worker_type == "macos":
            worker = MacOSInferenceActor.remote(self.config, self.databuffer)  # type: ignore
        elif worker_type == "cuda":
            worker = CUDAInferenceActor.remote(self.config, self.databuffer)  # type: ignore
        else:  # cpu
            worker = CPUInferenceActor.remote(self.config, self.databuffer)  # type: ignore
            
        await worker.initialize.remote()  # type: ignore
        
        self.inference_workers[worker_id] = worker
        self.worker_types[worker_id] = worker_type
        self.worker_health[worker_name] = {
            'status': 'healthy',
            'last_check': time.time(),
            'start_time': time.time(),
            'restart_count': self.restart_counts.get(worker_name, 0),
            'worker_type': worker_type
        }
        self.worker_load[worker_name] = 0
        
    # Slime-style async rollout generation methods
    async def async_generate_rollouts(self, episode_id: int, num_rollouts: int) -> List[ray.ObjectRef]:
        """
        Generate rollouts across multiple workers using Slime async pattern.
        
        Args:
            episode_id: Current episode identifier
            num_rollouts: Number of rollouts to generate
            
        Returns:
            List of ObjectRefs for rollout generation from workers
        """
        if not self.is_initialized:
            raise RuntimeError("InferenceGroup not initialized")
            
        # Distribute rollouts across healthy workers using load balancing
        rollout_refs = []
        healthy_workers = self._get_healthy_workers()
        
        if not healthy_workers:
            raise RuntimeError("No healthy inference workers available")
            
        # Distribute rollouts using round-robin with load balancing
        for rollout_idx in range(num_rollouts):
            worker_idx = self._select_worker_by_load(healthy_workers)
            worker = self.inference_workers[worker_idx]
            worker_name = f"inference_worker_{worker_idx}"
            
            # Generate single rollout
            rollout_ref = worker.generate_rollout.remote(episode_id, rollout_idx)  # type: ignore
            rollout_refs.append(rollout_ref)
            
            # Update load tracking
            self.worker_load[worker_name] += 1
            
        self.pending_operations[f'rollout_gen_{episode_id}'] = rollout_refs
        
        logger.info(f"Started async rollout generation: {num_rollouts} rollouts across {len(healthy_workers)} workers")
        return rollout_refs
        
    def _get_healthy_workers(self) -> List[int]:
        """Get list of healthy worker indices."""
        healthy_workers = []
        for i, worker in enumerate(self.inference_workers):
            worker_name = f"inference_worker_{i}"
            health = self.worker_health.get(worker_name, {})
            if health.get('status') == 'healthy':
                healthy_workers.append(i)
        return healthy_workers
        
    def _select_worker_by_load(self, healthy_workers: List[int]) -> int:
        """Select worker with lowest current load for load balancing."""
        if not healthy_workers:
            raise RuntimeError("No healthy workers available")
            
        # Find worker with minimum load
        min_load = float('inf')
        selected_worker = healthy_workers[0]
        
        for worker_idx in healthy_workers:
            worker_name = f"inference_worker_{worker_idx}"
            current_load = self.worker_load.get(worker_name, 0)
            
            if current_load < min_load:
                min_load = current_load
                selected_worker = worker_idx
                
        return selected_worker
        
    async def async_update_weights(self, weights: Dict[str, Any]) -> List[ray.ObjectRef]:
        """Update model weights across all workers using Slime async pattern."""
        update_refs = []
        for worker in self.inference_workers:
            update_refs.append(worker.update_model_weights.remote(weights))  # type: ignore
        return update_refs
        
    async def generate_rollouts_coordinated(self, episode_id: int, num_rollouts: int) -> List[Dict[str, Any]]:
        """
        Coordinated rollout generation with load balancing and fault tolerance.
        
        This follows Slime's distributed rollout pattern:
        1. Distribute rollouts across healthy workers
        2. Load balance based on worker capacity
        3. Handle failures gracefully with retries
        """
        if not self.is_initialized:
            raise RuntimeError("InferenceGroup not initialized")
            
        # Step 1: Generate rollouts asynchronously
        rollout_refs = await self.async_generate_rollouts(episode_id, num_rollouts)
        
        # Step 2: Wait for completion with timeout and retry logic
        completed_rollouts = []
        failed_rollouts = []
        
        for i, rollout_ref in enumerate(rollout_refs):
            try:
                rollout_data = await asyncio.wait_for(rollout_ref, timeout=60.0)
                completed_rollouts.append(rollout_data)
            except asyncio.TimeoutError:
                logger.warning(f"Rollout {i} timed out, retrying...")
                failed_rollouts.append(i)
            except Exception as e:
                logger.error(f"Rollout {i} failed: {e}")
                failed_rollouts.append(i)
                
        # Step 3: Retry failed rollouts on different workers
        if failed_rollouts:
            logger.info(f"Retrying {len(failed_rollouts)} failed rollouts")
            retry_refs = []
            healthy_workers = self._get_healthy_workers()
            
            for rollout_idx in failed_rollouts:
                if healthy_workers:
                    worker_idx = random.choice(healthy_workers)
                    worker = self.inference_workers[worker_idx]
                    retry_ref = worker.generate_rollout.remote(episode_id, rollout_idx)  # type: ignore
                    retry_refs.append(retry_ref)
                    
            # Wait for retries
            retry_results = await asyncio.gather(*retry_refs, return_exceptions=True)
            for result in retry_results:
                if not isinstance(result, Exception):
                    completed_rollouts.append(result)
                    
        # Update worker load tracking (decrease load after completion)
        for worker_name in self.worker_load:
            self.worker_load[worker_name] = max(0, self.worker_load[worker_name] - 1)
            
        logger.info(f"Coordinated rollout generation complete: {len(completed_rollouts)}/{num_rollouts} successful")
        return completed_rollouts
        
    async def connect_trainer(self, trainer_group: ray.ObjectRef) -> None:
        """Connect with trainer group for weight synchronization."""
        logger.info("InferenceGroup connected with trainer group")
        self.connected_trainer_group = trainer_group
        
    async def get_group_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the inference group."""
        healthy_workers = sum(1 for h in self.worker_health.values() if h.get('status') == 'healthy')
        
        # Count workers by type
        worker_type_counts = {}
        for worker_type in self.worker_types:
            worker_type_counts[worker_type] = worker_type_counts.get(worker_type, 0) + 1
            
        status = {
            'total_workers': len(self.inference_workers),
            'healthy_workers': healthy_workers,
            'worker_type_distribution': worker_type_counts,
            'mixed_hardware': self.mixed_hardware,
            'platform_optimized': self.platform_optimized,
            'worker_health': self.worker_health.copy(),
            'worker_load': self.worker_load.copy(),
            'restart_counts': self.restart_counts.copy(),
            'is_initialized': self.is_initialized,
            'pending_operations': len(self.pending_operations),
            'timestamp': time.time()
        }
        
        return status
        
    async def scale_workers(self, new_worker_count: int, prefer_gpu: bool = True) -> None:
        """Dynamically scale the number of inference workers."""
        if new_worker_count == len(self.inference_workers):
            return
            
        logger.info(f"Scaling InferenceGroup from {len(self.inference_workers)} to {new_worker_count} workers")
        
        if new_worker_count > len(self.inference_workers):
            # Scale up - add new workers
            current_count = len(self.inference_workers)
            
            for worker_id in range(current_count, new_worker_count):
                # Determine worker type for new workers
                if self.mixed_hardware and prefer_gpu:
                    worker_type = "cuda"
                elif self.platform_optimized:
                    worker_type = "macos"
                else:
                    worker_type = "cpu"
                    
                await self._create_single_worker(len(self.inference_workers), worker_type)
                self.inference_workers.append(None)  # type: ignore  # Placeholder, filled by _create_single_worker
                self.worker_types.append(worker_type)
                
        else:
            # Scale down
            workers_to_remove = self.inference_workers[new_worker_count:]
            for worker in workers_to_remove:
                if worker is not None:
                    ray.kill(worker)
                    
            self.inference_workers = self.inference_workers[:new_worker_count]
            self.worker_types = self.worker_types[:new_worker_count]
            
            # Clean up tracking for removed workers
            for worker_id in range(new_worker_count, len(self.worker_health)):
                worker_name = f"inference_worker_{worker_id}"
                if worker_name in self.worker_health:
                    del self.worker_health[worker_name]
                if worker_name in self.restart_counts:
                    del self.restart_counts[worker_name]
                if worker_name in self.worker_load:
                    del self.worker_load[worker_name]
                    
        self.num_workers = new_worker_count
        logger.info(f"InferenceGroup scaling complete: {len(self.inference_workers)} workers")
        
    async def shutdown(self) -> None:
        """Gracefully shutdown all inference workers."""
        logger.info("Shutting down InferenceGroup...")
        
        for worker in self.inference_workers:
            if worker is not None:
                try:
                    ray.kill(worker)
                except Exception as e:
                    logger.warning(f"Error killing inference worker: {e}")
                    
        self.inference_workers.clear()
        self.worker_types.clear()
        self.worker_health.clear()
        self.worker_load.clear()
        self.pending_operations.clear()
        
        logger.info("InferenceGroup shutdown complete") 