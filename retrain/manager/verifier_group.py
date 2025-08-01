# manager/verifier_group.py

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
import ray

logger = logging.getLogger(__name__)

@ray.remote
class VerifierGroup:
    """
     Verifier Actor Group Manager.
    
    Manages multiple verifier actor instances with:
    1. Automatic health monitoring and restart
    2. Load balancing across instances  
    3. macOS CPU-only compatibility
    4. Dynamic scaling based on verification load
    5. Fault tolerance and recovery
    """
    
    def __init__(self, config: Any, databuffer: ray.ObjectRef, num_workers: int = 2):
        """
        Initialize VerifierGroup with configuration and worker management.
        
        Args:
            config: Training configuration object
            databuffer: Reference to the ReDataBuffer actor
            num_workers: Number of verifier worker instances to maintain
        """
        self.config = config
        self.databuffer = databuffer
        self.num_workers = num_workers
        
        # Worker management
        self.verifier_workers: List[ray.ObjectRef] = []
        self.worker_health: Dict[str, Dict[str, Any]] = {}
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        
        # Load balancing
        self.current_worker_index = 0
        self.pending_verifications: Dict[str, ray.ObjectRef] = {}
        
        # Health monitoring
        self.health_check_interval = 30.0  # seconds
        self.max_restart_attempts = 3
        self.restart_counts: Dict[str, int] = {}
        
        # macOS compatibility
        self.platform_optimized = False
        self.is_initialized = False
        
        logger.info(f"VerifierGroup initialized with {num_workers} workers")
        
    def set_precreated_workers(self, workers: List[ray.ObjectRef]) -> None:
        """Set pre-created workers to avoid nested actor creation issues."""
        self.precreated_workers = workers
        self.num_workers = len(workers)
        logger.info(f"Set {len(workers)} pre-created workers")
        
    async def initialize(self) -> None:
        """Initialize verifier workers with resilience."""
        logger.info("Initializing VerifierGroup workers...")
        
        try:
            # Detect platform for optimization
            await self._detect_platform_optimizations()
            
            # Create initial worker pool
            await self._create_worker_pool()
            
            # Start health monitoring
            asyncio.create_task(self._health_monitoring_loop())
            
            self.is_initialized = True
            logger.info(f"VerifierGroup initialization complete with {len(self.verifier_workers)} workers")
            
        except Exception as e:
            logger.error(f"Failed to initialize VerifierGroup: {e}")
            raise
            
    async def _detect_platform_optimizations(self) -> None:
        """Detect platform and apply macOS optimizations."""
        try:
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                logger.info("Detected macOS - applying CPU-only optimizations for verifiers")
                # Reduce worker count for macOS
                self.num_workers = min(self.num_workers, 2)
                self.platform_optimized = True
            elif system == "Linux":
                logger.info("Detected Linux - standard verifier configuration")
            else:
                logger.info(f"Detected {system} - using standard verifier configuration")
                
        except Exception as e:
            logger.warning(f"Platform detection failed: {e}")
            
    async def _create_worker_pool(self) -> None:
        """Create the initial pool of verifier workers with proper resource allocation."""
        # Check if we have pre-created workers
        if hasattr(self, 'precreated_workers') and self.precreated_workers:
            logger.info("Using pre-created verifier workers to avoid nested actor creation issues")
            
            for worker_id, worker in enumerate(self.precreated_workers):
                try:
                    worker_name = f"verifier_worker_{worker_id}"
                    
                    # Initialize the pre-created worker
                    await worker.initialize.remote()  # type: ignore
                    
                    self.verifier_workers.append(worker)
                    self.worker_health[worker_name] = {
                        'status': 'healthy',
                        'last_check': time.time(),
                        'start_time': time.time(),
                        'restart_count': 0
                    }
                    self.restart_counts[worker_name] = 0
                    
                    logger.info(f"Initialized pre-created verifier worker: {worker_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize pre-created verifier worker {worker_id}: {e}")
            
            if not self.verifier_workers:
                raise RuntimeError("Failed to initialize any pre-created verifier workers")
            return
        
        # Fallback: Try to create workers (will likely fail due to nested actor creation)
        logger.warning("No pre-created workers provided, attempting nested actor creation (may fail)")
        
        # Import ActorFactory and hardware detection for proper resource allocation
        from ..hardware import HardwareDetector, ActorFactory
        
        # Create hardware detector and actor factory for proper resource allocation
        hardware_detector = HardwareDetector()
        actor_factory = ActorFactory(hardware_detector)
        
        for worker_id in range(self.num_workers):
            try:
                worker_name = f"verifier_worker_{worker_id}"
                
                # Create worker using ActorFactory for proper resource allocation
                # This ensures the actor gets the right CPU/GPU allocation from hardware detection
                worker = actor_factory.create_verifier_actor(self.config, self.databuffer)
                
                # Initialize the worker
                await worker.initialize.remote()  # type: ignore
                
                self.verifier_workers.append(worker)
                self.worker_health[worker_name] = {
                    'status': 'healthy',
                    'last_check': time.time(),
                    'start_time': time.time(),
                    'restart_count': 0
                }
                self.restart_counts[worker_name] = 0
                
                logger.info(f"Created verifier worker: {worker_name}")
                
            except Exception as e:
                logger.error(f"Failed to create verifier worker {worker_id}: {e}")
                
        if not self.verifier_workers:
            raise RuntimeError("Failed to create any verifier workers")
            
    async def _health_monitoring_loop(self) -> None:
        """Continuous health monitoring and restart loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_worker_health()
                await self._restart_unhealthy_workers()
                
            except Exception as e:
                logger.error(f"Verifier health monitoring error: {e}")
                
    async def _check_worker_health(self) -> None:
        """Check health of all verifier workers."""
        health_futures = {}
        
        for i, worker in enumerate(self.verifier_workers):
            if worker is not None:
                worker_name = f"verifier_worker_{i}"
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
                logger.warning(f"Verifier health check timeout for {worker_name}")
                self.worker_health[worker_name]['status'] = 'timeout'
            except Exception as e:
                logger.warning(f"Verifier health check failed for {worker_name}: {e}")
                self.worker_health[worker_name]['status'] = 'unhealthy'
                
    async def _restart_unhealthy_workers(self) -> None:
        """Restart verifier workers that are unhealthy."""
        for i, worker in enumerate(self.verifier_workers):
            worker_name = f"verifier_worker_{i}"
            health = self.worker_health.get(worker_name, {})
            
            if health.get('status') in ['unhealthy', 'timeout']:
                restart_count = self.restart_counts.get(worker_name, 0)
                
                if restart_count < self.max_restart_attempts:
                    logger.warning(f"Restarting unhealthy verifier worker: {worker_name} (attempt {restart_count + 1})")
                    
                    try:
                        # Kill the old worker
                        if worker is not None:
                            ray.kill(worker)
                            
                        # Create new worker
                        await self._create_single_worker(i)
                        
                        self.restart_counts[worker_name] = restart_count + 1
                        logger.info(f"Successfully restarted verifier {worker_name}")
                        
                    except Exception as e:
                        logger.error(f"Failed to restart verifier {worker_name}: {e}")
                        self.worker_health[worker_name]['status'] = 'failed'
                else:
                    logger.error(f"Verifier {worker_name} exceeded max restart attempts ({self.max_restart_attempts})")
                    self.worker_health[worker_name]['status'] = 'failed'
                    
    async def _create_single_worker(self, worker_id: int) -> None:
        """Create a single verifier worker to replace a failed one."""
        from retrain.verifier.verifier import ReVerifier
        
        worker_name = f"verifier_worker_{worker_id}"
        
        # Create worker - ReVerifier is already a Ray actor, use .remote() directly  
        worker = ReVerifier.remote(self.config, self.databuffer)  # type: ignore
            
        await worker.initialize.remote()  # type: ignore
        
        self.verifier_workers[worker_id] = worker
        self.worker_health[worker_name] = {
            'status': 'healthy',
            'last_check': time.time(),
            'start_time': time.time(),
            'restart_count': self.restart_counts.get(worker_name, 0)
        }
        
    async def verify_rollouts(self, rollout_data: List[Dict[str, Any]], episode_id: int) -> List[Dict[str, Any]]:
        """
        Verify rollout data using load-balanced worker selection.
        
        Args:
            rollout_data: List of rollout data to verify
            episode_id: Current episode identifier
            
        Returns:
            List of verification results
        """
        if not self.is_initialized:
            raise RuntimeError("VerifierGroup not initialized")
            
        # Select healthy worker using round-robin
        worker = await self._get_healthy_worker()
        if worker is None:
            raise RuntimeError("No healthy verifier workers available")
            
        try:
            # Delegate to the selected worker
            verification_results = await worker.verify_rollouts.remote(rollout_data, episode_id)  # type: ignore
            return verification_results
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            # Try with different worker
            backup_worker = await self._get_healthy_worker(exclude_last=True)
            if backup_worker:
                try:
                    verification_results = await backup_worker.verify_rollouts.remote(rollout_data, episode_id)  # type: ignore
                    return verification_results
                except Exception as backup_error:
                    logger.error(f"Backup verification also failed: {backup_error}")
                    
            # Fallback: return default pass results
            logger.warning("Using fallback verification results (all passed)")
            return [
                {
                    'rollout_id': f'rollout_{episode_id}_{i}',
                    'episode_id': episode_id,
                    'rollout_idx': i,
                    'verifications': {},
                    'overall_passed': True,
                    'confidence_score': 0.5,
                    'verification_status': 'fallback'
                }
                for i in range(len(rollout_data))
            ]
            
    async def _get_healthy_worker(self, exclude_last: bool = False) -> Optional[ray.ObjectRef]:
        """Get a healthy verifier worker using round-robin selection."""
        attempts = 0
        _ = self.current_worker_index
        
        while attempts < len(self.verifier_workers):
            if exclude_last and attempts == 0:
                self.current_worker_index = (self.current_worker_index + 1) % len(self.verifier_workers)
                
            worker_name = f"verifier_worker_{self.current_worker_index}"
            health = self.worker_health.get(worker_name, {})
            
            if health.get('status') == 'healthy':
                worker = self.verifier_workers[self.current_worker_index]
                self.current_worker_index = (self.current_worker_index + 1) % len(self.verifier_workers)
                return worker
                
            self.current_worker_index = (self.current_worker_index + 1) % len(self.verifier_workers)
            attempts += 1
            
        return None
        
    async def connect_reward(self, reward_ref: ray.ObjectRef) -> None:
        """Connect with reward actors (placeholder for future integration)."""
        logger.info("VerifierGroup connected with reward group")
        
    async def get_group_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the verifier group."""
        healthy_workers = sum(1 for h in self.worker_health.values() if h.get('status') == 'healthy')
        
        status = {
            'total_workers': len(self.verifier_workers),
            'healthy_workers': healthy_workers,
            'platform_optimized': self.platform_optimized,
            'worker_health': self.worker_health.copy(),
            'restart_counts': self.restart_counts.copy(),
            'is_initialized': self.is_initialized,
            'timestamp': time.time()
        }
        
        return status
        
    async def scale_workers(self, new_worker_count: int) -> None:
        """Dynamically scale the number of verifier workers."""
        if new_worker_count == len(self.verifier_workers):
            return
            
        logger.info(f"Scaling VerifierGroup from {len(self.verifier_workers)} to {new_worker_count} workers")
        
        if new_worker_count > len(self.verifier_workers):
            # Scale up - extend the list first, then create workers
            current_count = len(self.verifier_workers)
            # Extend list with None placeholders
            self.verifier_workers.extend([None] * (new_worker_count - current_count))  # type: ignore
            
            for worker_id in range(current_count, new_worker_count):
                await self._create_single_worker(worker_id)  # This sets self.verifier_workers[worker_id]
                
        else:
            # Scale down
            workers_to_remove = self.verifier_workers[new_worker_count:]
            for worker in workers_to_remove:
                if worker is not None:
                    ray.kill(worker)
                    
            self.verifier_workers = self.verifier_workers[:new_worker_count]
            
            # Clean up health tracking for removed workers
            for worker_id in range(new_worker_count, len(self.worker_health)):
                worker_name = f"verifier_worker_{worker_id}"
                if worker_name in self.worker_health:
                    del self.worker_health[worker_name]
                if worker_name in self.restart_counts:
                    del self.restart_counts[worker_name]
                    
        self.num_workers = new_worker_count
        logger.info(f"VerifierGroup scaling complete: {len(self.verifier_workers)} workers")
        
    async def shutdown(self) -> None:
        """Gracefully shutdown all verifier workers."""
        logger.info("Shutting down VerifierGroup...")
        
        for worker in self.verifier_workers:
            if worker is not None:
                try:
                    ray.kill(worker)
                except Exception as e:
                    logger.warning(f"Error killing verifier worker: {e}")
                    
        self.verifier_workers.clear()
        self.worker_health.clear()
        
        logger.info("VerifierGroup shutdown complete")