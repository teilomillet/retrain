# manager/reward_group.py

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
import ray

logger = logging.getLogger(__name__)

@ray.remote(num_cpus=1, num_gpus=0)
class RewardGroup:
    """
    Resilient Reward Actor Group Manager.
    
    Manages multiple reward actor instances with:
    1. Automatic health monitoring and restart
    2. Load balancing across instances
    3. macOS CPU-only compatibility
    4. Dynamic scaling based on load
    5. Fault tolerance and recovery
    """
    
    def __init__(self, config: Any, databuffer: ray.ObjectRef, num_workers: int = 2):
        """
        Initialize RewardGroup with configuration and worker management.
        
        Args:
            config: Training configuration object
            databuffer: Reference to the ReDataBuffer actor
            num_workers: Number of reward worker instances to maintain
        """
        self.config = config
        self.databuffer = databuffer
        self.num_workers = num_workers
        
        # Worker management
        self.reward_workers: List[ray.ObjectRef] = []
        self.worker_health: Dict[str, Dict[str, Any]] = {}
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        
        # Load balancing
        self.current_worker_index = 0
        self.pending_tasks: Dict[str, ray.ObjectRef] = {}
        
        # Health monitoring
        self.health_check_interval = 30.0  # seconds
        self.max_restart_attempts = 3
        self.restart_counts: Dict[str, int] = {}
        
        # macOS compatibility
        self.platform_optimized = False
        self.is_initialized = False
        
        logger.info(f"RewardGroup initialized with {num_workers} workers")
        
    async def initialize(self) -> None:
        """Initialize reward workers with resilience."""
        logger.info("Initializing RewardGroup workers...")
        
        try:
            # Detect platform for optimization
            await self._detect_platform_optimizations()
            
            # Create initial worker pool
            await self._create_worker_pool()
            
            # Start health monitoring
            asyncio.create_task(self._health_monitoring_loop())
            
            self.is_initialized = True
            logger.info(f"RewardGroup initialization complete with {len(self.reward_workers)} workers")
            
        except Exception as e:
            logger.error(f"Failed to initialize RewardGroup: {e}")
            raise
            
    async def _detect_platform_optimizations(self) -> None:
        """Detect platform and apply macOS optimizations."""
        try:
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                logger.info("Detected macOS - applying CPU-only optimizations")
                # Reduce worker count for macOS to avoid resource contention
                self.num_workers = min(self.num_workers, 2)
                self.platform_optimized = True
            elif system == "Linux":
                logger.info("Detected Linux - standard configuration")
            else:
                logger.info(f"Detected {system} - using standard configuration")
                
        except Exception as e:
            logger.warning(f"Platform detection failed: {e}")
            
    async def _create_worker_pool(self) -> None:
        """Create the initial pool of reward workers."""
        # Import ReReward class directly from reward module
        from retrain.reward.reward import ReReward
        
        for worker_id in range(self.num_workers):
            try:
                worker_name = f"reward_worker_{worker_id}"
                
                # Create worker - ReReward is already a Ray actor, use .remote() directly
                worker = ReReward.remote(self.config, self.databuffer)  # type: ignore
                
                # Initialize the worker
                await worker.initialize.remote()  # type: ignore
                
                self.reward_workers.append(worker)
                self.worker_health[worker_name] = {
                    'status': 'healthy',
                    'last_check': time.time(),
                    'start_time': time.time(),
                    'restart_count': 0
                }
                self.restart_counts[worker_name] = 0
                
                logger.info(f"Created reward worker: {worker_name}")
                
            except Exception as e:
                logger.error(f"Failed to create reward worker {worker_id}: {e}")
                
        if not self.reward_workers:
            raise RuntimeError("Failed to create any reward workers")
            
    async def _health_monitoring_loop(self) -> None:
        """Continuous health monitoring and restart loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_worker_health()
                await self._restart_unhealthy_workers()
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                
    async def _check_worker_health(self) -> None:
        """Check health of all workers."""
        health_futures = {}
        
        for i, worker in enumerate(self.reward_workers):
            if worker is not None:
                worker_name = f"reward_worker_{i}"
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
                logger.warning(f"Health check timeout for {worker_name}")
                self.worker_health[worker_name]['status'] = 'timeout'
            except Exception as e:
                logger.warning(f"Health check failed for {worker_name}: {e}")
                self.worker_health[worker_name]['status'] = 'unhealthy'
                
    async def _restart_unhealthy_workers(self) -> None:
        """Restart workers that are unhealthy."""
        for i, worker in enumerate(self.reward_workers):
            worker_name = f"reward_worker_{i}"
            health = self.worker_health.get(worker_name, {})
            
            if health.get('status') in ['unhealthy', 'timeout']:
                restart_count = self.restart_counts.get(worker_name, 0)
                
                if restart_count < self.max_restart_attempts:
                    logger.warning(f"Restarting unhealthy worker: {worker_name} (attempt {restart_count + 1})")
                    
                    try:
                        # Kill the old worker
                        if worker is not None:
                            ray.kill(worker)
                            
                        # Create new worker
                        await self._create_single_worker(i)
                        
                        self.restart_counts[worker_name] = restart_count + 1
                        logger.info(f"Successfully restarted {worker_name}")
                        
                    except Exception as e:
                        logger.error(f"Failed to restart {worker_name}: {e}")
                        self.worker_health[worker_name]['status'] = 'failed'
                else:
                    logger.error(f"Worker {worker_name} exceeded max restart attempts ({self.max_restart_attempts})")
                    self.worker_health[worker_name]['status'] = 'failed'
                    
    async def _create_single_worker(self, worker_id: int) -> None:
        """Create a single worker to replace a failed one."""
        from retrain.reward.reward import ReReward
        
        worker_name = f"reward_worker_{worker_id}"
        
        # Create worker - ReReward is already a Ray actor, use .remote() directly
        worker = ReReward.remote(self.config, self.databuffer)  # type: ignore
            
        await worker.initialize.remote()  # type: ignore
        
        self.reward_workers[worker_id] = worker
        self.worker_health[worker_name] = {
            'status': 'healthy',
            'last_check': time.time(),
            'start_time': time.time(),
            'restart_count': self.restart_counts.get(worker_name, 0)
        }
        
    async def compute_rewards(self, rollout_data: List[Dict[str, Any]], episode_id: int) -> List[float]:
        """
        Compute rewards using load-balanced worker selection.
        
        Args:
            rollout_data: List of rollout data to process
            episode_id: Current episode identifier
            
        Returns:
            List of computed rewards
        """
        if not self.is_initialized:
            raise RuntimeError("RewardGroup not initialized")
            
        # Select healthy worker using round-robin
        worker = await self._get_healthy_worker()
        if worker is None:
            raise RuntimeError("No healthy reward workers available")
            
        try:
            # Delegate to the selected worker
            rewards = await worker.compute_rewards.remote(rollout_data, episode_id)  # type: ignore
            return rewards
            
        except Exception as e:
            logger.error(f"Reward computation failed: {e}")
            # Try with different worker
            backup_worker = await self._get_healthy_worker(exclude_last=True)
            if backup_worker:
                try:
                    rewards = await backup_worker.compute_rewards.remote(rollout_data, episode_id)  # type: ignore
                    return rewards
                except Exception as backup_error:
                    logger.error(f"Backup reward computation also failed: {backup_error}")
                    
            # Fallback: return neutral rewards
            logger.warning("Using fallback neutral rewards")
            return [0.5] * len(rollout_data)
            
    async def _get_healthy_worker(self, exclude_last: bool = False) -> Optional[ray.ObjectRef]:
        """Get a healthy worker using round-robin selection."""
        attempts = 0
        _ = self.current_worker_index
        
        while attempts < len(self.reward_workers):
            if exclude_last and attempts == 0:
                self.current_worker_index = (self.current_worker_index + 1) % len(self.reward_workers)
                
            worker_name = f"reward_worker_{self.current_worker_index}"
            health = self.worker_health.get(worker_name, {})
            
            if health.get('status') == 'healthy':
                worker = self.reward_workers[self.current_worker_index]
                self.current_worker_index = (self.current_worker_index + 1) % len(self.reward_workers)
                return worker
                
            self.current_worker_index = (self.current_worker_index + 1) % len(self.reward_workers)
            attempts += 1
            
        return None
        
    async def connect_environment(self, environment_ref: ray.ObjectRef) -> None:
        """Connect with environment actors (placeholder for future integration)."""
        logger.info("RewardGroup connected with environment")
        
    async def get_group_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the reward group."""
        healthy_workers = sum(1 for h in self.worker_health.values() if h.get('status') == 'healthy')
        
        status = {
            'total_workers': len(self.reward_workers),
            'healthy_workers': healthy_workers,
            'platform_optimized': self.platform_optimized,
            'worker_health': self.worker_health.copy(),
            'restart_counts': self.restart_counts.copy(),
            'is_initialized': self.is_initialized,
            'timestamp': time.time()
        }
        
        return status
        
    async def scale_workers(self, new_worker_count: int) -> None:
        """Dynamically scale the number of workers."""
        if new_worker_count == len(self.reward_workers):
            return
            
        logger.info(f"Scaling RewardGroup from {len(self.reward_workers)} to {new_worker_count} workers")
        
        if new_worker_count > len(self.reward_workers):
            # Scale up - extend the list first, then create workers
            current_count = len(self.reward_workers)
            # Extend list to new size
            self.reward_workers.extend([None] * (new_worker_count - current_count))  # type: ignore  
            
            for worker_id in range(current_count, new_worker_count):
                await self._create_single_worker(worker_id)  # This sets self.reward_workers[worker_id]
                
        else:
            # Scale down
            workers_to_remove = self.reward_workers[new_worker_count:]
            for worker in workers_to_remove:
                if worker is not None:
                    ray.kill(worker)
                    
            self.reward_workers = self.reward_workers[:new_worker_count]
            
            # Clean up health tracking for removed workers
            for worker_id in range(new_worker_count, len(self.worker_health)):
                worker_name = f"reward_worker_{worker_id}"
                if worker_name in self.worker_health:
                    del self.worker_health[worker_name]
                if worker_name in self.restart_counts:
                    del self.restart_counts[worker_name]
                    
        self.num_workers = new_worker_count
        logger.info(f"RewardGroup scaling complete: {len(self.reward_workers)} workers")
        
    async def shutdown(self) -> None:
        """Gracefully shutdown all reward workers."""
        logger.info("Shutting down RewardGroup...")
        
        for worker in self.reward_workers:
            if worker is not None:
                try:
                    ray.kill(worker)
                except Exception as e:
                    logger.warning(f"Error killing reward worker: {e}")
                    
        self.reward_workers.clear()
        self.worker_health.clear()
        
        logger.info("RewardGroup shutdown complete")