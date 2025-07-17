# manager/trainer_group.py

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logger = logging.getLogger(__name__)

@ray.remote(num_cpus=1, num_gpus=0)
class TrainerGroup:
    """
    Trainer Actor Group Manager 
    
    Manages multiple training actor instances with:
    1. Distributed training coordination across multiple GPUs
    2. Automatic health monitoring and restart
    3. Weight synchronization between trainers
    4. Load balancing for parallel training steps
    5. Fault tolerance and recovery
    6. macOS CPU-only fallback support
    """
    
    def __init__(self, config: Any, databuffer: ray.ObjectRef, num_workers: int = 1, placement_group: Optional[PlacementGroup] = None):
        """
        Initialize TrainerGroup with configuration and worker management.
        
        Args:
            config: Training configuration object
            databuffer: Reference to the ReDataBuffer actor
            num_workers: Number of training worker instances to maintain
            placement_group: Optional Ray placement group for resource allocation
        """
        self.config = config
        self.databuffer = databuffer
        self.num_workers = num_workers
        self.placement_group = placement_group
        
        # Worker management
        self.trainer_workers: List[ray.ObjectRef] = []
        self.worker_health: Dict[str, Dict[str, Any]] = {}
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        
        # Training coordination
        self.current_worker_index = 0
        self.training_step_count = 0
        self.master_weights: Optional[Dict[str, Any]] = None
        
        # Health monitoring
        self.health_check_interval = 30.0  # seconds
        self.max_restart_attempts = 3
        self.restart_counts: Dict[str, int] = {}
        
        # Platform optimization
        self.platform_optimized = False
        self.is_initialized = False
        
        # Async coordination (following Slime pattern)
        self.pending_operations: Dict[str, List[ray.ObjectRef]] = {}
        
        logger.info(f"TrainerGroup initialized with {num_workers} workers")
        
    async def initialize(self) -> None:
        """Initialize training workers with resilience following Slime async pattern."""
        logger.info("Initializing TrainerGroup workers...")
        
        try:
            # Detect platform for optimization
            await self._detect_platform_optimizations()
            
            # Create worker pool using placement groups
            await self._create_worker_pool()
            
            # Initialize all workers in parallel (Slime async pattern)
            init_refs = await self._async_init_workers()
            await asyncio.gather(*[ref for ref in init_refs])
            
            # Start health monitoring
            asyncio.create_task(self._health_monitoring_loop())
            
            self.is_initialized = True
            logger.info(f"TrainerGroup initialization complete with {len(self.trainer_workers)} workers")
            
        except Exception as e:
            logger.error(f"Failed to initialize TrainerGroup: {e}")
            raise
            
    async def _detect_platform_optimizations(self) -> None:
        """Detect platform and apply optimizations."""
        try:
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                logger.info("Detected macOS - applying CPU-only training optimizations")
                self.num_workers = 1  # Single worker for macOS
                self.platform_optimized = True
            elif system == "Linux":
                logger.info("Detected Linux - multi-GPU training enabled")
            else:
                logger.info(f"Detected {system} - using standard configuration")
                
        except Exception as e:
            logger.warning(f"Platform detection failed: {e}")
            
    async def _create_worker_pool(self) -> None:
        """Create the initial pool of training workers with placement groups."""
        # Import ReTrainer class
        from retrain.trainer.trainer import ReTrainer
        
        for worker_id in range(self.num_workers):
            try:
                worker_name = f"trainer_worker_{worker_id}"
                
                # Create worker with placement group if available (following Slime pattern)
                if self.placement_group and not self.platform_optimized:
                    # Multi-GPU training with placement group
                    worker = ReTrainer.options(  # type: ignore
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=self.placement_group,
                            placement_group_bundle_index=worker_id,
                        ),
                        name=worker_name
                    ).remote(self.config, self.databuffer)
                else:
                    # Standard creation (single GPU or macOS)
                    worker = ReTrainer.remote(self.config, self.databuffer)  # type: ignore
                
                self.trainer_workers.append(worker)
                self.worker_health[worker_name] = {
                    'status': 'healthy',
                    'last_check': time.time(),
                    'start_time': time.time(),
                    'restart_count': 0
                }
                self.restart_counts[worker_name] = 0
                
                logger.info(f"Created trainer worker: {worker_name}")
                
            except Exception as e:
                logger.error(f"Failed to create trainer worker {worker_id}: {e}")
                
        if not self.trainer_workers:
            raise RuntimeError("Failed to create any trainer workers")
            
    async def _async_init_workers(self) -> List[ray.ObjectRef]:
        """Initialize all workers in parallel using Slime async pattern."""
        init_refs = []
        for worker in self.trainer_workers:
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
                logger.error(f"Trainer health monitoring error: {e}")
                
    async def _check_worker_health(self) -> None:
        """Check health of all training workers."""
        health_futures = {}
        
        for i, worker in enumerate(self.trainer_workers):
            if worker is not None:
                worker_name = f"trainer_worker_{i}"
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
                logger.warning(f"Trainer health check timeout for {worker_name}")
                self.worker_health[worker_name]['status'] = 'timeout'
            except Exception as e:
                logger.warning(f"Trainer health check failed for {worker_name}: {e}")
                self.worker_health[worker_name]['status'] = 'unhealthy'
                
    async def _restart_unhealthy_workers(self) -> None:
        """Restart training workers that are unhealthy."""
        for i, worker in enumerate(self.trainer_workers):
            worker_name = f"trainer_worker_{i}"
            health = self.worker_health.get(worker_name, {})
            
            if health.get('status') in ['unhealthy', 'timeout']:
                restart_count = self.restart_counts.get(worker_name, 0)
                
                if restart_count < self.max_restart_attempts:
                    logger.warning(f"Restarting unhealthy trainer worker: {worker_name} (attempt {restart_count + 1})")
                    
                    try:
                        # Kill the old worker
                        if worker is not None:
                            ray.kill(worker)
                            
                        # Create new worker
                        await self._create_single_worker(i)
                        
                        self.restart_counts[worker_name] = restart_count + 1
                        logger.info(f"Successfully restarted trainer {worker_name}")
                        
                    except Exception as e:
                        logger.error(f"Failed to restart trainer {worker_name}: {e}")
                        self.worker_health[worker_name]['status'] = 'failed'
                else:
                    logger.error(f"Trainer {worker_name} exceeded max restart attempts ({self.max_restart_attempts})")
                    self.worker_health[worker_name]['status'] = 'failed'
                    
    async def _create_single_worker(self, worker_id: int) -> None:
        """Create a single training worker to replace a failed one."""
        from retrain.trainer.trainer import ReTrainer
        
        worker_name = f"trainer_worker_{worker_id}"
        
        # Create replacement worker
        worker = ReTrainer.remote(self.config, self.databuffer)  # type: ignore
        await worker.initialize.remote()  # type: ignore
        
        self.trainer_workers[worker_id] = worker
        self.worker_health[worker_name] = {
            'status': 'healthy',
            'last_check': time.time(),
            'start_time': time.time(),
            'restart_count': self.restart_counts.get(worker_name, 0)
        }
        
    # Slime-style async training methods
    async def async_train_step(self, training_batch: Dict[str, Any], episode_id: int) -> List[ray.ObjectRef]:
        """
        Execute training step across all workers using Slime async pattern.
        
        Args:
            training_batch: Processed training data from DataBuffer
            episode_id: Current episode identifier
            
        Returns:
            List of ObjectRefs for training results from each worker
        """
        if not self.is_initialized:
            raise RuntimeError("TrainerGroup not initialized")
            
        # Execute training on all healthy workers in parallel (Slime pattern)
        train_refs = []
        for i, worker in enumerate(self.trainer_workers):
            worker_name = f"trainer_worker_{i}"
            health = self.worker_health.get(worker_name, {})
            
            if health.get('status') == 'healthy':
                train_refs.append(worker.train_step.remote(training_batch, episode_id))  # type: ignore
            else:
                logger.warning(f"Skipping unhealthy worker {worker_name} for training")
                
        if not train_refs:
            raise RuntimeError("No healthy training workers available")
            
        self.training_step_count += 1
        self.pending_operations[f'train_step_{episode_id}'] = train_refs
        
        logger.info(f"Started async training step {self.training_step_count} on {len(train_refs)} workers")
        return train_refs
        
    async def async_get_model_weights(self) -> List[ray.ObjectRef]:
        """Get model weights from all workers using Slime async pattern."""
        weight_refs = []
        for worker in self.trainer_workers:
            weight_refs.append(worker.get_model_weights.remote())  # type: ignore
        return weight_refs
        
    async def async_update_weights(self, weights: Dict[str, Any]) -> List[ray.ObjectRef]:
        """Update weights across all workers using Slime async pattern."""
        update_refs = []
        for worker in self.trainer_workers:
            update_refs.append(worker.update_model_weights.remote(weights))  # type: ignore
        return update_refs
        
    async def train_step_coordinated(self, training_batch: Dict[str, Any], episode_id: int) -> Dict[str, Any]:
        """
        Coordinated training step with weight synchronization.
        
        This follows a distributed training pattern where:
        1. All workers train on the same batch
        2. Results are aggregated
        3. Weights are synchronized across workers
        """
        if not self.is_initialized:
            raise RuntimeError("TrainerGroup not initialized")
            
        # Step 1: Execute training on all workers
        train_refs = await self.async_train_step(training_batch, episode_id)
        
        # Step 2: Wait for training completion and collect results
        training_results = await asyncio.gather(*train_refs)
        
        # Step 3: Aggregate metrics from all workers
        aggregated_metrics = self._aggregate_training_metrics(training_results)
        
        # Step 4: Get updated weights from master worker (rank 0)
        if len(self.trainer_workers) > 1:
            master_weights_ref = await self.async_get_model_weights()
            master_weights = await master_weights_ref[0]  # Use rank 0 as master
            
            # Step 5: Synchronize weights across all workers
            sync_refs = await self.async_update_weights(master_weights)
            await asyncio.gather(*sync_refs)
            
            self.master_weights = master_weights
            
        logger.info(f"Coordinated training step {self.training_step_count} completed")
        return aggregated_metrics
        
    def _aggregate_training_metrics(self, training_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate training metrics from multiple workers."""
        if not training_results:
            return {}
            
        # Average numerical metrics
        aggregated = {}
        numeric_keys = ['loss', 'policy_loss', 'value_loss', 'kl_loss', 'step_time']
        
        for key in numeric_keys:
            values = [result.get(key, 0.0) for result in training_results if key in result]
            if values:
                aggregated[f'avg_{key}'] = sum(values) / len(values)
                aggregated[f'min_{key}'] = min(values)
                aggregated[f'max_{key}'] = max(values)
                
        # Include metadata
        aggregated.update({
            'num_workers': len(training_results),
            'training_step': self.training_step_count,
            'episode_id': training_results[0].get('episode_id', 0),
            'timestamp': time.time()
        })
        
        return aggregated
        
    async def connect_inference(self, inference_group: ray.ObjectRef) -> None:
        """Connect with inference group for weight updates."""
        logger.info("TrainerGroup connected with inference group")
        self.connected_inference_group = inference_group
        
    async def get_group_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the training group."""
        healthy_workers = sum(1 for h in self.worker_health.values() if h.get('status') == 'healthy')
        
        status = {
            'total_workers': len(self.trainer_workers),
            'healthy_workers': healthy_workers,
            'platform_optimized': self.platform_optimized,
            'training_step_count': self.training_step_count,
            'worker_health': self.worker_health.copy(),
            'restart_counts': self.restart_counts.copy(),
            'is_initialized': self.is_initialized,
            'has_placement_group': self.placement_group is not None,
            'pending_operations': len(self.pending_operations),
            'timestamp': time.time()
        }
        
        return status
        
    async def scale_workers(self, new_worker_count: int) -> None:
        """Dynamically scale the number of training workers."""
        if new_worker_count == len(self.trainer_workers):
            return
            
        logger.info(f"Scaling TrainerGroup from {len(self.trainer_workers)} to {new_worker_count} workers")
        
        if new_worker_count > len(self.trainer_workers):
            # Scale up - extend the list first, then create workers
            current_count = len(self.trainer_workers)
            self.trainer_workers.extend([None] * (new_worker_count - current_count))  # type: ignore
            
            for worker_id in range(current_count, new_worker_count):
                await self._create_single_worker(worker_id)
                
        else:
            # Scale down
            workers_to_remove = self.trainer_workers[new_worker_count:]
            for worker in workers_to_remove:
                if worker is not None:
                    ray.kill(worker)
                    
            self.trainer_workers = self.trainer_workers[:new_worker_count]
            
            # Clean up health tracking for removed workers
            for worker_id in range(new_worker_count, len(self.worker_health)):
                worker_name = f"trainer_worker_{worker_id}"
                if worker_name in self.worker_health:
                    del self.worker_health[worker_name]
                if worker_name in self.restart_counts:
                    del self.restart_counts[worker_name]
                    
        self.num_workers = new_worker_count
        logger.info(f"TrainerGroup scaling complete: {len(self.trainer_workers)} workers")
        
    async def shutdown(self) -> None:
        """Gracefully shutdown all training workers."""
        logger.info("Shutting down TrainerGroup...")
        
        for worker in self.trainer_workers:
            if worker is not None:
                try:
                    ray.kill(worker)
                except Exception as e:
                    logger.warning(f"Error killing trainer worker: {e}")
                    
        self.trainer_workers.clear()
        self.worker_health.clear()
        self.pending_operations.clear()
        
        logger.info("TrainerGroup shutdown complete") 