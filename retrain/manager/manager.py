# manager/manager.py

import asyncio
import logging
import time
from typing import Dict, Optional, Any, List
import ray
from ray.util.placement_group import PlacementGroup, placement_group

from ..config_models import TrainingConfig
from ..hardware import HardwareDetector, ActorFactory

logger = logging.getLogger(__name__)

class ReManager:
    """
    Central orchestrator for the Retrain distributed training system.
    
    ReManager is the main entry point that:
    1. Detects hardware and initializes optimal Ray configuration
    2. Spawns actor groups (trainer, inference, reward, environment, verifier)  
    3. Creates and manages the DataBuffer actor
    4. Coordinates the training loop with smart resource allocation
    5. Handles health monitoring and performance optimization
    
    This follows the Ray-first architecture with smart hardware adaptation.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the ReManager with training configuration.
        
        Args:
            config: Complete training configuration including model, algorithm,
                   environment, and all other training parameters.
        """
        self.config = config
        self.actor_groups: Dict[str, Any] = {}
        self.placement_groups: Dict[str, PlacementGroup] = {}
        self.databuffer: Optional[ray.ObjectRef] = None
        self.training_active = False
        
        # Hardware detection and optimization
        self.hardware_detector = HardwareDetector()
        self.actor_factory = ActorFactory(self.hardware_detector)
        
        # Training state
        self.current_episode = 0
        self.training_metrics = {}
        
        logger.info(f"ReManager initialized with config: {config.experiment_name}")
        self.hardware_detector.print_summary()
        
    async def initialize(self) -> None:
        """
        Initialize Ray and all distributed components with hardware-optimized configuration.
        """
        logger.info("Initializing Ray and distributed components...")
        
        # Initialize Ray with hardware-optimized configuration
        await self._initialize_ray()
        
        # Create placement groups for resource isolation
        await self._create_placement_groups()
        
        # Initialize DataBuffer first (other actors depend on it)
        await self._initialize_databuffer()
        
        # Initialize all actor groups
        await self._initialize_actor_groups()
        
        # Setup inter-actor connections
        await self._setup_actor_connections()
        
        logger.info("ReManager initialization complete")
        
    async def _initialize_ray(self) -> None:
        """Initialize Ray with optimal configuration for detected hardware."""
        if not ray.is_initialized():
            ray_config = self.actor_factory.get_ray_init_config()
            
            logger.info(f"Initializing Ray with configuration: {ray_config}")
            
            ray.init(
                log_to_driver=ray_config.get('log_to_driver', True),
                logging_level=ray_config.get('logging_level', logging.INFO),
                object_store_memory=ray_config.get('object_store_memory', 2_000_000_000),
                num_cpus=ray_config.get('num_cpus'),
                num_gpus=ray_config.get('num_gpus'),
                local_mode=ray_config.get('local_mode', False),
            )
            
        # Analyze available resources after Ray initialization
        self._analyze_available_resources()
        
    def _analyze_available_resources(self) -> None:
        """Analyze available cluster resources for optimal actor placement."""
        try:
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            self.resource_stats = {
                'total_cpus': cluster_resources.get('CPU', 0),
                'total_gpus': cluster_resources.get('GPU', 0),
                'available_cpus': available_resources.get('CPU', 0),
                'available_gpus': available_resources.get('GPU', 0),
                'memory': cluster_resources.get('memory', 0),
                'object_store_memory': cluster_resources.get('object_store_memory', 0),
            }
            
            logger.info(f"Ray cluster resources: {self.resource_stats}")
            
        except Exception as e:
            logger.warning(f"Failed to analyze Ray resources: {e}")
            # Fallback to hardware detector recommendations
            self.resource_stats = {
                'total_cpus': self.hardware_detector.capabilities['cpu']['cpu_count'],
                'total_gpus': self.hardware_detector.capabilities['device']['device_count'],
                'available_cpus': self.hardware_detector.capabilities['cpu']['cpu_count'],
                'available_gpus': self.hardware_detector.capabilities['device']['device_count'],
                'memory': int(self.hardware_detector.capabilities['memory']['system_memory_gb'] * 1e9),
                'object_store_memory': int(self.hardware_detector.capabilities['memory']['system_memory_gb'] * 0.3 * 1e9),
            }
        
    async def _create_placement_groups(self) -> None:
        """
        Create placement groups using hardware-optimized bundles.
        """
        logger.info("Creating placement groups for actor isolation...")
        
        # Get optimal placement group bundles from actor factory
        bundles = self.actor_factory.get_placement_group_bundles()
        
        # Create placement groups with proper float types for Ray
        self.placement_groups = {}
        
        for actor_type, actor_bundles in bundles.items():
            try:
                # Convert to proper Ray bundle format
                ray_bundles: List[Dict[str, float]] = []
                for bundle in actor_bundles:
                    ray_bundle = {}
                    for resource, amount in bundle.items():
                        ray_bundle[resource] = float(amount)
                    ray_bundles.append(ray_bundle)
                
                # Determine strategy based on hardware
                strategy = "STRICT_PACK" if self.hardware_detector.capabilities['device']['device_count'] > 1 else "SPREAD"
                
                self.placement_groups[actor_type] = placement_group(
                    ray_bundles, 
                    strategy=strategy
                )
                
            except Exception as e:
                logger.warning(f"Failed to create placement group for {actor_type}: {e}")
                # Create minimal fallback placement group
                self.placement_groups[actor_type] = placement_group(
                    [{"CPU": 1.0}], 
                    strategy="SPREAD"
                )
        
        # Wait for placement groups to be ready
        for name, pg in self.placement_groups.items():
            try:
                await pg.ready()
                logger.info(f"Placement group '{name}' ready")
            except Exception as e:
                logger.warning(f"Placement group '{name}' failed to initialize: {e}")
            
    async def _initialize_databuffer(self) -> None:
        """Initialize the DataBuffer actor using the actor factory."""
        logger.info("Initializing DataBuffer actor...")
        
        try:
            self.databuffer = self.actor_factory.create_databuffer_actor(self.config)
            
            # Wait for databuffer to initialize
            await self.databuffer.initialize.remote()  # type: ignore
            logger.info("DataBuffer actor ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize DataBuffer: {e}")
            # Create basic databuffer without placement group
            from .databuffer import ReDataBuffer
            databuffer_config = self.hardware_detector.get_actor_config('databuffer')
            self.databuffer = ReDataBuffer.options(**databuffer_config).remote(self.config)  # type: ignore
            await self.databuffer.initialize.remote()  # type: ignore
            logger.info("DataBuffer actor ready (fallback mode)")
        
    async def _initialize_actor_groups(self) -> None:
        """Initialize all 4  actor groups using the smart actor factory."""
        logger.info("Initializing all 4  actor groups...")
        
        try:
            # Create databuffer first (required by all groups)
            self.databuffer = self.actor_factory.create_databuffer_actor(self.config)
            
            # Create ALL 4  GROUPS with smart hardware allocation
            
            # 1. Create  trainer group (distributed training across multiple GPUs)
            try:
                if self.hardware_detector.recommendations['deployment_type'] == 'development':
                    num_trainer_workers = 1  # Single worker for macOS
                else:
                    gpu_count = self.hardware_detector.capabilities['device']['gpu_count']
                    num_trainer_workers = min(4, max(1, gpu_count))  # One worker per GPU up to 4
                    
                self.actor_groups['trainer_group'] = self.actor_factory.create_trainer_group(
                    self.config, self.databuffer, num_trainer_workers
                )
                logger.info(f"Created  trainer group with {num_trainer_workers} workers")
                
            except Exception as e:
                logger.error(f"Failed to create trainer group: {e}")
                # Fallback to single trainer actor
                try:
                    self.actor_groups['trainer'] = self.actor_factory.create_trainer_actor(self.config)
                    logger.info("Using fallback single trainer actor")
                except Exception as fallback_error:
                    logger.warning(f"Trainer actor creation failed: {fallback_error}")

            # 2. Create  inference group (mixed CPU/GPU workers with load balancing)
            try:
                if self.hardware_detector.recommendations['deployment_type'] == 'development':
                    num_inference_workers = 2  # macOS: 2 workers for load balancing
                else:
                    # Mixed hardware: GPU workers + CPU workers for optimal load distribution
                    gpu_count = self.hardware_detector.capabilities['device']['gpu_count']
                    cpu_count = self.hardware_detector.capabilities['cpu']['cpu_count']
                    num_inference_workers = min(8, max(2, gpu_count + cpu_count // 4))
                    
                self.actor_groups['inference_group'] = self.actor_factory.create_inference_group(
                    self.config, self.databuffer, num_inference_workers
                )
                logger.info(f"Created  inference group with {num_inference_workers} workers")
                
            except Exception as e:
                logger.error(f"Failed to create inference group: {e}")
                # Fallback to single inference actor
                try:
                    self.actor_groups['inference'] = self.actor_factory.create_inference_actor(
                        self.config, self.databuffer
                    )
                    logger.info("Using fallback single inference actor")
                except Exception as fallback_error:
                    logger.warning(f"Inference actor creation failed: {fallback_error}")
            
            # 3. Create  reward group (multiple workers with auto-restart)
            try:
                if self.hardware_detector.recommendations['deployment_type'] == 'development':
                    num_reward_workers = 2  # Conservative for macOS
                else:
                    num_reward_workers = min(4, self.hardware_detector.capabilities['cpu']['cpu_count'] // 2)
                    
                self.actor_groups['reward_group'] = self.actor_factory.create_reward_group(
                    self.config, self.databuffer, num_reward_workers
                )
                logger.info(f"Created  reward group with {num_reward_workers} workers")
                
            except Exception as e:
                logger.error(f"Failed to create reward group: {e}")
                # Fallback to single reward actor
                try:
                    self.actor_groups['reward'] = self.actor_factory.create_reward_actor(
                        self.config, self.databuffer
                    )
                    logger.info("Using fallback single reward actor")
                except Exception as fallback_error:
                    logger.warning(f"Reward actor creation failed: {fallback_error}")
                    
            # 4. Create  verifier group (multiple workers with auto-restart)
            try:
                if self.hardware_detector.recommendations['deployment_type'] == 'development':
                    num_verifier_workers = 2  # Conservative for macOS
                else:
                    num_verifier_workers = min(3, self.hardware_detector.capabilities['cpu']['cpu_count'] // 3)
                    
                self.actor_groups['verifier_group'] = self.actor_factory.create_verifier_group(
                    self.config, self.databuffer, num_verifier_workers
                )
                logger.info(f"Created  verifier group with {num_verifier_workers} workers")
                
            except Exception as e:
                logger.error(f"Failed to create verifier group: {e}")
                # Fallback to single verifier actor
                try:
                    self.actor_groups['verifier'] = self.actor_factory.create_verifier_actor(
                        self.config, self.databuffer
                    )
                    logger.info("Using fallback single verifier actor")
                except Exception as fallback_error:
                    logger.warning(f"Verifier actor creation failed: {fallback_error}")
                    
            # Create environment actor (not  yet, but connected to all groups)
            try:
                self.actor_groups['environment'] = self.actor_factory.create_environment_actor(
                    self.config, self.databuffer
                )
            except ImportError:
                logger.warning("Environment actor creation failed, using placeholder")
          
            # Initialize ALL  GROUPS in parallel
            init_futures = []
            
            # Core actors that need initialization
            core_actors = ['trainer', 'inference', 'environment']
            for name in core_actors:
                if name in self.actor_groups and hasattr(self.actor_groups[name], 'initialize'):
                    init_futures.append(self.actor_groups[name].initialize.remote())
                     
            #  groups that need initialization
            _groups = ['trainer_group', 'inference_group', 'reward_group', 'verifier_group']
            for group_name in _groups:
                if group_name in self.actor_groups:
                    init_futures.append(self.actor_groups[group_name].initialize.remote())
                     
            if init_futures:
                await asyncio.gather(*init_futures)
                 
            logger.info("All 4  actor groups initialized successfully")
             
        except Exception as e:
            logger.error(f"Failed to initialize  actor groups: {e}")
            raise
        
    async def _setup_actor_connections(self) -> None:
        """Setup connections between different actor groups."""
        logger.info("Setting up inter-actor connections...")
        
        try:
            # Connect inference actors to trainer for weight updates
            if 'inference' in self.actor_groups and 'trainer' in self.actor_groups:
                await self.actor_groups['inference'].connect_trainer.remote(
                    self.actor_groups['trainer']
                )
                
            # Connect reward actors to environment (if both exist)
            if 'reward' in self.actor_groups and 'environment' in self.actor_groups:
                await self.actor_groups['reward'].connect_environment.remote(
                    self.actor_groups['environment']
                )
                
            # Connect verifiers to reward calculation (if both exist)
            if 'verifier' in self.actor_groups and 'reward' in self.actor_groups:
                await self.actor_groups['verifier'].connect_reward.remote(
                    self.actor_groups['reward']
                )
                
            logger.info("Inter-actor connections established")
            
        except Exception as e:
            logger.warning(f"Some inter-actor connections failed: {e}")
            # Continue without all connections - core functionality should still work
        
    async def train(self) -> Dict[str, Any]:
        """
        Execute the main training loop with all distributed components.
        
        Returns:
            Final training metrics and results.
        """
        logger.info(f"Starting training for {self.config.num_episodes} episodes")
        self.training_active = True
        
        try:
            # Training loop
            for episode in range(self.config.num_episodes):
                self.current_episode = episode
                
                logger.info(f"Starting episode {episode + 1}/{self.config.num_episodes}")
                
                # Execute single training episode
                episode_metrics = await self._execute_episode(episode)
                
                # Update training metrics
                self.training_metrics[f'episode_{episode}'] = episode_metrics
                
                # Health check and resource monitoring
                await self._health_check()
                
                # Optional: Save checkpoint periodically
                if (episode + 1) % 10 == 0:
                    await self._save_checkpoint(episode)
                    
            # Final results
            final_metrics = await self._finalize_training()
            
            logger.info("Training completed successfully")
            return final_metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            await self._handle_training_failure(e)
            raise
        finally:
            self.training_active = False
            
    async def _execute_episode(self, episode: int) -> Dict[str, Any]:
        """
        Execute a single training episode with  actor groups.
        
        Args:
            episode: Current episode number
            
        Returns:
            Metrics from this episode
        """
        episode_start_time = time.time()
        
        try:
            # Step 1: Generate rollouts (parallel inference)
            rollout_future = self.actor_groups['inference'].generate_rollouts.remote(
                episode_id=episode,
                batch_size=getattr(self.config, 'batch_size', 4)
            )
            
            # Step 2: Prepare environment (if available)
            env_future = None
            if 'environment' in self.actor_groups:
                env_future = self.actor_groups['environment'].prepare_episode.remote(episode)
            
            # Wait for rollouts to complete
            rollout_data = await rollout_future
            if env_future:
                await env_future
            
            # Step 3: Run verifiers using  verifier group
            verification_results = None
            if 'verifier_group' in self.actor_groups:
                try:
                    verification_future = self.actor_groups['verifier_group'].verify_rollouts.remote(
                        rollout_data=rollout_data,
                        episode_id=episode
                    )
                    verification_results = await verification_future
                    logger.info("Verification completed using  verifier group")
                except Exception as e:
                    logger.warning(f" verifier group failed: {e}")
                    # Fallback to single verifier if available
                    if 'verifier' in self.actor_groups:
                        verification_results = await self.actor_groups['verifier'].verify_rollouts.remote(
                            rollout_data=rollout_data,
                            episode_id=episode
                        )
            elif 'verifier' in self.actor_groups:
                verification_future = self.actor_groups['verifier'].verify_rollouts.remote(
                    rollout_data=rollout_data,
                    episode_id=episode
                )
                verification_results = await verification_future
            
            # Step 4: Compute rewards using  reward group
            rewards = None
            if 'reward_group' in self.actor_groups:
                try:
                    reward_future = self.actor_groups['reward_group'].compute_rewards.remote(
                        rollout_data=rollout_data,
                        episode_id=episode
                    )
                    rewards = await reward_future
                    logger.info("Rewards computed using  reward group")
                except Exception as e:
                    logger.warning(f" reward group failed: {e}")
                    # Fallback to single reward actor if available
                    if 'reward' in self.actor_groups:
                        rewards = await self.actor_groups['reward'].compute_rewards.remote(
                            rollout_data=rollout_data,
                            episode_id=episode
                        )
            elif 'reward' in self.actor_groups:
                reward_future = self.actor_groups['reward'].compute_rewards.remote(
                    rollout_data=rollout_data,
                    episode_id=episode
                )
                rewards = await reward_future
            else:
                # Fallback: simple random rewards for development
                rewards = [0.5] * len(rollout_data) if rollout_data else []
            
            # Step 5: Prepare training batch
            if hasattr(self.databuffer, 'prepare_training_batch'):
                training_batch = await self.databuffer.prepare_training_batch.remote(  # type: ignore
                    rollout_data=rollout_data,
                    rewards=rewards,
                    verification_results=verification_results,
                    episode_id=episode
                )
            else:
                # Simple fallback batch preparation
                training_batch = {
                    'input_ids': [[1, 2, 3, 4]] * len(rewards) if rewards else [[1, 2, 3, 4]],
                    'rewards': rewards or [0.5],
                    'episode_id': episode
                }
            
            # Step 6: Execute training step
            training_metrics = await self.actor_groups['trainer'].train_step.remote(
                training_batch=training_batch,
                episode_id=episode
            )
            
            # Step 7: Update model weights across inference actors
            await self.actor_groups['inference'].update_weights.remote(
                self.actor_groups['trainer']
            )
            
            episode_time = time.time() - episode_start_time
            
            episode_metrics = {
                'episode_time': episode_time,
                'training_metrics': training_metrics,
                'rollout_count': len(rollout_data) if rollout_data else 0,
                'verification_results': verification_results,
                'rewards_computed': len(rewards) if rewards else 0,
                'using__groups': {
                    'verifier_group': 'verifier_group' in self.actor_groups,
                    'reward_group': 'reward_group' in self.actor_groups
                },
                'timestamp': time.time()
            }
            
            logger.info(f"Episode {episode} completed in {episode_time:.2f}s with  groups")
            return episode_metrics
            
        except Exception as e:
            logger.error(f"Episode {episode} failed: {e}")
            # Return error metrics to continue training
            return {
                'episode_time': time.time() - episode_start_time,
                'error': str(e),
                'rollout_count': 0,
                'timestamp': time.time(),
                'using__groups': {
                    'verifier_group': 'verifier_group' in self.actor_groups,
                    'reward_group': 'reward_group' in self.actor_groups
                }
            }
        
    async def _health_check(self) -> None:
        """Monitor health of all actor groups including  groups."""
        try:
            health_futures = {}
            
            # Check individual actors
            for name, actor in self.actor_groups.items():
                if hasattr(actor, 'health_check'):
                    health_futures[name] = actor.health_check.remote()
                    
            # Check  group status
            if 'reward_group' in self.actor_groups:
                health_futures['reward_group_status'] = self.actor_groups['reward_group'].get_group_status.remote()
                
            if 'verifier_group' in self.actor_groups:
                health_futures['verifier_group_status'] = self.actor_groups['verifier_group'].get_group_status.remote()
                    
            # Add databuffer health check if available
            if hasattr(self.databuffer, 'health_check'):
                health_futures['databuffer'] = self.databuffer.health_check.remote()  # type: ignore
            
            # Collect health results with timeout
            health_results = {}
            for name, future in health_futures.items():
                try:
                    result = await asyncio.wait_for(future, timeout=10.0)
                    health_results[name] = result
                except asyncio.TimeoutError:
                    logger.warning(f"Health check timeout for {name}")
                    health_results[name] = {'status': 'timeout', 'timestamp': time.time()}
                except Exception as e:
                    logger.warning(f"Health check failed for {name}: {e}")
                    health_results[name] = {'status': 'error', 'error': str(e), 'timestamp': time.time()}
            
            # Log  group status
            for group_name in ['reward_group_status', 'verifier_group_status']:
                if group_name in health_results:
                    status = health_results[group_name]
                    if isinstance(status, dict):
                        healthy = status.get('healthy_workers', 0)
                        total = status.get('total_workers', 0)
                        group_type = group_name.replace('_status', '')
                        
                        if healthy == total and total > 0:
                            logger.info(f"{group_type}: All {total} workers healthy")
                        elif healthy > 0:
                            logger.warning(f"{group_type}: {healthy}/{total} workers healthy")
                        else:
                            logger.error(f"{group_type}: No healthy workers available!")
            
            # Check resource utilization
            try:
                current_resources = ray.available_resources()
                
                # Log resource warnings
                total_gpus = self.resource_stats.get('total_gpus', 0)
                total_cpus = self.resource_stats.get('total_cpus', 0)
                
                if total_gpus > 0 and current_resources.get('GPU', 0) < 0.1 * total_gpus:
                    logger.warning("Low GPU availability detected")
                    
                if current_resources.get('CPU', 0) < 0.2 * total_cpus:
                    logger.warning("Low CPU availability detected")
                    
            except Exception as e:
                logger.warning(f"Resource check failed: {e}")
                
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            
    async def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint."""
        try:
            checkpoint_path = f"{getattr(self.config, 'output_dir', './checkpoints')}/checkpoint_episode_{episode}"
            
            # Save trainer state
            if 'trainer' in self.actor_groups:
                await self.actor_groups['trainer'].save_checkpoint.remote(checkpoint_path)
                
            # Save databuffer state if available
            if hasattr(self.databuffer, 'save_state'):
                await self.databuffer.save_state.remote(checkpoint_path)  # type: ignore
                
            logger.info(f"Checkpoint saved at episode {episode}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
        
    async def _finalize_training(self) -> Dict[str, Any]:
        """Finalize training and collect final metrics."""
        try:
            # Save final model
            final_checkpoint = f"{getattr(self.config, 'output_dir', './checkpoints')}/final_model"
            
            if 'trainer' in self.actor_groups:
                await self.actor_groups['trainer'].save_checkpoint.remote(final_checkpoint)
            
            # Collect final metrics from all components
            final_metrics = {
                'total_episodes': getattr(self.config, 'num_episodes', 0),
                'experiment_name': getattr(self.config, 'experiment_name', 'unknown'),
                'training_metrics': self.training_metrics,
                'resource_stats': getattr(self, 'resource_stats', {}),
                'final_checkpoint': final_checkpoint,
                'hardware_summary': self.hardware_detector.capabilities['summary'],
                'deployment_type': self.hardware_detector.recommendations['deployment_type']
            }
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"Failed to finalize training: {e}")
            return {
                'error': str(e),
                'training_metrics': self.training_metrics
            }
        
    async def _handle_training_failure(self, error: Exception) -> None:
        """Handle training failures and cleanup."""
        logger.error(f"Handling training failure: {error}")
        
        # Save emergency checkpoint
        try:
            await self._save_checkpoint(self.current_episode)
        except Exception as checkpoint_error:
            logger.error(f"Failed to save emergency checkpoint: {checkpoint_error}")
            
    async def shutdown(self) -> None:
        """Gracefully shutdown all distributed components."""
        logger.info("Shutting down ReManager and all actors...")
        
        try:
            if self.databuffer and hasattr(self.databuffer, 'shutdown'):
                await self.databuffer.shutdown.remote()  # type: ignore
                
            for name, actor in self.actor_groups.items():
                try:
                    if hasattr(actor, 'shutdown'):
                        await actor.shutdown.remote()
                    logger.info(f"Actor group '{name}' shutdown successfully")
                except Exception as e:
                    logger.warning(f"Error shutting down '{name}': {e}")
                    
            # Cleanup placement groups
            for name, pg in self.placement_groups.items():
                try:
                    ray.util.remove_placement_group(pg)
                except Exception as e:
                    logger.warning(f"Error removing placement group '{name}': {e}")
                    
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
        logger.info("ReManager shutdown complete")
        
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and metrics."""
        return {
            'training_active': self.training_active,
            'current_episode': self.current_episode,
            'total_episodes': getattr(self.config, 'num_episodes', 0),
            'resource_stats': getattr(self, 'resource_stats', {}),
            'recent_metrics': list(self.training_metrics.values())[-5:] if self.training_metrics else [],
            'hardware_summary': self.hardware_detector.capabilities['summary'],
            'deployment_type': self.hardware_detector.recommendations['deployment_type'],
            'backend': self.hardware_detector.recommendations['backend']
        }
        
    def print_deployment_plan(self) -> None:
        """Print comprehensive deployment plan."""
        self.hardware_detector.print_summary()
        self.actor_factory.print_actor_plan()

async def get_comprehensive_status(self) -> Dict[str, Any]:
    """Get comprehensive training status including  groups."""
    base_status = self.get_training_status()
    
    # Add  group status
    _status = {}
    
    if 'reward_group' in self.actor_groups:
        try:
            reward_status = await self.actor_groups['reward_group'].get_group_status.remote()
            _status['reward_group'] = reward_status
        except Exception as e:
            _status['reward_group'] = {'error': str(e)}
            
    if 'verifier_group' in self.actor_groups:
        try:
            verifier_status = await self.actor_groups['verifier_group'].get_group_status.remote()
            _status['verifier_group'] = verifier_status
        except Exception as e:
            _status['verifier_group'] = {'error': str(e)}
    
    base_status['_groups'] = _status
    base_status['macOS_optimized'] = self.hardware_detector.capabilities['platform']['is_macos']
    
    return base_status

async def scale__groups(self, reward_workers: Optional[int] = None, verifier_workers: Optional[int] = None) -> None:
    """Dynamically scale  actor groups."""
    logger.info("Scaling  actor groups...")
    
    if reward_workers and 'reward_group' in self.actor_groups:
        try:
            await self.actor_groups['reward_group'].scale_workers.remote(reward_workers)
            logger.info(f"Scaled reward group to {reward_workers} workers")
        except Exception as e:
            logger.error(f"Failed to scale reward group: {e}")
            
    if verifier_workers and 'verifier_group' in self.actor_groups:
        try:
            await self.actor_groups['verifier_group'].scale_workers.remote(verifier_workers)
            logger.info(f"Scaled verifier group to {verifier_workers} workers")
        except Exception as e:
            logger.error(f"Failed to scale verifier group: {e}")
