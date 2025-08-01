"""
Smart Actor Factory for hardware-specific actor creation.

Creates optimal Ray actors based on detected hardware capabilities,
ensuring high performance across different platforms.
"""

import logging
from typing import Dict, Any, Optional
import ray

from .detector import HardwareDetector

logger = logging.getLogger(__name__)


class ActorFactory:
    """
    Smart factory for creating hardware-optimized Ray actors.
    
    Automatically selects appropriate actor implementations and
    resource configurations based on detected hardware.
    """
    
    def __init__(self, hardware_detector: HardwareDetector):
        """Initialize factory with hardware detection results."""
        self.detector = hardware_detector
        self.capabilities = hardware_detector.capabilities
        self.recommendations = hardware_detector.recommendations
        
        logger.info(f"ActorFactory initialized for {self.recommendations['deployment_type']} deployment")
        
    def create_trainer_actor(self, config: Any) -> ray.ObjectRef:
        """Create optimal trainer actor for detected hardware."""
        actor_config = self.detector.get_actor_config('trainer')
        deployment_type = self.recommendations['deployment_type']
        
        if deployment_type == 'development':
            # macOS/CPU development setup
            return self._create_development_trainer(config, actor_config)
        elif deployment_type == 'production':
            # Multi-GPU production setup
            return self._create_production_trainer(config, actor_config)
        else:
            # Single GPU or CPU setup
            return self._create_standard_trainer(config, actor_config)
            
    def create_inference_actor(self, config: Any, databuffer: ray.ObjectRef) -> ray.ObjectRef:
        """Create optimal inference actor for detected hardware."""
        actor_config = self.detector.get_actor_config('inference')
        deployment_type = self.recommendations['deployment_type']
        
        if deployment_type == 'development':
            # CPU inference for stable development
            return self._create_development_inference(config, databuffer, actor_config)
        else:
            # GPU inference for production
            return self._create_production_inference(config, databuffer, actor_config)
            
    def create_databuffer_actor(self, config: Any) -> ray.ObjectRef:
        """Create databuffer actor with appropriate resources."""
        actor_config = self.detector.get_actor_config('databuffer')
        
        from ..manager.databuffer import ReDataBuffer
        return ReDataBuffer.options(**actor_config).remote(config)  # type: ignore
        
    def create_reward_actor(self, config: Any, databuffer: ray.ObjectRef) -> ray.ObjectRef:
        """Create reward calculation actor."""
        actor_config = self.detector.get_actor_config('reward')
        
        from ..reward.reward import ReReward
        return ReReward.options(**actor_config).remote(config, databuffer)  # type: ignore
        
    def create_environment_actor(self, config: Any, databuffer: ray.ObjectRef) -> ray.ObjectRef:
        """Create environment actor for rollout execution."""
        actor_config = self.detector.get_actor_config('environment')
        
        from ..environment.environment import ReEnvironment
        return ReEnvironment.options(**actor_config).remote(config, databuffer)  # type: ignore
        
    def create_verifier_actor(self, config: Any, databuffer: ray.ObjectRef) -> ray.ObjectRef:
        """Create verifier actor for result validation."""
        actor_config = self.detector.get_actor_config('verifier')
        
        from ..verifier.verifier import ReVerifier
        return ReVerifier.options(**actor_config).remote(config, databuffer)  # type: ignore
        
    def create_reward_group(self, config: Any, databuffer: ray.ObjectRef, num_workers: Optional[int] = None) -> ray.ObjectRef:
        """Create  reward group with multiple workers."""
        if num_workers is None:
            # Auto-determine number of workers based on hardware
            ray_caps = self.detector.capabilities['ray']
            num_workers = ray_caps['optimal_actor_count']['reward']
            
        from ..manager.reward_group import RewardGroup
        
        if self.recommendations['deployment_type'] == 'development':
            # macOS: fewer workers for stability
            assert num_workers is not None
            num_workers = min(num_workers, 2)
            
        return RewardGroup.remote(config, databuffer, num_workers)  # type: ignore
        
    def create_verifier_group(self, config: Any, databuffer: ray.ObjectRef, num_workers: Optional[int] = None) -> ray.ObjectRef:
        """Create  verifier group with multiple workers."""
        if num_workers is None:
            # Auto-determine number of workers based on hardware
            ray_caps = self.detector.capabilities['ray']
            num_workers = ray_caps['optimal_actor_count']['verifier']
            
        from ..manager.verifier_group import VerifierGroup
        
        if self.recommendations['deployment_type'] == 'development':
            # macOS: fewer workers for stability
            assert num_workers is not None
            num_workers = min(num_workers, 2)
            
        return VerifierGroup.remote(config, databuffer, num_workers)  # type: ignore
        
    def create_trainer_group(self, config: Any, databuffer: ray.ObjectRef, num_workers: Optional[int] = None, 
                           placement_group: Any = None) -> ray.ObjectRef:
        """Create  trainer group with multiple workers."""
        if num_workers is None:
            # Auto-determine number of workers based on hardware
            gpu_count = self.detector.capabilities['device']['gpu_count']
            if self.recommendations['deployment_type'] == 'development':
                # macOS: single worker
                num_workers = 1
            else:
                # Multi-GPU: one worker per GPU up to 4
                num_workers = min(gpu_count, 4) if gpu_count > 0 else 1
                
        from ..manager.trainer_group import TrainerGroup
        
        return TrainerGroup.remote(config, databuffer, num_workers, placement_group)  # type: ignore
        
    def create_inference_group(self, config: Any, databuffer: ray.ObjectRef, num_workers: Optional[int] = None,
                             placement_group: Any = None) -> ray.ObjectRef:
        """Create  inference group with multiple workers and mixed hardware."""
        if num_workers is None:
            # Auto-determine number of workers based on hardware
            if self.recommendations['deployment_type'] == 'development':
                # macOS: 2 workers for load balancing
                num_workers = 2
            else:
                # Multi-GPU: more workers for parallel rollout generation
                gpu_count = self.detector.capabilities['device']['gpu_count']
                cpu_count = self.detector.capabilities['cpu']['cpu_count']
                # Mix of GPU and CPU workers
                num_workers = min(8, max(2, gpu_count + cpu_count // 4))
                
        from ..manager.inference_group import InferenceGroup
        
        return InferenceGroup.remote(config, databuffer, num_workers, placement_group)  # type: ignore
        
    def _create_development_trainer(self, config: Any, actor_config: Dict[str, Any]) -> ray.ObjectRef:
        """Create development trainer for macOS/CPU environments."""
        from ..trainer.trainer import ReTrainer
        # ReTrainer needs databuffer, so we create it first if needed
        databuffer = self.create_databuffer_actor(config)
        return ReTrainer.options(**actor_config).remote(config, databuffer)  # type: ignore
        
    def _create_production_trainer(self, config: Any, actor_config: Dict[str, Any]) -> ray.ObjectRef:
        """Create production trainer with MBridge backend."""
        from ..trainer.trainer import ReTrainer
        # ReTrainer needs databuffer, so we create it first if needed  
        databuffer = self.create_databuffer_actor(config)
        return ReTrainer.options(**actor_config).remote(config, databuffer)  # type: ignore
        
    def _create_standard_trainer(self, config: Any, actor_config: Dict[str, Any]) -> ray.ObjectRef:
        """Create standard trainer for single GPU setups."""
        from ..trainer.trainer import ReTrainer
        # ReTrainer needs databuffer, so we create it first if needed
        databuffer = self.create_databuffer_actor(config)
        return ReTrainer.options(**actor_config).remote(config, databuffer)  # type: ignore
        
    def _create_development_inference(self, config: Any, databuffer: ray.ObjectRef, 
                                    actor_config: Dict[str, Any]) -> ray.ObjectRef:
        """Create CPU inference actor for development."""
        from ..inference import ReInference
        return ReInference.options(**actor_config).remote(config, databuffer)  # type: ignore
        
    def _create_production_inference(self, config: Any, databuffer: ray.ObjectRef,
                                   actor_config: Dict[str, Any]) -> ray.ObjectRef:
        """Create GPU inference actor for production."""
        from ..inference import ReInference
        return ReInference.options(**actor_config).remote(config, databuffer)  # type: ignore
        
    def get_placement_group_bundles(self) -> Dict[str, list]:
        """Get optimal placement group bundles for all actor types."""
        actor_resources = self.recommendations['actor_resources']
        
        # Convert actor resources to placement group bundles
        bundles = {}
        
        for actor_type, resources in actor_resources.items():
            bundle = {}
            if resources['num_gpus'] > 0:
                bundle['GPU'] = float(resources['num_gpus'])
            bundle['CPU'] = float(resources['num_cpus'])
            
            bundles[actor_type] = [bundle]
            
        return bundles
        
    def get_ray_init_config(self) -> Dict[str, Any]:
        """Get optimal Ray initialization configuration."""
        config = self.detector.get_ray_init_config()
        
        # Remove runtime_env entirely - let Ray use the local installation
        # This avoids packaging issues and uses the installed retrain package
        
        # Add logging configuration for different deployment types
        if self.recommendations['deployment_type'] == 'development':
            config.update({
                'log_to_driver': True,
                'logging_level': logging.INFO,
                'local_mode': False,  # Keep distributed for testing
            })
        else:
            config.update({
                'log_to_driver': False,
                'logging_level': logging.WARNING,
                'local_mode': False,
            })
            
        return config
        
    def print_actor_plan(self) -> None:
        """Print the planned actor deployment strategy."""
        print(f"\n🏗️  Actor Deployment Plan ({self.recommendations['deployment_type']})")
        print("=" * 60)
        
        actor_resources = self.recommendations['actor_resources']
        
        for actor_type, resources in actor_resources.items():
            gpu_str = f"{resources['num_gpus']} GPU" if resources['num_gpus'] > 0 else "CPU"
            print(f"  📦 {actor_type.title()}: {gpu_str}, {resources['num_cpus']} CPU cores")
            
        total_gpus = sum(r['num_gpus'] for r in actor_resources.values())
        total_cpus = sum(r['num_cpus'] for r in actor_resources.values())
        
        print("\n📊 Total Resources Required:")
        print(f"  • GPUs: {total_gpus}")
        print(f"  • CPU Cores: {total_cpus}")
        print(f"  • Memory: {self.recommendations['ray_config']['object_store_memory'] / 1e9:.1f}GB object store")
        
        available_gpus = self.capabilities['device']['device_count']
        available_cpus = self.capabilities['cpu']['cpu_count']
        
        if total_gpus > available_gpus:
            print(f"  ⚠️  Warning: Requires {total_gpus} GPUs but only {available_gpus} available")
        if total_cpus > available_cpus:
            print(f"  ⚠️  Warning: Requires {total_cpus} CPU cores but only {available_cpus} available")
            
        print(f"\n🎯 Backend: {self.recommendations['backend']}")
        print(f"🎯 Model Size: {self.recommendations['model_size_class']}")
        
        if 'performance_tips' in self.recommendations:
            print("\n💡 Performance Tips:")
            for tip in self.recommendations['performance_tips']:
                print(f"  • {tip}")


class PerformanceTuner:
    """
    Performance tuning utilities for Ray actors.
    
    Provides runtime optimization suggestions and resource monitoring.
    """
    
    def __init__(self, hardware_detector: HardwareDetector):
        """Initialize performance tuner."""
        self.detector = hardware_detector
        self.capabilities = hardware_detector.capabilities
        
    def suggest_batch_size(self, model_size_class: str, device_type: str) -> int:
        """Suggest optimal batch size based on hardware and model."""
        if device_type == 'cpu':
            base_size = 2
        elif device_type == 'mps':
            base_size = 4
        elif device_type == 'cuda':
            gpu_memory = self.capabilities['device'].get('gpu_memory_gb', 8)
            if gpu_memory >= 24:
                base_size = 32
            elif gpu_memory >= 16:
                base_size = 16
            elif gpu_memory >= 8:
                base_size = 8
            else:
                base_size = 4
        else:
            base_size = 4
            
        # Adjust based on model size
        if model_size_class == 'large':
            return max(1, base_size // 4)
        elif model_size_class == 'medium':
            return max(1, base_size // 2)
        else:  # small
            return base_size
            
    def suggest_sequence_length(self, deployment_type: str, device_type: str) -> int:
        """Suggest optimal sequence length."""
        if deployment_type == 'development':
            return 512
        elif device_type == 'cpu':
            return 512
        elif device_type == 'mps':
            return 1024
        else:  # CUDA
            return 2048
            
    def get_memory_optimization_tips(self) -> list:
        """Get memory optimization recommendations."""
        tips = []
        
        device_type = self.capabilities['device']['primary_device']
        memory_gb = self.capabilities['memory']['system_memory_gb']
        
        if device_type == 'cpu' and memory_gb < 16:
            tips.extend([
                "Use gradient accumulation to reduce memory usage",
                "Consider model quantization (load_in_8bit=True)",
                "Reduce sequence length to 256-512 tokens"
            ])
            
        if device_type == 'mps':
            tips.extend([
                "Use mixed precision (fp16) for MPS acceleration",
                "Monitor unified memory usage",
                "Consider batch size 2-4 for optimal MPS utilization"
            ])
            
        if device_type == 'cuda':
            gpu_memory = self.capabilities['device'].get('gpu_memory_gb', 0)
            if gpu_memory < 12:
                tips.extend([
                    "Enable gradient checkpointing",
                    "Use DeepSpeed ZeRO for large models",
                    "Consider LoRA/QLoRA for parameter-efficient training"
                ])
                
        return tips
        
    def monitor_ray_performance(self) -> Dict[str, Any]:
        """Monitor Ray cluster performance metrics."""
        try:
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            utilization = {}
            for resource, total in cluster_resources.items():
                available = available_resources.get(resource, 0)
                utilization[resource] = {
                    'total': total,
                    'available': available,
                    'used': total - available,
                    'utilization_percent': ((total - available) / total * 100) if total > 0 else 0
                }
                
            return {
                'timestamp': ray.util.get_node_ip_address(),
                'cluster_utilization': utilization,
                'recommendations': self._generate_utilization_recommendations(utilization)
            }
            
        except Exception as e:
            logger.error(f"Failed to monitor Ray performance: {e}")
            return {}
            
    def _generate_utilization_recommendations(self, utilization: Dict[str, Any]) -> list:
        """Generate recommendations based on resource utilization."""
        recommendations = []
        
        gpu_util = utilization.get('GPU', {}).get('utilization_percent', 0)
        cpu_util = utilization.get('CPU', {}).get('utilization_percent', 0)
        
        if gpu_util > 90:
            recommendations.append("High GPU utilization - consider scaling out or reducing batch size")
        elif gpu_util < 30:
            recommendations.append("Low GPU utilization - consider increasing batch size or model complexity")
            
        if cpu_util > 85:
            recommendations.append("High CPU utilization - consider adding more CPU workers")
        elif cpu_util < 20:
            recommendations.append("Low CPU utilization - consider reducing CPU allocation")
            
        return recommendations 