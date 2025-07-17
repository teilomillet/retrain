"""
Resource allocation and management for Retrain distributed training.

Provides smart allocation of Ray resources across different actor types
based on hardware capabilities and workload requirements.
"""

import logging
from typing import Dict, Any, List
import ray

logger = logging.getLogger(__name__)


class ResourceAllocator:
    """
    Smart resource allocator for Ray actors.
    
    Manages optimal distribution of CPU, GPU, and memory resources
    across different actor types based on hardware capabilities.
    """
    
    def __init__(self, hardware_detector):
        """Initialize allocator with hardware detection results."""
        self.detector = hardware_detector
        self.capabilities = hardware_detector.capabilities
        self.recommendations = hardware_detector.recommendations
        
        # Resource tracking
        self.allocated_resources = {
            'GPU': 0.0,
            'CPU': 0.0,
            'memory': 0
        }
        
        self.actor_allocations = {}
        
    def allocate_for_actor_type(self, actor_type: str, count: int = 1) -> Dict[str, Any]:
        """
        Allocate optimal resources for a specific actor type.
        
        Args:
            actor_type: Type of actor ('trainer', 'inference', etc.)
            count: Number of actors of this type
            
        Returns:
            Resource allocation dictionary for Ray
        """
        base_allocation = self.recommendations['actor_resources'].get(
            actor_type, {'num_gpus': 0, 'num_cpus': 1}
        )
        
        # Scale allocation based on count
        allocation = {
            'num_gpus': base_allocation['num_gpus'],
            'num_cpus': base_allocation['num_cpus'],
            'memory': self._calculate_memory_per_actor(actor_type),
            'object_store_memory': self._calculate_object_store_memory(actor_type)
        }
        
        # Track allocation
        total_gpus = allocation['num_gpus'] * count
        total_cpus = allocation['num_cpus'] * count
        
        self.allocated_resources['GPU'] += total_gpus
        self.allocated_resources['CPU'] += total_cpus
        self.allocated_resources['memory'] += allocation['memory'] * count
        
        self.actor_allocations[actor_type] = {
            'count': count,
            'per_actor': allocation,
            'total': {
                'num_gpus': total_gpus,
                'num_cpus': total_cpus,
                'memory': allocation['memory'] * count
            }
        }
        
        logger.info(f"Allocated for {count}x {actor_type}: {allocation}")
        
        return allocation
        
    def _calculate_memory_per_actor(self, actor_type: str) -> int:
        """Calculate memory allocation per actor based on type and hardware."""
        total_memory = self.capabilities['memory']['system_memory_gb'] * 1e9
        
        memory_ratios = {
            'trainer': 0.3,      # 30% for training (model + gradients)
            'inference': 0.2,    # 20% for inference (model only)
            'databuffer': 0.15,  # 15% for data processing
            'environment': 0.05, # 5% for environment simulation
            'verifier': 0.05,    # 5% for verification
            'reward': 0.05       # 5% for reward computation
        }
        
        ratio = memory_ratios.get(actor_type, 0.05)
        return int(total_memory * ratio)
        
    def _calculate_object_store_memory(self, actor_type: str) -> int:
        """Calculate object store memory for inter-actor communication."""
        total_object_store = self.recommendations['ray_config']['object_store_memory']
        
        # Actors that need more object store for data sharing
        high_usage_actors = {'databuffer', 'trainer', 'inference'}
        
        if actor_type in high_usage_actors:
            return int(total_object_store * 0.2)  # 20% of object store
        else:
            return int(total_object_store * 0.05)  # 5% of object store
            
    def check_resource_availability(self) -> Dict[str, Any]:
        """Check if allocated resources fit within hardware limits."""
        available_gpus = self.capabilities['device']['device_count']
        available_cpus = self.capabilities['cpu']['cpu_count']
        available_memory = self.capabilities['memory']['system_memory_gb'] * 1e9
        
        status = {
            'gpu_ok': self.allocated_resources['GPU'] <= available_gpus,
            'cpu_ok': self.allocated_resources['CPU'] <= available_cpus,
            'memory_ok': self.allocated_resources['memory'] <= available_memory * 0.8,  # 80% limit
            'allocated': self.allocated_resources.copy(),
            'available': {
                'GPU': available_gpus,
                'CPU': available_cpus,
                'memory': available_memory
            },
            'utilization': {
                'GPU': (self.allocated_resources['GPU'] / max(available_gpus, 1)) * 100,
                'CPU': (self.allocated_resources['CPU'] / available_cpus) * 100,
                'memory': (self.allocated_resources['memory'] / available_memory) * 100
            }
        }
        
        status['overall_ok'] = status['gpu_ok'] and status['cpu_ok'] and status['memory_ok']
        
        return status
        
    def get_placement_group_strategy(self) -> str:
        """Get optimal placement group strategy based on resource allocation."""
        gpu_count = self.capabilities['device']['device_count']
        
        if gpu_count > 4:
            return "STRICT_PACK"  # Pack actors together for multi-GPU efficiency
        elif gpu_count > 1:
            return "PACK"  # Prefer packing but allow spread if needed
        else:
            return "SPREAD"  # Spread for CPU-only or single GPU
            
    def optimize_allocations(self) -> Dict[str, Any]:
        """Optimize resource allocations based on current usage."""
        status = self.check_resource_availability()
        
        optimizations = {
            'recommendations': [],
            'adjustments': {},
            'warnings': []
        }
        
        # Check for over-allocation
        if not status['overall_ok']:
            if not status['gpu_ok']:
                optimizations['warnings'].append(
                    f"GPU over-allocation: {status['allocated']['GPU']:.1f} > {status['available']['GPU']}"
                )
                optimizations['recommendations'].append("Reduce GPU allocation or use CPU fallback")
                
            if not status['cpu_ok']:
                optimizations['warnings'].append(
                    f"CPU over-allocation: {status['allocated']['CPU']:.1f} > {status['available']['CPU']}"
                )
                optimizations['recommendations'].append("Reduce CPU allocation per actor")
                
            if not status['memory_ok']:
                optimizations['warnings'].append(
                    f"Memory over-allocation: {status['allocated']['memory'] / 1e9:.1f}GB > {status['available']['memory'] / 1e9 * 0.8:.1f}GB"
                )
                optimizations['recommendations'].append("Reduce memory allocation or use memory-efficient models")
        
        # Check for under-utilization
        if status['utilization']['GPU'] < 50 and status['available']['GPU'] > 0:
            optimizations['recommendations'].append("Low GPU utilization - consider increasing batch size or model complexity")
            
        if status['utilization']['CPU'] < 30:
            optimizations['recommendations'].append("Low CPU utilization - consider increasing parallel actors")
            
        return optimizations
        
    def print_allocation_summary(self) -> None:
        """Print comprehensive resource allocation summary."""
        print("\n📊 Resource Allocation Summary")
        print("=" * 50)
        
        # Current allocations by actor type
        print("Actor Allocations:")
        for actor_type, allocation in self.actor_allocations.items():
            total = allocation['total']
            count = allocation['count']
            
            gpu_str = f"{total['num_gpus']:.1f} GPU" if total['num_gpus'] > 0 else "CPU"
            print(f"  • {actor_type} ({count}x): {gpu_str}, {total['num_cpus']:.1f} CPU, {total['memory'] / 1e9:.1f}GB RAM")
        
        # Resource utilization
        status = self.check_resource_availability()
        print("\nResource Utilization:")
        for resource, util in status['utilization'].items():
            status_emoji = "✅" if util <= 80 else "⚠️" if util <= 100 else "❌"
            print(f"  • {resource}: {util:.1f}% {status_emoji}")
            
        # Optimization recommendations
        optimizations = self.optimize_allocations()
        if optimizations['warnings']:
            print("\n⚠️ Warnings:")
            for warning in optimizations['warnings']:
                print(f"  • {warning}")
                
        if optimizations['recommendations']:
            print("\n💡 Recommendations:")
            for rec in optimizations['recommendations']:
                print(f"  • {rec}")
                
        # Hardware summary
        print("\nHardware Summary:")
        print(f"  • Platform: {self.capabilities['platform']['system']}")
        print(f"  • Device: {self.capabilities['device']['primary_device']}")
        print(f"  • GPUs: {self.capabilities['device']['device_count']}")
        print(f"  • CPU Cores: {self.capabilities['cpu']['cpu_count']}")
        print(f"  • Memory: {self.capabilities['memory']['system_memory_gb']:.1f}GB")


class DynamicResourceManager:
    """
    Dynamic resource management for runtime optimization.
    
    Monitors resource usage and adjusts allocations during training.
    """
    
    def __init__(self, allocator: ResourceAllocator):
        """Initialize with base resource allocator."""
        self.allocator = allocator
        self.usage_history = []
        self.adjustment_history = []
        
    def monitor_usage(self) -> Dict[str, Any]:
        """Monitor current Ray cluster resource usage."""
        try:
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            usage = {}
            for resource, total in cluster_resources.items():
                used = total - available_resources.get(resource, 0)
                usage[resource] = {
                    'total': total,
                    'used': used,
                    'available': available_resources.get(resource, 0),
                    'utilization': (used / total * 100) if total > 0 else 0
                }
            
            # Store usage history
            usage['timestamp'] = ray.util.get_node_ip_address()
            self.usage_history.append(usage)
            
            # Keep only recent history
            if len(self.usage_history) > 100:
                self.usage_history.pop(0)
                
            return usage
            
        except Exception as e:
            logger.error(f"Failed to monitor resource usage: {e}")
            return {}
            
    def suggest_adjustments(self) -> List[str]:
        """Suggest resource allocation adjustments based on usage patterns."""
        if len(self.usage_history) < 5:
            return ["Need more usage data for recommendations"]
            
        suggestions = []
        
        # Analyze recent usage patterns
        recent_usage = self.usage_history[-5:]
        
        # Calculate average utilization
        avg_gpu_util = sum(u.get('GPU', {}).get('utilization', 0) for u in recent_usage) / len(recent_usage)
        avg_cpu_util = sum(u.get('CPU', {}).get('utilization', 0) for u in recent_usage) / len(recent_usage)
        
        # Suggest adjustments
        if avg_gpu_util > 95:
            suggestions.append("High GPU utilization - consider reducing batch size or adding more GPUs")
        elif avg_gpu_util < 30:
            suggestions.append("Low GPU utilization - consider increasing batch size or model complexity")
            
        if avg_cpu_util > 90:
            suggestions.append("High CPU utilization - consider reducing parallel workers")
        elif avg_cpu_util < 20:
            suggestions.append("Low CPU utilization - consider increasing parallel processing")
            
        return suggestions if suggestions else ["Resource utilization looks optimal"] 