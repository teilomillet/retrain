"""
Hardware detection and capability assessment for Retrain.

Detects available hardware (CPU, MPS, CUDA) and recommends optimal
configurations for Ray actors and model backends.
"""

import logging
import platform
import psutil
from typing import Dict, Any, List
import torch

logger = logging.getLogger(__name__)


class HardwareDetector:
    """
    Comprehensive hardware detection for Retrain distributed training.
    
    Detects platform capabilities and recommends optimal configurations
    for Ray actors, model backends, and resource allocation.
    """
    
    def __init__(self):
        """Initialize hardware detector and analyze system capabilities."""
        self.capabilities = self._detect_all_capabilities()
        self.recommendations = self._generate_recommendations()
        
        logger.info(f"Hardware detected: {self.capabilities['summary']}")
        
    def _detect_all_capabilities(self) -> Dict[str, Any]:
        """Comprehensive hardware capability detection."""
        return {
            'platform': self._detect_platform(),
            'device': self._detect_device_info(),
            'memory': self._detect_memory_info(),
            'cpu': self._detect_cpu_info(), 
            'distributed': self._detect_distributed_capabilities(),
            'ray': self._detect_ray_capabilities(),
            'ml_frameworks': self._detect_ml_frameworks(),
            'summary': self._generate_summary()
        }
        
    def _detect_platform(self) -> Dict[str, Any]:
        """Detect platform and OS information."""
        system = platform.system().lower()
        
        platform_info = {
            'system': system,
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'is_macos': system == 'darwin',
            'is_linux': system == 'linux', 
            'is_windows': system == 'windows',
            'is_arm': platform.machine().lower() in ['arm64', 'aarch64'],
            'is_apple_silicon': system == 'darwin' and platform.machine() == 'arm64'
        }
        
        return platform_info
        
    def _detect_device_info(self) -> Dict[str, Any]:
        """Detect available compute devices and their capabilities."""
        device_info = {
            'primary_device': 'cpu',
            'available_devices': ['cpu'],
            'device_count': 0,
            'cuda_available': False,
            'mps_available': False,
            'gpu_memory_gb': 0.0,
            'cuda_devices': [],
            'recommended_device': 'cpu'
        }
        
        # Check CUDA availability
        if torch.cuda.is_available():
            device_info.update({
                'cuda_available': True,
                'primary_device': 'cuda',
                'device_count': torch.cuda.device_count(),
                'recommended_device': 'cuda'
            })
            device_info['available_devices'].append('cuda')
            
            # Get CUDA device details
            cuda_devices = []
            total_gpu_memory = 0.0
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device_details = {
                    'index': i,
                    'name': props.name,
                    'memory_gb': props.total_memory / 1e9,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multiprocessor_count': props.multiprocessor_count
                }
                cuda_devices.append(device_details)
                total_gpu_memory += device_details['memory_gb']
                
            device_info.update({
                'cuda_devices': cuda_devices,
                'gpu_memory_gb': total_gpu_memory
            })
            
        # Check MPS availability (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info.update({
                'mps_available': True,
                'primary_device': 'mps',
                'device_count': 1,
                'recommended_device': 'mps'
            })
            device_info['available_devices'].append('mps')
            
        return device_info
        
    def _detect_memory_info(self) -> Dict[str, Any]:
        """Detect system and GPU memory information."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = {
            'system_memory_gb': memory.total / 1e9,
            'available_memory_gb': memory.available / 1e9,
            'memory_percent_used': memory.percent,
            'swap_memory_gb': swap.total / 1e9,
            'recommended_memory_limit_gb': memory.total / 1e9 * 0.7,  # Reserve 30%
        }
        
        # Add GPU memory if available
        if self.capabilities.get('device', {}).get('cuda_available', False):
            memory_info['gpu_memory_gb'] = self.capabilities['device']['gpu_memory_gb']
            
        return memory_info
        
    def _detect_cpu_info(self) -> Dict[str, Any]:
        """Detect CPU capabilities and performance characteristics."""
        # Handle psutil functions that may return None
        cpu_count = psutil.cpu_count(logical=True) or 4
        physical_cores = psutil.cpu_count(logical=False) or 2
        
        # Safely get CPU frequency
        cpu_freq_max = None
        if hasattr(psutil, 'cpu_freq'):
            try:
                freq_info = psutil.cpu_freq()
                cpu_freq_max = freq_info.max if freq_info else None
            except (AttributeError, OSError):
                cpu_freq_max = None
        
        return {
            'cpu_count': cpu_count,
            'physical_cores': physical_cores,
            'cpu_freq_max': cpu_freq_max,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'recommended_cpu_actors': min(cpu_count, 8),  # Cap at 8
            'supports_multiprocessing': True
        }
        
    def _detect_distributed_capabilities(self) -> Dict[str, Any]:
        """Detect distributed training capabilities."""
        capabilities = {
            'backends_available': [],
            'recommended_backend': 'gloo',
            'supports_nccl': False,
            'supports_gloo': True,
            'supports_mpi': False
        }
        
        # Check NCCL (CUDA only)
        if torch.cuda.is_available():
            try:
                if hasattr(torch.distributed, 'is_nccl_available') and torch.distributed.is_nccl_available():
                    capabilities['backends_available'].append('nccl')
                    capabilities['supports_nccl'] = True
                    capabilities['recommended_backend'] = 'nccl'
            except Exception:
                pass
                
        # Gloo is always available
        capabilities['backends_available'].append('gloo')
        
        # Check MPI
        try:
            if hasattr(torch.distributed, 'is_mpi_available') and torch.distributed.is_mpi_available():
                capabilities['backends_available'].append('mpi')
                capabilities['supports_mpi'] = True
        except Exception:
            pass
            
        return capabilities
        
    def _detect_ray_capabilities(self) -> Dict[str, Any]:
        """Detect Ray-specific capabilities and resource recommendations."""
        device_count = self.capabilities.get('device', {}).get('device_count', 0)
        cpu_count = self.capabilities.get('cpu', {}).get('cpu_count', 4)
        
        return {
            'recommended_ray_resources': {
                'num_cpus': cpu_count,
                'num_gpus': device_count,
                'object_store_memory': int(self.capabilities.get('memory', {}).get('system_memory_gb', 8) * 0.3 * 1e9),
                'memory': int(self.capabilities.get('memory', {}).get('system_memory_gb', 8) * 0.5 * 1e9)
            },
            'placement_group_strategy': 'STRICT_PACK' if device_count > 1 else 'SPREAD',
            'optimal_actor_count': {
                'trainer': max(1, device_count),
                'inference': max(1, device_count if device_count > 0 else 2),
                'reward': min(4, cpu_count // 2),
                'environment': min(2, cpu_count // 4),
                'verifier': min(2, cpu_count // 4)
            }
        }
        
    def _detect_ml_frameworks(self) -> Dict[str, Any]:
        """Detect available ML frameworks and their capabilities."""
        # Get CUDA version safely
        cuda_version = None
        try:
            if torch.cuda.is_available():
                cuda_version = getattr(torch.version, 'cuda', None)  # type: ignore
        except AttributeError:
            cuda_version = None
        
        frameworks = {
            'torch_version': torch.__version__,
            'cuda_version': cuda_version,
            'mbridge_available': False,
            'transformers_available': False,
            'vllm_available': False,
            'recommended_backend': 'transformers'
        }
        
        # Check MBridge availability
        try:
            frameworks['mbridge_available'] = True
            if torch.cuda.is_available():
                frameworks['recommended_backend'] = 'mbridge'
        except ImportError:
            pass
            
        # Check Transformers availability  
        try:
            import transformers
            frameworks['transformers_available'] = True
            frameworks['transformers_version'] = transformers.__version__
        except ImportError:
            pass
            
        # Check VLLM availability
        try:
            import vllm  # type: ignore
            frameworks['vllm_available'] = True
            if torch.cuda.is_available():
                frameworks['vllm_version'] = vllm.__version__  # type: ignore
        except ImportError:
            pass
            
        return frameworks
        
    def _generate_summary(self) -> str:
        """Generate human-readable hardware summary."""
        platform = self.capabilities['platform']
        device = self.capabilities['device']
        memory = self.capabilities['memory']
        cpu = self.capabilities['cpu']
        
        if platform['is_apple_silicon']:
            return f"Apple Silicon Mac ({platform['machine']}) with {cpu['cpu_count']} cores, {memory['system_memory_gb']:.1f}GB RAM, MPS{'✓' if device['mps_available'] else '✗'}"
        elif device['cuda_available']:
            return f"{platform['system'].title()} with {device['device_count']} GPU(s) ({device['gpu_memory_gb']:.1f}GB VRAM), {cpu['cpu_count']} CPU cores, {memory['system_memory_gb']:.1f}GB RAM"
        else:
            return f"{platform['system'].title()} CPU-only with {cpu['cpu_count']} cores, {memory['system_memory_gb']:.1f}GB RAM"
            
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate optimal configuration recommendations based on detected hardware."""
        device = self.capabilities['device']
        platform = self.capabilities['platform']
        
        if platform['is_macos']:
            return self._generate_macos_recommendations()
        elif device['cuda_available'] and device['device_count'] >= 4:
            return self._generate_multi_gpu_recommendations()
        elif device['cuda_available']:
            return self._generate_single_gpu_recommendations() 
        else:
            return self._generate_cpu_recommendations()
            
    def _generate_macos_recommendations(self) -> Dict[str, Any]:
        """Generate macOS-specific recommendations."""
        device = self.capabilities['device']
        cpu = self.capabilities['cpu']
        memory = self.capabilities['memory']
        
        return {
            'deployment_type': 'development',
            'backend': 'transformers',
            'model_size_class': 'small',
            'ray_config': {
                'num_cpus': cpu['cpu_count'],
                'num_gpus': 0,  # Force CPU for stability
                'object_store_memory': int(memory['system_memory_gb'] * 0.2 * 1e9),
            },
            'actor_resources': {
                'trainer': {'num_gpus': 1 if device['mps_available'] else 0, 'num_cpus': 2},
                'inference': {'num_gpus': 0, 'num_cpus': 2},  # CPU inference for stability
                'environment': {'num_gpus': 0, 'num_cpus': 1},
                'verifier': {'num_gpus': 0, 'num_cpus': 1},
                'reward': {'num_gpus': 0, 'num_cpus': 1},
                'databuffer': {'num_gpus': 0, 'num_cpus': 2}
            },
            'model_recommendations': [
                'gpt2',
                'microsoft/DialoGPT-small',
                'microsoft/DialoGPT-medium'
            ],
            'training_config': {
                'batch_size': 4,
                'max_sequence_length': 512,
                'max_new_tokens': 100,
                'gradient_accumulation_steps': 2
            },
            'performance_tips': [
                'Use smaller models for faster iteration',
                'CPU inference provides stable development experience',
                'MPS training available if supported',
                'Ray provides parallelization benefits even on CPU'
            ]
        }
        
    def _generate_multi_gpu_recommendations(self) -> Dict[str, Any]:
        """Generate multi-GPU production recommendations."""
        device = self.capabilities['device']
        cpu = self.capabilities['cpu']
        memory = self.capabilities['memory']
        
        return {
            'deployment_type': 'production',
            'backend': 'mbridge',
            'model_size_class': 'large',
            'ray_config': {
                'num_cpus': cpu['cpu_count'],
                'num_gpus': device['device_count'],
                'object_store_memory': int(memory['system_memory_gb'] * 0.3 * 1e9),
            },
            'actor_resources': {
                'trainer': {'num_gpus': max(2, device['device_count'] // 2), 'num_cpus': 4},
                'inference': {'num_gpus': min(2, device['device_count'] // 2), 'num_cpus': 2},
                'environment': {'num_gpus': 0, 'num_cpus': 2},
                'verifier': {'num_gpus': 1, 'num_cpus': 2},
                'reward': {'num_gpus': 0, 'num_cpus': 4},
                'databuffer': {'num_gpus': 0, 'num_cpus': 4}
            },
            'model_recommendations': [
                'meta-llama/Llama-3.1-8B',
                'meta-llama/Llama-3.1-70B',
                'Qwen/Qwen2.5-32B'
            ],
            'training_config': {
                'batch_size': 32,
                'max_sequence_length': 2048,
                'max_new_tokens': 512,
                'gradient_accumulation_steps': 1
            }
        }
        
    def _generate_single_gpu_recommendations(self) -> Dict[str, Any]:
        """Generate single GPU recommendations."""
        cpu = self.capabilities['cpu']
        memory = self.capabilities['memory']
        
        return {
            'deployment_type': 'small_scale',
            'backend': 'mbridge',
            'model_size_class': 'medium',
            'ray_config': {
                'num_cpus': cpu['cpu_count'],
                'num_gpus': 1,
                'object_store_memory': int(memory['system_memory_gb'] * 0.3 * 1e9),
            },
            'actor_resources': {
                'trainer': {'num_gpus': 1, 'num_cpus': 2},
                'inference': {'num_gpus': 1, 'num_cpus': 2},
                'environment': {'num_gpus': 0, 'num_cpus': 2},
                'verifier': {'num_gpus': 0, 'num_cpus': 1},
                'reward': {'num_gpus': 0, 'num_cpus': 2},
                'databuffer': {'num_gpus': 0, 'num_cpus': 2}
            },
            'model_recommendations': [
                'microsoft/DialoGPT-large',
                'meta-llama/Llama-3.1-8B',
                'Qwen/Qwen2.5-7B'
            ],
            'training_config': {
                'batch_size': 8,
                'max_sequence_length': 1024,
                'max_new_tokens': 256,
                'gradient_accumulation_steps': 2
            }
        }
        
    def _generate_cpu_recommendations(self) -> Dict[str, Any]:
        """Generate CPU-only recommendations."""
        cpu = self.capabilities['cpu']
        memory = self.capabilities['memory']
        
        return {
            'deployment_type': 'cpu_only',
            'backend': 'transformers',
            'model_size_class': 'small',
            'ray_config': {
                'num_cpus': cpu['cpu_count'],
                'num_gpus': 0,
                'object_store_memory': int(memory['system_memory_gb'] * 0.2 * 1e9),
            },
            'actor_resources': {
                'trainer': {'num_gpus': 0, 'num_cpus': 2},
                'inference': {'num_gpus': 0, 'num_cpus': 2},
                'environment': {'num_gpus': 0, 'num_cpus': 1},
                'verifier': {'num_gpus': 0, 'num_cpus': 1},
                'reward': {'num_gpus': 0, 'num_cpus': 1},
                'databuffer': {'num_gpus': 0, 'num_cpus': 2}
            },
            'model_recommendations': [
                'gpt2',
                'microsoft/DialoGPT-small',
                'distilgpt2'
            ],
            'training_config': {
                'batch_size': 2,
                'max_sequence_length': 512,
                'max_new_tokens': 100,
                'gradient_accumulation_steps': 4
            }
        }
        
    def get_actor_config(self, actor_type: str) -> Dict[str, Any]:
        """Get optimal Ray actor configuration for specific actor type."""
        return self.recommendations['actor_resources'].get(actor_type, {'num_gpus': 0, 'num_cpus': 1})
        
    def get_ray_init_config(self) -> Dict[str, Any]:
        """Get optimal Ray initialization configuration."""
        return self.recommendations['ray_config']
        
    def get_backend_recommendation(self) -> str:
        """Get recommended backend (mbridge, transformers, etc.)."""
        return self.recommendations['backend']
        
    def get_model_recommendations(self) -> List[str]:
        """Get recommended models for this hardware."""
        return self.recommendations['model_recommendations']
        
    def should_use_distributed(self) -> bool:
        """Determine if distributed training should be used."""
        return self.capabilities['device']['device_count'] > 1
        
    def get_distributed_backend(self) -> str:
        """Get recommended distributed backend."""
        return self.capabilities['distributed']['recommended_backend']
        
    def print_summary(self) -> None:
        """Print comprehensive hardware summary and recommendations."""
        print("\n🔍 Hardware Detection Summary")
        print("=" * 50)
        print(f"Platform: {self.capabilities['summary']}")
        print(f"Backend: {self.recommendations['backend']}")
        print(f"Deployment Type: {self.recommendations['deployment_type']}")
        print(f"Model Size Class: {self.recommendations['model_size_class']}")
        
        print("\n📋 Recommended Models:")
        for model in self.recommendations['model_recommendations']:
            print(f"  • {model}")
            
        print("\n⚡ Ray Actor Resources:")
        for actor, resources in self.recommendations['actor_resources'].items():
            gpu_str = f"{resources['num_gpus']} GPU" if resources['num_gpus'] > 0 else "CPU"
            print(f"  • {actor}: {gpu_str}, {resources['num_cpus']} CPU cores")
            
        if 'performance_tips' in self.recommendations:
            print("\n💡 Performance Tips:")
            for tip in self.recommendations['performance_tips']:
                print(f"  • {tip}") 