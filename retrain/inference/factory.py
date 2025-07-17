"""
Factory function for creating hardware-appropriate inference actors.
"""

from typing import Any
import ray

from retrain.config_models import TrainingConfig
from retrain.hardware.detector import HardwareDetector


def create_inference_actor(config: TrainingConfig, databuffer: ray.ObjectRef) -> Any:
    """
    Factory function to create hardware-appropriate inference actor.
    
    Uses HardwareDetector to determine optimal configuration and backend.
    """
    from .macos import MacOSInferenceActor
    from .cuda import CUDAInferenceActor
    from .cpu import CPUInferenceActor
    
    hardware = HardwareDetector()
    platform = hardware.capabilities['platform']
    
    # Determine actor configuration based on hardware
    if platform['is_macos']:
        # macOS: CPU-only with Transformers backend
        return MacOSInferenceActor.remote(config, databuffer)  # type: ignore
    elif hardware.capabilities['device']['cuda_available']:
        # CUDA: GPU with MBridge backend  
        return CUDAInferenceActor.remote(config, databuffer)  # type: ignore
    else:
        # CPU-only: Transformers backend
        return CPUInferenceActor.remote(config, databuffer)  # type: ignore

