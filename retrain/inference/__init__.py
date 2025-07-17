"""
Retrain inference module with multi-backend support.

This module provides distributed inference for Retrain using Ray with support for:
- MBridge (CUDA GPU inference) 
- Transformers (CPU/macOS inference)
- Hardware-aware configuration

The module automatically selects the appropriate backend based on available hardware.
"""

from .models import RawRolloutData, GenerationResult
from .base import BaseInferenceActor
from .macos import MacOSInferenceActor
from .cuda import CUDAInferenceActor  
from .cpu import CPUInferenceActor
from .factory import create_inference_actor

# Backward compatibility - use factory function by default
def ReInference(config, databuffer):
    """Backward compatibility function that creates appropriate inference actor."""
    return create_inference_actor(config, databuffer)

__all__ = [
    'RawRolloutData',
    'GenerationResult', 
    'BaseInferenceActor',
    'MacOSInferenceActor',
    'CUDAInferenceActor',
    'CPUInferenceActor',
    'create_inference_actor',
    'ReInference'
]

