"""
Hardware detection and management for Retrain distributed training.

This module provides smart hardware detection, resource allocation, and
device-specific optimizations for Ray actors across different platforms.
"""

from .detector import HardwareDetector
from .allocator import ResourceAllocator  
from .factory import ActorFactory
from .optimizer import PerformanceOptimizer

__all__ = [
    'HardwareDetector',
    'ResourceAllocator', 
    'ActorFactory',
    'PerformanceOptimizer'
] 