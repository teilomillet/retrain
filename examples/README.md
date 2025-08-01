# Retrain Examples - Refactored Architecture

This directory contains examples showcasing the **refactored retrain architecture** with significant improvements:

- 🎯 **ReManager**: Centralized orchestrator with intelligent hardware detection
- 🏭 **Unified Actor System**: No more hardware-specific duplicated code
- 💾 **DataBuffer Integration**: Atomic operations at the manager level  
- ⚡ **Ray-First Design**: Distributed training with automatic resource optimization
- 🔧 **Hardware Detection**: Automatic optimization for different platforms

## Quick Start

```bash
cd examples/
# Simple example using refactored architecture
python simple_refactored_example.py

# Advanced example with hardware detection
python advanced_grpo_with_hardware_detection.py

# Hardware optimization demo
python hardware_optimization_example.py
```

## Examples Overview

### 1. **Simple Refactored Example** (`simple_refactored_example.py`)
- **Purpose**: Minimal example showcasing the new architecture
- **Features**: 
  - Uses `retrain.run()` with automatic ReManager setup
  - Hardware detection and optimization
  - Unified GRPO implementation (no hardware-specific actors)
  - DataBuffer coordination

### 2. **Advanced GRPO with Hardware Detection** (`advanced_grpo_with_hardware_detection.py`)
- **Purpose**: Full-featured example with ReManager
- **Features**:
  - Explicit ReManager usage and configuration
  - Advanced mathematical problem solving
  - Custom reward functions leveraging step_info
  - Hardware capability reporting

### 3. **Distributed Training Example** (`distributed_training_example.py`)
- **Purpose**: Demonstrates multi-actor distributed training
- **Features**:
  - Multiple actor groups (trainer, inference, reward, verifier)
  - SQL query generation with verification
  - Performance monitoring and optimization
  - Resource allocation based on hardware

### 4. **Hardware Optimization Example** (`hardware_optimization_example.py`)
- **Purpose**: Deep dive into hardware detection and optimization
- **Features**:
  - Comprehensive hardware capability detection
  - Actor factory optimization demonstration
  - Memory and batch size optimization
  - Platform-specific performance tips

### 5. **Legacy Examples** (Original Architecture)
- **Basic Example**: `run_example.py` + `simple_grpo_config.yaml`
- **Unsloth Integration**: `unsloth_lora_example.py`
- **FastMCP Integration**: `run_fastmcp_example.py` + `fastmcp_example_config.yaml`

## Key Architectural Improvements

### **Before Refactor** ❌
```python
# Multiple hardware-specific actors
MacOSGRPOActor()  # 400+ lines
CPUGRPOActor()    # 400+ lines  
CUDAGRPOActor()   # 400+ lines
# DataBuffer integration duplicated across algorithms
```

### **After Refactor** ✅
```python
# Single unified actor with auto-detection
GRPO()  # ~200 lines, handles all hardware
# DataBuffer operations centralized in ReManager
ReManager(config)  # Handles all coordination
```

## New Configuration Features

The refactored architecture supports enhanced configurations:

```yaml
# Hardware detection happens automatically
model:
  torch_dtype: "auto"  # ReManager optimizes based on hardware

# Unified algorithm configuration
algorithm:
  name: "grpo"  # Single implementation, no hardware variants
  
# Enhanced environment integration
environment:
  type: "smol_agent"  # Fully integrated with actor system
```

## Hardware Detection & Optimization

The new system automatically:

1. **Detects Platform**: macOS, Linux, Windows with specific optimizations
2. **Analyzes Resources**: CPU cores, memory, GPU availability
3. **Optimizes Configuration**: Batch sizes, actor allocation, memory usage
4. **Provides Recommendations**: Deployment type, model suggestions

Example output:
```
🔍 Hardware Detection Summary
Platform: Apple Silicon Mac (arm64) with 10 cores, 34.4GB RAM, MPS✓
Backend: transformers  
Deployment Type: development
Recommended Batch Size: 2
```

## Ray Integration Benefits

- **Automatic Resource Management**: Optimal actor placement
- **Fault Tolerance**: Actor restart and error handling
- **Scalability**: From single machine to cluster deployment
- **Monitoring**: Built-in performance metrics and logging

## Custom Components

### Custom Reward Functions
```python
from retrain.reward import reward

@reward(name="my_custom_reward")
def my_reward(prompt: str, completion: str, config_params: dict, **kwargs) -> float:
    step_info = kwargs.get("step_info", {})  # Enhanced info from actors
    # Your reward logic here
    return reward_score
```

### Hardware-Aware Configuration
```python
from retrain.hardware import HardwareDetector

detector = HardwareDetector()
recommendations = detector.recommendations
config["algorithm"]["hyperparameters"]["per_device_train_batch_size"] = recommendations["batch_size_recommendation"]
```

## Migration Guide

### From Old Architecture
1. Replace hardware-specific actor imports with unified ones
2. Use `ReManager` instead of direct actor creation
3. Leverage automatic hardware detection
4. Update reward functions to use enhanced `step_info`

### Performance Improvements
- **82% code reduction** in GRPO implementation  
- **Automatic hardware optimization**
- **Centralized DataBuffer operations**
- **Simplified actor management**

## Common Patterns

### Basic Training
```python
from retrain import run
results = await run(config=your_config)
```

### Advanced Training with ReManager
```python
from retrain.manager import ReManager
manager = ReManager(config)
await manager.initialize()
results = await manager.run_training()
await manager.shutdown()
```

### Hardware Optimization
```python
from retrain.hardware import HardwareDetector
detector = HardwareDetector()
optimized_config = detector.optimize_config(base_config)
```

## Troubleshooting

### macOS Ray Issues
```bash
export RAY_ENABLE_MAC_LARGE_OBJECT_STORE=1
```

### Memory Issues
- Reduce batch sizes for development
- Enable gradient checkpointing
- Use smaller models for testing

### GPU Issues
- Check CUDA availability with hardware detector
- Verify MPS support on Apple Silicon
- Monitor GPU memory usage

## Support

For issues with the refactored architecture:
1. Check hardware detection output
2. Review actor allocation recommendations  
3. Monitor resource utilization
4. Consult the main documentation

The refactored architecture provides much cleaner abstractions while maintaining full compatibility with existing use cases!