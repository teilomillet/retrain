# Slime Backend Integration

This document explains how to use Slime as a training backend in retrain, providing an alternative to the TRL backend for distributed RL training.

## Overview

[Slime](https://github.com/THUDM/slime) is a high-performance LLM post-training framework designed for RL scaling. It provides:

- **High-Performance Training**: Distributed training using Megatron + SGLang
- **Flexible Data Generation**: Server-based rollout generation with custom workflows  
- **Ray-based Distribution**: Automatic GPU allocation and multi-node support

The retrain framework now supports Slime as an alternative backend to TRL, giving users access to Slime's distributed capabilities through retrain's simplified interface.

## Installation

To use the Slime backend, install the optional Slime dependencies:

```bash
# Install retrain with Slime support
pip install -e .[slime]

# Or install manually
pip install "ray[default]>=2.0"
pip install -e ./slime  # If you have the Slime submodule
```

## Quick Start

Simply change your algorithm backend to use Slime:

```yaml
# Configuration file
algorithm:
  name: "grpo"
  backend: "slime"  # Change from 'trl' to 'slime'
  hyperparameters:
    learning_rate: 1e-5
    # All other parameters remain the same!
```

That's it! retrain will automatically:
- Set up Ray cluster with smart GPU allocation
- Create bridge system connecting retrain ↔ Slime
- Use your existing Environment and RewardCalculator
- Scale training across multiple GPUs/nodes

## Bridge System Architecture

The integration uses a sophisticated bridge system that connects retrain's components with Slime's distributed infrastructure:

### DataFormatBridge
**Purpose**: Converts between retrain and Slime data formats
- `RawRolloutData` ↔ `Sample` objects
- retrain conversation format ↔ Slime message format  
- retrain action/observation cycle ↔ Slime prompt/response format

### EnvironmentBridge
**Purpose**: Connects retrain Environment to Slime rollout generation
- Adapts step-by-step Environment interaction to batch rollout generation
- Handles async conversion between retrain's `env.step()` and Slime's parallel generation
- Manages episode state and conversation history reconstruction

### RewardBridge  
**Purpose**: Integrates retrain RewardCalculator with Slime reward system
- Converts retrain's multi-level reward calculation to Slime's sample-level rewards
- Handles step-level and rollout-level reward combination
- Maintains reward calculation sophistication while fitting Slime's interface

### RolloutBridge
**Purpose**: Main coordinator implementing Slime's custom rollout interface
- Orchestrates all bridge components
- Provides Slime-compatible rollout generation function
- Manages training vs evaluation mode differences

## Configuration Options

### Basic Configuration

```yaml
algorithm:
  name: "grpo"
  backend: "slime"
  hyperparameters:
    # Standard retrain parameters work unchanged
    learning_rate: 1e-5
    num_iterations: 20
    rollout_batch_size: 4
    global_batch_size: 16
```

### Advanced Slime Configuration

Use `slime_` prefix to access Slime's full parameter set:

```yaml
algorithm:
  name: "grpo"
  backend: "slime"
  hyperparameters:
    # Core RL parameters
    learning_rate: 1e-5
    rollout_batch_size: 4
    
    # Distributed configuration
    slime_actor_num_nodes: 2           # Number of training nodes
    slime_actor_num_gpus_per_node: 4   # GPUs per training node  
    slime_rollout_num_gpus: 8          # Total GPUs for rollout generation
    slime_colocate: false              # Whether to colocate training/inference
    
    # Generation parameters
    slime_rollout_temperature: 0.8
    slime_rollout_top_p: 0.9
    slime_rollout_max_response_len: 512
    
    # Model configuration
    slime_bf16: true                   # Use bfloat16 precision
    slime_gradient_checkpointing: true # Enable gradient checkpointing
    
    # System optimization
    slime_sglang_server_concurrency: 128
    slime_save_interval: 10            # Model checkpoint interval
    
    # Bridge-specific options
    slime_use_retrain_environment: true  # Use retrain Environment via bridge
    slime_use_retrain_rewards: true      # Use retrain RewardCalculator via bridge
```

## Edge Cases and Considerations

### 1. **Ray Cluster Management**

**Issue**: Ray cluster initialization and cleanup
**Solution**: SlimeTrainerAdapter automatically manages Ray lifecycle
```python
# Automatic Ray setup
ray.init(ignore_reinit_error=True)
# Automatic cleanup on exit/error
ray.shutdown()
```

**Edge Cases**:
- Multiple retrain processes: Each gets isolated Ray namespace
- Ray already initialized: Gracefully reuses existing cluster
- GPU allocation conflicts: Smart detection and fallback to available resources

### 2. **Environment Bridge Limitations**

**Issue**: retrain's step-by-step Environment vs Slime's batch generation
**Handled**:
- Environment bridge simulates step-by-step interaction for batch generation
- Conversation history properly reconstructed for each turn
- Tool interactions properly bridged between paradigms

**Edge Cases**:
- Very long episodes: Memory management with checkpointing
- Environment errors: Graceful fallback and error propagation
- Tool availability mismatch: Dynamic tool discovery and validation

### 3. **Reward Calculation Bridging**

**Issue**: retrain's sophisticated reward system vs Slime's sample-level rewards  
**Solution**: RewardBridge maintains full reward calculation complexity
```python
# retrain's multi-level rewards → Slime sample rewards
step_rewards = reward_calculator.process_step_rewards(sample)
rollout_rewards = reward_calculator.process_rollout_rewards(trajectory)
final_reward = combine_rewards(step_rewards, rollout_rewards)
sample.reward = final_reward
```

**Edge Cases**:
- Reward calculation errors: Sample-level fallbacks prevent training crash  
- Mismatched reward scales: Automatic normalization and scaling
- Memory constraints: Batch reward processing with chunking

### 4. **Data Format Conversion**

**Issue**: TypedDict strict typing vs runtime dictionary creation
**Status**: Functional at runtime, minor linter warnings (safe to ignore)

**Issue**: Sample object compatibility across Slime versions
**Solution**: Graceful fallback with local Sample definition if import fails

### 5. **Distributed Training Considerations**

**Ray Resource Management**:
- Auto-detection of available GPUs
- Intelligent placement group creation
- Fallback to smaller configurations if resources unavailable

**Network Communication**:
- SGLang server health monitoring
- Automatic retry on communication failures
- Distributed component synchronization

**Multi-Node Setup**:
- Automatic node discovery via Ray
- Load balancing across nodes
- Fault tolerance for node failures

### 6. **Memory and Performance**

**Large Model Handling**:
- Automatic model sharding across GPUs
- Gradient accumulation for large batch sizes
- Memory-efficient checkpointing

**Batch Size Scaling**:
- Dynamic batch size adjustment based on available memory
- Automatic gradient accumulation when needed
- OOM prevention with fallback strategies

## Troubleshooting

### Common Issues

1. **"Slime not available" Warning**
```bash
# Install Slime dependencies
pip install -e .[slime]
pip install -e ./slime
```

2. **Ray Initialization Errors**
```bash
# Check Ray status
ray status
# Clean restart if needed
ray stop && ray start --head
```

3. **GPU Allocation Issues**
```python
# Check available GPUs
import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
```

4. **Bridge Component Errors**
- DataBridge: Check data format compatibility
- EnvironmentBridge: Verify Environment interface implementation  
- RewardBridge: Ensure RewardCalculator configuration is valid

### Performance Optimization

1. **Increase Parallelism**:
```yaml
slime_rollout_num_gpus: 8      # More GPUs for generation
slime_sglang_server_concurrency: 256  # Higher concurrency
```

2. **Memory Optimization**:
```yaml
slime_bf16: true               # Use half precision
slime_gradient_checkpointing: true  # Save memory
```

3. **Network Optimization**:
```yaml
slime_colocate: true          # Reduce network overhead
```

## Examples

### Basic Example
```bash
# Run with Slime backend
python retrain/run.py --config examples/slime_grpo_config.yaml
```

### Advanced Example with Bridges
```bash
# Run comprehensive bridge system example
python examples/slime_bridge_example.py
```

## Comparison: TRL vs Slime Backend

| Feature | TRL Backend | Slime Backend |
|---------|-------------|---------------|
| **Scaling** | Single process | Multi-node distributed |
| **Model Format** | HuggingFace | Megatron (auto-converted) |
| **Generation** | In-process | SGLang server-based |
| **Memory** | Limited to single GPU | Sharded across GPUs |
| **Performance** | Good for small scale | Optimized for large scale |
| **Setup** | Simple | Automatic via bridges |
| **Compatibility** | All retrain features | Full compatibility via bridges |

## Migration Guide

**From TRL to Slime**: 
1. Change `backend: "slime"` in your config
2. Optionally tune Slime-specific parameters with `slime_` prefix  
3. No other changes needed - bridges handle everything!

**Resource Requirements**:
- Minimum: 2 GPUs (1 for training, 1 for rollout)
- Recommended: 4+ GPUs for optimal performance
- Multi-node: 8+ GPUs across 2+ nodes

The Slime backend with bridge system provides a seamless upgrade path from TRL to high-performance distributed training while maintaining full compatibility with retrain's ecosystem! 🚀 