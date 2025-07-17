# manager/databuffer.py

import logging
import pickle
import time
from typing import Any, Dict, List, Optional
import ray
import torch

from ..config_models import TrainingConfig

logger = logging.getLogger(__name__)

@ray.remote(num_cpus=2, num_gpus=0)
class ReDataBuffer:
    """
    Distributed DataBuffer actor for managing training data flow.
    
    The ReDataBuffer is the central data coordination hub that:
    1. Manages memory for large datasets and rollout buffers
    2. Converts between inference and training data formats
    3. Batches data optimally for different algorithms 
    4. Handles sample storage and retrieval
    5. Coordinates data flow between all actor groups
    
    This runs as a separate Ray actor to isolate heavy memory operations
    from the lightweight ReManager coordinator.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the DataBuffer with training configuration.
        
        Args:
            config: Complete training configuration
        """
        self.config = config
        
        # Data storage
        self.rollout_buffer: List[Dict[str, Any]] = []
        self.training_batches: Dict[int, Dict[str, Any]] = {}
        self.evaluation_data: Dict[int, Dict[str, Any]] = {}
        
        # Metadata and statistics
        self.episode_metadata: Dict[int, Dict[str, Any]] = {}
        self.buffer_stats = {
            'total_samples': 0,
            'total_rollouts': 0,
            'memory_usage': 0,
            'processing_time': {}
        }
        
        # Sample management
        self.sample_index = 0
        self.max_buffer_size = 10000  # Configurable buffer size
        
        logger.info("ReDataBuffer initialized")
        
    async def initialize(self) -> None:
        """Initialize the DataBuffer actor."""
        logger.info("Initializing DataBuffer...")
        
        # Initialize any external dependencies
        await self._setup_data_processing()
        
        logger.info("DataBuffer initialization complete")
        
    async def _setup_data_processing(self) -> None:
        """Setup data processing pipelines and converters."""
        # Initialize data converters based on algorithm
        algorithm_name = self.config.algorithm.name.lower()
        
        if algorithm_name == "grpo":
            self.data_converter = self._grpo_data_converter
        elif algorithm_name == "rloo":
            self.data_converter = self._rloo_data_converter
        else:
            self.data_converter = self._default_data_converter
            
        logger.info(f"Data converter setup for algorithm: {algorithm_name}")
        
    async def store_rollout_data(self, 
                                rollout_data: List[Dict[str, Any]], 
                                episode_id: int) -> str:
        """
        Store rollout data from inference actors.
        
        Args:
            rollout_data: List of rollout samples from inference
            episode_id: Current episode identifier
            
        Returns:
            Storage identifier for the rollout batch
        """
        start_time = time.time()
        
        # Add metadata to each sample
        processed_rollouts = []
        for i, sample in enumerate(rollout_data):
            processed_sample = {
                'sample_id': self.sample_index,
                'episode_id': episode_id,
                'rollout_index': i,
                'timestamp': time.time(),
                'data': sample
            }
            processed_rollouts.append(processed_sample)
            self.sample_index += 1
            
        # Store in buffer
        self.rollout_buffer.extend(processed_rollouts)
        
        # Update statistics
        self.buffer_stats['total_samples'] += len(rollout_data)
        self.buffer_stats['total_rollouts'] += 1
        self.buffer_stats['processing_time']['store_rollout'] = time.time() - start_time
        
        # Manage buffer size
        await self._manage_buffer_size()
        
        storage_id = f"rollout_{episode_id}_{int(time.time())}"
        logger.info(f"Stored {len(rollout_data)} rollout samples for episode {episode_id}")
        
        return storage_id
        
    async def prepare_training_batch(self,
                                   rollout_data: List[Dict[str, Any]],
                                   rewards: List[float],
                                   verification_results: Dict[str, Any],
                                   episode_id: int) -> Dict[str, Any]:
        """
        Prepare a training batch by combining rollouts, rewards, and verification results.
        
        Args:
            rollout_data: Raw rollout data from inference
            rewards: Computed rewards for each rollout
            verification_results: Results from verifiers
            episode_id: Current episode identifier
            
        Returns:
            Formatted training batch ready for the trainer
        """
        start_time = time.time()
        
        # Convert to training format using algorithm-specific converter
        training_batch = await self.data_converter(
            rollout_data=rollout_data,
            rewards=rewards,
            verification_results=verification_results,
            episode_id=episode_id
        )
        
        # Store the training batch
        self.training_batches[episode_id] = training_batch
        
        # Update metadata
        self.episode_metadata[episode_id] = {
            'rollout_count': len(rollout_data),
            'reward_stats': {
                'mean': sum(rewards) / len(rewards) if rewards else 0,
                'min': min(rewards) if rewards else 0,
                'max': max(rewards) if rewards else 0
            },
            'verification_stats': verification_results,
            'batch_preparation_time': time.time() - start_time
        }
        
        logger.info(f"Training batch prepared for episode {episode_id}")
        return training_batch
        
    async def _grpo_data_converter(self,
                                 rollout_data: List[Dict[str, Any]],
                                 rewards: List[float],
                                 verification_results: Dict[str, Any],
                                 episode_id: int) -> Dict[str, Any]:
        """Convert data to GRPO training format."""
        training_batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
            'rewards': rewards,
            'episode_id': episode_id,
            'verification_passed': verification_results.get('passed', [True] * len(rollout_data))
        }
        
        for sample in rollout_data:
            # Extract tokenized data from rollout
            if 'tokens' in sample:
                training_batch['input_ids'].append(sample['tokens'])
            if 'attention_mask' in sample:
                training_batch['attention_mask'].append(sample['attention_mask'])
            if 'labels' in sample:
                training_batch['labels'].append(sample['labels'])
                
        # Convert to tensors if using PyTorch
        if training_batch['input_ids'] and isinstance(training_batch['input_ids'][0], list):
            training_batch['input_ids'] = torch.tensor(training_batch['input_ids'])
            if training_batch['attention_mask']:
                training_batch['attention_mask'] = torch.tensor(training_batch['attention_mask'])
                
        return training_batch
        
    async def _rloo_data_converter(self,
                                 rollout_data: List[Dict[str, Any]],
                                 rewards: List[float],
                                 verification_results: Dict[str, Any],
                                 episode_id: int) -> Dict[str, Any]:
        """Convert data to RLOO training format."""
        # RLOO-specific data formatting
        training_batch = {
            'rollouts': rollout_data,
            'rewards': rewards,
            'advantages': [],  # Will be computed by RLOO trainer
            'episode_id': episode_id,
            'verification_results': verification_results
        }
        
        return training_batch
        
    async def _default_data_converter(self,
                                    rollout_data: List[Dict[str, Any]],
                                    rewards: List[float],
                                    verification_results: Dict[str, Any],
                                    episode_id: int) -> Dict[str, Any]:
        """Default data converter for unknown algorithms."""
        return {
            'rollout_data': rollout_data,
            'rewards': rewards,
            'verification_results': verification_results,
            'episode_id': episode_id
        }
        
    async def get_training_batch(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a stored training batch."""
        return self.training_batches.get(episode_id)
        
    async def store_evaluation_data(self, 
                                  eval_data: Dict[str, Any], 
                                  episode_id: int) -> None:
        """Store evaluation data for later analysis."""
        self.evaluation_data[episode_id] = eval_data
        logger.info(f"Evaluation data stored for episode {episode_id}")
        
    async def get_buffer_statistics(self) -> Dict[str, Any]:
        """Get current buffer statistics and memory usage."""
        import sys
        
        memory_usage = 0
        memory_usage += sys.getsizeof(self.rollout_buffer)
        memory_usage += sys.getsizeof(self.training_batches)
        memory_usage += sys.getsizeof(self.evaluation_data)
        
        self.buffer_stats['memory_usage'] = memory_usage
        self.buffer_stats['rollout_buffer_size'] = len(self.rollout_buffer)
        self.buffer_stats['training_batches_count'] = len(self.training_batches)
        
        return self.buffer_stats
        
    async def _manage_buffer_size(self) -> None:
        """Manage buffer size to prevent memory overflow."""
        while len(self.rollout_buffer) > self.max_buffer_size:
            # Remove oldest samples
            removed_sample = self.rollout_buffer.pop(0)
            logger.debug(f"Removed old sample {removed_sample.get('sample_id')} from buffer")
            
    async def save_state(self, checkpoint_path: str) -> None:
        """Save DataBuffer state to checkpoint."""
        state = {
            'episode_metadata': self.episode_metadata,
            'buffer_stats': self.buffer_stats,
            'sample_index': self.sample_index,
            'config': self.config.dict()
        }
        
        checkpoint_file = f"{checkpoint_path}/databuffer_state.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
            
        logger.info(f"DataBuffer state saved to {checkpoint_file}")
        
    async def load_state(self, checkpoint_path: str) -> None:
        """Load DataBuffer state from checkpoint."""
        checkpoint_file = f"{checkpoint_path}/databuffer_state.pkl"
        
        try:
            with open(checkpoint_file, 'rb') as f:
                state = pickle.load(f)
                
            self.episode_metadata = state.get('episode_metadata', {})
            self.buffer_stats = state.get('buffer_stats', {})
            self.sample_index = state.get('sample_index', 0)
            
            logger.info(f"DataBuffer state loaded from {checkpoint_file}")
        except FileNotFoundError:
            logger.warning(f"Checkpoint file not found: {checkpoint_file}")
            
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        stats = await self.get_buffer_statistics()
        
        health_status = {
            'status': 'healthy',
            'memory_usage_mb': stats['memory_usage'] / (1024 * 1024),
            'buffer_size': stats['rollout_buffer_size'],
            'total_samples_processed': stats['total_samples'],
            'timestamp': time.time()
        }
        
        # Check for potential issues
        if stats['memory_usage'] > 8_000_000_000:  # 8GB warning
            health_status['status'] = 'warning'
            health_status['warning'] = 'High memory usage detected'
            
        return health_status
        
    async def shutdown(self) -> None:
        """Gracefully shutdown the DataBuffer."""
        logger.info("Shutting down DataBuffer...")
        
        # Clear large data structures
        self.rollout_buffer.clear()
        self.training_batches.clear()
        self.evaluation_data.clear()
        
        logger.info("DataBuffer shutdown complete")
