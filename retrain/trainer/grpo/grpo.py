"""
GRPO Implementation - Clean, Unified Architecture

Single GRPO implementation with:
- Auto hardware detection (macOS/MPS, CUDA, CPU)  
- Backend auto-selection (Transformers, MBridge, Unsloth)
- Pure algorithm focus (no databuffer concerns)
- Ray-first architecture for distributed training

Uses non-actor base class approach for Ray compatibility.
"""

import logging
import time
from typing import Dict, Any
import os

# Ray imports
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import configuration
from ...config_models import TrainingConfig

# Hardware detection
try:
    import platform
    HAS_PLATFORM = True
except ImportError:
    HAS_PLATFORM = False

# Backend detection
try:
    HAS_MBRIDGE = True
except ImportError:
    HAS_MBRIDGE = False

try:
    from unsloth import FastLanguageModel  # type: ignore
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class BaseGRPO:
    """
    Base GRPO class - pure algorithm implementation without Ray decorators.
    
    This allows inheritance for DRGRPO while maintaining Ray compatibility.
    Contains core GRPO algorithm with hardware detection and backend selection.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize base GRPO with automatic hardware and backend detection.
        
        Args:
            config: Complete training configuration
        """
        self.config = config
        self.device = self._detect_device()
        self.backend = self._detect_backend()
        
        # Core model components
        self.model = None
        self.tokenizer = None
        self.value_head = None
        self.optimizer = None
        
        # Training state
        self.is_initialized = False
        self.training_step = 0
        self.model_weights = {}
        
        # GRPO-specific hyperparameters
        self.clip_range = getattr(config.algorithm, 'clip_range', 0.2)
        self.value_clip_range = getattr(config.algorithm, 'value_clip_range', 0.2)
        self.entropy_coef = getattr(config.algorithm, 'entropy_coef', 0.01)
        self.value_coef = getattr(config.algorithm, 'value_coef', 0.5)
        self.gamma = getattr(config.algorithm, 'gamma', 0.99)
        self.gae_lambda = getattr(config.algorithm, 'gae_lambda', 0.95)
        
        logger.info(f"BaseGRPO initialized - Device: {self.device}, Backend: {self.backend}")
        
    def _detect_device(self) -> str:
        """Detect optimal device for training."""
        if HAS_PLATFORM and platform.system() == "Darwin":
            # macOS - prefer MPS if available
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        elif torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        else:
            return "cpu"
            
    def _detect_backend(self) -> str:
        """Detect optimal backend for model loading."""
        # Priority: Unsloth > MBridge > Transformers
        if HAS_UNSLOTH:
            logger.info("Using Unsloth backend for optimized training")
            return "unsloth"
        elif HAS_MBRIDGE:
            logger.info("Using MBridge backend for distributed inference")
            return "mbridge"
        elif HAS_TRANSFORMERS:
            logger.info("Using Transformers backend")
            return "transformers"
        else:
            raise RuntimeError("No supported backend found. Install transformers, unsloth, or mbridge.")
            
    async def initialize(self) -> None:
        """Initialize model, tokenizer, and training components."""
        try:
            logger.info(f"Initializing GRPO with {self.backend} backend on {self.device}")
            
            # Load model and tokenizer based on backend
            await self._load_model_and_tokenizer()
            
            # Add value head for GRPO
            self._add_value_head()
            
            # Setup optimizer
            self._setup_optimizer()
            
            # Initialize model weights cache
            self.model_weights = await self._extract_model_weights()
            
            self.is_initialized = True
            logger.info("GRPO initialization complete")
            
        except Exception as e:
            logger.error(f"GRPO initialization failed: {e}")
            raise
            
    async def _load_model_and_tokenizer(self) -> None:
        """Load model and tokenizer based on detected backend."""
        model_path = self.config.model.name_or_path
        
        if self.backend == "unsloth":
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(  # type: ignore
                model_name=model_path,
                max_seq_length=getattr(self.config.model, 'max_seq_length', 2048),
                dtype=torch.float16 if self.device != "cpu" else torch.float32,
                load_in_4bit=getattr(self.config.model, 'load_in_4bit', False)
            )
            
        elif self.backend == "mbridge":
            # MBridge initialization
            from mbridge import create_model  # type: ignore
            self.model = create_model(model_path, device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore
            
        elif self.backend == "transformers":
            self.model = AutoModel.from_pretrained(  # type: ignore
                model_path,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                trust_remote_code=getattr(self.config.model, 'trust_remote_code', False)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore
            
        # Move model to device
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)  # type: ignore
            
    def _add_value_head(self) -> None:
        """Add value head for GRPO value function estimation."""
        try:
            # Get model hidden size
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):  # type: ignore
                hidden_size = self.model.config.hidden_size  # type: ignore
            else:
                # Default fallback
                hidden_size = 768
                
            # Create value head
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 1)
            ).to(self.device)
            
            logger.info(f"Value head added with hidden size: {hidden_size}")
            
        except Exception as e:
            logger.warning(f"Failed to add value head: {e}")
            # Create minimal fallback value head
            self.value_head = nn.Linear(768, 1).to(self.device)
            
    def _setup_optimizer(self) -> None:
        """Setup optimizer for GRPO training."""
        # Get learning rate from config
        learning_rate = self.config.algorithm.hyperparameters.get('learning_rate', 1e-5)
        
        # Combine model and value head parameters
        params = []
        if hasattr(self.model, 'parameters'):
            params.extend(self.model.parameters())  # type: ignore
        if self.value_head is not None:
            params.extend(self.value_head.parameters())
            
        # Create optimizer
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate)
        logger.info(f"Optimizer setup complete with LR: {learning_rate}")
        
    async def train_step(self, training_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute GRPO training step with pure algorithm logic.
        
        Args:
            training_batch: Preprocessed training data
            
        Returns:
            Training metrics
        """
        if not self.is_initialized:
            raise RuntimeError("GRPO not initialized")
            
        step_start_time = time.time()
        
        try:
            # Extract batch components
            input_ids = torch.tensor(training_batch['input_ids']).to(self.device)
            attention_mask = torch.tensor(training_batch.get('attention_mask', [])).to(self.device)
            rewards = torch.tensor(training_batch.get('rewards', [])).to(self.device)
            old_log_probs = torch.tensor(training_batch.get('old_log_probs', [])).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                # Get current model outputs
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)  # type: ignore
                
                # Extract logits and compute log probabilities
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get value estimates
                if self.value_head is not None:
                    hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                    values = self.value_head(hidden_states).squeeze(-1)
                else:
                    values = torch.zeros_like(rewards)
                    
            # Compute advantages using GAE
            advantages = self._compute_advantages(rewards, values)
            
            # GRPO loss computation
            policy_loss = self._compute_policy_loss(log_probs, old_log_probs, advantages)
            value_loss = self._compute_value_loss(values, rewards, advantages)
            entropy_loss = self._compute_entropy_loss(log_probs)
            
            # Combined loss
            total_loss = (policy_loss + 
                         self.value_coef * value_loss - 
                         self.entropy_coef * entropy_loss)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()  # type: ignore
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.value_head.parameters()),  # type: ignore
                max_norm=1.0
            )
            
            self.optimizer.step()  # type: ignore
            
            # Update training state
            self.training_step += 1
            step_time = time.time() - step_start_time
            
            # Prepare metrics
            metrics = {
                'loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'mean_advantage': advantages.mean().item(),
                'mean_reward': rewards.mean().item(),
                'step_time': step_time,
                'training_step': self.training_step,
                'device': self.device,
                'backend': self.backend
            }
            
            logger.info(f"GRPO step {self.training_step} completed - Loss: {total_loss.item():.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"GRPO training step failed: {e}")
            raise
            
    def _compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute GAE advantages."""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
                
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
    def _compute_policy_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, 
                           advantages: torch.Tensor) -> torch.Tensor:
        """Compute GRPO policy loss with group-relative optimization."""
        # Compute probability ratios
        ratio = torch.exp(log_probs - old_log_probs)
        
        # GRPO group-relative normalization
        normalized_advantages = self._group_relative_normalization(advantages)
        
        # Clipped surrogate loss
        surr1 = ratio * normalized_advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * normalized_advantages
        
        return -torch.min(surr1, surr2).mean()
        
    def _group_relative_normalization(self, advantages: torch.Tensor) -> torch.Tensor:
        """Apply group-relative normalization for GRPO."""
        # Normalize advantages within the batch (group)
        mean_adv = advantages.mean()
        std_adv = advantages.std() + 1e-8
        
        return (advantages - mean_adv) / std_adv
        
    def _compute_value_loss(self, values: torch.Tensor, rewards: torch.Tensor, 
                          advantages: torch.Tensor) -> torch.Tensor:
        """Compute value function loss."""
        # Target values
        returns = rewards + advantages
        
        # Clipped value loss
        if hasattr(self, 'value_clip_range'):
            clipped_values = values + torch.clamp(
                values - values.detach(),
                -self.value_clip_range,
                self.value_clip_range
            )
            loss1 = F.mse_loss(values, returns)
            loss2 = F.mse_loss(clipped_values, returns)
            return torch.max(loss1, loss2)
        else:
            return F.mse_loss(values, returns)
            
    def _compute_entropy_loss(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy regularization loss."""
        # Approximate entropy from log probabilities
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.mean()
        
    async def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights for inference coordination."""
        if not self.is_initialized:
            raise RuntimeError("GRPO not initialized")
            
        weights = {}
        
        # Model weights
        if hasattr(self.model, 'state_dict'):
            weights['model'] = {k: v.cpu() for k, v in self.model.state_dict().items()}  # type: ignore
            
        # Value head weights
        if self.value_head is not None:
            weights['value_head'] = {k: v.cpu() for k, v in self.value_head.state_dict().items()}  # type: ignore
            
        self.model_weights = weights
        return weights
        
    async def _extract_model_weights(self) -> Dict[str, torch.Tensor]:
        """Extract model weights during initialization."""
        return await self.get_model_weights()
        
    async def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save GRPO checkpoint."""
        try:
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save model
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(f"{checkpoint_path}/model")  # type: ignore
            else:
                torch.save(self.model.state_dict(), f"{checkpoint_path}/model.pth")  # type: ignore
                
            # Save value head
            if self.value_head is not None:
                torch.save(self.value_head.state_dict(), f"{checkpoint_path}/value_head.pth")
                
            # Save training state
            training_state = {
                'training_step': self.training_step,
                'backend': self.backend,
                'device': self.device,
                'config': self.config.dict()
            }
            
            torch.save(training_state, f"{checkpoint_path}/training_state.pth")
            logger.info(f"GRPO checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save GRPO checkpoint: {e}")
            raise
            
    async def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load GRPO checkpoint."""
        try:
            # Load training state
            training_state = torch.load(f"{checkpoint_path}/training_state.pth")
            self.training_step = training_state.get('training_step', 0)
            
            # Load model
            if hasattr(self.model, 'load_pretrained'):
                self.model.load_pretrained(f"{checkpoint_path}/model")  # type: ignore
            else:
                self.model.load_state_dict(torch.load(f"{checkpoint_path}/model.pth"))  # type: ignore
                
            # Load value head
            value_head_path = f"{checkpoint_path}/value_head.pth"
            if os.path.exists(value_head_path) and self.value_head is not None:
                self.value_head.load_state_dict(torch.load(value_head_path))
                
            logger.info(f"GRPO checkpoint loaded from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load GRPO checkpoint: {e}")
            raise
            
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for monitoring."""
        return {
            'is_initialized': self.is_initialized,
            'training_step': self.training_step,
            'device': self.device,
            'backend': self.backend,
            'memory_usage': self._get_memory_usage()
        }
        
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if self.device.startswith('cuda'):
            return {
                'gpu_allocated': torch.cuda.memory_allocated() / 1e9,
                'gpu_cached': torch.cuda.memory_reserved() / 1e9
            }
        elif self.device == 'mps':
            return {
                'mps_allocated': torch.mps.current_allocated_memory() / 1e9 if hasattr(torch.mps, 'current_allocated_memory') else 0
            }
        else:
            return {'cpu_memory': 0.0}
            
    async def shutdown(self) -> None:
        """Clean shutdown of GRPO actor."""
        try:
            logger.info("Shutting down GRPO actor...")
            
            # Clear GPU memory
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                
            logger.info("GRPO shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during GRPO shutdown: {e}")


# Ray Actor - Clean GRPO implementation
@ray.remote(num_cpus=2, num_gpus=0)
class GRPO(BaseGRPO):
    """
    GRPO Ray Actor - extends BaseGRPO with Ray remote capabilities.
    
    Single class with auto hardware detection, no databuffer concerns.
    Pure algorithm focus with distributed training coordination.
    """
    pass  # Inherits all functionality from BaseGRPO


 