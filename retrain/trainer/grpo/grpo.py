
import logging
import time
import os
from typing import Any, Dict
import ray
import torch
import torch.nn.functional as F

from ...config_models import TrainingConfig

logger = logging.getLogger(__name__)


# ==========================================
# Base GRPO Actor with Common Logic
# ==========================================

class BaseGRPOActor:
    """Base GRPO implementation with hardware-agnostic algorithm logic."""
    
    def __init__(self, config: TrainingConfig, databuffer: ray.ObjectRef):
        """Initialize GRPO actor with configuration and databuffer reference."""
        self.config = config
        self.databuffer = databuffer
        
        # Model components (will be initialized in initialize())
        self.model = None
        self.optimizer = None
        self.reference_model = None
        
        # GRPO algorithm parameters
        self.eps_clip = getattr(config.algorithm, 'eps_clip', 0.2)
        self.eps_clip_high = getattr(config.algorithm, 'eps_clip_high', 0.28)
        self.kl_coef = getattr(config.algorithm, 'kl_coef', 0.0)
        self.use_kl_loss = getattr(config.algorithm, 'use_kl_loss', True)
        
        # Training state tracking
        self.training_step_count = 0
        self.current_weights = None
        self.is_initialized = False
        self.step_metrics = []
        
        logger.info(f"{self.__class__.__name__} initialized")
        
    async def train_step(self, training_batch: Dict[str, Any], episode_id: int) -> Dict[str, Any]:
        """Execute a single GRPO training step (hardware-agnostic)."""
        if not self.is_initialized or self.model is None:
            raise RuntimeError("GRPOActor not initialized")
            
        step_start_time = time.time()
        
        try:
            # Extract and prepare data
            input_ids = training_batch['input_ids']
            rewards = torch.tensor(training_batch['rewards'], dtype=torch.float32)
            
            # Convert to tensors and move to device
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)
                 
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            rewards = rewards.to(device)
            
            # Get old policy log probabilities and values
            with torch.no_grad():
                old_outputs = self.model(input_ids=input_ids)
                old_logits = old_outputs.logits
                old_log_probs = F.log_softmax(old_logits, dim=-1)
                
                # Get values from value head
                values = self._extract_values(old_outputs, rewards)
                    
            # Compute GRPO advantages
            advantages = self._compute_grpo_advantages(rewards, values)
            
            # Training forward pass
            if self.optimizer is None:
                raise RuntimeError("Optimizer not initialized")
                
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Compute losses
            policy_loss = self._compute_policy_loss(log_probs, old_log_probs, advantages)
            value_loss = self._compute_value_loss(values, rewards)
            
            kl_loss = torch.tensor(0.0, device=device)
            if self.use_kl_loss and self.reference_model is not None:
                kl_loss = self._compute_kl_loss(log_probs, input_ids)
                 
            total_loss = policy_loss + 0.5 * value_loss + self.kl_coef * kl_loss
            
            # Backward pass and optimization
            total_loss.backward()
            if hasattr(self.model, 'parameters'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update state and metrics
            self.training_step_count += 1
            step_time = time.time() - step_start_time
            
            metrics = {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'kl_loss': kl_loss.item(),
                'total_loss': total_loss.item(),
                'advantages_mean': advantages.mean().item(),
                'step_time': step_time,
                'training_step': self.training_step_count,
                'episode_id': episode_id
            }
            
            self.step_metrics.append(metrics)
            if len(self.step_metrics) > 50:
                self.step_metrics.pop(0)
                 
            self.current_weights = self._extract_model_weights()
            
            logger.info(f"GRPO step {self.training_step_count}: loss={total_loss.item():.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            raise
             
    def _compute_grpo_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute GRPO group-relative advantages."""
        # Group baseline (key insight of GRPO)
        group_baseline = rewards.mean()
        advantages = rewards - group_baseline
        
        # Normalize for training stability
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return advantages
         
    def _compute_policy_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, 
                           advantages: torch.Tensor) -> torch.Tensor:
        """Compute PPO-style clipped policy loss."""
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip_high) * advantages
        return -torch.min(surr1, surr2).mean()
         
    def _compute_value_loss(self, values: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Compute value function loss."""
        return F.mse_loss(values, rewards)
         
    def _compute_kl_loss(self, log_probs: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence with reference model."""
        if self.reference_model is None:
            return torch.tensor(0.0, device=log_probs.device)
             
        with torch.no_grad():
            ref_outputs = self.reference_model(input_ids=input_ids)
            ref_log_probs = F.log_softmax(ref_outputs.logits, dim=-1)
             
        return F.kl_div(log_probs, ref_log_probs, log_target=True, reduction='batchmean')
         
    def _extract_model_weights(self) -> Dict[str, torch.Tensor]:
        """Extract model weights for inference actors."""
        if self.model is None:
            return {}
            
        weights = {}
        assert self.model is not None  # Type assertion for linter
        for name, param in self.model.state_dict().items():  # type: ignore
            if "_extra_state" not in name:
                weights[name] = param.detach().cpu().clone()
                    
        return weights

    async def get_current_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights for DataBuffer synchronization."""
        return self.current_weights.copy() if self.current_weights else {}

    async def update_weights(self, new_weights: Dict[str, torch.Tensor]) -> None:
        """Update model weights from DataBuffer."""
        if not self.is_initialized or not new_weights or self.model is None:
            return
            
        try:
            if self.model is not None:  # Explicit type guard
                model_state = self.model.state_dict()
                
                for name, param in model_state.items():  # type: ignore
                    if "_extra_state" not in name and name in new_weights:
                        param.data.copy_(new_weights[name].to(param.device))
                            
            self.current_weights = new_weights.copy()
            logger.info("Model weights updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update weights: {e}")

    # Abstract methods to be implemented by hardware-specific subclasses
    async def initialize(self) -> None:
        """Initialize model, optimizer, and hardware-specific components."""
        raise NotImplementedError("Must be implemented by subclass")
        
    def _extract_values(self, outputs: Any, rewards: torch.Tensor) -> torch.Tensor:
        """Extract value predictions from model outputs."""
        raise NotImplementedError("Must be implemented by subclass")


# ==========================================
# macOS GRPO Actor (Transformers + MPS)
# ==========================================

@ray.remote(num_cpus=2, num_gpus=0)
class MacOSGRPOActor(BaseGRPOActor):
    """macOS-optimized GRPO using Transformers backend with MPS support."""
    
    def __init__(self, config: TrainingConfig, databuffer: ray.ObjectRef):
        super().__init__(config, databuffer)
        self.device = "cpu"  # Will be set to "mps" if available
        self.backend = "transformers"
        
    async def initialize(self) -> None:
        """Initialize GRPO for macOS with Transformers backend."""
        logger.info("Initializing macOS GRPO with Transformers backend...")
        
        # Detect optimal device for macOS
        await self._detect_macos_device()
        
        # Initialize model using Transformers
        await self._initialize_transformers_model()
        
        # Initialize optimizer
        await self._initialize_optimizer()
        
        # Initialize reference model if needed
        if self.use_kl_loss:
            await self._initialize_reference_model()
            
        # Extract initial weights
        self.current_weights = self._extract_model_weights()
        
        self.is_initialized = True
        logger.info(f"macOS GRPO initialization complete on device: {self.device}")
        
    async def _detect_macos_device(self) -> None:
        """Detect optimal device for macOS (MPS vs CPU)."""
        try:
            import torch
            import platform
            
            if platform.system() != "Darwin":
                logger.warning("MacOSGRPOActor being used on non-macOS platform")
            
            # Check for MPS availability (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using MPS (Apple Silicon) for GRPO training")
            else:
                self.device = "cpu"
                logger.info("Using CPU for GRPO training (MPS not available)")
                
        except Exception as e:
            logger.warning(f"Device detection failed: {e}, using CPU")
            self.device = "cpu"
            
    async def _initialize_transformers_model(self) -> None:
        """Initialize model using Transformers backend."""
        try:
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            import torch
            
            model_path = getattr(self.config.model, 'hf_checkpoint_path', None) or self.config.model.name_or_path
            
            # Load model with appropriate device and dtype
            if self.device == "mps":
                # For Apple Silicon, use float16 to save memory
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=getattr(self.config.model, 'trust_remote_code', True)
                )
                self.model = self.model.to(self.device)
            else:
                # For CPU, use float32 for stability
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=getattr(self.config.model, 'trust_remote_code', True)
                )
                
            # Add value head for GRPO (simple linear layer)
            self._add_value_head()
            
            # Set model to training mode
            self.model.train()
                
            logger.info("Transformers model loaded with value head for macOS GRPO")
            
        except Exception as e:
            logger.error(f"Failed to initialize Transformers model: {e}")
            raise
            
    def _add_value_head(self) -> None:
        """Add value head to the model for GRPO."""
        try:
            import torch.nn as nn
            
            # Get hidden size from model config
            hidden_size = self.model.config.hidden_size
            
            # Add value head as a new module
            self.model.value_head = nn.Linear(hidden_size, 1)
            
            # Initialize value head weights
            nn.init.normal_(self.model.value_head.weight, std=0.02)
            nn.init.zeros_(self.model.value_head.bias)
            
            logger.info(f"Added value head with hidden_size={hidden_size}")
            
        except Exception as e:
            logger.error(f"Failed to add value head: {e}")
            raise
            
    def _extract_values(self, outputs: Any, rewards: torch.Tensor) -> torch.Tensor:
        """Extract value predictions from Transformers model."""
        try:
            # Get last hidden state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = outputs.last_hidden_state
                
            # Pass through value head
            values = self.model.value_head(hidden_states)
            return values.squeeze(-1).mean(dim=1)  # Average over sequence length
            
        except Exception as e:
            logger.warning(f"Failed to extract values: {e}, using zero values")
            return torch.zeros_like(rewards)
            
    async def _initialize_optimizer(self) -> None:
        """Initialize optimizer for Transformers model."""
        try:
            learning_rate = getattr(self.config.algorithm, 'learning_rate', 1e-5)  # Lower LR for CPU
            weight_decay = getattr(self.config.algorithm, 'weight_decay', 0.01)
            beta1 = getattr(self.config.algorithm, 'adam_beta1', 0.9)
            beta2 = getattr(self.config.algorithm, 'adam_beta2', 0.999)
            
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(beta1, beta2)
            )
            
            logger.info(f"Optimizer initialized for macOS: lr={learning_rate}")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            raise
            
    async def _initialize_reference_model(self) -> None:
        """Initialize reference model for KL divergence."""
        try:
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            
            model_path = getattr(self.config.model, 'hf_checkpoint_path', None) or self.config.model.name_or_path
            
            # Load reference model (no value head)
            if self.device == "mps":
                self.reference_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                self.reference_model = self.reference_model.to(self.device)
            else:
                self.reference_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                
            # Freeze reference model
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
            
            logger.info("Reference model initialized for macOS")
            
        except Exception as e:
            logger.error(f"Failed to initialize reference model: {e}")
            raise


# ==========================================
# CPU GRPO Actor (Transformers)
# ==========================================

@ray.remote(num_gpus=0, num_cpus=4)
class CPUGRPOActor(BaseGRPOActor):
    """CPU-only GRPO using Transformers backend."""
    
    def __init__(self, config: TrainingConfig, databuffer: ray.ObjectRef):
        super().__init__(config, databuffer)
        self.device = "cpu"
        self.backend = "transformers"
        
    async def initialize(self) -> None:
        """Initialize GRPO for CPU with Transformers backend."""
        logger.info("Initializing CPU GRPO with Transformers backend...")
        
        # Initialize model using Transformers
        await self._initialize_transformers_model()
        
        # Initialize optimizer
        await self._initialize_optimizer()
        
        # Initialize reference model if needed
        if self.use_kl_loss:
            await self._initialize_reference_model()
            
        # Extract initial weights
        self.current_weights = self._extract_model_weights()
        
        self.is_initialized = True
        logger.info("CPU GRPO initialization complete")
        
    async def _initialize_transformers_model(self) -> None:
        """Initialize model using Transformers backend for CPU."""
        try:
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            import torch
            
            model_path = getattr(self.config.model, 'hf_checkpoint_path', None) or self.config.model.name_or_path
            
            # CPU-optimized model loading
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,  # CPU-friendly dtype
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=getattr(self.config.model, 'trust_remote_code', True)
            )
            
            # Add value head for GRPO
            self._add_value_head()
            
            # Set model to training mode
            self.model.train()
                
            logger.info("Transformers model loaded for CPU GRPO with value head")
            
        except Exception as e:
            logger.error(f"Failed to initialize Transformers model: {e}")
            raise
            
    def _add_value_head(self) -> None:
        """Add value head to the model for GRPO."""
        try:
            import torch.nn as nn
            
            # Get hidden size from model config
            hidden_size = self.model.config.hidden_size
            
            # Add value head as a new module
            self.model.value_head = nn.Linear(hidden_size, 1)
            
            # Initialize value head weights
            nn.init.normal_(self.model.value_head.weight, std=0.02)
            nn.init.zeros_(self.model.value_head.bias)
            
            logger.info(f"Added value head with hidden_size={hidden_size}")
            
        except Exception as e:
            logger.error(f"Failed to add value head: {e}")
            raise
            
    def _extract_values(self, outputs: Any, rewards: torch.Tensor) -> torch.Tensor:
        """Extract value predictions from Transformers model."""
        try:
            # Get last hidden state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = outputs.last_hidden_state
                
            # Pass through value head
            values = self.model.value_head(hidden_states)
            return values.squeeze(-1).mean(dim=1)  # Average over sequence length
            
        except Exception as e:
            logger.warning(f"Failed to extract values: {e}, using zero values")
            return torch.zeros_like(rewards)
            
    async def _initialize_optimizer(self) -> None:
        """Initialize optimizer for CPU training."""
        try:
            learning_rate = getattr(self.config.algorithm, 'learning_rate', 1e-5)  # Lower LR for CPU
            weight_decay = getattr(self.config.algorithm, 'weight_decay', 0.01)
            beta1 = getattr(self.config.algorithm, 'adam_beta1', 0.9)
            beta2 = getattr(self.config.algorithm, 'adam_beta2', 0.999)
            
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(beta1, beta2)
            )
            
            logger.info(f"Optimizer initialized for CPU: lr={learning_rate}")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            raise
            
    async def _initialize_reference_model(self) -> None:
        """Initialize reference model for KL divergence."""
        try:
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            
            model_path = getattr(self.config.model, 'hf_checkpoint_path', None) or self.config.model.name_or_path
            
            # Load reference model (no value head) 
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
                
            # Freeze reference model
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
            
            logger.info("Reference model initialized for CPU")
            
        except Exception as e:
            logger.error(f"Failed to initialize reference model: {e}")
            raise


# ==========================================
# CUDA GRPO Actor (MBridge - Cleaned Up)
# ==========================================

@ray.remote(
    num_gpus=1,
    num_cpus=2,
    runtime_env={
        "env_vars": {
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_CUMEM_ENABLE": "0",
        }
    }
)
class CUDAGRPOActor(BaseGRPOActor):
    """CUDA-optimized GRPO using MBridge backend for production training."""
    
    def __init__(self, config: TrainingConfig, databuffer: ray.ObjectRef):
        super().__init__(config, databuffer)
        self.bridge = None
        self.device = "cuda"
        self.backend = "mbridge"
        
    async def initialize(self) -> None:
        """Initialize GRPO for CUDA with MBridge backend."""
        logger.info("Initializing CUDA GRPO with MBridge backend...")
        
        # Only initialize distributed if truly needed (multi-GPU)
        if self._needs_distributed():
            await self._initialize_distributed()
        
        # Initialize model using MBridge
        await self._initialize_mbridge_model()
        
        # Initialize optimizer
        await self._initialize_optimizer()
        
        # Initialize reference model if needed
        if self.use_kl_loss:
            await self._initialize_reference_model()
            
        # Extract initial weights
        self.current_weights = self._extract_model_weights()
        
        self.is_initialized = True
        logger.info("CUDA GRPO initialization complete")
        
    def _needs_distributed(self) -> bool:
        """Check if distributed training is actually needed."""
        tp_size = getattr(self.config.model, 'tensor_parallel_size', 1)
        pp_size = getattr(self.config.model, 'pipeline_parallel_size', 1)
        return tp_size > 1 or pp_size > 1
        
    async def _initialize_distributed(self) -> None:
        """Initialize distributed environment only if needed for multi-GPU."""
        try:
            # Check if already initialized
            if (torch.distributed.is_available() and 
                hasattr(torch.distributed, 'is_initialized') and 
                torch.distributed.is_initialized()):
                logger.info("Distributed already initialized")
                return
                
            from megatron.core import parallel_state as mpu  # type: ignore
            from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # type: ignore
            
            # Simple distributed setup for MBridge
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            
            if (hasattr(torch.distributed, 'is_initialized') and 
                not torch.distributed.is_initialized() and
                hasattr(torch.distributed, 'init_process_group')):
                torch.distributed.init_process_group("nccl")
            
            # Initialize Megatron parallel state
            tp_size = getattr(self.config.model, 'tensor_parallel_size', 1)
            pp_size = getattr(self.config.model, 'pipeline_parallel_size', 1)
            
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=tp_size,
                pipeline_model_parallel_size=pp_size,
            )
            
            model_parallel_cuda_manual_seed(42)
            logger.info(f"MBridge distributed initialized: TP={tp_size}, PP={pp_size}")
            
        except Exception as e:
            logger.warning(f"Distributed initialization failed: {e}, continuing without")
            
    async def _initialize_mbridge_model(self) -> None:
        """Initialize model using MBridge backend."""
        try:
            from mbridge import AutoBridge
            from mbridge.utils.post_creation_callbacks import make_value_model
            
            model_path = getattr(self.config.model, 'hf_checkpoint_path', None) or self.config.model.name_or_path
            self.bridge = AutoBridge.from_pretrained(model_path)
            
            # Get model with value head
            self.model = self.bridge.get_model(
                weight_path=model_path,
                bf16=True,
                post_model_creation_callbacks=[make_value_model]  # type: ignore
            )
            
            # Handle VPP case
            if isinstance(self.model, list):
                logger.info(f"VPP enabled: received {len(self.model)} model chunks")
                self.primary_model = self.model[0]
                for model_chunk in self.model:
                    model_chunk.train()
            else:
                logger.info("Single model received")
                self.primary_model = self.model
                self.model.train()
                
            logger.info("MBridge model loaded with value head for CUDA GRPO")
            
        except Exception as e:
            logger.error(f"Failed to initialize MBridge model: {e}")
            raise
            
    def _extract_values(self, outputs: Any, rewards: torch.Tensor) -> torch.Tensor:
        """Extract value predictions from MBridge model."""
        try:
            # MBridge adds value head as output_layer
            if hasattr(self.primary_model, 'output_layer'):
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    hidden_states = outputs.hidden_states[-1]
                else:
                    hidden_states = outputs.last_hidden_state
                values = self.primary_model.output_layer(hidden_states).squeeze(-1)
                return values.mean(dim=1)  # Average over sequence length
            else:
                logger.warning("No value head found in MBridge model, using zero values")
                return torch.zeros_like(rewards)
                
        except Exception as e:
            logger.warning(f"Failed to extract values: {e}, using zero values")
            return torch.zeros_like(rewards)
            
    async def _initialize_optimizer(self) -> None:
        """Initialize optimizer for CUDA training."""
        try:
            learning_rate = getattr(self.config.algorithm, 'learning_rate', 1e-6)
            weight_decay = getattr(self.config.algorithm, 'weight_decay', 0.1)
            beta1 = getattr(self.config.algorithm, 'adam_beta1', 0.9)
            beta2 = getattr(self.config.algorithm, 'adam_beta2', 0.98)
            
            # Get parameters from all model chunks if VPP
            if isinstance(self.model, list):
                parameters = []
                for model_chunk in self.model:
                    parameters.extend(model_chunk.parameters())
            else:
                parameters = self.model.parameters()
                
            self.optimizer = torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(beta1, beta2)
            )
            
            logger.info(f"Optimizer initialized for CUDA: lr={learning_rate}")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            raise
            
    async def _initialize_reference_model(self) -> None:
        """Initialize reference model using MBridge."""
        try:
            from mbridge import AutoBridge
            
            model_path = getattr(self.config.model, 'hf_checkpoint_path', None) or self.config.model.name_or_path
            ref_bridge = AutoBridge.from_pretrained(model_path)
            
            # Get reference model without value head
            self.reference_model = ref_bridge.get_model(
                weight_path=model_path,
                bf16=True
            )
            
            # Handle VPP and freeze
            if isinstance(self.reference_model, list):
                ref_primary = self.reference_model[0]
            else:
                ref_primary = self.reference_model
                
            for param in ref_primary.parameters():
                param.requires_grad = False
            ref_primary.eval()
            
            logger.info("Reference model initialized for CUDA")
            
        except Exception as e:
            logger.error(f"Failed to initialize reference model: {e}")
            raise


# ==========================================
# Factory Function for Hardware-Aware GRPO
# ==========================================

def create_grpo_actor(config: TrainingConfig, databuffer: ray.ObjectRef):
    """
    Factory function to create hardware-appropriate GRPO actor.
    
    Uses HardwareDetector to determine optimal configuration and backend.
    """
    from ...hardware.detector import HardwareDetector
    
    hardware = HardwareDetector()
    platform = hardware.capabilities['platform']
    
    # Determine actor type based on hardware capabilities
    if platform['is_macos']:
        # macOS: CPU/MPS with Transformers backend
        logger.info("Creating MacOSGRPOActor for macOS environment")
        return MacOSGRPOActor.remote(config, databuffer)  # type: ignore
    elif hardware.capabilities['device']['cuda_available']:
        # CUDA: GPU with MBridge backend
        logger.info("Creating CUDAGRPOActor for CUDA environment")  
        return CUDAGRPOActor.remote(config, databuffer)  # type: ignore
    else:
        # CPU-only: Transformers backend
        logger.info("Creating CPUGRPOActor for CPU-only environment")
        return CPUGRPOActor.remote(config, databuffer)  # type: ignore


# Backward compatibility - use factory function by default
def GRPOActor(config, databuffer):
    """Backward compatibility function that creates appropriate GRPO actor."""
    return create_grpo_actor(config, databuffer)

