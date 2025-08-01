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
import math
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
    import mbridge  # type: ignore
    HAS_MBRIDGE = True
except ImportError:
    HAS_MBRIDGE = False

try:
    from unsloth import FastLanguageModel  # type: ignore
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
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
        
        # Adaptive gradient management for stable initial training
        self.gradient_warm_up_steps = 10  # Allow larger gradients in first 10 steps
        self.base_gradient_threshold = 50.0  # Base threshold for extreme gradients
        self.warm_up_gradient_threshold = 200.0  # Higher threshold during warm-up
        
        # GRPO-specific hyperparameters from config.algorithm.hyperparameters
        hyperparams = getattr(config.algorithm, 'hyperparameters', {})
        self.clip_range = hyperparams.get('clip_range', 0.2)
        self.value_clip_range = hyperparams.get('value_clip_range', 0.2)  
        self.entropy_coef = hyperparams.get('entropy_coef', 0.01)
        self.value_coef = hyperparams.get('value_coef', 0.5)
        self.gamma = hyperparams.get('gamma', 0.99)
        self.gae_lambda = hyperparams.get('gae_lambda', 0.95)
        self.learning_rate = hyperparams.get('learning_rate', 1e-4)
        
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
            self._load_model_and_tokenizer()
            
            # Add value head for GRPO
            self._add_value_head()
            
            # Setup optimizer
            self._setup_optimizer()
            
            # Set initialized flag first to allow weight extraction
            self.is_initialized = True
            
            # Initialize model weights cache (now safe to call get_model_weights)
            self.model_weights = await self._extract_model_weights()
            
            logger.info("GRPO initialization complete")
            
        except Exception as e:
            logger.error(f"GRPO initialization failed: {e}")
            raise
            
    def _load_model_and_tokenizer(self) -> None:
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
            # CRITICAL: Use AutoModelForCausalLM for language modeling tasks
            # AutoModel lacks the lm_head needed for token prediction
            self.model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                model_path,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                trust_remote_code=getattr(self.config.model, 'trust_remote_code', False)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore
            
        # Move model to device and ensure gradients are enabled
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)  # type: ignore
            
        # Ensure model parameters require gradients for training
        if hasattr(self.model, 'parameters'):
            for param in self.model.parameters():  # type: ignore
                param.requires_grad_(True)
            
    def _add_value_head(self) -> None:
        """Add value head for GRPO value function estimation."""
        try:
            # Get model hidden size
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):  # type: ignore
                hidden_size = self.model.config.hidden_size  # type: ignore
            else:
                # Default fallback
                hidden_size = 768
                
            # Determine dtype to match model precision (critical for MPS compatibility)
            model_dtype = torch.float32  # Default fallback
            if hasattr(self.model, 'dtype'):
                model_dtype = self.model.dtype  # type: ignore
            elif hasattr(self.model, 'parameters'):
                # Get dtype from first model parameter
                try:
                    first_param = next(self.model.parameters())  # type: ignore
                    model_dtype = first_param.dtype
                except StopIteration:
                    pass
                
            # Create value head with matching dtype (prevents MPS f16+f32 conflicts)
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 1)
            ).to(device=self.device, dtype=model_dtype)  # Explicit dtype matching
            
            logger.info(f"Value head added with hidden size: {hidden_size}, dtype: {model_dtype}")
            
        except Exception as e:
            logger.warning(f"Failed to add value head: {e}")
            # Create minimal fallback value head with safe dtype
            fallback_dtype = torch.float32 if self.device == "cpu" else torch.float16
            self.value_head = nn.Linear(768, 1).to(device=self.device, dtype=fallback_dtype)
            
    def _setup_optimizer(self) -> None:
        """Setup optimizer for GRPO training."""
        # Use learning rate from hyperparameters (already extracted in __init__)
        learning_rate = self.learning_rate
        
        # Combine model and value head parameters
        params = []
        if hasattr(self.model, 'parameters'):
            params.extend(self.model.parameters())  # type: ignore
        if self.value_head is not None:
            params.extend(self.value_head.parameters())
            
        # Create optimizer
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate)
        logger.info(f"Optimizer setup complete with LR: {learning_rate}")
        
    def _reset_optimizer_state(self) -> None:
        """Reset optimizer state to recover from corrupted gradients/momentum.
        
        This method clears accumulated momentum and state that may be corrupted
        when NaN/Inf gradients are detected, preventing corruption propagation.
        Critical for recovery after large gradient explosions (Episode 0 → Episode 1+ issue).
        """
        try:
            logger.info("Resetting optimizer state to recover from gradient corruption")
            
            # Clear optimizer state (momentum, variance estimates, etc.)
            if hasattr(self.optimizer, 'state'):
                self.optimizer.state.clear()  # type: ignore
            
            # Reinitialize optimizer with fresh state to break corruption chain
            learning_rate = self.learning_rate
            params = []
            if hasattr(self.model, 'parameters'):
                params.extend(self.model.parameters())  # type: ignore
            if self.value_head is not None:
                params.extend(self.value_head.parameters())
                
            self.optimizer = torch.optim.AdamW(params, lr=learning_rate)
            
            logger.info("Optimizer state reset complete - fresh momentum/variance estimates")
            
        except Exception as e:
            logger.warning(f"Failed to reset optimizer state: {e}")
            # Fallback: continue with potentially corrupted state
        
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
            # Extract batch components with correct dtypes and proper tensor handling
            # Use detach().clone() for existing tensors to avoid warnings
            raw_input_ids = training_batch['input_ids']
            if isinstance(raw_input_ids, torch.Tensor):
                input_ids = raw_input_ids.detach().clone().to(dtype=torch.long, device=self.device)
            else:
                input_ids = torch.tensor(raw_input_ids, dtype=torch.long, device=self.device)
            
            raw_attention_mask = training_batch.get('attention_mask', [])
            if isinstance(raw_attention_mask, torch.Tensor):
                attention_mask = raw_attention_mask.detach().clone().to(dtype=torch.long, device=self.device)
            else:
                attention_mask = torch.tensor(raw_attention_mask, dtype=torch.long, device=self.device)
                
            raw_rewards = training_batch.get('rewards', [])
            if isinstance(raw_rewards, torch.Tensor):
                rewards = raw_rewards.detach().clone().to(dtype=torch.float32, device=self.device)
            else:
                rewards = torch.tensor(raw_rewards, dtype=torch.float32, device=self.device)
                
            raw_old_log_probs = training_batch.get('old_log_probs', [])
            if isinstance(raw_old_log_probs, torch.Tensor):
                old_log_probs = raw_old_log_probs.detach().clone().to(dtype=torch.float32, device=self.device)
            else:
                old_log_probs = torch.tensor(raw_old_log_probs, dtype=torch.float32, device=self.device)
            
            # Forward pass - Enable gradients for training
            # CRITICAL FIX: ALL backends require explicit output_hidden_states=True for value head computation
            # Transformers, MBridge, and Unsloth all need hidden states to be explicitly requested
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True  # ESSENTIAL: Required for all backends to provide hidden states
            )  # type: ignore

            # Extract logits and compute log probabilities
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            log_probs_full = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
            
            # Extract action log probabilities for the actual tokens
            # Shift input_ids for autoregressive training (predict next token)
            if input_ids.size(1) > 1:
                target_ids = input_ids[:, 1:]  # Remove first token
                log_probs_full_shifted = log_probs_full[:, :-1, :]  # Remove last prediction
            else:
                target_ids = input_ids  # Use as-is for single token
                log_probs_full_shifted = log_probs_full
            
            # Gather the log probabilities of the actual tokens
            log_probs = torch.gather(log_probs_full_shifted, -1, target_ids.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]
            
            # Critical: Validate current log probabilities before proceeding
            # Early detection prevents NaN propagation through the entire training step
            if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                logger.error("NaN/Inf detected in current log_probs from model forward pass")
                logger.error(f"  Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
                logger.error("Model may be corrupted - attempting recovery")
                
                # CRITICAL: Attempt model parameter recovery
                try:
                    # Reset model parameters to previous safe state if available
                    if hasattr(self, 'model_weights') and self.model_weights:
                        logger.info("Attempting to restore model from last known good state")
                        await self._restore_model_weights(self.model_weights)
                        
                        # Reset optimizer state to prevent corrupted momentum
                        self._reset_optimizer_state()
                        
                        # Try forward pass again with restored model
                        outputs_restored = self.model(input_ids=input_ids, attention_mask=attention_mask)  # type: ignore
                        logits_restored = outputs_restored.logits if hasattr(outputs_restored, 'logits') else outputs_restored[0]
                        
                        if not (torch.isnan(logits_restored).any() or torch.isinf(logits_restored).any()):
                            logger.info("Model recovery successful - continuing with restored parameters")
                            # Update outputs to use restored results
                            outputs = outputs_restored
                            logits = logits_restored
                            log_probs_full = F.log_softmax(logits, dim=-1)
                            
                            # Recompute log_probs with restored model output
                            if input_ids.size(1) > 1:
                                target_ids = input_ids[:, 1:]  # Remove first token
                                log_probs_full_shifted = log_probs_full[:, :-1, :]  # Remove last prediction
                            else:
                                target_ids = input_ids  # Use as-is for single token
                                log_probs_full_shifted = log_probs_full
                            
                            log_probs = torch.gather(log_probs_full_shifted, -1, target_ids.unsqueeze(-1)).squeeze(-1)
                        else:
                            raise RuntimeError("Model still corrupted after restoration attempt")
                    else:
                        raise RuntimeError("No previous model state available for recovery")
                        
                except Exception as recovery_error:
                    logger.error(f"Model recovery failed: {recovery_error}")
                    logger.error("Skipping training step to prevent further corruption")
                    return {
                        'loss': float('nan'),
                        'policy_loss': float('nan'), 
                        'value_loss': float('nan'),
                        'entropy_loss': float('nan'),
                        'mean_advantage': 0.0,
                        'mean_reward': rewards.mean().item() if not torch.isnan(rewards).any() else 0.0,
                        'step_time': time.time() - step_start_time,
                        'training_step': self.training_step,
                        'device': self.device,
                        'backend': self.backend,
                        'error': 'corrupted_model_logits_recovery_failed'
                    }
            
            # Apply the same autoregressive shifting to old_log_probs to match current log_probs
            # This ensures both tensors represent the same token positions for accurate policy ratios
            if input_ids.size(1) > 1:
                # Apply same shifting as current log_probs: remove first token position
                # This aligns old_log_probs with the autoregressive next-token prediction pattern
                old_log_probs_shifted = old_log_probs[:, 1:]  # Remove first position to match target_ids
            else:
                # Single token case - no shifting needed
                old_log_probs_shifted = old_log_probs
            
            # Handle any remaining shape mismatch after proper shifting alignment
            if old_log_probs_shifted.shape != log_probs.shape:
                logger.warning(f"Old log probs shape {old_log_probs_shifted.shape} doesn't match current shape {log_probs.shape} after autoregressive alignment")
                # Apply minimal correction - this should rarely happen with proper alignment
                if old_log_probs_shifted.size(1) > log_probs.size(1):
                    old_log_probs_shifted = old_log_probs_shifted[:, :log_probs.size(1)]  # Trim excess
                elif old_log_probs_shifted.size(1) < log_probs.size(1):
                    # Pad with zeros (should be rare with proper shifting)
                    padding_size = log_probs.size(1) - old_log_probs_shifted.size(1)
                    padding = torch.zeros(old_log_probs_shifted.size(0), padding_size, 
                                       device=self.device, dtype=old_log_probs_shifted.dtype)
                    old_log_probs_shifted = torch.cat([old_log_probs_shifted, padding], dim=1)
            
            # Use the properly aligned old log probabilities for policy ratio computation
            old_log_probs = old_log_probs_shifted
            
            # Critical: Validate old log probabilities after alignment
            # Corrupted old_log_probs lead to extreme probability ratios and NaN losses
            if torch.isnan(old_log_probs).any() or torch.isinf(old_log_probs).any():
                logger.error("NaN/Inf detected in old_log_probs after autoregressive alignment")
                logger.error(f"  Raw old_log_probs stats: min={raw_old_log_probs.min().item():.4f}, max={raw_old_log_probs.max().item():.4f}")
                logger.error("Training batch contains corrupted old log probabilities")
                return {
                    'loss': float('nan'),
                    'policy_loss': float('nan'), 
                    'value_loss': float('nan'),
                    'entropy_loss': float('nan'),
                    'mean_advantage': 0.0,
                    'mean_reward': rewards.mean().item() if not torch.isnan(rewards).any() else 0.0,
                    'step_time': time.time() - step_start_time,
                    'training_step': self.training_step,
                    'device': self.device,
                    'backend': self.backend,
                    'error': 'corrupted_old_logprobs'
                }
            
            # Get value estimates with numerical stability safeguards
            if self.value_head is not None:
                try:
                    # CRITICAL FIX: Standardized hidden states extraction across all backends
                    # All backends now provide outputs.hidden_states due to output_hidden_states=True
                    hidden_states = None
                    
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        # STANDARD APPROACH: Extract final layer hidden states (works for all backends)
                        # hidden_states is a tuple/list where [-1] contains the final layer representations
                        hidden_states = outputs.hidden_states[-1]
                        logger.debug(f"Extracted hidden_states from .hidden_states[-1]: {hidden_states.shape}")
                        
                    elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                        # LEGACY FALLBACK: Some custom models might still provide this attribute
                        hidden_states = outputs.last_hidden_state
                        logger.debug(f"Extracted hidden_states from .last_hidden_state: {hidden_states.shape}")
                        
                    else:
                        # FINAL FALLBACK: Try outputs[0] but validate dimensions first
                        # This should rarely be needed with output_hidden_states=True
                        potential_hidden_states = outputs[0]
                        
                        # DIMENSION VALIDATION: Check if this could be hidden states vs logits
                        expected_hidden_size = getattr(self.model.config, 'hidden_size', 768)
                        actual_last_dim = potential_hidden_states.shape[-1]
                        
                        if actual_last_dim == expected_hidden_size:
                            # Dimensions match: likely hidden states
                            hidden_states = potential_hidden_states
                            logger.debug(f"Using outputs[0] as hidden_states (dims match): {hidden_states.shape}")
                        else:
                            # Dimensions don't match: likely logits (vocab_size), not hidden states
                            logger.error(f"Backend {self.backend}: All hidden states extraction methods failed")
                            logger.error(f"  outputs.hidden_states: {hasattr(outputs, 'hidden_states')}")
                            logger.error(f"  outputs.last_hidden_state: {hasattr(outputs, 'last_hidden_state')}")
                            logger.error(f"  outputs[0] shape: {potential_hidden_states.shape} (expected {expected_hidden_size})")
                            logger.error("This suggests a fundamental backend compatibility issue")
                            hidden_states = None

                    # Proceed only if we successfully extracted valid hidden states
                    if hidden_states is not None:
                        # Final dimension validation before value head computation
                        expected_hidden_size = getattr(self.model.config, 'hidden_size', 768)
                        actual_hidden_size = hidden_states.shape[-1]
                        
                        if actual_hidden_size != expected_hidden_size:
                            logger.error(f"Hidden states dimension mismatch:")
                            logger.error(f"  Model config hidden_size: {expected_hidden_size}")
                            logger.error(f"  Actual hidden_states shape: {hidden_states.shape}")
                            logger.error(f"  Cannot proceed with value head computation")
                            values = torch.zeros_like(rewards, requires_grad=True)
                        
                        # Critical: Validate hidden states before value head computation
                        elif torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                            logger.warning("NaN/Inf detected in hidden states, using zero values")
                            values = torch.zeros_like(rewards, requires_grad=True)
                        else:
                            # Value head computation with proper dimension handling
                            # hidden_states shape: [batch_size, seq_len, hidden_size]
                            # We need to reduce to [batch_size, hidden_size] for value head input
                            
                            logger.debug(f"Hidden states original shape: {hidden_states.shape}, dims: {hidden_states.dim()}")
                            
                            # Handle different hidden states dimensions properly
                            hidden_states_reduced = None
                            if hidden_states.dim() == 3:
                                # Take the mean across sequence dimension to get per-sample representation
                                # This gives us [batch_size, hidden_size] suitable for value head
                                hidden_states_reduced = hidden_states.mean(dim=1)  # Average over sequence length
                                logger.debug(f"3D case: Reduced shape {hidden_states_reduced.shape}")
                            elif hidden_states.dim() == 2:
                                # Already in correct shape [batch_size, hidden_size]
                                hidden_states_reduced = hidden_states
                                logger.debug(f"2D case: Using original shape {hidden_states_reduced.shape}")
                            else:
                                # Unexpected dimensions - this should be avoided for value head compatibility
                                logger.error(f"Unexpected hidden_states shape {hidden_states.shape} with {hidden_states.dim()} dimensions")
                                logger.error("Cannot proceed with value head computation - falling back to zero values")
                                values = torch.zeros_like(rewards, requires_grad=True)
                                
                            # Only proceed with value head if we have properly shaped hidden states
                            if hidden_states_reduced is not None:
                                # Now apply value head with correct input dimensions
                                value_outputs = self.value_head(hidden_states_reduced).squeeze(-1)
                                
                                # Validate value head outputs before proceeding
                                if torch.isnan(value_outputs).any() or torch.isinf(value_outputs).any():
                                    logger.warning("NaN/Inf detected in value head output, using zero values")
                                    values = torch.zeros_like(rewards, requires_grad=True)
                                else:
                                    # For GRPO, we need per-sample values, not per-token
                                    # Take mean across sequence dimension to get [batch_size] shape
                                    if value_outputs.dim() > 1:
                                        values = value_outputs.mean(dim=-1)  # Average across sequence
                                    else:
                                        values = value_outputs
                                        
                                    # Final validation after shape processing
                                    if torch.isnan(values).any() or torch.isinf(values).any():
                                        logger.warning("NaN/Inf detected in processed values, using zero values")
                                        values = torch.zeros_like(rewards, requires_grad=True)
                                    else:
                                        # Ensure values matches rewards shape exactly  
                                        if values.shape != rewards.shape:
                                            logger.warning(f"Value shape {values.shape} doesn't match rewards shape {rewards.shape}, reshaping...")
                                            values = values.view(rewards.shape)
                                            
                                        # Apply value bounds to prevent extreme estimates that cause GAE instability
                                        # Clamp values to reasonable range based on typical reward scales
                                        values = torch.clamp(values, min=-100.0, max=100.0)
                            else:
                                # hidden_states_reduced is None - set fallback values
                                logger.warning("Could not reduce hidden states to proper dimensions")
                                values = torch.zeros_like(rewards, requires_grad=True)
                    else:
                        # CRITICAL FIX: Handle the case when hidden_states is None
                        # This happens when we can't extract valid hidden states from any backend
                        logger.warning(f"Backend {self.backend}: Could not extract hidden states for value head computation")
                        logger.warning("Falling back to zero values - this may impact training quality")
                        values = torch.zeros_like(rewards, requires_grad=True)
                        
                except Exception as e:
                    logger.warning(f"Value head computation failed: {e}, using zero values")
                    values = torch.zeros_like(rewards, requires_grad=True)
            else:
                values = torch.zeros_like(rewards, requires_grad=True)  # Enable gradients even for fallback
                    
            # Compute advantages using GAE
            advantages = self._compute_advantages(rewards, values)
            
            # GRPO loss computation
            policy_loss = self._compute_policy_loss(log_probs, old_log_probs, advantages)
            value_loss = self._compute_value_loss(values, rewards, advantages)
            entropy_loss = self._compute_entropy_loss(log_probs_full_shifted)  # Use shifted full distribution for entropy
            
            # Combined loss with adaptive coefficient scaling for initial training stability
            # Reduce value loss impact during warm-up to prevent gradient explosion
            if self.training_step < self.gradient_warm_up_steps:
                # Warm-up: reduce value coefficient to prevent untrained value head from dominating
                warm_up_value_coef = self.value_coef * 0.1  # 10% of normal value coefficient
                warm_up_entropy_coef = self.entropy_coef * 0.5  # 50% of normal entropy coefficient
                logger.debug(f"Warm-up mode: value_coef={warm_up_value_coef:.3f}, entropy_coef={warm_up_entropy_coef:.3f}")
            else:
                # Normal training: full coefficients
                warm_up_value_coef = self.value_coef
                warm_up_entropy_coef = self.entropy_coef
            
            total_loss = (policy_loss + 
                         warm_up_value_coef * value_loss - 
                         warm_up_entropy_coef * entropy_loss)
            
            # Numerical stability check: detect NaN/inf values before backward pass
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f"NaN/Inf detected in total_loss: {total_loss.item()}")
                logger.warning(f"  policy_loss: {policy_loss.item()}, value_loss: {value_loss.item()}, entropy_loss: {entropy_loss.item()}")
                logger.warning("Skipping optimization step to prevent gradient corruption")
                # Return NaN metrics to indicate the issue
                return {
                    'loss': float('nan'),
                    'policy_loss': float('nan'), 
                    'value_loss': float('nan'),
                    'entropy_loss': float('nan'),
                    'mean_advantage': advantages.mean().item(),
                    'mean_reward': rewards.mean().item(),
                    'step_time': time.time() - step_start_time,
                    'training_step': self.training_step,
                    'device': self.device,
                    'backend': self.backend
                }
            
            # Backward pass and optimization
            self.optimizer.zero_grad()  # type: ignore
            total_loss.backward()
            
            # Gradient clipping with gradient norm monitoring
            all_params = list(self.model.parameters()) + list(self.value_head.parameters())  # type: ignore
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            
            # Adaptive gradient health monitoring with warm-up training support
            # Use higher threshold during initial training to allow model to stabilize
            current_threshold = (self.warm_up_gradient_threshold if self.training_step < self.gradient_warm_up_steps 
                               else self.base_gradient_threshold)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logger.error(f"NaN/Inf gradients detected (norm: {grad_norm})")
                logger.error("Model parameters are corrupted - reinitializing optimizer state")
                
                # Critical: Reset optimizer state to prevent accumulated corruption
                self.optimizer.zero_grad()  # Clear corrupted gradients
                self._reset_optimizer_state()  # Reset momentum/state that may be corrupted
                
                return {
                    'loss': total_loss.item(),
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(), 
                    'entropy_loss': entropy_loss.item(),
                    'mean_advantage': advantages.mean().item(),
                    'mean_reward': rewards.mean().item(),
                    'step_time': time.time() - step_start_time,
                    'training_step': self.training_step,
                    'device': self.device,
                    'backend': self.backend,
                    'grad_norm': float('nan'),
                    'optimizer_skipped': True,
                    'recovery_action': 'optimizer_state_reset'
                }
            elif grad_norm > current_threshold:
                # Extreme gradient explosion - skip only if beyond adaptive threshold
                warm_up_status = "WARM-UP" if self.training_step < self.gradient_warm_up_steps else "NORMAL"
                logger.warning(f"Extreme gradient norm detected: {grad_norm:.4f} > {current_threshold:.1f} ({warm_up_status} mode)")
                logger.warning("Skipping optimizer step to prevent potential model corruption")
                self.optimizer.zero_grad()  # Clear extreme gradients
                
                return {
                    'loss': total_loss.item(),
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(), 
                    'entropy_loss': entropy_loss.item(),
                    'mean_advantage': advantages.mean().item(),
                    'mean_reward': rewards.mean().item(),
                    'step_time': time.time() - step_start_time,
                    'training_step': self.training_step,
                    'device': self.device,
                    'backend': self.backend,
                    'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                    'optimizer_skipped': True,
                    'recovery_action': 'extreme_gradient_skip',
                    'gradient_threshold': current_threshold
                }
            elif grad_norm > 10.0:
                warm_up_status = "WARM-UP" if self.training_step < self.gradient_warm_up_steps else "NORMAL"
                logger.info(f"Large gradient norm detected: {grad_norm:.4f} (clipped to 1.0) - {warm_up_status} mode")
            
            # Apply optimizer step only with healthy gradients
            self.optimizer.step()  # type: ignore
            
            # Update training state
            self.training_step += 1
            step_time = time.time() - step_start_time
            
            # Save model weights after successful training step for recovery purposes
            try:
                if self.training_step % 1 == 0:  # Save every step initially for safety
                    self.model_weights = await self._extract_model_weights()
                    logger.debug(f"Model weights saved after successful step {self.training_step}")
            except Exception as e:
                logger.warning(f"Failed to save model weights for recovery: {e}")
            
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
                'backend': self.backend,
                'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                'advantage_std': advantages.std().item(),  # Monitor advantage variance
                'optimizer_skipped': False
            }
            
            logger.info(f"GRPO step {self.training_step} completed - Loss: {total_loss.item():.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"GRPO training step failed: {e}")
            raise
            
    def _compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute GAE advantages with robust numerical stability and NaN prevention."""
        
        # Input validation to prevent NaN propagation in GAE computation
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            logger.warning("NaN/Inf detected in rewards, using zero advantages")
            return torch.zeros_like(rewards, requires_grad=True)
            
        if torch.isnan(values).any() or torch.isinf(values).any():
            logger.warning("NaN/Inf detected in values, using zero advantages")
            return torch.zeros_like(rewards, requires_grad=True)
        
        batch_size = rewards.shape[0]
        
        # Build advantages list first (no in-place operations on tensors with gradients)
        advantages_list = []
        gae = torch.tensor(0.0, device=self.device, dtype=rewards.dtype)  # Start with scalar
        
        # GAE computation in reverse order with numerical safeguards
        for i in reversed(range(batch_size)):
            if i == batch_size - 1:
                next_value = torch.tensor(0.0, device=self.device, dtype=values.dtype)
            else:
                next_value = values[i + 1]
                
            # Ensure all values are scalars for the computation
            # Extract scalar values from tensors (handles both 0-dim and multi-dim tensors)
            reward_i = rewards[i].item() if isinstance(rewards[i], torch.Tensor) else rewards[i]
            value_i = values[i].item() if isinstance(values[i], torch.Tensor) else values[i]
            next_val = next_value.item() if isinstance(next_value, torch.Tensor) else next_value
            gae_val = gae.item() if isinstance(gae, torch.Tensor) else gae
            
            # Critical: Validate extracted scalars for NaN/Inf before computation
            if not (math.isfinite(reward_i) and math.isfinite(value_i) and 
                    math.isfinite(next_val) and math.isfinite(gae_val)):
                logger.warning(f"Non-finite values in GAE computation at step {i}: "
                             f"reward={reward_i}, value={value_i}, next_val={next_val}, gae={gae_val}")
                logger.warning("Using zero advantages to prevent corruption")
                return torch.zeros_like(rewards, requires_grad=True)
            
            # Pure scalar computation with bounds checking (avoids MPS shape conflicts)
            delta = reward_i + self.gamma * next_val - value_i
            gae_val = delta + self.gamma * self.gae_lambda * gae_val
            
            # Apply bounds to prevent GAE explosion that leads to NaN in normalization
            gae_val = max(min(gae_val, 1000.0), -1000.0)  # Reasonable bounds for GAE values
            
            # Store scalar in list (no in-place tensor operations)
            advantages_list.append(gae_val)
            # Convert scalar back to tensor for next iteration (ensures proper type)
            gae = torch.scalar_tensor(gae_val, device=self.device, dtype=rewards.dtype)
        
        # Reverse the list to get correct order (we computed in reverse)
        advantages_list.reverse()
        
        # Convert to tensor (this creates a new tensor, not in-place operation)
        # Preserves gradients properly for autograd computation graph
        advantages = torch.tensor(advantages_list, device=self.device, dtype=rewards.dtype, requires_grad=True)
        
        # Final validation of computed advantages
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            logger.warning("NaN/Inf detected in computed advantages, using zero advantages")
            return torch.zeros_like(rewards, requires_grad=True)
            
        return advantages
        
    def _compute_policy_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, 
                           advantages: torch.Tensor) -> torch.Tensor:
        """Compute GRPO policy loss with robust numerical stability and NaN prevention."""
        # log_probs shape: [batch_size, seq_len]
        # old_log_probs shape: [batch_size, seq_len] 
        # advantages shape: [batch_size]
        
        # Step 1: Validate input tensors for NaN/Inf (early detection prevents propagation)
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            logger.warning("NaN/Inf detected in current log_probs, using fallback policy loss")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        if torch.isnan(old_log_probs).any() or torch.isinf(old_log_probs).any():
            logger.warning("NaN/Inf detected in old_log_probs, using fallback policy loss") 
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        # Step 2: Compute log probability differences with bounds checking
        # Clamp the difference to prevent exp() overflow/underflow that causes NaN
        log_ratio = log_probs - old_log_probs  # [batch_size, seq_len]
        
        # Critical: Bound log ratios to prevent exp() numerical issues
        # exp(-10) ≈ 4.5e-5 (very small but safe), exp(10) ≈ 22026 (large but manageable)
        log_ratio_clamped = torch.clamp(log_ratio, min=-10.0, max=10.0)
        
        # Log bounds violations for debugging
        extreme_ratios = (log_ratio.abs() > 10.0).sum().item()
        if extreme_ratios > 0:
            logger.debug(f"Clamped {extreme_ratios} extreme log ratios (|ratio| > 10.0)")
        
        # Step 3: Compute probability ratios with numerical safety
        ratio = torch.exp(log_ratio_clamped)  # [batch_size, seq_len] - now bounded between ~4.5e-5 and ~22026
        
        # Step 4: Validate ratio tensor before proceeding
        if torch.isnan(ratio).any() or torch.isinf(ratio).any():
            logger.warning("NaN/Inf detected in probability ratios despite clamping, using fallback")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Step 5: GRPO group-relative normalization with built-in stability
        normalized_advantages = self._group_relative_normalization(advantages)  # [batch_size]
        
        # Step 6: Validate normalized advantages
        if torch.isnan(normalized_advantages).any() or torch.isinf(normalized_advantages).any():
            logger.warning("NaN/Inf detected in normalized advantages, using fallback")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Step 7: Expand advantages to match sequence dimension for token-level computation
        normalized_advantages_expanded = normalized_advantages.unsqueeze(1).expand(-1, log_probs.size(1))  # [batch_size, seq_len]
        
        # Step 8: Compute surrogate losses with additional ratio bounds (PPO-style clipping)
        # Apply more aggressive clipping to prevent extreme policy updates
        surr1 = ratio * normalized_advantages_expanded
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * normalized_advantages_expanded
        
        # Step 9: Final validation before computing loss
        policy_loss_raw = -torch.min(surr1, surr2).mean()
        
        if torch.isnan(policy_loss_raw) or torch.isinf(policy_loss_raw):
            logger.warning("NaN/Inf detected in final policy loss computation, using fallback")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return policy_loss_raw
        
    def _group_relative_normalization(self, advantages: torch.Tensor) -> torch.Tensor:
        """Apply group-relative normalization for GRPO with numerical stability."""
        # Normalize advantages within the batch (group)
        mean_adv = advantages.mean()
        std_adv = advantages.std()
        
        # Enhanced numerical stability: only normalize if std is significant
        # This prevents division by near-zero values that cause NaN gradients
        if std_adv < 1e-6:
            # When advantages have very low variance, just center them
            # This maintains GRPO's group-relative property without numerical instability
            logger.debug(f"Low advantage variance ({std_adv:.2e}), using mean-centering only")
            return advantages - mean_adv
        else:
            # Standard GRPO normalization with safe epsilon
            return (advantages - mean_adv) / (std_adv + 1e-8)
        
    def _compute_value_loss(self, values: torch.Tensor, rewards: torch.Tensor, 
                          advantages: torch.Tensor) -> torch.Tensor:
        """Compute value function loss with MPS dtype compatibility."""
        # Ensure all tensors have compatible dtypes for MPS operations
        target_dtype = values.dtype  # Use value head's dtype as reference
        
        # Convert rewards and advantages to match values dtype
        rewards_converted = rewards.to(dtype=target_dtype, device=values.device)
        advantages_converted = advantages.to(dtype=target_dtype, device=values.device)
        
        # Target values (ensure same dtype)
        returns = rewards_converted + advantages_converted
        
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
            
    def _compute_entropy_loss(self, log_probs_full: torch.Tensor) -> torch.Tensor:
        """Compute entropy regularization loss from full vocabulary log probabilities."""
        # log_probs_full shape: [batch_size, seq_len, vocab_size]
        # Compute entropy across vocabulary dimension
        probs = torch.exp(log_probs_full)
        entropy = -(probs * log_probs_full).sum(dim=-1)  # [batch_size, seq_len]
        return entropy.mean()  # Average across batch and sequence
        
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
    
    async def _restore_model_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Restore model weights from saved state."""
        if not weights or self.model is None:
            raise RuntimeError("No weights to restore or model not available")
            
        try:
            if hasattr(self.model, 'load_state_dict'):
                # Filter weights to match current model structure
                current_state = self.model.state_dict()  # type: ignore
                filtered_weights = {}
                
                for name, param in weights.items():
                    if name in current_state and current_state[name].shape == param.shape:
                        filtered_weights[name] = param.to(device=self.device)
                    else:
                        logger.warning(f"Skipping weight {name} due to shape mismatch or missing key")
                
                # Restore filtered weights
                self.model.load_state_dict(filtered_weights, strict=False)  # type: ignore
                logger.info(f"Restored {len(filtered_weights)} model parameters from saved state")
                
        except Exception as e:
            logger.error(f"Failed to restore model weights: {e}")
            raise RuntimeError(f"Model weight restoration failed: {e}")
        
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


 