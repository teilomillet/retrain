"""
CPU-only inference actor using Transformers backend.
"""

import logging
from typing import Dict, Any, List
import torch
import ray
import asyncio

from retrain.config_models import TrainingConfig
from .base import BaseInferenceActor
from .models import GenerationResult

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=0, num_cpus=4)
class CPUInferenceActor(BaseInferenceActor):
    """CPU-only inference actor using Transformers backend."""
    
    def __init__(self, config: TrainingConfig, databuffer: ray.ObjectRef):
        super().__init__()
        self.config = config
        self.databuffer = databuffer
        self.backend = "transformers"
        self.device = "cpu"
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        self.generation_config = {
            'max_length': 100,
            'temperature': 0.7,
            'do_sample': True,
            'pad_token_id': None  # Will be set after tokenizer initialization
        }

    async def initialize(self) -> None:
        """Initialize the inference engine."""
        if self.is_initialized:
            return
        
        try:
            await self._initialize_model()
            self.is_initialized = True
            logger.info("CPUInferenceActor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CPUInferenceActor: {e}")
            raise

    async def generate_rollout(self, episode_id: int, rollout_idx: int) -> Dict[str, Any]:
        """Generate a single rollout."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Get prompts from databuffer or generate test prompts
            prompts = [f"Test prompt for episode {episode_id}, rollout {rollout_idx}"]
            
            # Generate responses efficiently
            responses = []
            for prompt in prompts:
                result = await self._generate_with_backend(prompt)
                responses.append(result.response)
            
            rollout_data = {
                'episode_id': episode_id,
                'rollout_idx': rollout_idx,
                'prompts': prompts,
                'responses': responses,
                'backend': self.backend,
                'device': self.device,
                'timestamp': asyncio.get_event_loop().time()
            }
            
            return rollout_data
            
        except Exception as e:
            logger.error(f"Failed to generate rollout: {e}")
            return {
                'episode_id': episode_id,
                'rollout_idx': rollout_idx,
                'error': str(e),
                'backend': self.backend,
                'device': self.device
            }

    async def generate_batch(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[str]:
        """Generate responses for a batch of prompts."""
        if not self.is_initialized:
            await self.initialize()
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized")
        
        try:
            # Update generation config with sampling params
            current_config = self.generation_config.copy()
            current_config.update(sampling_params)
            
            responses = []
            for prompt in prompts:
                # Use efficient single-prompt generation to avoid batch complexity
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **current_config
                    )
                
                # Extract response efficiently
                response_tokens = outputs.sequences[0][len(inputs['input_ids'][0]):]
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                responses.append(response)
            
            return responses
            
        except Exception as e:
            logger.error(f"Failed to generate batch: {e}")
            return [f"Error: {str(e)}" for _ in prompts]

    async def update_model_weights(self, model_weights: Dict[str, Any]) -> None:
        """Update model weights with minimal copying."""
        if not self.is_initialized:
            await self.initialize()
        
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            # Efficient weight update - avoid unnecessary copying
            # Convert weights to CPU tensors if they're not already
            cpu_weights = {}
            for key, weight in model_weights.items():
                if isinstance(weight, torch.Tensor):
                    # Move to CPU only if necessary, avoid copying if already on CPU
                    if weight.device.type != 'cpu':
                        cpu_weights[key] = weight.cpu()
                    else:
                        cpu_weights[key] = weight
                else:
                    cpu_weights[key] = weight
            
            # Load state dict with strict=False to handle partial updates
            self.model.load_state_dict(cpu_weights, strict=False)
            logger.info("Model weights updated efficiently with minimal copying")
            
        except Exception as e:
            logger.error(f"Failed to update model weights: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check actor health and return status."""
        try:
            import psutil
            
            # Get platform info
            platform_info = await self._get_platform_info()
            
            health_data = {
                'status': 'healthy',
                'is_initialized': self.is_initialized,
                'backend': self.backend,
                'device': self.device,
                'model_loaded': self.model is not None,
                'tokenizer_loaded': self.tokenizer is not None,
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'torch_version': torch.__version__,
                'timestamp': asyncio.get_event_loop().time()
            }
            
            health_data.update(platform_info)
            return health_data
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'backend': self.backend,
                'device': self.device,
                'timestamp': asyncio.get_event_loop().time()
            }

    async def _initialize_model(self) -> None:
        """Initialize model using Transformers backend for CPU."""
        try:
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            from transformers.models.auto.tokenization_auto import AutoTokenizer
            
            model_path = getattr(self.config.model, 'hf_checkpoint_path', None) or self.config.model.name_or_path
            
            # CPU-optimized model loading with minimal memory usage
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=getattr(self.config.model, 'trust_remote_code', True)
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Update generation config with pad token
            self.generation_config['pad_token_id'] = self.tokenizer.pad_token_id
            
            # Load model with CPU optimizations and memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,  # CPU-friendly dtype
                device_map="cpu",
                low_cpu_mem_usage=True,  # Minimize peak memory usage
                trust_remote_code=getattr(self.config.model, 'trust_remote_code', True)
            )
            
            # Set model to eval mode to avoid gradient computation
            self.model.eval()
            
            # Ensure model stays on CPU to avoid weight movement
            for param in self.model.parameters():
                if param.device.type != 'cpu':
                    param.data = param.data.cpu()
            
            logger.info(f"Transformers model loaded for CPU inference on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Transformers model: {e}")
            raise

    async def _generate_with_backend(self, prompt: str) -> GenerationResult:
        """Generate response using Transformers backend."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized")
            
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Ensure inputs are on CPU
        inputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Extract response efficiently
        response_tokens = outputs.sequences[0][len(inputs['input_ids'][0]):]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Extract logprobs if available
        logprobs = None
        if hasattr(outputs, 'scores') and outputs.scores:
            logprobs = torch.stack(outputs.scores, dim=1).log_softmax(dim=-1)
        
        return GenerationResult(
            response=response,
            tokens=response_tokens.tolist(),
            logprobs=logprobs
        )

    async def _apply_weights(self, new_weights: Dict[str, torch.Tensor]) -> None:
        """Apply new weights to the model with minimal copying."""
        try:
            # Efficient weight application - reuse update_model_weights logic
            await self.update_model_weights(new_weights)
            
        except Exception as e:
            logger.error(f"Failed to apply weights: {e}")
            raise

    async def _get_platform_info(self) -> Dict[str, Any]:
        """Get CPU-specific platform information."""
        try:
            import psutil
            
            # Safely access cpu_freq since it might be unavailable on some platforms (e.g. certain macOS builds)
            freq = psutil.cpu_freq() if hasattr(psutil, "cpu_freq") else None

            return {
                'platform': 'CPU',
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': freq._asdict() if freq else None,
                'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                'total_memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            }
        except Exception as e:
            logger.warning(f"Failed to get platform info: {e}")
            return {'platform': 'CPU'}

    async def _save_backend_checkpoint(self, checkpoint_path: str) -> None:
        """Save checkpoint using Transformers."""
        if self.model is not None and self.tokenizer is not None:
            self.model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)
        else:
            raise ValueError("No model or tokenizer available for checkpointing")

    async def _cleanup_backend(self) -> None:
        """Perform CPU-specific cleanup."""
        # Clear model references to free memory
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        self.is_initialized = False
        logger.info("CPU backend cleanup completed")

