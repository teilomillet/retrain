"""
CPU-only inference actor using Transformers backend.
"""

import logging
from typing import Dict, Any
import torch
import ray

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
        self.generation_config = {
            'max_length': 100,
            'temperature': 0.7,
            'do_sample': True,
            'pad_token_id': None  # Will be set after tokenizer initialization
        }

    async def _initialize_model(self) -> None:
        """Initialize model using Transformers backend for CPU."""
        try:
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            from transformers.models.auto.tokenization_auto import AutoTokenizer
            
            model_path = getattr(self.config.model, 'hf_checkpoint_path', None) or self.config.model.name_or_path
            
            # CPU-optimized model loading
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=getattr(self.config.model, 'trust_remote_code', True)
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with CPU optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,  # CPU-friendly dtype
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=getattr(self.config.model, 'trust_remote_code', True)
            )
            
            self.model.eval()
            logger.info(f"Transformers model loaded for CPU inference on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Transformers model: {e}")
            raise

    async def _generate_with_backend(self, prompt: str) -> GenerationResult:
        """Generate response using Transformers backend."""
        inputs = self.tokenizer(prompt, return_tensors="pt")  # type: ignore
        
        with torch.no_grad():
            outputs = self.model.generate(  # type: ignore
                **inputs,
                **self.generation_config
            )
        
        # Extract response
        response_tokens = outputs.sequences[0][len(inputs['input_ids'][0]):]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)  # type: ignore
        
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
        """Apply new weights to the model."""
        try:
            self.model.load_state_dict(new_weights, strict=False)  # type: ignore
            logger.info("New weights applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply weights: {e}")
            raise

    async def _get_platform_info(self) -> Dict[str, Any]:
        """Get CPU-specific platform information."""
        return {
            'platform': 'CPU'
        }

    async def _save_backend_checkpoint(self, checkpoint_path: str) -> None:
        """Save checkpoint using Transformers."""
        if self.model:
            self.model.save_pretrained(checkpoint_path)  # type: ignore
            self.tokenizer.save_pretrained(checkpoint_path)  # type: ignore
        else:
            raise ValueError("No model available for checkpointing")

    async def _cleanup_backend(self) -> None:
        """Perform CPU-specific cleanup."""
        # No special cleanup needed for CPU
        pass

