"""
Base inference actor with shared functionality.
"""

import ray
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class BaseInferenceActor(ABC):
    """Base class for all inference actors with common interface"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the inference engine"""
        pass
    
    @abstractmethod
    async def generate_rollout(self, episode_id: int, rollout_idx: int) -> Dict[str, Any]:
        """Generate a single rollout"""
        pass
    
    @abstractmethod
    async def generate_batch(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[str]:
        """Generate responses for a batch of prompts"""
        pass
    
    @abstractmethod
    async def update_model_weights(self, model_weights: Dict[str, Any]) -> None:
        """Update model weights from training"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check actor health and return status"""
        pass


@ray.remote(num_cpus=1, num_gpus=0)
class CPUInferenceActor(BaseInferenceActor):
    """CPU-only inference actor for basic models and fallback scenarios"""
    
    def __init__(self, config: Any, databuffer: ray.ObjectRef):
        self.config = config
        self.databuffer = databuffer
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        
        logger.info("CPUInferenceActor initialized")
    
    async def initialize(self) -> None:
        """Initialize CPU-based inference engine"""
        try:
            # Import here to avoid GPU dependencies
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            from transformers.models.auto.tokenization_auto import AutoTokenizer
            import torch
            
            model_name = getattr(self.config, 'model_name', 'microsoft/DialoGPT-medium')
            
            # Load model on CPU
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.is_initialized = True
            logger.info(f"CPUInferenceActor initialized with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CPUInferenceActor: {e}")
            raise
    
    async def generate_rollout(self, episode_id: int, rollout_idx: int) -> Dict[str, Any]:
        """Generate a single rollout using CPU inference"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Get prompts from databuffer or generate test prompts
            prompts = [f"Test prompt for episode {episode_id}, rollout {rollout_idx}"]
            
            # Generate responses
            responses = await self.generate_batch(prompts, {
                'max_length': 100,
                'temperature': 0.7,
                'do_sample': True
            })
            
            rollout_data = {
                'episode_id': episode_id,
                'rollout_idx': rollout_idx,
                'prompts': prompts,
                'responses': responses,
                'backend': 'cpu',
                'timestamp': asyncio.get_event_loop().time()
            }
            
            return rollout_data
            
        except Exception as e:
            logger.error(f"Failed to generate rollout: {e}")
            return {
                'episode_id': episode_id,
                'rollout_idx': rollout_idx,
                'error': str(e),
                'backend': 'cpu'
            }
    
    async def generate_batch(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[str]:
        """Generate responses for a batch of prompts using CPU"""
        if not self.is_initialized:
            await self.initialize()
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized")
        
        try:
            import torch
            
            responses = []
            for prompt in prompts:
                # Tokenize input
                inputs = self.tokenizer.encode(prompt, return_tensors='pt')
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=sampling_params.get('max_length', 100),
                        temperature=sampling_params.get('temperature', 0.7),
                        do_sample=sampling_params.get('do_sample', True),
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the original prompt from response
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
                
                responses.append(response)
            
            return responses
            
        except Exception as e:
            logger.error(f"Failed to generate batch: {e}")
            return [f"Error: {str(e)}" for _ in prompts]
    
    async def update_model_weights(self, model_weights: Dict[str, Any]) -> None:
        """Update model weights (CPU implementation)"""
        try:
            if self.model is not None:
                # Simple weight update for CPU model
                self.model.load_state_dict(model_weights, strict=False)
                logger.info("Model weights updated successfully")
        except Exception as e:
            logger.error(f"Failed to update model weights: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check actor health"""
        try:
            import psutil
            import torch
            
            return {
                'status': 'healthy',
                'is_initialized': self.is_initialized,
                'backend': 'cpu',
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'torch_version': torch.__version__,
                'model_loaded': self.model is not None,
                'timestamp': asyncio.get_event_loop().time()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'backend': 'cpu',
                'timestamp': asyncio.get_event_loop().time()
            }
