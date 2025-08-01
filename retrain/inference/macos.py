"""
macOS-optimized inference actor using CPU and Transformers backend.
"""

import ray
import asyncio
import logging
from typing import Dict, List, Any

from .base import BaseInferenceActor

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=2, num_gpus=0)
class MacOSInferenceActor(BaseInferenceActor):
    """macOS-compatible inference actor with MPS support and CPU fallback"""
    
    def __init__(self, config: Any, databuffer: ray.ObjectRef):
        self.config = config
        self.databuffer = databuffer
        self.model = None
        self.tokenizer = None
        self.device = "cpu"  # Will be set to "mps" if available
        self.is_initialized = False
        
        logger.info("MacOSInferenceActor initialized")
    
    async def initialize(self) -> None:
        """Initialize macOS-compatible inference engine with lazy model loading"""
        try:
            import torch
            import platform
            
            # Check if we're actually on macOS
            if platform.system() != "Darwin":
                logger.warning("MacOSInferenceActor being used on non-macOS platform")
            
            # Check for MPS availability (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using MPS (Apple Silicon) for inference")
            else:
                self.device = "cpu"
                logger.info("Using CPU for inference (MPS not available)")
            
            # Store model name but don't load yet (lazy loading)
            # This prevents simultaneous heavy model downloads during actor group initialization
            self.model_name = getattr(self.config, 'model_name', 'microsoft/DialoGPT-medium')
            self.model = None  # Will be loaded on first inference call
            self.tokenizer = None
            
            self.is_initialized = True
            logger.info(f"MacOSInferenceActor initialized (lazy) - model {self.model_name} will load on first use")
            
        except Exception as e:
            logger.error(f"Failed to initialize MacOSInferenceActor: {e}")
            raise
    
    async def _initialize_model(self, model_name: str) -> None:
        """Initialize model for macOS/Apple Silicon"""
        try:
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            from transformers.models.auto.tokenization_auto import AutoTokenizer
            import torch
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with appropriate device and dtype
            if self.device == "mps":
                # For Apple Silicon, use float16 to save memory
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
            else:
                # For CPU, use float32 for stability
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set model to eval mode
            self.model.eval()
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    async def generate_rollout(self, episode_id: int, rollout_idx: int) -> Dict[str, Any]:
        """Generate a single rollout using macOS-optimized inference"""
        if not self.is_initialized:
            await self.initialize()
        
        # Lazy model loading - load model on first inference call
        # This prevents simultaneous model downloads during actor initialization
        if self.model is None or self.tokenizer is None:
            await self._initialize_model(self.model_name)
        
        try:
            # Get prompts from databuffer or generate test prompts
            prompts = [f"Test prompt for episode {episode_id}, rollout {rollout_idx}"]
            
            # Generate responses
            responses = await self.generate_batch(prompts, {
                'max_tokens': 100,
                'temperature': 0.7
            })
            
            rollout_data = {
                'episode_id': episode_id,
                'rollout_idx': rollout_idx,
                'prompts': prompts,
                'responses': responses,
                'backend': f'macos_{self.device}',
                'timestamp': asyncio.get_event_loop().time()
            }
            
            return rollout_data
            
        except Exception as e:
            logger.error(f"Failed to generate rollout: {e}")
            return {
                'episode_id': episode_id,
                'rollout_idx': rollout_idx,
                'error': str(e),
                'backend': f'macos_{self.device}'
            }
    
    async def generate_batch(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[str]:
        """Generate responses for a batch of prompts using macOS-optimized inference"""
        if not self.is_initialized:
            await self.initialize()
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized")
        
        try:
            import torch
            
            responses = []
            for prompt in prompts:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors='pt')
                
                # Move to appropriate device
                if self.device == "mps":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate response with device-appropriate settings
                with torch.no_grad():
                    if self.device == "mps":
                        # For MPS, use more conservative settings
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=sampling_params.get('max_tokens', 100),
                            temperature=sampling_params.get('temperature', 0.7),
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            top_p=sampling_params.get('top_p', 0.9),
                            use_cache=True  # Important for MPS performance
                        )
                    else:
                        # For CPU, standard settings
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=sampling_params.get('max_tokens', 100),
                            temperature=sampling_params.get('temperature', 0.7),
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            top_p=sampling_params.get('top_p', 0.9)
                        )
                
                # Decode response (remove input prompt)
                input_length = inputs['input_ids'].shape[1]
                response_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                responses.append(response.strip())
            
            return responses
            
        except Exception as e:
            logger.error(f"Failed to generate batch: {e}")
            return [f"Error: {str(e)}" for _ in prompts]
    
    async def update_model_weights(self, model_weights: Dict[str, Any]) -> None:
        """Update model weights (macOS implementation)"""
        try:
            if self.model is not None:
                # Simple weight update
                self.model.load_state_dict(model_weights, strict=False)
                logger.info("Model weights updated successfully")
        except Exception as e:
            logger.error(f"Failed to update model weights: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check actor health with macOS-specific information"""
        try:
            import torch
            import psutil
            import platform
            
            health_data = {
                'status': 'healthy',
                'is_initialized': self.is_initialized,
                'backend': f'macos_{self.device}',
                'device': self.device,
                'platform': platform.platform(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'timestamp': asyncio.get_event_loop().time()
            }
            
            # Add macOS/MPS specific info
            if self.device == "mps":
                health_data.update({
                    'mps_available': True,
                    'torch_version': torch.__version__,
                })
            else:
                health_data.update({
                    'mps_available': False,
                    'using_cpu_fallback': True,
                })
            
            # Add memory usage if model is loaded
            if self.model is not None:
                try:
                    # Try to estimate model memory usage
                    model_params = sum(p.numel() for p in self.model.parameters())
                    health_data['model_parameters'] = model_params
                except Exception as e:
                    logger.warning(f"Failed to get model parameters: {e}")
                    pass
                
            return health_data
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'backend': f'macos_{self.device}',
                'timestamp': asyncio.get_event_loop().time()
            }


