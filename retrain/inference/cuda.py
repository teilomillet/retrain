"""
CUDA-optimized inference actor using GPU and MBridge backend.
"""

import ray
import asyncio
import logging
from typing import Dict, List, Any

from .base import BaseInferenceActor

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1, num_gpus=1)
class CUDAInferenceActor(BaseInferenceActor):
    """CUDA-based inference actor for GPU inference using vLLM or transformers"""
    
    def __init__(self, config: Any, databuffer: ray.ObjectRef):
        self.config = config
        self.databuffer = databuffer
        self.model = None
        self.tokenizer = None
        self.vllm_engine = None
        self.is_initialized = False
        self.backend_type = "transformers"  # or "vllm"
        
        logger.info("CUDAInferenceActor initialized")
    
    async def initialize(self) -> None:
        """Initialize CUDA-based inference engine"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available for CUDAInferenceActor")
            
            model_name = getattr(self.config, 'model_name', 'microsoft/DialoGPT-medium')
            
            # Try to use vLLM first, fall back to transformers
            try:
                await self._initialize_vllm(model_name)
                self.backend_type = "vllm"
                logger.info(f"CUDAInferenceActor initialized with vLLM: {model_name}")
            except Exception as e:
                logger.warning(f"vLLM initialization failed: {e}, falling back to transformers")
                await self._initialize_transformers(model_name)
                self.backend_type = "transformers"
                logger.info(f"CUDAInferenceActor initialized with transformers: {model_name}")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize CUDAInferenceActor: {e}")
            raise
    
    async def _initialize_vllm(self, model_name: str) -> None:
        """Initialize using vLLM for high-performance inference"""
        try:
            from vllm import LLM  # type: ignore
            
            # Initialize vLLM engine
            self.vllm_engine = LLM(
                model=model_name,
                tensor_parallel_size=1,
                dtype="float16",
                gpu_memory_utilization=0.8,
                max_model_len=2048
            )
            
            # Get tokenizer from vLLM
            self.tokenizer = self.vllm_engine.get_tokenizer()
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise
    
    async def _initialize_transformers(self, model_name: str) -> None:
        """Initialize using transformers as fallback"""
        try:
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            from transformers.models.auto.tokenization_auto import AutoTokenizer
            import torch
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model on GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True
            )
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Failed to initialize transformers: {e}")
            raise
    
    async def generate_rollout(self, episode_id: int, rollout_idx: int) -> Dict[str, Any]:
        """Generate a single rollout using CUDA inference"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Get prompts from databuffer or generate test prompts
            prompts = [f"Test prompt for episode {episode_id}, rollout {rollout_idx}"]
            
            # Generate responses using the appropriate backend
            responses = await self.generate_batch(prompts, {
                'max_tokens': 100,
                'temperature': 0.7
            })
            
            rollout_data = {
                'episode_id': episode_id,
                'rollout_idx': rollout_idx,
                'prompts': prompts,
                'responses': responses,
                'backend': f'cuda_{self.backend_type}',
                'timestamp': asyncio.get_event_loop().time()
            }
            
            return rollout_data
            
        except Exception as e:
            logger.error(f"Failed to generate rollout: {e}")
            return {
                'episode_id': episode_id,
                'rollout_idx': rollout_idx,
                'error': str(e),
                'backend': f'cuda_{self.backend_type}'
            }
    
    async def generate_batch(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[str]:
        """Generate responses for a batch of prompts using CUDA"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if self.backend_type == "vllm":
                return await self._generate_with_vllm(prompts, sampling_params)
            else:
                return await self._generate_with_transformers(prompts, sampling_params)
                
        except Exception as e:
            logger.error(f"Failed to generate batch: {e}")
            return [f"Error: {str(e)}" for _ in prompts]
    
    async def _generate_with_vllm(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[str]:
        """Generate using vLLM engine"""
        if self.vllm_engine is None:
            raise RuntimeError("vLLM engine not initialized")
        
        try:
            from vllm import SamplingParams  # type: ignore
            
            # Convert sampling params to vLLM format
            vllm_params = SamplingParams(
                temperature=sampling_params.get('temperature', 0.7),
                top_p=sampling_params.get('top_p', 0.9),
                max_tokens=sampling_params.get('max_tokens', 100),
            )
            
            # Generate responses
            outputs = self.vllm_engine.generate(prompts, vllm_params)
            
            # Extract text responses
            responses = []
            for output in outputs:
                response_text = output.outputs[0].text if output.outputs else ""
                responses.append(response_text.strip())
            
            return responses
            
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise
    
    async def _generate_with_transformers(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[str]:
        """Generate using transformers as fallback"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Transformers model or tokenizer not initialized")
        
        try:
            import torch
            
            responses = []
            for prompt in prompts:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors='pt')  # type: ignore
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
                # Generate response
                with torch.no_grad():
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
                response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                responses.append(response.strip())
            
            return responses
            
        except Exception as e:
            logger.error(f"Transformers generation failed: {e}")
            raise
    
    async def update_model_weights(self, model_weights: Dict[str, Any]) -> None:
        """Update model weights (CUDA implementation)"""
        try:
            if self.backend_type == "transformers" and self.model is not None:
                # Simple weight update for transformers model
                self.model.load_state_dict(model_weights, strict=False)
                logger.info("Model weights updated successfully (transformers)")
            elif self.backend_type == "vllm":
                logger.warning("vLLM weight updates not supported yet")
        except Exception as e:
            logger.error(f"Failed to update model weights: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check actor health"""
        try:
            import torch
            import psutil
            
            health_data = {
                'status': 'healthy',
                'is_initialized': self.is_initialized,
                'backend': f'cuda_{self.backend_type}',
                'cuda_available': torch.cuda.is_available(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'timestamp': asyncio.get_event_loop().time()
            }
            
            # Add GPU info if available
            if torch.cuda.is_available():
                health_data.update({
                    'gpu_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'gpu_memory_allocated': torch.cuda.memory_allocated(),
                    'gpu_memory_reserved': torch.cuda.memory_reserved(),
                })
                
            return health_data
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'backend': f'cuda_{self.backend_type}',
                'timestamp': asyncio.get_event_loop().time()
            }
