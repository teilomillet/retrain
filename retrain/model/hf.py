from typing import Any, Dict, Tuple
from loguru import logger # Added for logging

from .model import Model, ModelObject, TokenizerObject

# Try to import peft components, but don't fail if not installed yet
# The error will be raised within the peft method if called without installation.
try:
    from peft import get_peft_model, LoraConfig, TaskType # Assuming Lora for now
except ImportError:
    LoraConfig = None # Placeholder
    get_peft_model = None
    TaskType = None


class HuggingFaceModel(Model):
    """Loads models and tokenizers using the Hugging Face transformers library."""

    def load(self, model_config: Dict[str, Any]) -> Tuple[ModelObject, TokenizerObject]:
        """Loads a model and tokenizer using transformers.AutoModel/AutoTokenizer."""
        try:
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            from transformers.models.auto.tokenization_auto import AutoTokenizer
        except ImportError:
            logger.error("[HuggingFaceModel.load] HuggingFace transformers library not found. Please install it: pip install transformers torch")
            raise ImportError(
                "HuggingFace transformers library not found. "
                "Please install it: pip install transformers torch"
            )

        model_name = model_config.get("name_or_path")
        if not model_name:
            logger.error("[HuggingFaceModel.load] 'name_or_path' must be specified in model_config.")
            raise ValueError("'name_or_path' must be specified in model_config for HuggingFaceLoader.")

        # Extract relevant kwargs for AutoTokenizer, defaulting trust_remote_code
        tokenizer_kwargs = {
            "revision": model_config.get("revision"),
            "trust_remote_code": model_config.get("trust_remote_code", True),
        }
        tokenizer_kwargs = {k: v for k, v in tokenizer_kwargs.items() if v is not None}

        # Extract relevant kwargs for AutoModel, defaulting trust_remote_code
        model_kwargs = {
            "revision": model_config.get("revision"),
            "trust_remote_code": model_config.get("trust_remote_code", True),
            "torch_dtype": model_config.get("torch_dtype"), # e.g., torch.bfloat16
            "low_cpu_mem_usage": model_config.get("low_cpu_mem_usage", True),
            # Add quantization args if applicable (e.g., load_in_8bit, load_in_4bit)
            "load_in_8bit": model_config.get("load_in_8bit", False),
            "load_in_4bit": model_config.get("load_in_4bit", False),
        }
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        logger.debug(f"[HuggingFaceModel.load] Loading tokenizer: {model_name} with args: {tokenizer_kwargs}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

        logger.debug(f"[HuggingFaceModel.load] Loading model: {model_name} with args: {model_kwargs}")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # You might need to handle device placement here or later, e.g.:
        # model = model.to(model_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        logger.info(f"[HuggingFaceModel.load] HuggingFace model and tokenizer loaded successfully for '{model_name}'.")
        return model, tokenizer

    def peft(self, model: ModelObject, peft_config_dict: Dict[str, Any]) -> ModelObject:
        """Applies PEFT adaptations using the 'peft' library."""
        if get_peft_model is None or LoraConfig is None or TaskType is None:
            logger.error("[HuggingFaceModel.peft] PEFT library not found. Please install it: pip install peft")
            raise ImportError(
                "PEFT library not found. Please install it: pip install peft"
            )

        # Example: Creating a LoraConfig. Adapt as needed for other PEFT types.
        # We expect peft_config_dict to contain args for LoraConfig like r, lora_alpha, etc.
        # Defaulting task_type to CAUSAL_LM, might need to be configurable.
        peft_config_dict.setdefault("task_type", TaskType.CAUSAL_LM)

        # Ensure necessary keys for LoraConfig are present or raise error
        required_keys = ["r", "lora_alpha"] # Example minimal keys for LoRA
        if not all(key in peft_config_dict for key in required_keys):
             logger.error(f"[HuggingFaceModel.peft] peft_config_dict must contain keys for LoraConfig: {required_keys}. Provided: {peft_config_dict}")
             raise ValueError(f"peft_config_dict must contain keys for LoraConfig: {required_keys}")

        logger.info(f"[HuggingFaceModel.peft] Applying PEFT (LoRA) config: {peft_config_dict}")
        try:
            config = LoraConfig(**peft_config_dict)
            peft_model = get_peft_model(model, config)
            logger.info("[HuggingFaceModel.peft] PEFT model created successfully.")
            logger.debug("[HuggingFaceModel.peft] Printing trainable parameters (output to stdout)...")
            peft_model.print_trainable_parameters() # Useful helper from peft
            return peft_model
        except Exception as e:
            logger.error(f"[HuggingFaceModel.peft] Error applying PEFT config {peft_config_dict}: {e}", exc_info=True)
            raise ValueError(f"Failed to apply PEFT configuration: {peft_config_dict}") from e 