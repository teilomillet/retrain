from typing import Any, Dict, Tuple
from loguru import logger # Added for logging

from .model import Model, ModelObject, TokenizerObject


class UnslothModel(Model):
    """Loads models and tokenizers using the Unsloth library for optimized performance."""

    def load(self, model_config: Dict[str, Any]) -> Tuple[ModelObject, TokenizerObject]:
        """Loads a model and tokenizer using unsloth.FastLanguageModel."""
        try:
            # Use FastLanguageModel as it includes get_peft_model
            from unsloth import FastLanguageModel
        except ImportError:
            logger.error("[UnslothModel.load] Unsloth library not found. Please install it: pip install unsloth")
            raise ImportError(
                "Unsloth library not found. Please install it: pip install unsloth"
            )

        model_name = model_config.get("name_or_path")
        if not model_name:
            logger.error("[UnslothModel.load] 'name_or_path' must be specified in model_config.")
            raise ValueError("'name_or_path' must be specified in model_config for UnslothLoader.")

        # Extract relevant kwargs for FastLanguageModel.from_pretrained
        unsloth_kwargs = {
            "max_seq_length": model_config.get("max_seq_length"),
            "dtype": model_config.get("dtype"), # e.g., None, torch.float16, torch.bfloat16
            "load_in_4bit": model_config.get("load_in_4bit", False),
            "token": model_config.get("token"), # For gated models
            "device_map": model_config.get("device_map", "auto"),
            "trust_remote_code": model_config.get("trust_remote_code", True),
            # Add other Unsloth-specific args from model_config as needed
        }
        unsloth_kwargs = {k: v for k, v in unsloth_kwargs.items() if v is not None}

        logger.debug(f"[UnslothModel.load] Loading Unsloth FastLanguageModel: {model_name} with args: {unsloth_kwargs}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            **unsloth_kwargs
        )

        logger.info(f"[UnslothModel.load] Unsloth model '{model_name}' and tokenizer loaded successfully.")
        return model, tokenizer

    def peft(self, model: ModelObject, peft_config_dict: Dict[str, Any]) -> ModelObject:
        """Applies PEFT adaptations using unsloth.FastLanguageModel.get_peft_model."""
        get_peft_model_func = None
        try:
            if hasattr(model, 'get_peft_model') and callable(model.get_peft_model):
                get_peft_model_func = model.get_peft_model
            else:
                 from unsloth import FastLanguageModel # Import here if needed for fallback
                 if hasattr(FastLanguageModel, 'get_peft_model'):
                      get_peft_model_func = FastLanguageModel.get_peft_model
                 else:
                      logger.error("[UnslothModel.peft] Model object and FastLanguageModel class lack a callable 'get_peft_model' method.")
                      raise AttributeError("Model object does not have a callable 'get_peft_model' method, and neither does FastLanguageModel class.")
        except ImportError:
             logger.error("[UnslothModel.peft] Unsloth library not found. Needed for PEFT application. Please install it: pip install unsloth")
             raise ImportError(
                "Unsloth library not found. Please install it: pip install unsloth"
            )

        logger.info("[UnslothModel.peft] Attempting to apply Unsloth PEFT configuration.")
        logger.debug(f"[UnslothModel.peft] Applying Unsloth PEFT config details: {peft_config_dict}")
        try:
            if 'r' not in peft_config_dict:
                logger.error(f"[UnslothModel.peft] peft_config_dict must contain at least the 'r' key for Unsloth PEFT. Provided: {peft_config_dict}")
                raise ValueError("peft_config_dict must contain at least the 'r' key for Unsloth PEFT.")

            peft_model = get_peft_model_func(
                model, 
                **peft_config_dict 
            )
            logger.info("[UnslothModel.peft] Unsloth PEFT model created successfully.")
            # Unsloth model might have its own way to show trainable parameters.
            # For example, it might be printed automatically or via a different method like model.print_trainable_parameters().
            # Consider adding a debug log if such a method is called.
            return peft_model
        except Exception as e:
            logger.error(f"[UnslothModel.peft] Error applying Unsloth PEFT config {peft_config_dict}: {e}", exc_info=True)
            raise ValueError(f"Failed to apply Unsloth PEFT configuration: {peft_config_dict}") from e 