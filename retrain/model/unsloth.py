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
        """
        Applies PEFT adaptations using Unsloth's FastLanguageModel.get_peft_model.

        This method relies on the get_peft_model method typically found on
        Unsloth FastLanguageModel instances.
        """
        is_instance_method = False
        get_peft_model_func = None # Ensure it's defined

        # Prioritize using the get_peft_model method directly from the model instance.
        # This is the standard way for models loaded via unsloth.FastLanguageModel.
        if hasattr(model, 'get_peft_model') and callable(model.get_peft_model):
            get_peft_model_func = model.get_peft_model
            is_instance_method = True
            logger.debug("[UnslothModel.peft] Using model.get_peft_model (instance method).")
        else:
            # Fallback: Try to use FastLanguageModel.get_peft_model (class method).
            # This requires unsloth to be importable.
            try:
                from unsloth import FastLanguageModel # Import for fallback
                if hasattr(FastLanguageModel, 'get_peft_model') and callable(FastLanguageModel.get_peft_model):
                    get_peft_model_func = FastLanguageModel.get_peft_model
                    logger.warning("[UnslothModel.peft] Using FastLanguageModel.get_peft_model (class method) as fallback.")
                else:
                    logger.error("[UnslothModel.peft] Model object lacks 'get_peft_model' method, and FastLanguageModel class does not provide it either.")
                    raise AttributeError("No callable 'get_peft_model' method found on model instance or Unsloth FastLanguageModel class.")
            except ImportError:
                logger.error("[UnslothModel.peft] Unsloth library not found. Needed for PEFT application with FastLanguageModel.get_peft_model fallback. Please install it: pip install unsloth")
                raise ImportError(
                    "Unsloth library not found. Please install it: pip install unsloth"
                )

        logger.info("[UnslothModel.peft] Attempting to apply Unsloth PEFT configuration.")
        logger.debug(f"[UnslothModel.peft] Applying Unsloth PEFT config details: {peft_config_dict}")
        
        try:
            # Unsloth's get_peft_model (for LoRA) typically requires at least 'r' (rank).
            # Other parameters like target_modules, lora_alpha, lora_dropout, bias, etc.,
            # are passed via **peft_config_dict. The user is responsible for providing a
            # complete and valid configuration for Unsloth.
            if 'r' not in peft_config_dict:
                logger.error(f"[UnslothModel.peft] peft_config_dict must contain at least the 'r' (rank) key for Unsloth LoRA. Provided: {peft_config_dict}")
                raise ValueError("peft_config_dict must contain at least the 'r' key for Unsloth LoRA.")

            # Call Unsloth's PEFT function.
            # If it's an instance method (model.get_peft_model), it's called as: model.get_peft_model(**config)
            # If it's a class method (FastLanguageModel.get_peft_model), it's called as: FastLanguageModel.get_peft_model(model, **config)
            if is_instance_method:
                peft_model = get_peft_model_func(**peft_config_dict)
            else:
                # This case implies get_peft_model_func refers to FastLanguageModel.get_peft_model
                if get_peft_model_func is None: # Should not happen if logic above is correct
                     logger.error("[UnslothModel.peft] Internal error: get_peft_model_func is None before call.")
                     raise RuntimeError("Internal error: PEFT function resolver failed.")
                peft_model = get_peft_model_func(model, **peft_config_dict)

            logger.info("[UnslothModel.peft] Unsloth PEFT model created successfully.")
            # After applying PEFT, Unsloth models often have a method to show trainable parameters,
            # e.g., model.print_trainable_parameters().
            # This would typically be called by the user or a higher-level training script if detailed output is needed.
            return peft_model
        except Exception as e:
            logger.error(f"[UnslothModel.peft] Error applying Unsloth PEFT config {peft_config_dict}: {e}", exc_info=True)
            raise ValueError(f"Failed to apply Unsloth PEFT configuration: {peft_config_dict}") from e 