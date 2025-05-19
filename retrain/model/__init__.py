# retrain/model/__init__.py
from typing import Any, Dict, Optional, Tuple

# Import the base model loader class and type aliases first
from .model import Model, ModelObject, TokenizerObject

# Import concrete model loader implementations
from .hf import HuggingFaceModel
from .unsloth import UnslothModel

# Import loguru for logging
from loguru import logger

# Added get_model factory function
SUPPORTED_MODEL_TYPES = {
    "hf": HuggingFaceModel,
    "huggingface": HuggingFaceModel,
    "unsloth": UnslothModel,
}

def get_model(
    model_type: str,
    model_name_or_path: str,
    model_config_overrides: Optional[Dict[str, Any]] = None,
    peft_config: Optional[Dict[str, Any]] = None,
) -> Tuple[ModelObject, TokenizerObject]:
    """Factory function to instantiate, load, and optionally apply PEFT to a model.

    Args:
        model_type: The type of model to load (e.g., "hf", "unsloth").
        model_name_or_path: The identifier for the model (e.g., "meta-llama/Llama-2-7b-hf").
        model_config_overrides: Dictionary of parameters to pass to the model loader's
                               `load` method, overriding defaults or adding specifics.
                               This will be merged with a base config including 'name_or_path'.
        peft_config: Optional dictionary of PEFT configuration parameters.
                     If provided, PEFT will be applied to the loaded model.

    Returns:
        A tuple containing the loaded (and possibly PEFT-adapted) model object
        and the loaded tokenizer object.

    Raises:
        ValueError: If the model_type is not supported or configurations are invalid.
    """
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            f"Supported types are: {list(SUPPORTED_MODEL_TYPES.keys())}"
        )

    model_loader_class = SUPPORTED_MODEL_TYPES[model_type]
    model_loader_instance: Model = model_loader_class() # Instantiate the loader

    # Prepare the main model_config dictionary for the load method
    # Start with name_or_path, then add overrides.
    effective_model_config = {"name_or_path": model_name_or_path}
    if model_config_overrides:
        effective_model_config.update(model_config_overrides)

    # Load the base model and tokenizer
    logger.debug(f"[get_model] Attempting to load model of type '{model_type}'. Name/Path: '{model_name_or_path}'")
    logger.debug(f"[get_model] Effective model config for loader: {effective_model_config}")
    loaded_model, tokenizer = model_loader_instance.load(model_config=effective_model_config)
    logger.debug(f"[get_model] Base model and tokenizer loaded successfully for '{model_name_or_path}'.")

    # Apply PEFT if a PEFT configuration is provided
    if peft_config:
        logger.debug(f"[get_model] Attempting to apply PEFT config for '{model_name_or_path}'. PEFT Config: {peft_config}")
        final_model = model_loader_instance.peft(model=loaded_model, peft_config_dict=peft_config)
        logger.debug(f"[get_model] PEFT applied successfully for '{model_name_or_path}'.")
    else:
        final_model = loaded_model
        logger.debug(f"[get_model] No PEFT config provided for '{model_name_or_path}'. Returning base model.")

    return final_model, tokenizer

# Define what gets imported with a wildcard import `from .model import *`
# This also helps linters and IDEs understand the public API of the package.
__all__ = [
    "Model",
    "ModelObject",
    "TokenizerObject",
    "HuggingFaceModel",
    "UnslothModel",
    "get_model",
]
