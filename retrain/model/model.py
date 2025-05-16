from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

# Type alias for the loaded model object (can be HF transformer, Unsloth FastModel, etc.)
ModelObject = Any
# Type alias for the loaded tokenizer object
TokenizerObject = Any


class Model(ABC):
    """
    Abstract Base Class for model loaders.

    Defines the interface for loading a pre-trained model and its
    associated tokenizer, and optionally applying PEFT adaptations.
    """

    @abstractmethod
    def load(self, model_config: Dict[str, Any]) -> Tuple[ModelObject, TokenizerObject]:
        """
        Loads the model and tokenizer based on the provided configuration.

        Args:
            model_config: A dictionary containing configuration specific to
                          the model to be loaded. Expected keys might include
                          'name_or_path', 'revision', 'load_in_4bit',
                          'max_seq_length', 'trust_remote_code', etc.,
                          depending on the loader implementation.

        Returns:
            A tuple containing the loaded model object and the loaded
            tokenizer object.

        Raises:
            ValueError: If the configuration is missing required keys or
                        contains invalid values for the specific loader.
            ImportError: If required libraries (e.g., transformers, unsloth)
                         are not installed.
            Exception: For other underlying loading errors (e.g., network issues,
                       model file corruption).
        """
        pass

    @abstractmethod
    def peft(self, model: ModelObject, peft_config_dict: Dict[str, Any]) -> ModelObject:
        """
        Applies PEFT (Parameter-Efficient Fine-Tuning) adaptations to the loaded model.

        This method should be implemented by subclasses to handle the specific
        PEFT library integration (e.g., using peft.get_peft_model for Hugging Face
        models or unsloth.FastModel.get_peft_model for Unsloth models).

        Args:
            model: The loaded model object returned by the `load` method.
            peft_config_dict: A dictionary containing the PEFT configuration
                              parameters (e.g., r, lora_alpha, target_modules,
                              task_type, etc.). The exact structure depends on the
                              PEFT method and the underlying model library.

        Returns:
            The PEFT-adapted model object.

        Raises:
            NotImplementedError: If the specific subclass does not support PEFT.
            ValueError: If the `peft_config_dict` is invalid or incompatible.
            ImportError: If required PEFT libraries (e.g., peft, unsloth) are
                         not installed.
        """
        pass 