import abc
import asyncio
import json
import pathlib
from typing import Any, Dict, List, Optional, AsyncIterator

from retrain.utils.logging_utils import get_logger

logger = get_logger(__name__)

class PromptSource(abc.ABC):
    """
    Abstract Base Class for prompt sources.

    A prompt source provides an asynchronous way to get initial prompts for environment rollouts.
    It supports asynchronous iteration.
    """

    @abc.abstractmethod
    async def get_prompt(self) -> Optional[str]:
        """
        Fetch the next prompt.

        Returns:
            Optional[str]: The next prompt string, or None if no more prompts are available.
        """
        pass

    @abc.abstractmethod
    async def reset(self) -> None:
        """
        Reset the prompt source to its beginning.
        Allows re-iteration over the prompts if the source is finite.
        For infinite sources, this might have no effect or re-initialize a stream.
        """
        pass

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        prompt = await self.get_prompt()
        if prompt is None:
            raise StopAsyncIteration
        return prompt


class CfgPromptSource(PromptSource):
    """
    A PromptSource implementation that is configured via a dictionary.
    It can load prompts from a list, a file, or a Hugging Face dataset.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the CfgPromptSource.

        Args:
            config: A dictionary with configuration options.
                Expected keys:
                - "type": str - One of "list", "file", "hf_dataset".
                - For "list":
                    - "prompts": List[str] - A list of prompt strings.
                - For "file":
                    - "file_path": str - Path to the prompt file.
                    - "file_format": Optional[str] - "txt_lines" (default) or "json_list".
                - For "hf_dataset":
                    - "dataset_name_or_path": str - Name or path of the HF dataset.
                    - "column_name": str - The column containing the prompt text.
                    - "split": Optional[str] - Dataset split (e.g., "train", "test"). Defaults to "train".
                    - "streaming": Optional[bool] - Whether to stream the dataset. Defaults to False.
                                                  Streaming is recommended for very large datasets.
                    - "dataset_config_name": Optional[str] - Specific dataset config if needed.
                    - "shuffle_buffer_size": Optional[int] - Buffer size for shuffling (if streaming and shuffling).
        """
        self._config = config
        self.source_type: Optional[str] = self._config.get("type")

        self._prompts_data: List[str] = []
        self._current_index: int = 0

        self._hf_dataset_iterator: Optional[AsyncIterator] = None # For hf_dataset type, stores the async iterator
        self._hf_dataset_instance = None # Stores the loaded dataset object for potential re-iteration

        if self.source_type == "list":
            source_config_dict = self._config.get("source_config", {})
            self._prompts_data = source_config_dict.get("prompts", [])
            if not isinstance(self._prompts_data, list) or not all(isinstance(p, str) for p in self._prompts_data):
                logger.error(f"For 'list' type, 'prompts' must be a list of strings under 'source_config'. Found: {self._prompts_data}")
                raise ValueError("Invalid 'prompts' configuration for CfgPromptSource type 'list'.")
            logger.info(f"CfgPromptSource: Initialized with {len(self._prompts_data)} prompts from list.")
        elif self.source_type == "file":
            source_config_dict = self._config.get("source_config", {})
            file_path_str = source_config_dict.get("file_path")
            if not file_path_str:
                raise ValueError("Missing 'file_path' in 'source_config' for CfgPromptSource type 'file'.")
            
            file_path = pathlib.Path(file_path_str)
            file_format = source_config_dict.get("file_format", "txt_lines").lower()

            if not file_path.is_file():
                raise FileNotFoundError(f"Prompt file not found: {file_path}")

            try:
                if file_format == "txt_lines":
                    with open(file_path, "r", encoding="utf-8") as f:
                        self._prompts_data = [line.strip() for line in f if line.strip()]
                elif file_format == "json_list":
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, list) or not all(isinstance(p, str) for p in data):
                        logger.error(f"JSON file {file_path} must contain a list of strings. Found: {type(data)}")
                        raise ValueError("Invalid JSON format: Expected a list of strings.")
                    self._prompts_data = data
                else:
                    raise ValueError(f"Unsupported file_format: {file_format}. Choose 'txt_lines' or 'json_list'.")
                logger.info(f"CfgPromptSource: Initialized with {len(self._prompts_data)} prompts from file '{file_path}' (format: {file_format}).")
            except Exception as e:
                logger.error(f"Error loading prompts from file {file_path}: {e}")
                raise
        elif self.source_type == "hf_dataset":
            # Validation of hf_dataset config keys
            source_config_dict = self._config.get("source_config", {})
            required_hf_keys = ["dataset_name_or_path", "column_name"]
            for key in required_hf_keys:
                if key not in source_config_dict:
                    raise ValueError(f"Missing required configuration key '{key}' in 'source_config' for CfgPromptSource type 'hf_dataset'.")
            logger.info("CfgPromptSource: Configured for Hugging Face dataset. Will attempt to load on first get_prompt().")
            # Actual loading is deferred to _load_hf_dataset_iterator_if_needed
        else:
            raise ValueError(f"Unsupported prompt_source type: '{self.source_type}'. Choose from 'list', 'file', 'hf_dataset'.")

    def __len__(self) -> int:
        """
        Returns the number of prompts available from the source, if known.

        Raises:
            TypeError: If the length cannot be determined for the configured source type (e.g., streaming hf_dataset).
        """
        if self.source_type == "list" or self.source_type == "file":
            return len(self._prompts_data)
        elif self.source_type == "hf_dataset":
            # If not streaming and the dataset has been loaded and has a length
            if (not self._config.get("streaming", False) and
                self._hf_dataset_instance is not None and
                hasattr(self._hf_dataset_instance, "__len__")):
                try:
                    return len(self._hf_dataset_instance) # type: ignore
                except TypeError: # Handle cases where __len__ might exist but still raise TypeError (e.g. some iterables)
                    pass # Fall through to raise TypeError below
            
            # For streaming datasets or if length is otherwise unavailable
            raise TypeError(
                f"Length not available for Hugging Face dataset prompt source (type: '{self.source_type}') "
                f"when streaming or if the loaded dataset does not support __len__(). "
                f"Please specify 'max_steps' in your TRL trainer configuration (e.g., GRPOConfig)."
            )
        else:
            # Should not be reached if constructor validation is thorough
            raise TypeError(f"Length not determinable for unknown prompt_source type: '{self.source_type}'.")

    def get_all_prompts_sync(self) -> List[str]:
        """
        Returns all prompts as a list synchronously.

        This method is suitable for 'list' and 'file' types where all data is loaded
        during initialization.

        Raises:
            NotImplementedError: If the source type is 'hf_dataset', as it might be streaming
                                 or require asynchronous loading.
            ValueError: If the source type is unknown or data is not loaded.
        Returns:
            List[str]: A list of all prompt strings.
        """
        if self.source_type == "list" or self.source_type == "file":
            if self._prompts_data is not None: # Data should be loaded at __init__
                return list(self._prompts_data)
            else:
                # This case should ideally not be reached if __init__ is successful
                logger.error(f"CfgPromptSource ({self.source_type}): Data not loaded, cannot get all prompts synchronously.")
                raise ValueError(f"Data not loaded for CfgPromptSource type '{self.source_type}'.")
        elif self.source_type == "hf_dataset":
            logger.error("CfgPromptSource (hf_dataset): get_all_prompts_sync is not supported for 'hf_dataset' type due to potential streaming or async loading.")
            raise NotImplementedError("get_all_prompts_sync is not supported for 'hf_dataset' type.")
        else:
            logger.error(f"CfgPromptSource: Unknown source type '{self.source_type}' for get_all_prompts_sync.")
            raise ValueError(f"Unknown source type: {self.source_type}")

    async def _load_hf_dataset_iterator_if_needed(self) -> None:
        """
        Helper method to load and prepare the Hugging Face dataset iterator
        if it hasn't been loaded yet.
        This method is designed to be called by get_prompt.
        """
        if self.source_type == "hf_dataset" and self._hf_dataset_iterator is None:
            logger.info("Attempting to load Hugging Face dataset...")
            source_config_dict = self._config.get("source_config", {})
            try:
                from datasets import load_dataset # type: ignore
                logger.debug("'datasets' library imported successfully.")
            except ImportError:
                logger.error("The 'datasets' library is required for 'hf_dataset' prompt source type. Please install it: pip install datasets")
                raise ImportError("The 'datasets' library is required for 'hf_dataset' prompt source type. Please install it: pip install datasets")

            dataset_name = source_config_dict["dataset_name_or_path"]
            split = source_config_dict.get("split", "train")
            streaming = source_config_dict.get("streaming", False)
            name = source_config_dict.get("dataset_config_name") # `name` is the arg for load_dataset for config name
            shuffle_buffer_size = source_config_dict.get("shuffle_buffer_size")

            try:
                logger.info(f"Loading dataset: {dataset_name}, split: {split}, streaming: {streaming}, config_name: {name}")
                # Store the dataset instance if not streaming, to allow for reset if the dataset object supports re-iteration
                # For streaming, each call to iter() on the dataset object typically yields a new stream.
                if not streaming:
                    if self._hf_dataset_instance is None:
                         self._hf_dataset_instance = load_dataset(dataset_name, name=name, split=split, streaming=streaming)
                    dataset_to_iterate = self._hf_dataset_instance
                else: # Always reload for streaming to get a fresh stream, or if _hf_dataset_instance is None
                    dataset_to_iterate = load_dataset(dataset_name, name=name, split=split, streaming=streaming)
                    if shuffle_buffer_size and hasattr(dataset_to_iterate, 'shuffle'):
                        # Note: shuffle on a streaming dataset returns a new dataset instance
                        # The `shuffle` method for IterableDataset expects buffer_size as a positional argument.
                        dataset_to_iterate = dataset_to_iterate.shuffle(shuffle_buffer_size)


                if not hasattr(dataset_to_iterate, "__iter__") and not hasattr(dataset_to_iterate, "__aiter__"):
                     logger.error(f"Loaded dataset object of type {type(dataset_to_iterate)} is not iterable nor async iterable.")
                     raise TypeError("HuggingFace dataset is not iterable. This should not happen.")

                # Create an async iterator.
                # If the dataset is already an async iterable, use it directly.
                # Otherwise, wrap a sync iterator in an async generator.
                # This part might need adjustment if `datasets` provides native async iterators for streaming.
                # For now, assume a synchronous iterator from `iter()` and wrap it.
                
                # Simplification: Assume iter() gives a sync iterator, and we wrap it.
                # Real async support for HF datasets might evolve.
                sync_iterator = iter(dataset_to_iterate)

                async def _async_gen_wrapper(sync_iter):
                    loop = asyncio.get_event_loop()
                    while True:
                        try:
                            # Run the synchronous next() in a thread pool executor
                            # to avoid blocking the event loop.
                            item = await loop.run_in_executor(None, next, sync_iter)
                            yield item
                        except StopIteration:
                            break
                        except Exception as e: # Catch other potential errors during iteration
                            logger.error(f"Error during dataset iteration: {e}")
                            raise

                self._hf_dataset_iterator = _async_gen_wrapper(sync_iterator)
                logger.info(f"Hugging Face dataset '{dataset_name}' loaded and iterator created.")

            except Exception as e:
                logger.error(f"Failed to load or iterate Hugging Face dataset '{dataset_name}': {e}")
                raise

    async def get_prompt(self) -> Optional[str]:
        """Fetches the next prompt based on the configuration."""
        if self.source_type == "list" or self.source_type == "file":
            if self._current_index < len(self._prompts_data):
                prompt = self._prompts_data[self._current_index]
                self._current_index += 1
                return prompt
            else:
                return None  # No more prompts
        elif self.source_type == "hf_dataset":
            await self._load_hf_dataset_iterator_if_needed()
            if self._hf_dataset_iterator is None: # Should not happen if _load_hf_dataset_iterator_if_needed succeeded
                return None

            try:
                # item = await self._hf_dataset_iterator.__anext__() # Prefer direct anext if available
                # Using the wrapper means it's an async generator, so `anext` works
                item = await self._hf_dataset_iterator.__anext__()

                source_config_dict = self._config.get("source_config", {})
                column_name = source_config_dict["column_name"]
                if column_name not in item:
                    keys = list(item.keys()) if isinstance(item, dict) else []
                    logger.error(f"Column '{column_name}' not found in dataset item. Available keys: {keys}")
                    # Skip this item and try next, or raise error? For now, raise.
                    raise KeyError(f"Column '{column_name}' not found in dataset item. Available keys: {keys}")
                
                prompt_text = item[column_name]
                if not isinstance(prompt_text, str):
                    logger.warning(f"Value in column '{column_name}' is not a string (type: {type(prompt_text)}). Attempting to convert. Value: {prompt_text}")
                    try:
                        prompt_text = str(prompt_text)
                    except Exception as e_str:
                        logger.error(f"Could not convert value from column '{column_name}' to string: {e_str}")
                        # Skip this item and try next, or raise error? For now, raise.
                        raise TypeError(f"Value in column '{column_name}' (type: {type(prompt_text)}) could not be converted to string.")
                return prompt_text
            except StopAsyncIteration:
                logger.debug("Reached end of Hugging Face dataset.")
                return None # End of dataset
            except Exception as e:
                logger.error(f"Error fetching prompt from Hugging Face dataset: {e}")
                # Depending on the error, we might want to stop or try to recover.
                # For now, re-raise to signal a problem.
                raise
        else:
            # Should have been caught in __init__, but as a safeguard:
            logger.error(f"Invalid source_type '{self.source_type}' encountered in get_prompt.")
            return None 

    async def reset(self) -> None:
        """Resets the prompt source to its beginning."""
        if self.source_type == "list" or self.source_type == "file":
            self._current_index = 0
            logger.info(f"CfgPromptSource ({self.source_type}) reset to beginning.")
        elif self.source_type == "hf_dataset":
            # For HF datasets, resetting means re-creating the iterator.
            # If not streaming and _hf_dataset_instance is stored, we can re-use it.
            # If streaming, a new stream will be fetched.
            self._hf_dataset_iterator = None 
            # _hf_dataset_instance remains as is, _load_hf_dataset_iterator_if_needed will use it or reload if streaming.
            logger.info("CfgPromptSource (hf_dataset) reset. Will re-initialize iterator on next get_prompt().")
        else:
            logger.warning(f"Reset called on CfgPromptSource with unknown type: {self.source_type}")

def get_prompt_source(config: Dict[str, Any]) -> PromptSource:
    """Factory function to create a PromptSource instance from a configuration.

    Currently, this factory specifically returns a CfgPromptSource instance,
    as CfgPromptSource is designed to handle various underlying source types
    (list, file, hf_dataset) based on its own internal configuration.

    Args:
        config: A dictionary compatible with CfgPromptSource initialization.
                It should contain a "type" key (e.g., "list", "file", "hf_dataset")
                and other necessary keys based on that type (e.g., "prompts" for "list",
                "file_path" for "file", etc.)

    Returns:
        An instance of a PromptSource (specifically CfgPromptSource).

    Raises:
        ValueError: If the configuration is invalid for CfgPromptSource.
        FileNotFoundError: If a configured prompt file is not found.
        ImportError: If 'datasets' library is needed but not installed.
    """
    # CfgPromptSource's __init__ already handles the logic of parsing 'type'
    # and other config details. We just pass the config through.
    logger.info(f"[get_prompt_source] Creating CfgPromptSource with config: {config}")
    return CfgPromptSource(config=config)

__all__ = [
    "PromptSource",
    "CfgPromptSource",
    "get_prompt_source",
]

# Example Usage (for testing, not part of the library's main flow)
async def main_example():

    # 1. List example
    print("\n--- List Prompt Source Example ---")
    list_config = {
        "type": "list",
        "prompts": ["Hello from list!", "This is the second prompt.", "And a third one."]
    }
    list_source = CfgPromptSource(list_config)
    async for prompt in list_source:
        print(f"  ListPrompt: {prompt}")
    
    print("\nAttempting reset and re-iteration for list source:")
    await list_source.reset()
    async for prompt in list_source:
        print(f"  ListPrompt (after reset): {prompt}")


    # 2. File example (txt_lines)
    print("\n\n--- File Prompt Source Example (txt_lines) ---")
    # Create a dummy prompt file
    dummy_txt_file = pathlib.Path("dummy_prompts.txt")
    with open(dummy_txt_file, "w") as f:
        f.write("First line from file\n")
        f.write("Second line from file\n")
        f.write("\n") # Empty line, should be skipped
        f.write("Third after empty\n")
    
    file_config_txt = {"type": "file", "file_path": str(dummy_txt_file)}
    file_source_txt = CfgPromptSource(file_config_txt)
    async for prompt in file_source_txt:
        print(f"  FilePrompt (txt): {prompt}")
    dummy_txt_file.unlink() # Clean up


    # 3. File example (json_list)
    print("\n\n--- File Prompt Source Example (json_list) ---")
    dummy_json_file = pathlib.Path("dummy_prompts.json")
    with open(dummy_json_file, "w") as f:
        json.dump(["JSON prompt 1!", "JSON prompt 2, with \"quotes\"."], f)

    file_config_json = {"type": "file", "file_path": str(dummy_json_file), "file_format": "json_list"}
    file_source_json = CfgPromptSource(file_config_json)
    async for prompt in file_source_json:
        print(f"  FilePrompt (json): {prompt}")
    dummy_json_file.unlink() # Clean up

    # 4. Hugging Face Dataset Example (requires 'datasets' library and internet)
    #    This part will only run if 'datasets' is installed.
    print("\n\n--- Hugging Face Dataset Example ---")
    hf_config = {
        "type": "hf_dataset",
        "dataset_name_or_path": "glue", # Using a small, well-known dataset
        "dataset_config_name": "mrpc",  # Specify a configuration of GLUE
        "column_name": "sentence1",     # Column to get prompts from
        "split": "validation",          # Use validation split (usually smaller)
        "streaming": False              # Test non-streaming first
    }
    try:
        hf_source = CfgPromptSource(hf_config)
        print("Iterating first few prompts from HF dataset (limit 3 for example):")
        count = 0
        async for prompt in hf_source:
            print(f"  HF Prompt: {prompt[:80]}...") # Print first 80 chars
            count += 1
            if count >= 3:
                break
        
        print("\nAttempting reset and re-iteration for HF source (limit 2):")
        await hf_source.reset()
        count = 0
        async for prompt in hf_source:
            print(f"  HF Prompt (after reset): {prompt[:80]}...")
            count += 1
            if count >= 2:
                break

    except ImportError:
        print("  Skipping Hugging Face dataset example: 'datasets' library not installed.")
    except Exception as e:
        print(f"  Error during Hugging Face dataset example: {e}")

if __name__ == "__main__":
    # HACK: Need to import asyncio for the example to run
    # No longer needed here as main_example imports it if run directly
    # and the library itself uses asyncio where necessary.
    asyncio.run(main_example()) 