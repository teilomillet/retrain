from typing import Any, List, Optional, Dict, Callable, Tuple
from collections import defaultdict
import torch
import inspect
from loguru import logger

# Try importing TRL components. Handle potential ImportError if trl is not installed.
# This allows the framework to function even if specific TRL backends aren't available.
try:
    from trl import GRPOTrainer as ActualTrlGRPOTrainer # type: ignore[no-redef]
    from trl import GRPOConfig as ActualTrlGRPOConfig # type: ignore[no-redef]
    from transformers import TrainingArguments # type: ignore[no-redef] # For checkpointing logic check
    from trl.trainer.utils import exact_div # For batching
    _trl_available = True
except ImportError:
    # Define dummy classes if TRL is not available to avoid runtime errors on class definition.
    # Instantiation with these dummies should be prevented by _validate_and_setup_backend if TRL is truly needed.
    class ActualTrlGRPOConfig: # type: ignore[no-redef]
        num_train_epochs: float = 1.0
        logging_steps: int = 10
        save_steps: int = 500
        num_iterations: int = 1
        # report_to: Optional[List[str]] = None
        # peft_config: Optional[Dict[str, Any]] = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            pass

    class ActualTrlGRPOTrainer: # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
        def train(self, *args: Any, **kwargs: Any) -> Any: return None
        def step(self, *args: Any, **kwargs: Any) -> Any: pass
        def save_model(self, *args: Any, **kwargs: Any) -> None: pass
        train_dataset: Any = None
        state: Any = None

    class TrainingArguments: # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    exact_div = None # type: ignore
    _trl_available = False

# Import Hugging Face Dataset, with a dummy fallback.
try:
    from datasets import Dataset # type: ignore[no-redef]
    _datasets_available = True
except ImportError:
    class Dataset: # type: ignore[no-redef]
        @staticmethod
        def from_list(data: List[Dict[str, Any]]) -> Optional['Dataset']:
             if not _datasets_available: # Ensure dummy is only used if real one failed
                if data:
                    dummy_ds = object.__new__(Dataset) # Create instance without calling __init__
                    dummy_ds._data = data # type: ignore
                    dummy_ds.features = {k: type(v) for k, v in data[0].items()} if data else {} # type: ignore
                    return dummy_ds # type: ignore
                return None # Return None for empty list to match one path of HF's .from_list
             return None

        def __len__(self) -> int:
            return len(self._data) # type: ignore

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return self._data[idx] # type: ignore

        features: Dict[str, Any] = {}


    _datasets_available = False

from retrain.reward.calculator import RewardCalculator
from retrain.reward.types import ProcessedTrajectory, SingleTurnDataForTrainer, RawRolloutData
from retrain.utils.metrics import MetricsTracker, GenerationMetrics, TrainingBatchMetrics
from retrain.trainer.trainer import (
    BaseTrainer, AlgorithmConfig, ModelObject, ExperienceBatch, 
    TrainingMetrics, PromptSource, RewardFunction
)
# Factory for creating TRL-compatible batch reward functions
from ...reward import create_grpo_batch_reward_func, REWARD_REGISTRY
from ...verifier import VERIFIER_REGISTRY

from ...environment.environment import Environment
from ...reward.calculator import RewardCalculator
from ...reward.types import SingleTurnDataForTrainer, RawRolloutData, ProcessedTrajectory

# TRL GRPOTrainer and its config
try:
    from trl import GRPOTrainer as TRL_GRPOTrainer # Specific alias
    from trl import GRPOConfig
    _trl_available = True
except ImportError:
    TRL_GRPOTrainer = None # type: ignore
    GRPOConfig = None # type: ignore
    _trl_available = False

# For checking if datasets is available (TRL GRPOTrainer requires it for prompt dataset)
try:
    import datasets
    _datasets_available = True
except ImportError:
    datasets = None # type: ignore
    _datasets_available = False

class GRPOTrainerAdapter(BaseTrainer):
    """
    Adapter for the TRL GRPO (Group Relative Policy Optimization) trainer.
    
    Uses a `retrain.Environment` for rollouts and `retrain.RewardCalculator`
    to process these rollouts before passing data to `trl.GRPOTrainer`.
    """

    trl_trainer: Optional[ActualTrlGRPOTrainer]
    environment: Environment
    reward_calculator: RewardCalculator # Primary source of rewards for TRL processing.
    actual_trl_reward_functions: List[Callable] # Stores TRL-compatible reward functions from run.py.
    metrics_tracker: MetricsTracker  # Track environment-specific metrics

    # Adapter-specific parameters (not part of ActualTrlGRPOConfig)
    _sampling_params: Dict[str, Any]
    _max_tokens_per_generation: int
    _temperature_for_logprobs: float
    _max_steps_this_trainer_train_call: int

    def __init__(self,
                 model: ModelObject,
                 algorithm_config: AlgorithmConfig,
                 environment: Environment,
                 reward_calculator: RewardCalculator,
                 prompt_source: PromptSource,
                 reward_functions: List[Callable], # Actual TRL-compatible functions
                 tokenizer: Optional[Any] = None,
                 reference_model: Optional[ModelObject] = None,
                 sampling_params: Optional[Dict[str, Any]] = None,
                 max_tokens_per_generation: Optional[int] = None,
                 temperature_for_logprobs: Optional[float] = None,
                 max_steps_this_trainer_train_call: Optional[int] = None,
                 metrics_window_size: Optional[int] = None,  # For MetricsTracker
                 **trainer_specific_kwargs: Any): # For TRL GRPOTrainer constructor
        """
        Initializes the TRL GRPO adapter.
        """
        self.environment = environment
        self.reward_calculator = reward_calculator
        # These TRL-compatible reward functions will be used to initialize the underlying TRL GRPOTrainer.
        self.actual_trl_reward_functions = reward_functions
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(window_size=metrics_window_size or 100)
        
        logger.debug(f"[GRPOTrainerAdapter.__init__] Received {len(reward_functions)} actual TRL reward functions.")
        logger.debug(f"[GRPOTrainerAdapter.__init__] Initialized MetricsTracker with window size {self.metrics_tracker.window_size}")

        if _trl_available and isinstance(algorithm_config, ActualTrlGRPOConfig):
            logger.debug(f"    TRL algorithm_config.loss_type: {getattr(algorithm_config, 'loss_type', 'N/A')}")
            logger.debug(f"    TRL algorithm_config.max_completion_length: {getattr(algorithm_config, 'max_completion_length', 'N/A')}")
            logger.debug(f"    TRL algorithm_config.num_generations: {getattr(algorithm_config, 'num_generations', 'N/A')}")
            logger.debug(f"    TRL algorithm_config.beta (KL coeff): {getattr(algorithm_config, 'beta', 'N/A')}")
            logger.debug(f"    TRL algorithm_config.reward_weights: {getattr(algorithm_config, 'reward_weights', 'N/A')}")
        elif not _trl_available:
            logger.debug(f"    (Using dummy ActualTrlGRPOConfig as TRL is not available)")
            logger.debug(f"    dummy algorithm_config.num_iterations: {getattr(algorithm_config, 'num_iterations', 'N/A')}")

        logger.debug(f"[GRPOTrainerAdapter.__init__] Adapter received trainer_specific_kwargs before pop: {trainer_specific_kwargs}")

        self._sampling_params = sampling_params if sampling_params is not None else {}
        self._max_tokens_per_generation = max_tokens_per_generation if max_tokens_per_generation is not None else 256
        self._temperature_for_logprobs = temperature_for_logprobs if temperature_for_logprobs is not None else 0.7
        self._max_steps_this_trainer_train_call = max_steps_this_trainer_train_call if max_steps_this_trainer_train_call is not None else 100

        # BaseTrainer still expects `reward_functions` in its __init__ signature.
        # RewardCalculator now encapsulates this logic, but we pass the actual TRL functions
        # for potential use by BaseTrainer or for consistency.
        super().__init__(
            model=model,
            algorithm_config=algorithm_config,
            reward_functions=self.actual_trl_reward_functions,
            prompt_source=prompt_source,
            tokenizer=tokenizer,
            reference_model=reference_model,
            **trainer_specific_kwargs
        )
        logger.debug(f"[GRPOTrainerAdapter.__init__] Adapter _sampling_params: {self._sampling_params}")
        logger.debug(f"[GRPOTrainerAdapter.__init__] Adapter _max_tokens_per_generation: {self._max_tokens_per_generation}")
        logger.debug(f"[GRPOTrainerAdapter.__init__] Adapter _temperature_for_logprobs: {self._temperature_for_logprobs}")
        logger.debug(f"[GRPOTrainerAdapter.__init__] Adapter _max_steps_this_trainer_train_call: {self._max_steps_this_trainer_train_call}")
        logger.debug(f"[GRPOTrainerAdapter.__init__] Adapter remaining kwargs passed to BaseTrainer: {trainer_specific_kwargs}")

    # Setup and Validation

    def _validate_and_setup_backend(self) -> None:
        """
        Orchestrates validation and setup of the TRL GRPOTrainer.
        Called by BaseTrainer's __init__.
        """
        self._check_dependencies()
        self._validate_input_components()
        self._initialize_underlying_trainer()

    def _check_dependencies(self) -> None:
        """Checks if TRL is installed."""
        if not _trl_available:
            raise ImportError(
                "TRL library is not installed, but is required for GRPOTrainer (using TRL backend). "
                "Please install TRL (`pip install trl transformers accelerate`)."
            )

    def _validate_input_components(self) -> None:
        """Validates necessary configuration and components."""
        if not _datasets_available:
            raise ImportError("The `datasets` library is required for GRPOTrainer to construct training batches. Please install it (`pip install datasets`).")
        if not isinstance(self.algorithm_config, ActualTrlGRPOConfig):
            raise TypeError(
                f"Configuration must be an instance of trl.GRPOConfig, got {type(self.algorithm_config)}."
            )

        if not isinstance(self.environment, Environment):
            raise TypeError(
                f"Environment must be an instance of retrain.environment.Environment, got {type(self.environment)}."
            )
        if not hasattr(self.environment, 'rollout'):
            raise AttributeError(
                "The provided environment does not have a 'rollout' method, which is required."
            )
        if not isinstance(self.reward_calculator, RewardCalculator):
            raise TypeError(
                f"Reward calculator must be an instance of retrain.reward.RewardCalculator, got {type(self.reward_calculator)}."
            )
            
        # TRL's GRPOTrainer requires a dataset for prompts during its initialization.
        if self.prompt_source is None:
            raise ValueError("A 'prompt_source' (e.g., HF Dataset) must be provided for TRL GRPOTrainer.")

    def _initialize_underlying_trainer(self) -> None:
        """
        Instantiates the `trl.GRPOTrainer`.
        Relies on `_validate_input_components` for prior checks.
        """
        hf_train_dataset: Optional[Dataset] = None
        try:
            # TRL GRPOTrainer expects a dataset of prompts for its internal data loader.
            # We use self.prompt_source for our adapter's loop; TRL needs a dataset for its own setup.
            try:
                if self.prompt_source:
                    prompts_list = self.prompt_source.get_all_prompts_sync()
                    # TRL GRPOTrainer expects a dataset with a "prompt" column.
                    hf_train_dataset = Dataset.from_list([{"prompt": p} for p in prompts_list])
                else:
                    logger.warning("GRPOTrainerAdapter (TRL backend): prompt_source is None. Using empty dataset for TRL init.")
                    hf_train_dataset = Dataset.from_list([]) # type: ignore[assignment]
            except (NotImplementedError, ValueError) as e:
                # This can occur if prompt_source is streaming-only and cannot provide all prompts synchronously.
                logger.warning(f"GRPOTrainerAdapter (TRL backend): Could not convert prompt_source to a static HF Dataset for TRL init: {e}. Using empty dataset.")
                hf_train_dataset = Dataset.from_list([]) # type: ignore[assignment]
            except ImportError as e_ds:
                logger.error(f"GRPOTrainerAdapter (TRL backend): Failed to create HF Dataset, `datasets` library import issue: {e_ds}")
                raise RuntimeError("Failed to create HF dataset for TRL GRPOTrainer due to missing `datasets` library.") from e_ds
            except AttributeError as e_attr: # Handles prompt_source being None.
                 logger.error(f"GRPOTrainerAdapter (TRL backend): Error accessing prompt_source: {e_attr}. This might happen if prompt_source is None. Using empty dataset.")
                 hf_train_dataset = Dataset.from_list([]) # type: ignore[assignment]


            logger.debug("[GRPOTrainerAdapter._initialize_underlying_trainer] Initializing TRL_GRPOTrainer with:")
            logger.debug(f"  model: {'Provided' if self.model else 'None'}")
            logger.debug(f"  tokenizer: {'Provided' if self.tokenizer else 'None'} (type: {type(self.tokenizer) if self.tokenizer else 'N/A'})")
            logger.debug(f"  train_dataset (for TRL init): {'Provided and has length' if hf_train_dataset and len(hf_train_dataset) > 0 else ('Provided but empty' if hf_train_dataset is not None else 'None')}")
            
            # Pass the TRL-compatible reward functions.
            logger.debug(f"  reward_funcs (for TRL GRPOTrainer constructor): {len(self.actual_trl_reward_functions)} functions provided.")
            logger.debug(f"[GRPOTrainerAdapter._initialize_underlying_trainer] Additional **kwargs for TrlGRPOTrainer (from self.trainer_specific_kwargs): {self.trainer_specific_kwargs}")

            # TRL's GRPOTrainer constructor expects the `model` to be the raw unwrapped model.
            self.trl_trainer = ActualTrlGRPOTrainer(
                model=self.model.model, # Pass the underlying model object
                args=self.algorithm_config,
                processing_class=self.tokenizer,
                train_dataset=hf_train_dataset,
                reward_funcs=self.actual_trl_reward_functions,
            )
            logger.info("TRL GRPOTrainer instantiated successfully.")
        except Exception as e:
            logger.error(f"Error instantiating TRL GRPOTrainer: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize TRL GRPOTrainer instance: {e}") from e

    # Core Training Methods

    async def _generate_and_process_rollout(self, initial_prompt: str) -> Tuple[List[SingleTurnDataForTrainer], GenerationMetrics]:
        """Helper to perform one rollout and process it with the reward calculator."""
        current_sampling_params = self._sampling_params.copy()

        # Ensure adapter's configured max_tokens_per_generation is in sampling_params
        # for environments that expect it there (e.g., SmolAgentEnv).
        if self._max_tokens_per_generation is not None:
            current_sampling_params["max_tokens_to_generate"] = self._max_tokens_per_generation
            logger.debug(f"[GRPOTrainerAdapter._generate_and_process_rollout] Ensured 'max_tokens_to_generate={self._max_tokens_per_generation}' in sampling_params for environment.")
        else:
            logger.debug("[GRPOTrainerAdapter._generate_and_process_rollout] self._max_tokens_per_generation is None. Relying on environment or its sampling_params default.")

        logger.debug(f"[GRPOTrainerAdapter._generate_and_process_rollout] Initial prompt: {initial_prompt[:100]}")
        logger.debug(f"[GRPOTrainerAdapter._generate_and_process_rollout] Final sampling_params for env.rollout: {current_sampling_params}")

        env_rollout_kwargs = {
            "initial_prompt": initial_prompt,
            "llm_model": self.model,
            "tokenizer": self.tokenizer,
            "sampling_params": current_sampling_params
        }

        # Dynamically add max_tokens_to_generate or max_tokens_this_turn if the environment's
        # rollout method explicitly defines them as top-level (non-keyword-only) arguments.
        # This is mainly for compatibility if an environment doesn't read it from sampling_params.
        try:
            env_rollout_sig = inspect.signature(self.environment.rollout)
            if ("max_tokens_to_generate" in env_rollout_sig.parameters and
                    env_rollout_sig.parameters["max_tokens_to_generate"].kind != inspect.Parameter.KEYWORD_ONLY):
                if self._max_tokens_per_generation is not None and "max_tokens_to_generate" not in current_sampling_params:
                    env_rollout_kwargs["max_tokens_to_generate"] = self._max_tokens_per_generation
                    logger.debug(f"Additionally passing top-level 'max_tokens_to_generate' to {type(self.environment).__name__}.rollout")
            elif ("max_tokens_this_turn" in env_rollout_sig.parameters and
                    env_rollout_sig.parameters["max_tokens_this_turn"].kind != inspect.Parameter.KEYWORD_ONLY):
                if self._max_tokens_per_generation is not None and "max_tokens_this_turn" not in current_sampling_params:
                    env_rollout_kwargs["max_tokens_this_turn"] = self._max_tokens_per_generation
                    logger.debug(f"Additionally passing top-level 'max_tokens_this_turn' to {type(self.environment).__name__}.rollout")
        except AttributeError:
            logger.error(f"Could not inspect signature of {type(self.environment).__name__}.rollout. Cannot dynamically pass certain top-level args.")

        raw_rollout_data: RawRolloutData = await self.environment.rollout(**env_rollout_kwargs)
        
        logger.debug(f"[GRPOTrainerAdapter._generate_and_process_rollout] RawRolloutData type from env.rollout: {type(raw_rollout_data)}")

        # Extract generation metrics from rollout metadata
        generation_metrics = raw_rollout_data.get("rollout_metadata", {}).get("generation_metrics")
        if generation_metrics is None:
            # Fallback: create basic metrics if environment doesn't provide them
            logger.warning("[GRPOTrainerAdapter._generate_and_process_rollout] No generation_metrics found in rollout_metadata, creating fallback metrics")
            generation_metrics = GenerationMetrics(
                parsing_success=True,  # Assume success if no data
                tool_calls_attempted=0,
                tool_calls_successful=0,
                final_answer_provided=False,
                total_reward=sum(raw_rollout_data.get("intrinsic_rewards_per_turn", [])),
                reward_per_step=raw_rollout_data.get("intrinsic_rewards_per_turn", []),
                num_turns=len(raw_rollout_data.get("executed_llm_actions", [])),
                completion_reason="unknown"
            )

        # Track tools used
        tools_used = raw_rollout_data.get("rollout_metadata", {}).get("tools_used", [])
        for tool_name in tools_used:
            self.metrics_tracker.track_tool_usage(tool_name)

        # RewardCalculator.process_rollouts structures data into SingleTurnDataForTrainer.
        # The `final_combined_reward` from this step is NOT used by TRL directly;
        # TRL uses its configured reward_funcs with "infos" from the processed data.
        results_batch: List[ProcessedTrajectory] = await self.reward_calculator.process_rollouts([raw_rollout_data])

        if not results_batch:
            logger.warning(f"RewardCalculator.process_rollouts returned no data for prompt: '{initial_prompt[:100]}'.")
            return [], generation_metrics
        
        processed_trajectory: List[SingleTurnDataForTrainer] = results_batch[0]
        return processed_trajectory, generation_metrics

    async def train(self, total_training_iterations: Optional[int] = None) -> TrainingMetrics:
        """
        Executes the training loop using environment rollouts and reward calculation.
        Constructs a HF Dataset from processed rollouts and calls TRL GRPOTrainer's train method.
        """
        if self.trl_trainer is None:
            raise RuntimeError("TRL Trainer not initialized. Cannot start training.")
        if not _datasets_available:
            raise RuntimeError("Datasets library not available, cannot proceed with training.")

        if not total_training_iterations or total_training_iterations <= 0:
            total_training_iterations = self._max_steps_this_trainer_train_call
            logger.info(f"Using training iterations for this GRPOTrainerAdapter.train call: {total_training_iterations} (from adapter config or default).")

        if self.prompt_source is None or not hasattr(self.prompt_source, '__anext__'):
            raise TypeError(f"prompt_source must be a non-None async iterator, got {type(self.prompt_source)}")

        overall_metrics_accumulator: Dict[str, List[float]] = defaultdict(list)
        batch_generation_metrics: List[GenerationMetrics] = []  # Collect metrics for this training run

        # Start metrics tracking
        self.metrics_tracker.start_training()
        
        logger.info(f"GRPOTrainerAdapter: Starting training for {total_training_iterations} iterations (rollout batches).")

        for iteration in range(total_training_iterations):
            logger.info(f"Iteration {iteration + 1}/{total_training_iterations}")
            try:
                initial_prompt = await self.prompt_source.__anext__() # type: ignore
            except StopAsyncIteration:
                logger.warning("Warning: Ran out of initial prompts from prompt_source. Resetting and re-iterating.")
                await self.prompt_source.reset() # type: ignore
                try:
                    initial_prompt = await self.prompt_source.__anext__() # type: ignore
                except StopAsyncIteration:
                    logger.error("Prompt source is empty or not re-iterable after reset and exhaustion.")
                    raise ValueError("Prompt source is empty or not re-iterable after reset and exhaustion.")

            try:
                processed_trajectory, generation_metrics = await self._generate_and_process_rollout(initial_prompt)
                
                # Track the generation metrics
                self.metrics_tracker.track_generation(generation_metrics)
                batch_generation_metrics.append(generation_metrics)
                
            except RuntimeError as e:
                logger.error(f"ERROR during async rollout/processing: {e}. Iteration {iteration + 1} will be skipped.", exc_info=True)
                continue # Skip to next iteration if this specific rollout fails critically

            if not processed_trajectory:
                logger.warning(f"Warning: Iteration {iteration + 1} produced no processable data. Skipping TRL training for this batch.")
                continue

            dataset_data: List[Dict[str, Any]] = []
            for turn_data in processed_trajectory:
                entry: Dict[str, Any] = {
                    "prompt": turn_data["prompt_text"],
                    "completion": turn_data["completion_text"],
                    # "reward" field is not added; TRL calculates it using its reward_funcs.
                }
                old_logprobs = turn_data.get("old_per_token_logps")
                if old_logprobs is not None:
                    if isinstance(old_logprobs, torch.Tensor):
                        entry["old_logprobs"] = torch.nan_to_num(old_logprobs.clone().detach().to(dtype=torch.float32), nan=-1e9)
                    else:
                        try:
                            entry["old_logprobs"] = torch.nan_to_num(torch.tensor(old_logprobs, dtype=torch.float32), nan=-1e9)
                        except Exception as e:
                            logger.warning(f"Failed to convert old_logprobs to tensor: {e}. Skipping for this entry. Prompt: {turn_data['prompt_text'][:50]}")
                else:
                    logger.warning(f"old_per_token_logps is None for a sample. TRL might require this. Prompt: {turn_data['prompt_text'][:50]}")
                
                # CRITICAL: Add "infos" (original_step_info) for TRL-compatible reward functions.
                # TRL passes extra columns from the dataset as kwargs to its reward_funcs.
                original_step_info = turn_data.get("auxiliary_data", {}).get("original_step_info")
                if original_step_info is not None:
                    entry["infos"] = original_step_info
                else:
                    logger.warning(f"original_step_info is missing in turn_data for a sample. Reward function might not work as expected. Prompt: {turn_data['prompt_text'][:50]}")
                    entry["infos"] = {} # Provide an empty dict to avoid KeyErrors if reward func expects 'infos'

                dataset_data.append(entry)

            if not dataset_data:
                logger.warning(f"Iteration {iteration + 1} resulted in empty dataset_data after processing trajectory. Skipping.")
                continue
            
            if dataset_data:
                sample_log_entry = dataset_data[0]
                log_msg = "[GRPOTrainerAdapter.train] Sample entry for TRL Dataset:\n  {\n"
                for key, value in sample_log_entry.items():
                    if key == 'old_logprobs' and isinstance(value, torch.Tensor):
                        log_msg += f"    '{key}': torch.Tensor(shape={value.shape}, dtype={value.dtype}),\n"
                    elif key == 'infos' and isinstance(value, dict):
                        log_msg += f"    '{key}': {{infos_keys: {list(value.keys())}}},\n"
                    else:
                        log_msg += f"    '{key}': {repr(value)[:100] + ('...' if len(repr(value)) > 100 else '')},\n"
                log_msg += "  }"
                logger.debug(log_msg)

            hf_dataset: Optional[Dataset] = Dataset.from_list(dataset_data) # type: ignore[assignment]
            
            if hf_dataset is None:
                logger.warning(f"  Iteration {iteration + 1}: HF Dataset creation returned None. Skipping TRL training for this batch.")
                continue

            logger.info(f"  Iteration {iteration + 1}: Created HF Dataset with {len(hf_dataset)} entries for TRL.")

            current_algo_config = self.algorithm_config # type: ignore[assignment]
            if not isinstance(current_algo_config, ActualTrlGRPOConfig):
                logger.error(f"Algorithm config is not of type ActualTrlGRPOConfig. Type: {type(current_algo_config)}")
                raise TypeError("Algorithm config is not a TRL GRPOConfig.")


            original_num_train_epochs = current_algo_config.num_train_epochs
            original_logging_steps = current_algo_config.logging_steps
            original_save_steps = current_algo_config.save_steps
            
            current_algo_config.num_train_epochs = float(current_algo_config.num_iterations)
            current_algo_config.logging_steps = 1 # Ensure TRL logs at every step for this batch.
            if current_algo_config.num_iterations <= 5: # Configure save_steps for less frequent saving during inner loops.
                 current_algo_config.save_steps = current_algo_config.num_iterations + 1

            logger.debug(f"  GRPOTrainerAdapter: TRL Config before train call: logging_steps={current_algo_config.logging_steps}, num_train_epochs={current_algo_config.num_train_epochs}, num_iterations={current_algo_config.num_iterations}")
            logger.info(f"  Calling TRL trainer.train() for {current_algo_config.num_train_epochs} epochs over current batch of {len(hf_dataset)} items.")
            
            original_trl_train_dataset = self.trl_trainer.train_dataset
            self.trl_trainer.train_dataset = hf_dataset # Temporarily set the dataset for TRL
            logger.debug(f"  Temporarily set trl_trainer.train_dataset. Features: {self.trl_trainer.train_dataset.features if self.trl_trainer.train_dataset else 'None'}")


            try:
                train_output = self.trl_trainer.train() # train_output could be None from dummy TRL trainer
                logger.debug(f"  GRPOTrainerAdapter: TRL train_output: {train_output}")
                if train_output:
                    logger.debug(f"  GRPOTrainerAdapter: TRL train_output.training_loss: {getattr(train_output, 'training_loss', 'N/A')}, train_output.metrics: {getattr(train_output, 'metrics', 'N/A')}")

                trl_batch_metrics: Optional[Dict[str, float]] = None
                training_loss_val: Optional[float] = None

                if train_output is not None:
                    if hasattr(train_output, 'metrics') and train_output.metrics:
                        trl_batch_metrics = train_output.metrics
                    elif hasattr(train_output, 'training_loss'):
                        loss_attr = getattr(train_output, 'training_loss')
                        if isinstance(loss_attr, (float, int)): # Ensure training_loss is float or convertible
                            training_loss_val = float(loss_attr)
                        else:
                            logger.warning(f"train_output.training_loss was not a number: {loss_attr}")
                
                if trl_batch_metrics:
                    logger.info(f"  TRL Batch Training Metrics: {trl_batch_metrics}")
                    for key, value in trl_batch_metrics.items():
                        overall_metrics_accumulator[key].append(value)
                elif training_loss_val is not None:
                     overall_metrics_accumulator["training_loss"].append(training_loss_val)
                     logger.info(f"  TRL Batch Training Loss: {training_loss_val}")
                else:
                    logger.info("  TRL Batch Training: No metrics or training_loss reported from train_output.")


            except Exception as e:
                logger.error(f"  Error during TRL trainer.train() on batch: {e}", exc_info=True)
                raise
            finally:
                self.trl_trainer.train_dataset = original_trl_train_dataset # Restore TRL's original dataset
                logger.debug("  Restored trl_trainer.train_dataset.")
                current_algo_config.num_train_epochs = original_num_train_epochs
                current_algo_config.logging_steps = original_logging_steps
                current_algo_config.save_steps = original_save_steps

        logger.info("GRPOTrainerAdapter: Main training loop finished.")
        
        # Compute final metrics including environment-specific ones
        final_metrics = {
            key: sum(values) / len(values) if values else 0
            for key, values in overall_metrics_accumulator.items()
        }
        
        # Add environment-specific metrics
        env_metrics = self.metrics_tracker.get_current_metrics()
        final_metrics.update(env_metrics)
        
        # Add training progress summary
        progress_metrics = self.metrics_tracker.get_training_progress_summary()
        final_metrics.update(progress_metrics)
        
        # Log batch summary if we have generation metrics
        if batch_generation_metrics:
            batch_metrics: TrainingBatchMetrics = self.metrics_tracker.compute_batch_metrics(batch_generation_metrics)
            self.metrics_tracker.log_batch_summary(total_training_iterations, batch_metrics)
            
            # Add batch metrics to final metrics
            final_metrics.update({
                "batch/parsing_success_rate": batch_metrics.parsing_success_rate,
                "batch/tool_success_rate": batch_metrics.tool_success_rate,
                "batch/final_answer_rate": batch_metrics.final_answer_rate,
                "batch/mean_total_reward": batch_metrics.mean_total_reward,
                "batch/mean_turns_per_rollout": batch_metrics.mean_turns_per_rollout,
            })
        
        # Fallback: Try to get metrics from TRL state if no direct metrics were captured
        # and trl_trainer and its state are available.
        if not overall_metrics_accumulator and self.trl_trainer and hasattr(self.trl_trainer, 'state') and self.trl_trainer.state and hasattr(self.trl_trainer.state, 'log_history'):
            for log_entry in self.trl_trainer.state.log_history:
                for key, value in log_entry.items():
                    if isinstance(value, (int, float)): # Only numeric values
                        overall_metrics_accumulator[f"avg_{key}"].append(value)
            trl_metrics = {
                key: sum(values) / len(values) if values else 0
                for key, values in overall_metrics_accumulator.items()
            }
            final_metrics.update(trl_metrics)

        logger.info(f"Final aggregated metrics: {final_metrics}")
        return final_metrics

    async def step(self, experience_batch: ExperienceBatch) -> TrainingMetrics:
        """
        Processes a pre-defined experience batch asynchronously.
        The experience_batch is expected to be List[SingleTurnDataForTrainer-like dicts].
        """
        if self.trl_trainer is None:
            raise RuntimeError("TRL Trainer not initialized.")
        if not _datasets_available:
            raise RuntimeError("Datasets library not available for GRPOTrainerAdapter.step.")

        if not isinstance(experience_batch, list) or \
           not all(isinstance(item, dict) for item in experience_batch) or \
           not experience_batch:
            raise ValueError("experience_batch for step should be a non-empty List[SingleTurnDataForTrainer-like dicts]")

        dataset_data: List[Dict[str, Any]] = []
        for turn_data in experience_batch:
            entry: Dict[str, Any] = {
                "prompt": turn_data["prompt_text"],
                "completion": turn_data["completion_text"],
                # "reward" is not added; TRL calculates rewards via reward_funcs.
            }
            old_logprobs = turn_data.get("old_per_token_logps")
            if old_logprobs is not None:
                if isinstance(old_logprobs, torch.Tensor):
                    entry["old_logprobs"] = torch.nan_to_num(old_logprobs.clone().detach().to(dtype=torch.float32), nan=-1e9)
                else:
                    try:
                        temp_tensor = torch.tensor(old_logprobs, dtype=torch.float32)
                        entry["old_logprobs"] = torch.nan_to_num(temp_tensor, nan=-1e9)
                    except Exception as e:
                        logger.warning(f"Failed to convert old_logprobs to tensor in GRPOTrainerAdapter.step: {e}. Skipping. Prompt: {turn_data.get('prompt_text', '')[:50]}")
            
            # Add "infos" (original_step_info) for TRL reward functions, similar to the train method.
            original_step_info = turn_data.get("auxiliary_data", {}).get("original_step_info")
            if original_step_info is not None:
                entry["infos"] = original_step_info
            else:
                logger.warning(f"original_step_info is missing for a sample in GRPOTrainerAdapter.step. Prompt: {turn_data.get('prompt_text', '')[:50]}")
                entry["infos"] = {}

            dataset_data.append(entry)
        
        hf_dataset: Optional[Dataset] = Dataset.from_list(dataset_data) # type: ignore[assignment]

        if hf_dataset is None:
            logger.warning("GRPOTrainerAdapter.step(): HF Dataset creation returned None. Cannot perform step.")
            return {}

        logger.info(f"GRPOTrainerAdapter.step(): Created HF Dataset with {len(hf_dataset)} entries.")

        current_algo_config = self.algorithm_config
        if not isinstance(current_algo_config, ActualTrlGRPOConfig):
            logger.error(f"Algorithm config is not ActualTrlGRPOConfig in step. Type: {type(current_algo_config)}")
            raise TypeError("Algorithm config is not a TRL GRPOConfig during step.")

        original_num_train_epochs = current_algo_config.num_train_epochs
        original_logging_steps = current_algo_config.logging_steps
        original_save_steps = current_algo_config.save_steps
        
        current_algo_config.num_train_epochs = float(current_algo_config.num_iterations)
        current_algo_config.logging_steps = 1 # Log at every step for this batch.
        if current_algo_config.num_iterations <= 5:
            current_algo_config.save_steps = current_algo_config.num_iterations + 1
        
        logger.debug(f"  GRPOTrainerAdapter (step): TRL Config before train call: logging_steps={current_algo_config.logging_steps}, num_train_epochs={current_algo_config.num_train_epochs}, num_iterations={current_algo_config.num_iterations}")
        step_metrics: TrainingMetrics = {}
        try:
            logger.info(f"  Calling TRL trainer.train() for {current_algo_config.num_train_epochs} epochs over current batch of {len(hf_dataset)} items (via step method).")
            
            original_trl_train_dataset = self.trl_trainer.train_dataset
            self.trl_trainer.train_dataset = hf_dataset # Temporarily set dataset
            logger.debug(f"  Temporarily set trl_trainer.train_dataset for step. Features: {self.trl_trainer.train_dataset.features if self.trl_trainer.train_dataset else 'None'}")

            train_output = self.trl_trainer.train()
            logger.debug(f"  GRPOTrainerAdapter (step): TRL train_output: {train_output}")
            if train_output:
                logger.debug(f"  GRPOTrainerAdapter (step): TRL train_output.training_loss: {getattr(train_output, 'training_loss', 'N/A')}, train_output.metrics: {getattr(train_output, 'metrics', 'N/A')}")
            
            if train_output is not None:
                if hasattr(train_output, 'metrics') and train_output.metrics:
                    step_metrics = train_output.metrics
                elif hasattr(train_output, 'training_loss'):
                    loss_attr = getattr(train_output, 'training_loss')
                    if isinstance(loss_attr, (float, int)):
                        step_metrics = {"training_loss": float(loss_attr)}
                    else:
                        logger.warning(f"train_output.training_loss was not a number in step: {loss_attr}")
            else:
                logger.info("  TRL Batch Training (step): No metrics or training_loss reported from train_output.")

        except Exception as e:
            logger.error(f"  Error during TRL trainer.train() on batch (via step method): {e}", exc_info=True)
            # Error is logged; finally block will execute, then current (likely empty) step_metrics are returned.
        finally:
            self.trl_trainer.train_dataset = original_trl_train_dataset # Restore original dataset
            logger.debug("  Restored trl_trainer.train_dataset after step.")
            current_algo_config.num_train_epochs = original_num_train_epochs
            current_algo_config.logging_steps = original_logging_steps
            current_algo_config.save_steps = original_save_steps
            
        logger.info(f"  TRL Step Training Metrics: {step_metrics}")
        return step_metrics

    # Checkpointing Methods

    def save_checkpoint(self, checkpoint_directory: str) -> None:
        """
        Saves the model state using the underlying TRL trainer's method.
        """
        if self.trl_trainer is None:
            logger.error("TRL Trainer not initialized. Cannot save checkpoint.")
            raise RuntimeError("TRL Trainer not initialized. Cannot save checkpoint.")
            
        logger.info(f"GRPOTrainerAdapter (TRL backend): Saving checkpoint to {checkpoint_directory}")
        try:
            self.trl_trainer.save_model(checkpoint_directory)
            # Note: TRL's save_model typically saves model weights.
            # Full trainer state (optimizer, etc.) may require a separate save_state() if TRL provides it
            # and if needed for an exact resume beyond just model weights.
        except Exception as e:
            logger.error(f"GRPOTrainerAdapter (TRL backend): Error saving checkpoint - {e}", exc_info=True)
            raise RuntimeError(f"Failed to save checkpoint to {checkpoint_directory}: {e}") from e

    def load_checkpoint(self, checkpoint_directory: str) -> None:
        """
        Advises on TRL checkpoint loading.
        
        Directly loading state onto an existing TRL Trainer object is not typical.
        Loading is usually handled via configuration (e.g., `resume_from_checkpoint`
        in TrainingArguments or initializing the base model from the checkpoint)
        *before* creating this trainer adapter.
        """
        if not _trl_available:
             raise ImportError("TRL is required to assess checkpoint loading.")

        logger.info(f"GRPOTrainerAdapter (TRL backend): Checkpoint loading information ({checkpoint_directory}) - "
              "TRL typically handles this via `resume_from_checkpoint=True` in its `train()` call "
              "(using the path set in TrainingArguments), or by initializing the base model "
              f"from '{checkpoint_directory}' before creating the trainer adapter.")
        # This method is informational and does not modify the live trainer state.
        pass


