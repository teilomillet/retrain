from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable

# Configuration for the specific learning algorithm (e.g., GRPOConfig, PPOConfig)
AlgorithmConfig = Any
# The AI model object being trained
ModelObject = Any
# A batch of interaction data (e.g., prompts, responses, rewards)
ExperienceBatch = Any
# Dictionary to store metrics from training
TrainingMetrics = Dict[str, Any]
# Source for prompts, could be a dataset, a list, or a generator
PromptSource = Any
# Defines how to calculate rewards.
# Args: prompts (List[str]), completions (List[str]), **kwargs
# Returns: List of rewards (float)
RewardFunction = Callable[[List[str], List[str], Dict[str, Any]], List[float]]


class BaseTrainer(ABC):
    """
    Abstract Base for all training backends.

    This class defines a standard way for the main system (Orchestrator)
    to interact with different Reinforcement Learning algorithms. It aims to
    hide the specific details of each underlying library (like TRL).
    """

    def __init__(self,
                 model: ModelObject,
                 algorithm_config: AlgorithmConfig,
                 reward_functions: List[RewardFunction],
                 prompt_source: Optional[PromptSource] = None,
                 tokenizer: Optional[Any] = None,
                 reference_model: Optional[ModelObject] = None,
                 **trainer_specific_kwargs: Any):
        """
        Initializes the training backend.

        This is where you provide all the static components and configurations
        needed for the trainer to be set up.

        Args:
            model: The core AI model (e.g., a language model) to be trained.
            algorithm_config: Specific settings for the learning algorithm itself
                              (e.g., learning rate, batch size for the algorithm).
            reward_functions: A list of functions that will score the model\'s
                              outputs. This is crucial for RL.
            prompt_source: Optional. Where the trainer should get its initial
                           prompts or questions if it manages its own data generation
                           (e.g., a Hugging Face Dataset, a list of strings).
            tokenizer: Optional. The tokenizer associated with the model, often needed
                       by backend libraries.
            reference_model: Optional. A baseline version of the model, used by some
                             algorithms (like PPO or GRPO) for stabilization.
            **trainer_specific_kwargs: For any other settings that are highly specific
                                       to one particular type of trainer backend and
                                       don\'t fit the common parameters.
        """
        self.model = model
        self.algorithm_config = algorithm_config
        self.reward_functions = reward_functions
        self.prompt_source = prompt_source
        self.tokenizer = tokenizer
        self.reference_model = reference_model
        self.trainer_specific_kwargs = trainer_specific_kwargs

        # This internal method is called to validate inputs and
        # perform the actual setup of the underlying trainer library.
        self._validate_and_setup_backend()

    @abstractmethod
    def _validate_and_setup_backend(self) -> None:
        """
        Internal method: Validates all provided components and configurations.
        It then performs the actual setup of the specific underlying training
        engine (e.g., initializes TRL\'s GRPOTrainer).

        This should raise appropriate errors (e.g., TypeError, ValueError)
        if the configuration or components are unsuitable for this backend.
        This method is called automatically at the end of `__init__`.
        """
        pass

    @abstractmethod
    async def train(self, total_training_iterations: Optional[int] = None) -> TrainingMetrics:
        """
        Starts and executes the main training process asynchronously.

        The trainer might run its own internal loop (generating data, calculating
        rewards, and updating the model) or it might expect data to be fed
        if designed differently (though `step` is more for that).

        Args:
            total_training_iterations: Optional. An indication of how long or how many
                                      cycles the training should run. The specific
                                      meaning (e.g., epochs, steps, episodes)
                                      can be interpreted by the backend.

        Returns:
            A dictionary containing a summary of training results or final metrics.
        """
        pass

    @abstractmethod
    async def step(self, experience_batch: ExperienceBatch) -> TrainingMetrics:
        """
        Performs a single learning update using a provided batch of experience asynchronously.

        This method is primarily for scenarios where the main system (Orchestrator)
        is controlling the experience generation loop and feeds data to the trainer
        incrementally.

        Note: Not all backend libraries or algorithms are easily adapted to an
              external step-by-step control if they are designed for a monolithic
              `train()` call. In such cases, this method might raise a
              `NotImplementedError` or have limited functionality.

        Args:
            experience_batch: A package of data from recent model interactions,
                              typically including prompts, the model\'s responses,
                              and the scores (rewards) it received.

        Returns:
            A dictionary containing metrics specific to this single learning step.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint_directory: str) -> None:
        """Saves the current state of the trainer and model."""
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_directory: str) -> None:
        """Loads the trainer and model state from a checkpoint."""
        pass 