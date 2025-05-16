from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List, TYPE_CHECKING
import gymnasium as gym

# Import the new types
from retrain.environment.types import LLMAction, EnvironmentObservation
from retrain.reward.types import RawRolloutData

# Type alias for clarity, can be refined later (e.g., to a specific dict structure or Pydantic model)
# ActionType = Any 
# ObservationType = Any

# Placeholders for model and tokenizer types, to be replaced with actual imports
ModelObject = Any
TokenizerObject = Any
SamplingParams = Dict[str, Any]

class Environment(ABC):
    """
    Abstract Base Class for all environments in the `retrain` library.

    This class defines a standard interface for how reinforcement learning agents
    (primarily Language Models in this context) interact with a task or world.

    """

    # Metadata like in gym.Env
    metadata: Dict[str, Any] = {'render_modes': []}
    reward_range: Tuple[float, float] = (-float('inf'), float('inf'))
    action_space: Any = None       
    observation_space: Any = None    

    def __init__(self, **kwargs: Any):
        """
        Initializes the environment.
        Specific environment configurations can be passed via kwargs.
        """
        # Basic initialization, can be extended by subclasses
        pass

    async def setup(self) -> None:
        """
        Perform any necessary asynchronous setup for the environment after initialization.
        This can include loading resources, configuring tools, establishing connections, etc.
        This default implementation does nothing.
        Subclasses should override this method if they have setup requirements.
        """
        pass

    @abstractmethod
    async def step(self, action: LLMAction) -> Tuple[EnvironmentObservation, float, bool, bool, Dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics asynchronously.

        When the agent takes an action, the environment responds with the next
        observation, a reward, and flags indicating if the episode has
        terminated or been truncated.

        Args:
            action (LLMAction): An action provided by the agent, processed into a standard structure.
                                For LLMs, this could be generated text, a tool call, etc.

        Returns:
            Tuple[EnvironmentObservation, float, bool, bool, Dict[str, Any]]: A tuple containing:
                - observation (EnvironmentObservation): The agent's observation of the current environment state.
                - reward (float): The amount of reward returned after the previous action.
                - terminated (bool): Whether the agent reached a terminal state (episode end due to task completion).
                - truncated (bool): Whether the episode was ended by an external condition (e.g., max steps).
                - info (Dict[str, Any]): Auxiliary diagnostic information (helpful for debugging, learning).
        """
        pass

    @abstractmethod
    async def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[EnvironmentObservation, Dict[str, Any]]:
        """
        Resets the environment to an initial state asynchronously and returns the initial observation.

        This method should be called at the beginning of every new episode.

        Args:
            seed (Optional[int]): The seed that is used to initialize the environment's
                                  random number generator(s). This is for reproducibility.
            options (Optional[Dict[str, Any]]): Additional information to specify how to reset the environment
                                                (e.g., starting with a specific prompt or task configuration).

        Returns:
            Tuple[EnvironmentObservation, Dict[str, Any]]: A tuple containing:
                - observation (EnvironmentObservation): The initial observation of the space.
                - info (Dict[str, Any]): Auxiliary diagnostic information.
        """
        pass

    def render(self) -> None:
        """
        Renders the environment.

        The set of supported modes varies per environment. For LLM environments,
        this might involve printing the current conversation history or task state
        to the console.
        
        This default implementation does nothing.
        """
        pass

    def close(self) -> None:
        """
        Performs any necessary cleanup.

        Environments will automatically close when garbage collected or when the
        program exits. This method allows for manual immediate cleanup.
        This default implementation does nothing.
        """
        pass

    # Potentially add __str__ or __repr__ for easier debugging of env instances
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>" 

    # Retrain Specific Optional Methods
    async def rollout(
        self,
        initial_prompt: str, # Or List[Dict[str, str]] for initial messages
        llm_model: ModelObject,
        tokenizer: TokenizerObject,
        sampling_params: SamplingParams,
        max_tokens_to_generate: int = 256
    ) -> RawRolloutData:
        """
        Performs a full rollout in the environment using the provided LLM until 
        the episode ends (terminated or truncated). This method encapsulates the 
        interaction loop between the LLM and the environment.

        This is an optional method; not all environments might support or require
        a self-contained rollout logic (e.g., if the trainer manages the loop).

        Args:
            initial_prompt: The initial message or prompt to start the conversation.
            llm_model: The language model object to use for generating responses.
            tokenizer: The tokenizer associated with the llm_model.
            sampling_params: Parameters for the LLM's sampling process.
            max_tokens_to_generate: Max tokens for each LLM generation call within the rollout.

        Returns:
            RawRolloutData: A tuple containing the full conversation history,
                            executed LLM actions, intrinsic environment rewards per step,
                            observations per step, info dicts per step, and
                            optionally, the old_per_token_logps for the entire completion.
        
        Raises:
            NotImplementedError: If the environment does not implement this method.
        """
        raise NotImplementedError("Rollout method not implemented for this environment.")

    # Potentially other environment-specific utility methods
    # def get_available_tools(self) -> List[Dict[str, Any]]:
    #     raise NotImplementedError 