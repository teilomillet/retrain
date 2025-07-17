from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List

# Import the new types
from retrain.environment.types import LLMAction, EnvironmentObservation
from retrain.reward.types import RawRolloutData


import logging
import time
import ray

logger = logging.getLogger(__name__)

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




@ray.remote(num_cpus=2, num_gpus=0)
class ReEnvironment:
    """
    Environment Manager Ray Actor.
    
    Manages environment instances and coordinates rollout execution
    across different environment types (FastMCP, SmolAgent, Spider2).
    
    This actor handles:
    1. Environment initialization and management
    2. Rollout processing and coordination
    3. Environment state management
    4. Integration with DataBuffer for data flow
    """
    
    def __init__(self, config: Any, databuffer: ray.ObjectRef):
        """
        Initialize ReEnvironment actor with configuration and databuffer reference.
        
        Args:
            config: Training configuration object
            databuffer: Reference to the ReDataBuffer actor for data coordination
        """
        self.config = config
        self.databuffer = databuffer
        self.environments: Dict[str, Environment] = {}
        self.environment_stats: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False
        
        # Environment coordination state
        self.current_episode_id = None
        self.active_rollouts: Dict[str, Dict[str, Any]] = {}
        
        logger.info("ReEnvironment actor initialized")
        
    async def initialize(self) -> None:
        """Initialize environment instances based on configuration."""
        logger.info("Initializing ReEnvironment...")
        
        try:
            # Get environment configuration
            env_config = getattr(self.config, 'environment', None)
            if not env_config:
                # Create default environment configuration
                env_config = {
                    'default': {
                        'type': 'fastmcp_env',
                        'config': {}
                    }
                }
                logger.info("Using default environment configuration")
            
            # Initialize environments based on config
            if hasattr(env_config, 'type'):
                # Single environment configuration
                await self._initialize_single_environment('default', env_config)
            else:
                # Multiple environment configuration
                for env_name, env_params in env_config.items():
                    await self._initialize_single_environment(env_name, env_params)
                    
            self.is_initialized = True
            logger.info(f"ReEnvironment initialized with {len(self.environments)} environments")
            
        except Exception as e:
            logger.error(f"Failed to initialize ReEnvironment: {e}")
            raise
            
    async def _initialize_single_environment(self, env_name: str, env_config: Any) -> None:
        """Initialize a single environment instance."""
        try:
            env_type = getattr(env_config, 'type', 'fastmcp_env')
            env_specific_config = getattr(env_config, 'env_specific_config', {})
            
            # Import get_environment function
            from . import get_environment
            
            # Create environment instance
            env_instance = await get_environment(env_type, env_specific_config)
            self.environments[env_name] = env_instance
            
            # Initialize stats for this environment
            self.environment_stats[env_name] = {
                'type': env_type,
                'rollouts_processed': 0,
                'average_episode_time': 0.0,
                'success_rate': 0.0,
                'last_reset_time': 0.0,
                'status': 'ready'
            }
            
            logger.info(f"Environment '{env_name}' ({env_type}) initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize environment '{env_name}': {e}")
            raise
            
    async def prepare_episode(self, episode_id: int) -> Dict[str, Any]:
        """
        Prepare environments for new episode.
        
        Args:
            episode_id: Current episode identifier
            
        Returns:
            Dictionary with environment preparation results
        """
        if not self.is_initialized:
            raise RuntimeError("ReEnvironment not initialized")
            
        logger.info(f"Preparing environments for episode {episode_id}")
        self.current_episode_id = episode_id
        
        results = {}
        
        for env_name, env in self.environments.items():
            try:
                start_time = time.time()
                
                # Reset environment for new episode
                obs, info = await env.reset(
                    options={'episode_id': episode_id}
                )
                
                reset_time = time.time() - start_time
                self.environment_stats[env_name]['last_reset_time'] = reset_time
                self.environment_stats[env_name]['status'] = 'ready'
                
                results[env_name] = {
                    'observation': obs,
                    'info': info,
                    'reset_time': reset_time,
                    'status': 'ready'
                }
                
                logger.debug(f"Environment '{env_name}' prepared in {reset_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to prepare environment '{env_name}': {e}")
                self.environment_stats[env_name]['status'] = 'error'
                results[env_name] = {
                    'error': str(e),
                    'status': 'error'
                }
                
        logger.info(f"Episode {episode_id} preparation complete: {len(results)} environments")
        return results
        
    async def process_rollouts(self, rollout_data: List[Dict[str, Any]], episode_id: int) -> List[Dict[str, Any]]:
        """
        Process rollouts through environments.
        
        Args:
            rollout_data: List of rollout data to process
            episode_id: Current episode identifier
            
        Returns:
            List of processed rollout results
        """
        if not self.is_initialized:
            raise RuntimeError("ReEnvironment not initialized")
            
        logger.info(f"Processing {len(rollout_data)} rollouts for episode {episode_id}")
        
        processed_rollouts = []
        
        for i, rollout in enumerate(rollout_data):
            try:
                processed_rollout = await self._process_single_rollout(rollout, episode_id, i)
                processed_rollouts.append(processed_rollout)
                
            except Exception as e:
                logger.error(f"Failed to process rollout {i}: {e}")
                # Create error rollout
                error_rollout = rollout.copy()
                error_rollout.update({
                    'processing_error': str(e),
                    'processing_status': 'error',
                    'episode_id': episode_id,
                    'rollout_idx': i
                })
                processed_rollouts.append(error_rollout)
                
        # Update statistics
        await self._update_environment_stats(processed_rollouts, episode_id)
        
        logger.info(f"Completed processing {len(processed_rollouts)} rollouts")
        return processed_rollouts
        
    async def _process_single_rollout(self, rollout: Dict[str, Any], episode_id: int, rollout_idx: int) -> Dict[str, Any]:
        """Process a single rollout through appropriate environment."""
        start_time = time.time()
        
        # Determine which environment to use
        env_name = rollout.get('environment', 'default')
        if env_name not in self.environments:
            env_name = 'default'
            
        if env_name not in self.environments:
            # Use first available environment
            env_name = next(iter(self.environments.keys()))
            
        environment = self.environments[env_name]
        
        # Extract rollout components
        conversation_history = rollout.get('conversation_history', [])
        executed_actions = rollout.get('executed_llm_actions', [])
        
        # Process the rollout data for environment context
        processed_rollout = rollout.copy()
        processed_rollout.update({
            'environment_used': env_name,
            'episode_id': episode_id,
            'rollout_idx': rollout_idx,
            'processing_time': 0.0,
            'processing_status': 'success',
            'environment_observations': [],
            'environment_rewards': [],
            'final_state': {}
        })
        
        try:
            # If the environment supports analysis of completed rollouts
            if hasattr(environment, 'analyze_rollout'):
                analysis = await environment.analyze_rollout(  # type: ignore
                    conversation_history=conversation_history,
                    executed_actions=executed_actions
                )
                processed_rollout['environment_analysis'] = analysis
                
            # Extract environment-specific information
            if hasattr(environment, 'get_final_state'):
                final_state = await environment.get_final_state()  # type: ignore
                processed_rollout['final_state'] = final_state
                
            # Calculate environment-specific metrics
            if conversation_history:
                processed_rollout['conversation_length'] = len(conversation_history)
                processed_rollout['action_count'] = len(executed_actions)
                
                # Simple success heuristic based on conversation completion
                last_message = conversation_history[-1] if conversation_history else {}
                processed_rollout['environment_success'] = not last_message.get('error', False)
                
        except Exception as e:
            logger.warning(f"Environment processing failed for rollout {rollout_idx}: {e}")
            processed_rollout['processing_status'] = 'partial'
            processed_rollout['processing_warning'] = str(e)
            
        processing_time = time.time() - start_time
        processed_rollout['processing_time'] = processing_time
        
        return processed_rollout
        
    async def _update_environment_stats(self, processed_rollouts: List[Dict[str, Any]], episode_id: int) -> None:
        """Update environment statistics based on processed rollouts."""
        env_rollout_counts = {}
        env_success_counts = {}
        
        for rollout in processed_rollouts:
            env_name = rollout.get('environment_used', 'default')
            
            # Count rollouts per environment
            env_rollout_counts[env_name] = env_rollout_counts.get(env_name, 0) + 1
            
            # Count successes
            if rollout.get('environment_success', False):
                env_success_counts[env_name] = env_success_counts.get(env_name, 0) + 1
                
        # Update stats for each environment
        for env_name in env_rollout_counts:
            if env_name in self.environment_stats:
                stats = self.environment_stats[env_name]
                
                # Update rollout count
                old_count = stats['rollouts_processed']
                new_count = old_count + env_rollout_counts[env_name]
                stats['rollouts_processed'] = new_count
                
                # Update success rate
                successes = env_success_counts.get(env_name, 0)
                if new_count > 0:
                    old_success_rate = stats['success_rate']
                    new_success_rate = ((old_success_rate * old_count) + successes) / new_count
                    stats['success_rate'] = new_success_rate
                    
    async def get_environment_status(self) -> Dict[str, Any]:
        """Get current status of all environments."""
        status = {
            'initialized': self.is_initialized,
            'current_episode': self.current_episode_id,
            'environment_count': len(self.environments),
            'environments': {}
        }
        
        for env_name, stats in self.environment_stats.items():
            env_status = stats.copy()
            
            # Add runtime information
            if env_name in self.environments:
                env = self.environments[env_name]
                env_status['class_name'] = type(env).__name__
                env_status['available'] = True
            else:
                env_status['available'] = False
                
            status['environments'][env_name] = env_status
            
        return status
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        health_status = {
            'status': 'healthy',
            'initialized': self.is_initialized,
            'environment_count': len(self.environments),
            'total_rollouts_processed': sum(
                stats['rollouts_processed'] for stats in self.environment_stats.values()
            ),
            'average_success_rate': 0.0,
            'timestamp': time.time()
        }
        
        # Calculate average success rate
        if self.environment_stats:
            total_success = sum(stats['success_rate'] for stats in self.environment_stats.values())
            health_status['average_success_rate'] = total_success / len(self.environment_stats)
            
        # Check for environment issues
        failed_envs = [
            name for name, stats in self.environment_stats.items()
            if stats['status'] == 'error'
        ]
        
        if failed_envs:
            health_status['status'] = 'warning'
            health_status['failed_environments'] = failed_envs
            
        return health_status
        
    async def shutdown(self) -> None:
        """Gracefully shutdown all environments."""
        logger.info("Shutting down ReEnvironment...")
        
        for env_name, env in self.environments.items():
            try:
                if hasattr(env, 'close'):
                    env.close()
                logger.info(f"Environment '{env_name}' closed successfully")
            except Exception as e:
                logger.warning(f"Error closing environment '{env_name}': {e}")
                
        self.environments.clear()
        self.environment_stats.clear()
        self.is_initialized = False
        
        logger.info("ReEnvironment shutdown complete") 