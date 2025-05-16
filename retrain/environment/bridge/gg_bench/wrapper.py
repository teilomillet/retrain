import asyncio
from typing import Any, Dict, Tuple, Optional, TypeVar

from retrain.environment.environment import Environment
from retrain.environment.types import LLMAction, EnvironmentObservation

# Type variable for the synchronous environment, can be refined with a Protocol later
SyncEnv = TypeVar("SyncEnv")


class AsyncGGBenchBridgeWrapper(Environment):
    """
    Wraps a synchronous, fully-wrapped gg-bench environment to make it
    compatible with retrain's asynchronous Environment interface.

    It handles the async/sync transition and delegates control methods
    to the underlying gg-bench environment, particularly for self-play
    mechanisms managed by AlternatingAgentEnv.
    """

    def __init__(self, sync_env: SyncEnv):
        """
        Initializes the bridge wrapper.

        Args:
            sync_env: The synchronous gg-bench environment, already wrapped
                      with gg-bench's standard wrappers (TimeoutEnv,
                      MetadataEnv, AlternatingAgentEnv).
        """
        super().__init__()
        self.sync_env: SyncEnv = sync_env
        
        self.action_space: Any = getattr(self.sync_env, 'action_space', None)
        self.observation_space: Any = getattr(self.sync_env, 'observation_space', None)
        
        self._loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()

    def _action_to_sync(self, action: LLMAction) -> Any:
        """
        Converts retrain's LLMAction to the action type expected by the sync env.
        Placeholder for specific conversion logic.
        """
        return action

    def _obs_from_sync(self, sync_obs: Any) -> EnvironmentObservation:
        """
        Converts the sync env's observation to retrain's EnvironmentObservation type.
        Placeholder for specific conversion logic.
        """
        return sync_obs

    async def step(self, action: LLMAction) -> Tuple[EnvironmentObservation, float, bool, bool, Dict[str, Any]]:
        """Asynchronously steps through the synchronous environment."""
        sync_action = self._action_to_sync(action)

        obs_sync, reward, terminated, truncated, info = await self._loop.run_in_executor(
            None, 
            self.sync_env.step, 
            sync_action
        )
        
        env_observation = self._obs_from_sync(obs_sync)
        
        return env_observation, float(reward), bool(terminated), bool(truncated), info

    async def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[EnvironmentObservation, Dict[str, Any]]:
        """Asynchronously resets the synchronous environment."""
        
        def _sync_reset_with_kwargs():
            # sourcery skip: instance-method-must-be-public
            # This is an internal helper, okay to be private-like.
            return self.sync_env.reset(seed=seed, options=options)

        obs_sync, info = await self._loop.run_in_executor(
            None,
            _sync_reset_with_kwargs
        )
        
        env_observation = self._obs_from_sync(obs_sync)
        return env_observation, info

    def render(self) -> None:
        """Renders the synchronous environment."""
        if hasattr(self.sync_env, 'render'):
            self.sync_env.render()
        else:
            super().render()

    def close(self) -> None:
        """Closes the synchronous environment."""
        if hasattr(self.sync_env, 'close'):
            self.sync_env.close()
        else:
            super().close()

    async def add_opposing_agent(self, agent: Any, resample: bool = True) -> None:
        """
        Adds an opposing agent to the underlying AlternatingAgentEnv.
        Essential for self-play.
        """
        if not hasattr(self.sync_env, 'add_opposing_agent'):
            raise AttributeError(
                "Wrapped sync_env lacks 'add_opposing_agent'. Ensure it's AlternatingAgentEnv."
            )
        
        await self._loop.run_in_executor(
            None,
            self.sync_env.add_opposing_agent,
            agent,
            resample
        )

    async def set_epsilon(self, epsilon: float) -> None:
        """
        Sets epsilon for the opposing agent in AlternatingAgentEnv.
        """
        if not hasattr(self.sync_env, 'set_epsilon'):
            raise AttributeError(
                "Wrapped sync_env lacks 'set_epsilon'. Ensure it's AlternatingAgentEnv."
            )
        
        await self._loop.run_in_executor(
            None,
            self.sync_env.set_epsilon,
            epsilon
        )