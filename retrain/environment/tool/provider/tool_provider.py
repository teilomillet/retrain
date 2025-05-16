from abc import ABC, abstractmethod
from typing import List, Any

from ...tool import Tool # Up three levels to retrain.environment.tool.tool

class ToolProvider(ABC):
    """
    Abstract Base Class for tool providers.

    Tool providers are responsible for discovering or supplying a list of 
    tool instances that an environment can use.
    """

    @abstractmethod
    async def discover_tools(self, **kwargs: Any) -> List[Tool]:
        """
        Discovers and returns a list of tool instances.

        This method should be implemented by concrete providers to fetch,
        generate, or otherwise assemble a list of Tool-compliant objects.

        Args:
            **kwargs: Provider-specific arguments that might be needed for discovery
                      (e.g., client instances, configuration paths).

        Returns:
            A list of Tool instances.
        """
        pass 