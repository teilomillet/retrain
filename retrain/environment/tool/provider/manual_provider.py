from typing import List, Any
from loguru import logger # Added for logging

from ...tool import Tool # From retrain.environment.tool.tool
from .tool_provider import ToolProvider # From retrain.environment.tool.provider.tool_provider

class ManualToolProvider(ToolProvider):
    """
    A ToolProvider that serves a predefined list of tool instances.

    This is useful for:
    - Providing already instantiated and configured tools.
    - Injecting mock tools for testing.
    - Scenarios where tools are created dynamically outside the @tool registry.
    """
    def __init__(self, tools: List[Tool]):
        """
        Initializes the provider with a list of tool instances.

        Args:
            tools: A list of objects that are instances of classes inheriting from Tool.
        """
        if not isinstance(tools, list) or not all(isinstance(t, Tool) for t in tools):
            # ValueError for incorrect setup is appropriate and should remain.
            raise ValueError("ManualToolProvider must be initialized with a list of Tool instances.")
        self._tools = tools
        # DEBUG log for successful initialization, confirming the number of tools loaded.
        logger.debug(f"[ManualToolProvider] Initialized with {len(self._tools)} predefined tools.")

    async def discover_tools(self, **kwargs: Any) -> List[Tool]:
        """
        Returns the predefined list of tool instances provided at initialization.

        Args:
            **kwargs: Not used by this provider.

        Returns:
            The list of Tool instances.
        """
        # DEBUG log for the discovery process, indicating the action and number of tools.
        logger.debug(f"[ManualToolProvider] Discovering tools: Returning {len(self._tools)} predefined tools.")
        return self._tools 