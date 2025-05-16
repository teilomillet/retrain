from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class Tool(ABC):
    """
    Abstract Base Class for all tools.

    Tools are components that an LLM can decide to call to perform specific actions
    or retrieve information from external systems.

    When implementing a custom tool, its `__init__` method should accept `name` and
    `description` arguments (typically passed by the system during instantiation based
    on registration information or provider logic) and any tool-specific configuration
    parameters. The `__init__` should then call `super().__init__(name, description)`.
    For example:
    ```python
    class MyCustomTool(Tool):
        def __init__(self, name: str, description: str, api_key: str, timeout: int = 30):
            super().__init__(name, description)
            self.api_key = api_key
            self.timeout = timeout
    ```
    """

    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        """The unique name of the tool, used by the LLM to identify it."""
        return self._name

    @property
    def description(self) -> str:
        """A description of what the tool does, used to inform the LLM."""
        return self._description

    @abstractmethod
    async def execute(self, tool_input: Optional[Dict[str, Any]]) -> Any:
        """
        Executes the tool with the given input.

        Args:
            tool_input: A dictionary containing parameters for the tool, 
                        as specified by the LLM.

        Returns:
            Any: The result of the tool's execution. This should be serializable 
                 to be included in the EnvironmentObservation.

        Raises:
            ToolExecutionError: If the tool encounters an error during execution.
        """
        pass 

    @abstractmethod
    async def get_schema(self) -> Dict[str, Any]:
        """
        Returns a schema describing the tool, its purpose, and its input parameters.

        The schema should ideally follow JSON Schema conventions for describing parameters
        to provide a standardized way for LLMs to understand how to use the tool.

        Example structure:
        ```json
        {
            "name": "my_tool_name",
            "description": "A brief description of what the tool does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "Description of param1."},
                    "param2": {"type": "integer", "description": "Description of param2."}
                },
                "required": ["param1"]
            }
        }
        ```

        Returns:
            Dict[str, Any]: A dictionary representing the tool's schema.
        """
        pass 