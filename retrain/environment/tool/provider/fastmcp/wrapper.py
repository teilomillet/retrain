from typing import Optional, Dict, Any, TYPE_CHECKING
from loguru import logger # Added for logging

from ...tool import Tool # Up three levels to retrain.environment.tool.base
from ...error import ToolExecutionError # Up three levels to retrain.environment.tool.error

if TYPE_CHECKING:
    # Import for type hinting only to avoid circular dependencies or heavy imports at runtime
    # This assumes fastmcp.Client is the correct type from the fastmcp library
    # We don't have direct access to the fastmcp library's types here, so we use a string
    # or a type alias if it were defined globally for the project.
    # For now, let's use Any or a forward reference if possible, or just assume it's passed.
    # Since we don't have the actual fastmcp.Client type, we'll use Any for now in the type hint.
    # A better approach would be to have fastmcp library installed and import its client type.
    # As a placeholder for where the actual fastmcp client type would be imported:
    # from fastmcp.client import Client as FastMCPClientType
    FastMCPClientType = Any # Placeholder type

class GenericFastMCPWrapper(Tool):
    """
    A generic wrapper for tools discovered from a FastMCP server.
    This class adapts a FastMCP tool to the BaseTool interface.
    """
    def __init__(
        self,
        mcp_tool_name: str,        # The actual name of the tool on the MCP server
        mcp_tool_description: str, # Description from the MCP server for this tool
        fastmcp_client: 'FastMCPClientType', # The FastMCP client instance
        # The 'name' for BaseTool can be prefixed to avoid collisions, e.g., "fastmcp_" + mcp_tool_name
        # Or we can decide that the provider ensures unique names.
        # For now, let's assume the provider might pass a distinct 'name' for BaseTool registration.
        tool_interface_name: Optional[str] = None, # The name to register this wrapper under in BaseTool sense
        parameters_schema: Optional[Dict[str, Any]] = None # JSON schema for the tool's parameters
    ):
        # If tool_interface_name is not provided, use the mcp_tool_name, possibly prefixed.
        # This name is what our BaseTool interface will use.
        name_for_base_tool = tool_interface_name if tool_interface_name else f"fastmcp_{mcp_tool_name}"
        super().__init__(name=name_for_base_tool, description=mcp_tool_description)
        self._mcp_tool_name_actual = mcp_tool_name
        self._fastmcp_client = fastmcp_client
        self._parameters_schema = parameters_schema if parameters_schema else {"type": "object", "properties": {}}

    async def execute(self, tool_input: Optional[Dict[str, Any]]) -> Any:
        """
        Executes the wrapped FastMCP tool.

        Args:
            tool_input: Arguments for the FastMCP tool. The `fastmcp.Client.call_tool`
                        expects arguments as a dictionary.

        Returns:
            The processed result from `fastmcp_client.call_tool` (typically a string).

        Raises:
            ToolExecutionError: If `fastmcp_client.call_tool` raises an exception.
        """
        import mcp.types

        if not self._fastmcp_client:
            raise ToolExecutionError(
                f"FastMCP client not available for tool '{self.name}' (MCP tool: '{self._mcp_tool_name_actual}')."
            )
        
        mcp_arguments = tool_input if tool_input is not None else {}

        try:
            # Downgraded from INFO: Routine operation start.
            logger.debug(f"[GenericFastMCPWrapper] Attempting to execute MCP tool: '{self._mcp_tool_name_actual}'.")
            async with self._fastmcp_client as client: 
                raw_result_list = await client.call_tool(name=self._mcp_tool_name_actual, arguments=mcp_arguments)
            
                processed_parts = []
                if isinstance(raw_result_list, list):
                    for item in raw_result_list:
                        if isinstance(item, mcp.types.TextContent) and hasattr(item, 'text'):
                            processed_parts.append(item.text)
                        elif isinstance(item, mcp.types.ImageContent):
                            processed_parts.append(f"[ImageContent: {getattr(item, 'uri', 'no_uri')}]")
                        elif isinstance(item, mcp.types.EmbeddedResource):
                            processed_parts.append(f"[EmbeddedResource: {getattr(item, 'uri', 'no_uri')}]")
                        else:
                            processed_parts.append(str(item))
                
                    final_result = "\n".join(processed_parts) if processed_parts else None
                elif raw_result_list is None:
                    final_result = None 
                    logger.debug(f"[GenericFastMCPWrapper] MCP tool '{self._mcp_tool_name_actual}' returned no data (None).")
                else:
                    logger.warning(f"[GenericFastMCPWrapper] Unexpected raw result type from MCP tool '{self._mcp_tool_name_actual}': {type(raw_result_list)}. Attempting string conversion.")
                    final_result = str(raw_result_list)

            # Downgraded from INFO: Routine success logs.
            if final_result is not None:
                 logger.debug(f"[GenericFastMCPWrapper] Successfully executed MCP tool '{self._mcp_tool_name_actual}' and processed its output.")
            else:
                 # This case is already covered by the specific debug log for `raw_result_list is None` if that was the path.
                 # If processed_parts was empty but raw_result_list was an empty list, this is still valid.
                 logger.debug(f"[GenericFastMCPWrapper] Successfully executed MCP tool '{self._mcp_tool_name_actual}', which yielded no processable output.")
            return final_result
        
        except Exception as e:
            logger.error(f"Error executing FastMCP tool '{self.name}' (MCP tool: '{self._mcp_tool_name_actual}'): {e}", exc_info=True)
            raise ToolExecutionError(
                f"Error executing FastMCP tool '{self.name}' (MCP tool: '{self._mcp_tool_name_actual}'): {e}"
            ) from e 

    async def get_schema(self) -> Dict[str, Any]:
        """
        Returns a schema describing the tool, its purpose, and its input parameters,
        based on information from the FastMCP server.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._parameters_schema
        } 