from typing import List, Any, TYPE_CHECKING, Dict
from loguru import logger # Added for logging

from ..tool_provider import ToolProvider # Up one level to retrain.environment.tool.providers.base
from ...tool import Tool # Up three levels to retrain.environment.tool.base
from .wrapper import GenericFastMCPWrapper

if TYPE_CHECKING:
    # As in wrapper.py, placeholder for actual FastMCPClient type
    # from fastmcp.client import Client as FastMCPClientType
    FastMCPClientType = Any # Placeholder type
    # Also, placeholder for the type of items returned by fastmcp_client.list_tools()
    # Assuming it's a list of objects/structs with 'name' and 'description' attributes.
    MCPToolInfoType = Any # Placeholder type

class FastMCPToolProvider(ToolProvider):
    """
    A ToolProvider that discovers tools from a FastMCP server instance.
    It uses GenericFastMCPWrapper to adapt these tools to the BaseTool interface.
    """
    def __init__(self, client: 'FastMCPClientType', prefix: str = "fastmcp_"):
        if client is None:
            # This ValueError is good: clear, direct, and prevents operation with invalid state.
            raise ValueError("FastMCPToolProvider requires a valid FastMCP client instance.")
        self._fastmcp_client = client
        self.prefix = prefix # Initialize prefix attribute

    async def discover_tools(self) -> List[Tool]:
        """
        Discovers tools from the FastMCP server by calling its list_tools method.
        Wraps them in FastMCPToolWrapper.
        Returns a list of Tool instances.
        """
        discovered_tools_dict: Dict[str, Tool] = {}
        if not self._fastmcp_client:
            logger.warning("[FastMCPToolProvider] No FastMCP client configured; cannot discover tools.")
            return [] # Return empty list

        try:
            # Downgraded from INFO: Routine operation start.
            logger.debug(f"[FastMCPToolProvider] Attempting to discover tools from FastMCP client: {self._fastmcp_client}")
            async with self._fastmcp_client: # Ensure client connection context
                raw_tools = await self._fastmcp_client.list_tools()
                # Downgraded from INFO: Expected successful step.
                logger.debug(f"[FastMCPToolProvider] Received {len(raw_tools)} raw tool definitions from server.")
                
                for tool_info in raw_tools:
                    tool_name = getattr(tool_info, 'name', None)
                    if not tool_name:
                        logger.warning(f"[FastMCPToolProvider] Discovered a tool definition with no name, skipping: {tool_info}")
                        continue

                    prefixed_tool_name = f"{self.prefix}{tool_name}"
                    
                    tool_parameters_schema = None
                    schema_source_log_message = "unknown (defaulted to empty)" 
                    schema_obtained_via_primary_method = False

                    raw_input_schema_object = getattr(tool_info, 'inputSchema', None)

                    if isinstance(raw_input_schema_object, dict):
                        if raw_input_schema_object.get('type') or raw_input_schema_object.get('properties') or raw_input_schema_object.get('anyOf'):
                            tool_parameters_schema = raw_input_schema_object
                            schema_source_log_message = "direct from tool_info.inputSchema (dict)"
                            schema_obtained_via_primary_method = True
                            if tool_parameters_schema.get('title') == 'Inputschema': 
                                tool_parameters_schema.pop('title')
                    elif raw_input_schema_object is not None: 
                        has_model_json_schema_attr = hasattr(raw_input_schema_object, 'model_json_schema')
                        if has_model_json_schema_attr:
                            model_json_schema_method = getattr(raw_input_schema_object, 'model_json_schema')
                            if callable(model_json_schema_method):
                                try:
                                    tool_parameters_schema = raw_input_schema_object.model_json_schema()
                                    if isinstance(tool_parameters_schema, dict) and tool_parameters_schema.get('title') == 'Inputschema':
                                        tool_parameters_schema.pop('title')
                                    schema_source_log_message = "via tool_info.inputSchema.model_json_schema()"
                                    # This is a primary path if inputSchema is a Pydantic model.
                                    schema_obtained_via_primary_method = True 
                                except Exception as e_is_model:
                                    logger.warning(f"[FastMCPToolProvider] Tool '{tool_name}': Error calling model_json_schema() on inputSchema (type: {type(raw_input_schema_object)}): {e_is_model}. Will attempt other schema sources.")

                    if tool_parameters_schema is None: # Indicates primary methods above didn't yield a schema or raw_input_schema_object was None
                        schema_obtained_via_primary_method = False # Explicitly mark as not primary path if we reach here
                        if hasattr(tool_info, 'model_json_schema') and callable(getattr(tool_info, 'model_json_schema')):
                            try:
                                full_tool_definition_schema = tool_info.model_json_schema()
                                if isinstance(full_tool_definition_schema, dict) and \
                                   isinstance(full_tool_definition_schema.get('properties'), dict):
                                    potential_schema = full_tool_definition_schema['properties'].get('inputSchema')
                                    if isinstance(potential_schema, dict) and \
                                       (potential_schema.get("properties") or potential_schema.get("anyOf") or potential_schema.get("allOf") or potential_schema.get("oneOf")):
                                        tool_parameters_schema = potential_schema
                                        schema_source_log_message = "via tool_info.model_json_schema()['properties']['inputSchema'] (fallback)"
                            except Exception as e_pyd_fallback:
                                logger.warning(f"[FastMCPToolProvider] Tool '{tool_name}': Error processing tool_info.model_json_schema() for secondary fallback: {e_pyd_fallback}. Will attempt other schema sources.")

                    log_level_for_schema_source = logger.debug # Default to DEBUG for primary methods
                    if tool_parameters_schema is None:
                        tool_parameters_schema = {"type": "object", "properties": {}}
                        schema_source_log_message = "default empty schema (no valid source found)"
                        logger.warning(f"[FastMCPToolProvider] Tool '{tool_name}': No input schema found or derived; using default empty schema. Tool may not function as expected if parameters are required.")
                        log_level_for_schema_source = logger.info # INFO because a warning was issued
                    elif not schema_obtained_via_primary_method and "fallback" in schema_source_log_message:
                        # If schema was obtained via a fallback, log it as INFO as it's a deviation.
                        log_level_for_schema_source = logger.info 
                    
                    # Use the determined log level for schema source announcement.
                    log_level_for_schema_source(f"[FastMCPToolProvider] Tool '{tool_name}': Input schema resolution: {schema_source_log_message}.")
                        
                    wrapper = GenericFastMCPWrapper(
                        mcp_tool_name=tool_name,
                        mcp_tool_description=getattr(tool_info, 'description', ""),
                        fastmcp_client=self._fastmcp_client,
                        tool_interface_name=prefixed_tool_name, 
                        parameters_schema=tool_parameters_schema
                    )
                    discovered_tools_dict[prefixed_tool_name] = wrapper
                    # Downgraded from INFO: Routine successful wrapping of a tool.
                    logger.debug(f"[FastMCPToolProvider] Successfully wrapped tool '{tool_name}' (ID: '{prefixed_tool_name}').")

        except Exception as e:
            logger.error(f"[FastMCPToolProvider] Critical error during tool discovery from client ({type(self._fastmcp_client)}): {e}. Check client configuration and server status.")
        
        if not discovered_tools_dict:
            logger.warning(f"[FastMCPToolProvider] No tools were discovered or loaded from client: {self._fastmcp_client}. Please check server connection and ensure tools are published.")
        else:
            # Downgraded from INFO: Routine summary of successful operation.
            logger.debug(f"[FastMCPToolProvider] Successfully discovered and wrapped {len(discovered_tools_dict)} tools.")
        return list(discovered_tools_dict.values())

    async def execute_tool(
        self,
        tool_wrapper: Tool,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Executes a tool via the FastMCP client.
        The FastMCPToolWrapper should handle the actual call.
        """
        if not isinstance(tool_wrapper, GenericFastMCPWrapper):
            # This ValueError is good and retained.
            raise ValueError("execute_tool in FastMCPToolProvider expects a FastMCPToolWrapper.")

        # The client connection for execution should be handled by the wrapper's execute method
        # or the wrapper should be given a connected client.
        # For now, assuming the wrapper's execute method handles the 'async with self.client:'
        logger.info(f"[FastMCPToolProvider] Attempting to execute tool '{tool_wrapper._mcp_tool_name_actual}'.")
        # Logging arguments directly can be too verbose or expose sensitive data in production.
        # Consider logging keys or a hash if detailed argument info is needed for debugging common issues.
        # For now, removed arguments from this INFO log.
        try:
            # The wrapper's execute method needs to manage the client context for the call
            return await tool_wrapper.execute(arguments)
        except Exception as e:
            logger.error(f"[FastMCPToolProvider] Error executing tool '{tool_wrapper._mcp_tool_name_actual}': {e}. Ensure the tool server is operational and arguments are valid.")
            raise # Re-raise to allow higher-level error handling 