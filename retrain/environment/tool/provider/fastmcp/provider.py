from typing import List, Any, TYPE_CHECKING, Dict
from loguru import logger # Added for logging

from ..tool_provider import ToolProvider # Up one level to retrain.environment.tool.providers.base
from ...tool import Tool # Up three levels to retrain.environment.tool.base
from .wrapper import GenericFastMCPWrapper

class ToolExecutionError(Exception):
    """Exception raised when tool execution fails."""
    pass

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
        
        # Manual schema definitions for common MCP tools that don't provide proper schemas.
        # This fixes the issue where MCP servers (like mcp-alchemy) don't provide input schemas,
        # causing LLMs to not know how to call the tools correctly.
        self._manual_schemas = {
            "all_table_names": {
                "type": "object",
                "properties": {},
                "description": "Returns all table names in the database. No parameters required."
            },
            "filter_table_names": {
                "type": "object", 
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "Substring to search for in table names"
                    }
                },
                "required": ["q"],
                "description": "Find tables matching a substring pattern"
            },
            "schema_definitions": {
                "type": "object",
                "properties": {
                    "table_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of table names to get schema definitions for"
                    }
                },
                "required": ["table_names"],
                "description": "Get detailed schema information for specified tables including columns, types, and relationships"
            },
            "execute_query": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "SQL query to execute"
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional parameters for parameterized queries",
                        "additionalProperties": True
                    }
                },
                "required": ["query"],
                "description": "Execute SQL query and return results in vertical format"
            }
        }

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
                    
                    # Improved schema extraction with better error handling and logging
                    tool_parameters_schema = None
                    schema_source_log_message = "unknown (defaulted to empty)" 
                    schema_obtained_via_primary_method = False
                    schema_extraction_errors = []  # Track errors for better debugging

                    raw_input_schema_object = getattr(tool_info, 'inputSchema', None)

                    # Primary method: Check if inputSchema is already a dict
                    if isinstance(raw_input_schema_object, dict):
                        if raw_input_schema_object.get('type') or raw_input_schema_object.get('properties') or raw_input_schema_object.get('anyOf'):
                            tool_parameters_schema = raw_input_schema_object
                            schema_source_log_message = "direct from tool_info.inputSchema (dict)"
                            schema_obtained_via_primary_method = True
                            if tool_parameters_schema.get('title') == 'Inputschema': 
                                tool_parameters_schema.pop('title')
                        else:
                            schema_extraction_errors.append("inputSchema dict exists but appears empty or invalid")
                    
                    # Alternative method: Check if inputSchema is a Pydantic model
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
                                    schema_obtained_via_primary_method = True 
                                except Exception as e_is_model:
                                    error_msg = f"Error calling model_json_schema() on inputSchema: {e_is_model}"
                                    schema_extraction_errors.append(error_msg)
                                    logger.warning(f"[FastMCPToolProvider] Tool '{tool_name}': {error_msg}")

                    # Fallback method: Try to extract from full tool model
                    if tool_parameters_schema is None:
                        schema_obtained_via_primary_method = False
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
                                    else:
                                        schema_extraction_errors.append("Fallback method found inputSchema but it appears invalid")
                                else:
                                    schema_extraction_errors.append("Fallback method: tool model_json_schema() invalid structure")
                            except Exception as e_pyd_fallback:
                                error_msg = f"Error in fallback model_json_schema(): {e_pyd_fallback}"
                                schema_extraction_errors.append(error_msg)
                                logger.warning(f"[FastMCPToolProvider] Tool '{tool_name}': {error_msg}")

                    # Final handling of schema extraction results
                    log_level_for_schema_source = logger.debug
                    if tool_parameters_schema is None:
                        # Check if we have a manual schema definition for this tool
                        # This addresses the common issue where MCP servers don't provide proper schemas
                        if tool_name in self._manual_schemas:
                            tool_parameters_schema = self._manual_schemas[tool_name].copy()
                            schema_source_log_message = "manual schema definition (MCP server provided no schema)"
                            log_level_for_schema_source = logger.info
                            logger.info(f"[FastMCPToolProvider] Tool '{tool_name}': Using manual schema definition to fix missing MCP server schema.")
                        else:
                            tool_parameters_schema = {"type": "object", "properties": {}}
                            schema_source_log_message = "default empty schema (no valid source found)"
                            
                            # Enhanced warning with extraction error details
                            warning_msg = (
                                f"[FastMCPToolProvider] Tool '{tool_name}': No input schema found or derived; "
                                f"using default empty schema. Tool may not function as expected if parameters are required."
                            )
                            if schema_extraction_errors:
                                warning_msg += f" Extraction errors: {'; '.join(schema_extraction_errors)}"
                            
                            logger.warning(warning_msg)
                            log_level_for_schema_source = logger.info
                    elif not schema_obtained_via_primary_method and "fallback" in schema_source_log_message:
                        log_level_for_schema_source = logger.info 
                    
                    # Enhanced logging with schema validation
                    schema_info_msg = f"[FastMCPToolProvider] Tool '{tool_name}': Input schema resolution: {schema_source_log_message}."
                    if tool_parameters_schema and tool_parameters_schema.get("properties"):
                        param_names = list(tool_parameters_schema["properties"].keys())
                        schema_info_msg += f" Parameters: {param_names}"
                    else:
                        schema_info_msg += " No parameters defined."
                    
                    log_level_for_schema_source(schema_info_msg)
                        
                    wrapper = GenericFastMCPWrapper(
                        mcp_tool_name=tool_name,
                        mcp_tool_description=getattr(tool_info, 'description', ""),
                        fastmcp_client=self._fastmcp_client,
                        tool_interface_name=prefixed_tool_name, 
                        parameters_schema=tool_parameters_schema
                    )
                    discovered_tools_dict[prefixed_tool_name] = wrapper
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
            # Fix: Use safer string formatting to avoid issues with quotes in exception messages
            # The original f-string failed when exception message contained {'param': 'CRYPTO'} 
            # because single quotes were interpreted as format string delimiters
            # This happens when MCP server returns validation errors with quoted parameter names
            error_msg = "Error executing FastMCP tool '%s' (MCP tool: '%s'): %s" % (
                tool_wrapper.name, 
                tool_wrapper._mcp_tool_name_actual, 
                str(e)
            )
            logger.error(error_msg, exc_info=True)
            
            # Also use safer formatting for the ToolExecutionError message
            # to ensure consistency and avoid similar issues downstream
            safe_error_message = "Error executing FastMCP tool '%s' (MCP tool: '%s'): %s" % (
                tool_wrapper.name, 
                tool_wrapper._mcp_tool_name_actual, 
                str(e)
            )
            raise ToolExecutionError(safe_error_message) from e 

 