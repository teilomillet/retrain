# FastMCP Tool Integration (`retrain.environment.tool.providers.fastmcp`)

This subpackage provides the components necessary to integrate tools from a Fast Model Control Plane (FastMCP) system into `retrain` environments. It allows tools managed externally by a FastMCP server to be discovered and used as standard `BaseTool` instances within the `retrain` framework.

## Core Components

### 1. `FastMCPToolProvider` (in `provider.py`)

`FastMCPToolProvider` is a concrete implementation of `ToolProvider` designed to discover tools available on a FastMCP server.

**Key Features:**

*   **Inheritance**: It inherits from `retrain.environment.tool.providers.base.ToolProvider`.
*   **Initialization**: 
    *   `__init__(self, client: FastMCPClientType)`: The constructor expects an instance of a FastMCP client (the `FastMCPClientType` is currently an `Any` placeholder, representing what would typically be `fastmcp.client.Client`). This client is used to communicate with the FastMCP server.
*   **Tool Discovery**:
    *   `async def discover_tools(self, **kwargs: Any) -> List[BaseTool]`:
        *   This method is responsible for querying the FastMCP server for its available tools. It is intended to call a method like `list_tools()` on the provided `fastmcp_client`.
        *   For each tool description retrieved from FastMCP (assumed to have at least `name` and `description` attributes), it instantiates a `GenericFastMCPWrapper`.
        *   **Mocking Note**: In the current implementation, the actual call to `fastmcp_client.list_tools()` is commented out and replaced with a mock list of tool information objects. This facilitates development and testing without a live FastMCP server.
*   **Return Value**: It returns a list of `GenericFastMCPWrapper` instances, each representing a discovered FastMCP tool adapted to the `BaseTool` interface.

### 2. `GenericFastMCPWrapper` (in `wrapper.py`)

`GenericFastMCPWrapper` acts as an adapter class that makes an individual FastMCP tool conform to the `BaseTool` interface required by `retrain` environments.

**Key Features:**

*   **Inheritance**: It inherits from `retrain.environment.tool.base.BaseTool`.
*   **Initialization**:
    *   `__init__(self, mcp_tool_name: str, mcp_tool_description: str, fastmcp_client: FastMCPClientType, tool_interface_name: Optional[str] = None)`:
        *   `mcp_tool_name`: The actual name of the tool as it is known on the FastMCP server.
        *   `mcp_tool_description`: The description of the tool provided by the FastMCP server.
        *   `fastmcp_client`: The FastMCP client instance, used to execute the tool.
        *   `tool_interface_name`: An optional name for this tool within the `retrain` environment. If not provided, it defaults to `f"fastmcp_{mcp_tool_name}"`. This becomes the `name` property of the `BaseTool`.
*   **Tool Execution**:
    *   `async def execute(self, tool_input: Optional[Dict[str, Any]]) -> Any`:
        *   This method implements the actual execution logic by calling the FastMCP server.
        *   It is intended to use the `fastmcp_client` to invoke the specific FastMCP tool (identified by `self._mcp_tool_name_actual`) with the provided `tool_input` (which are the arguments for the FastMCP tool).
        *   The expected client method is `fastmcp_client.call_tool(name=..., arguments=...)`.
        *   It includes basic error handling, wrapping exceptions from the client call in a `ToolExecutionError`.
        *   **Mocking Note**: Similar to the provider, the actual call to `fastmcp_client.call_tool()` is currently commented out and replaced with a print statement and a mock result. It also contains placeholder logic for processing various content types that a real FastMCP tool might return (e.g., text, images).

## Mocking and Dependencies

Currently, this package does not have a hard dependency on a `fastmcp` library. The interactions with the `fastmcp.Client` are designed based on an anticipated interface and are **mocked directly within `provider.py` and `wrapper.py`**. 

*   Type hints like `FastMCPClientType` and `MCPToolInfoType` are used as placeholders (often `Any`).
*   Calls to `list_tools()` and `call_tool()` are simulated with print statements and example return structures.

This approach allows for the development and testing of the integration logic without requiring a live FastMCP instance or the `fastmcp` client library itself.

## Usage

To use this provider, an instance of `FastMCPToolProvider` would be created with a configured FastMCP client. This provider can then be passed to an environment, which would call `discover_tools()` to get a list of FastMCP tools ready for use.

```python
# Hypothetical usage (assuming a real or mock FastMCP client)
# from retrain.environment.tool.providers.fastmcp import FastMCPToolProvider

# class MockFastMCPClient:
#     async def list_tools(self):
#         print("[MockFastMCPClient] list_tools called")
#         return [
#             type('ToolInfo', (), {'name': 'example_mcp_tool', 'description': 'An example tool from MCP.'})()
#         ]
#     async def call_tool(self, name, arguments):
#         print(f"[MockFastMCPClient] call_tool called: {name} with {arguments}")
#         return f"Result from {name}"

# fastmcp_client_instance = MockFastMCPClient()
# provider = FastMCPToolProvider(client=fastmcp_client_instance)

# async def main():
#     tools = await provider.discover_tools()
#     for t in tools:
#         print(f"Discovered Tool: {t.name} - {t.description}")
#         if t.name == "fastmcp_example_mcp_tool":
#             result = await t.execute({"param": "value"})
#             print(f"Execution result: {result}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
```

This setup allows `retrain` to leverage externally managed toolsets via the FastMCP system, enhancing its capabilities through dynamic tool integration. 